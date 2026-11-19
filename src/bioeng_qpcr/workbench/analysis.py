"""Curve processing and SOP-based QC; quantification reuses the existing engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from bioeng_qpcr.calculations.delta_delta_cq import delta_delta_cq
from bioeng_qpcr.calculations.efficiency import fit_efficiency_from_standard_curve
from bioeng_qpcr.calculations.normalized_quantity import normalized_quantity
from bioeng_qpcr.domain import Assay, Experiment, Sample, Well
from bioeng_qpcr.workbench.model import AnalysisSettings, Project, validate_settings


def threshold_ct(points: list, settings: dict) -> tuple[float | None, str]:
    """Baseline-subtracted, interpolated threshold crossing (not an inflection)."""
    if len(points) < 4:
        return None, "未提供足够的原始扩增点"
    values = np.asarray(points, dtype=float)
    values = values[np.argsort(values[:, 0])]
    x, y = values.T
    if np.any(np.diff(x) <= 0):
        raise ValueError("扩增周期必须唯一且递增")
    baseline = (x >= settings["baseline_start"]) & (x <= settings["baseline_end"])
    if baseline.sum() < 3:
        return None, "基线区间不足三个数据点"
    corrected = y - float(np.mean(y[baseline]))
    threshold = settings["threshold"]
    for i in range(1, len(x)):
        if (
            x[i - 1] >= settings["baseline_end"]
            and corrected[i - 1] < threshold <= corrected[i]
        ):
            ct = x[i - 1] + (threshold - corrected[i - 1]) / (
                corrected[i] - corrected[i - 1]
            ) * (x[i] - x[i - 1])
            return float(ct), "基线校正后的线性插值阈值 Ct"
    return None, "基线结束后未出现有效阈值穿越；不以最大周期代替未检出"


def melting_peaks(points: list, relative_prominence: float) -> dict:
    """Descriptive -dF/dT peaks; missing curves are never synthesized."""
    if len(points) < 5:
        return {
            "tm": None,
            "peaks": [],
            "derivative": [],
            "note": "未提供足够的原始熔解点",
        }
    grouped: dict[float, list[float]] = defaultdict(list)
    for temperature, fluorescence in points:
        grouped[temperature].append(fluorescence)
    temperatures = np.array(sorted(grouped), dtype=float)
    fluorescence = np.array([np.mean(grouped[t]) for t in temperatures], dtype=float)
    if len(temperatures) < 5:
        return {"tm": None, "peaks": [], "derivative": [], "note": "独立温度点不足"}
    # Fixed, disclosed smoothing; not the full RDML/LinRegPCR MCA algorithm.
    smoothed = (
        savgol_filter(fluorescence, 7, 2) if len(temperatures) >= 7 else fluorescence
    )
    derivative = -np.gradient(smoothed, temperatures)
    height = float(np.max(derivative))
    peaks, _ = find_peaks(
        derivative,
        prominence=max(height, 0) * relative_prominence,
        height=max(height, 0) * relative_prominence,
        distance=3,
    )
    peaks = (
        sorted(peaks, key=lambda index: derivative[index], reverse=True)
        if height > 0
        else []
    )
    return {
        "tm": float(temperatures[peaks[0]]) if len(peaks) else None,
        "peaks": [
            {"temperature_c": float(temperatures[i]), "height": float(derivative[i])}
            for i in peaks
        ],
        "derivative": np.column_stack((temperatures, derivative)).tolist(),
        "note": (
            "按温度排序 / 重复温度取均值；7 点二阶 Savitzky–Golay 平滑后计算 -dF/dT；"
            "峰不等于产物特异性认证"
        ),
    }


def calculate_curves(project: Project) -> dict:
    validate_settings(project)
    run = project.active_run
    if not run or not run["measurements"]:
        raise ValueError("请先完成采集或文件导入")
    rows = []
    for measurement in run["measurements"]:
        if project.curve_settings["ct_method"] == "instrument":
            ct = measurement["instrument_cq"]
            note = "沿用源文件报告 Ct；缺失时不回填或猜测"
        else:
            ct, note = threshold_ct(
                measurement["amplification"], project.curve_settings
            )
        melt = melting_peaks(
            measurement["melting"], project.curve_settings["melt_prominence"]
        )
        rows.append(
            {
                "well": measurement["well"],
                "assay_id": measurement["assay_id"],
                "cq": ct,
                "ct_note": note,
                "instrument_cq": measurement["instrument_cq"],
                **melt,
            }
        )
    return {"run_id": run["id"], "settings": dict(project.curve_settings), "rows": rows}


def quality_control(project: Project) -> dict:
    rows = project.outputs["curves"]["rows"]
    policy = project.qc_policy
    samples = {row["id"]: row for row in project.samples}
    assays = {row["id"]: row for row in project.assays}
    assignments = {(row["well"], row["assay_id"]): row for row in project.plate}
    checks, groups = [], defaultdict(list)

    def add(rule: str, status: str, subject: str, message: str, evidence: Any = None):
        checks.append(
            {
                "rule": rule,
                "status": status,
                "subject": subject,
                "message": message,
                "evidence": evidence,
            }
        )

    measured = {(row["well"], row["assay_id"]): row for row in rows}
    for key, assignment in assignments.items():
        sample, assay = samples[assignment["sample_id"]], assays[key[1]]
        row = measured.get(key)
        if row is None:
            add("DATA", "FAIL", key[0], "设计孔没有采集记录，不静默排除")
            continue
        groups[(sample["id"], assay["id"])].append(row)
        if sample["kind"] in {"ntc", "negative", "nrt"}:
            status = (
                "PASS"
                if row["cq"] is None
                else "FAIL"
                if row["cq"] <= policy["ntc_max_cq"]
                else "WARN"
            )
            add(
                "NEGATIVE",
                status,
                f"{key[0]} / {assay['name']}",
                "阴性对照：未检出通过；阈值以内扩增失败；晚期扩增需复核",
                {"cq": row["cq"], "sop_cutoff": policy["ntc_max_cq"]},
            )
        elif sample["kind"] == "positive":
            good = (
                row["cq"] is not None
                and policy["positive_min_cq"] <= row["cq"] <= policy["positive_max_cq"]
            )
            add(
                "POSITIVE",
                "PASS" if good else "FAIL",
                key[0],
                "阳性对照必须落在 SOP 声明范围内",
                {
                    "cq": row["cq"],
                    "range": [policy["positive_min_cq"], policy["positive_max_cq"]],
                },
            )
        elif row["cq"] is None:
            add("DETECTION", "WARN", key[0], "未检出；定量时会保留缺失原因")
        if assay["chemistry"] == "dye":
            if row["tm"] is None:
                add("MELT", "NOT_ASSESSABLE", key[0], "缺少可用熔解峰，不能判定特异性")
            elif len(row["peaks"]) > 1:
                add(
                    "MELT",
                    "WARN",
                    key[0],
                    "存在多个显著熔解峰，需结合产物与阴性对照复核",
                    row["peaks"],
                )
            elif assay["expected_tm"] is not None:
                good = abs(row["tm"] - assay["expected_tm"]) <= policy["tm_tolerance_c"]
                add(
                    "MELT",
                    "PASS" if good else "FAIL",
                    key[0],
                    "主峰 Tm 与检测项目预期值比较",
                    {"tm": row["tm"], "expected": assay["expected_tm"]},
                )
            else:
                add(
                    "MELT",
                    "WARN",
                    key[0],
                    "已找到单主峰，但未声明预期 Tm，特异性仍需人工复核",
                    {"tm": row["tm"]},
                )
    for aid, assay in assays.items():
        used_samples = [
            samples[a["sample_id"]] for a in project.plate if a["assay_id"] == aid
        ]
        for rule, kinds in (
            ("NEGATIVE", {"ntc", "negative"}),
            ("POSITIVE", {"positive"}),
        ):
            if not any(s["kind"] in kinds for s in used_samples):
                add(
                    rule,
                    "NOT_ASSESSABLE",
                    assay["name"],
                    "板设计未包含该类对照；不以普通样品代替",
                )
        if (
            not assay["forward"]
            or not assay["reverse"]
            or (assay["chemistry"] == "probe" and not assay["probe"])
        ):
            add(
                "ASSAY_METADATA",
                "WARN",
                assay["name"],
                "引物 / 探针信息不完整，报告保留证据缺口",
            )
    for (sid, aid), group in groups.items():
        if samples[sid]["kind"] in {"ntc", "negative", "nrt"}:
            continue
        values = [row["cq"] for row in group if row["cq"] is not None]
        sd = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        status = (
            "WARN"
            if sd is None or len(values) != len(group)
            else "FAIL"
            if sd > policy["replicate_max_sd"]
            else "PASS"
        )
        add(
            "REPLICATES",
            status,
            f"{samples[sid]['name']} / {assays[aid]['name']}",
            "同一样品 / 检测项目的技术重复一致性",
            {
                "n_total": len(group),
                "n_detected": len(values),
                "sd": sd,
                "sop_max_sd": policy["replicate_max_sd"],
            },
        )
    counts = {
        status: sum(c["status"] == status for c in checks)
        for status in ("PASS", "WARN", "FAIL", "NOT_ASSESSABLE")
    }
    return {
        "checks": checks,
        "counts": counts,
        "blocked": counts["FAIL"] > 0,
        "policy": dict(policy),
        "run_id": project.active_run_id,
        "note": "SOP 质控结果，不是 MIQE 完整合规认证；缺失证据不算通过",
    }


def quantify(project: Project) -> dict:
    settings = asdict(AnalysisSettings(**project.analysis_settings))
    validate_settings(project)
    if project.outputs["qc"]["blocked"]:
        raise ValueError("质控存在 FAIL，请先解决问题再生成定量结论；问题报告仍可导出")
    target = settings["target_id"]
    assays = {a["id"]: a for a in project.assays}
    if target not in assays:
        raise ValueError("请选择有效的目标检测项目")
    assignments = {(a["well"], a["assay_id"]): a for a in project.plate}
    curves = project.outputs["curves"]["rows"]
    grouped = defaultdict(list)
    for row in curves:
        sid = assignments[(row["well"], row["assay_id"])]["sample_id"]
        grouped[(sid, row["assay_id"])].append(row["cq"])
    warnings = ["技术重复先汇总；结果不构成组间生物学统计推断"]
    if any(project.outputs["qc"]["counts"][s] for s in ("WARN", "NOT_ASSESSABLE")):
        warnings.append("质控有警告或无法评估项目，报告需连同证据缺口审阅")
    if settings["mode"] == "relative":
        corrected = settings["relative_method"] == "efficiency_corrected"
        refs = settings["reference_ids"] or [settings["reference_id"]]
        if any(
            r not in assays or r == target or not assays[r]["reference"] for r in refs
        ):
            raise ValueError("请选择与目标不同、已标记为内参的检测项目")
        if not corrected and not settings["efficiency_confirmed"]:
            raise ValueError(
                "ΔΔCt 需要显式确认目标与所有内参效率接近且约为 100%；"
                "否则请选择效率校正并提供实测效率"
            )
        efficiencies = {}
        for aid in (target, *refs):
            evidence = settings["efficiencies"].get(aid, {})
            if corrected and (
                evidence.get("value") is None or not evidence.get("source", "").strip()
            ):
                raise ValueError(f"{assays[aid]['name']} 缺少实测扩增效率或效率依据")
            efficiencies[aid] = evidence["value"] if corrected else 1.0
            if corrected and not 0.9 <= efficiencies[aid] <= 1.1:
                warnings.append(
                    f"{assays[aid]['name']} 的实测效率不在演示参考范围 90%–110%，"
                    "请按 SOP 复核"
                )
        warnings.append("多内参不等于已证明稳定性；内参适用性与效率依据需由实验者审阅")
        eligible = {s["id"]: s for s in project.samples if s["kind"] == "unknown"}
        calibrator = settings["calibrator_id"]
        if calibrator not in eligible:
            raise ValueError("校准样品必须是普通生物样品，而非阴性或标准品")
        wells, exclusions = [], {}
        for sid in eligible:
            for aid in (target, *refs):
                values = grouped.get((sid, aid), [])
                if not values or any(v is None for v in values):
                    exclusions[sid] = (
                        "目标或内参数据缺失 / 含未检出技术重复；未静默剔除孔"
                    )
                else:
                    wells.append(
                        Well(
                            position=f"{sid}:{aid}",
                            sample_name=sid,
                            target_name=aid,
                            cq=float(np.mean(values)),
                        )
                    )
        if calibrator in exclusions:
            raise ValueError("校准样品的目标或内参数据不完整")
        experiment = Experiment(
            name=project.name,
            wells=wells,
            samples={
                sid: Sample(name=sid, group=s.get("group") or None)
                for sid, s in eligible.items()
            },
            assays={
                aid: Assay(
                    name=aid, efficiency=efficiencies[aid], is_reference=aid in refs
                )
                for aid in (target, *refs)
            },
        )
        results = []
        try:
            baseline = normalized_quantity(experiment, calibrator, target, refs)
            for sid, sample in eligible.items():
                if sid in exclusions:
                    results.append(
                        {
                            "sample_id": sid,
                            "sample": sample["name"],
                            "status": "excluded",
                            "reason": exclusions[sid],
                        }
                    )
                    continue
                quantity = normalized_quantity(experiment, sid, target, refs)
                log_fold = (
                    quantity.log2_normalized_quantity
                    - baseline.log2_normalized_quantity
                )
                fold = 2.0**log_fold
                if (
                    not np.isfinite(fold)
                    or fold <= 0
                    or quantity.normalized_quantity <= 0
                ):
                    raise ValueError("相对定量超出数值范围，请检查 Ct 与效率")
                details = asdict(quantity)
                if not corrected:
                    delta = quantity.mean_cq_target - float(
                        np.mean(list(quantity.mean_cq_references.values()))
                    )
                    details.update(delta_cq=delta, delta_delta_cq=-log_fold)
                    if len(refs) == 1:
                        details.update(
                            asdict(
                                delta_delta_cq(
                                    experiment, sid, target, calibrator, refs[0]
                                )
                            )
                        )
                results.append(
                    {
                        **details,
                        "sample_id": sid,
                        "sample": sample["name"],
                        "status": "ok",
                        "fold_change": fold,
                        "log2_fold_change": log_fold,
                    }
                )
        except OverflowError as exc:
            raise ValueError("相对定量超出数值范围，请检查 Ct 与效率") from exc
        return {
            "mode": "relative",
            "relative_method": settings["relative_method"],
            "method": "效率校正标准化量 / 校准样品；多内参数量取几何均值"
            if corrected
            else "2^(-ΔΔCt), Livak；多内参在等效率假设下合并",
            "reference_ids": refs,
            "rows": results,
            "warnings": warnings,
            "settings": settings,
            "run_id": project.active_run_id,
        }

    standards = [
        s
        for s in project.samples
        if s["kind"] == "standard" and (s["id"], target) in grouped
    ]
    if len({s["quantity"] for s in standards}) < 3:
        raise ValueError("绝对定量至少需要三个不同正标准量水平")
    units = {s["unit"].strip() for s in standards}
    if len(units) != 1:
        raise ValueError("标准品单位必须一致；请先显式换算")
    concentrations, values = [], []
    for standard in standards:
        observations = grouped[(standard["id"], target)]
        if any(v is None for v in observations):
            raise ValueError("标准曲线含未检出重复孔；请审查标准范围，不能静默删除")
        concentrations.extend([standard["quantity"]] * len(observations))
        values.extend(observations)
    fit = fit_efficiency_from_standard_curve(
        np.log10(concentrations), np.asarray(values), target_name=target
    )
    if fit.r_squared < 0.98 or not 0.9 <= fit.efficiency <= 1.1:
        warnings.append(
            "标准曲线 R² < 0.98 或效率不在演示参考范围 90%–110%，请按实验室 SOP 复核"
        )
    lower, upper = min(concentrations), max(concentrations)
    results = []
    for sample in project.samples:
        if sample["kind"] != "unknown":
            continue
        observations = grouped.get((sample["id"], target), [])
        if not observations or any(v is None for v in observations):
            results.append(
                {
                    "sample_id": sample["id"],
                    "sample": sample["name"],
                    "status": "excluded",
                    "reason": "缺少 Ct 或存在未检出的技术重复",
                }
            )
            continue
        mean_ct = float(np.mean(observations))
        log_quantity = (mean_ct - fit.intercept) / fit.slope
        if not -300 <= log_quantity <= 300:
            raise ValueError("标准曲线反算超出数值范围，请检查标准量、斜率与未知样品")
        quantity = float(10**log_quantity)
        results.append(
            {
                "sample_id": sample["id"],
                "sample": sample["name"],
                "mean_cq": mean_ct,
                "quantity": quantity,
                "unit": next(iter(units)),
                "status": "extrapolated" if not lower <= quantity <= upper else "ok",
            }
        )
    return {
        "mode": "absolute",
        "method": "Ct = slope × log10(quantity) + intercept",
        "rows": results,
        "standard_curve": asdict(fit),
        "standard_points": list(zip(concentrations, values, strict=True)),
        "calibration_range": [lower, upper],
        "unit": next(iter(units)),
        "warnings": warnings,
        "settings": dict(settings),
        "run_id": project.active_run_id,
    }
