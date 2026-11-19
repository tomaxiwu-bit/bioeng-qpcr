"""Browser entry points reuse the desktop scientific engines and archive hooks."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass

import numpy as np
from flask import Blueprint, jsonify, request

from bioeng_qpcr.web.app import (
    _ensure_matplotlib,
    _plot_payload,
    _remove,
    _save_upload,
)

blueprint = Blueprint("kinetics", __name__)


def finite_json(value):
    """Undefined intervals/criteria stay undefined, never become NaN JSON or zero."""
    if is_dataclass(value):
        return finite_json(asdict(value))
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@blueprint.errorhandler(ValueError)
@blueprint.errorhandler(RuntimeError)
@blueprint.errorhandler(OSError)
def invalid(exc):
    return jsonify(ok=False, error=str(exc)), 400


@blueprint.errorhandler(ImportError)
def dependency(exc):
    return jsonify(ok=False, error=f"当前环境缺少分析依赖：{exc}"), 503


def choice(name, default, allowed):
    value = request.form.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name} 必须为 {', '.join(allowed)}")
    return value


def figure(title, fig, stem):
    if stem.startswith("fba-"):
        fig.axes[0].set_xlabel("Flux (model units)")
    return _plot_payload(title, fig, stem, publication=True)


def table(title, rows):
    return {"title": title, "columns": list(rows[0]) if rows else [], "rows": rows}


@blueprint.post("/api/enzyme")
def enzyme():
    from bioeng_qpcr.enzyme_kinetics import (
        compare_inhibition_models,
        fit_michaelis_menten,
        read_kinetics_csv,
        validate_km_range,
        validate_substrate_span,
    )
    from bioeng_qpcr.plotting.enzyme_kinetics import (
        plot_global_inhibition,
        plot_michaelis_menten,
    )

    _ensure_matplotlib()
    mode = choice("mode", "inhibition", ("mm", "inhibition"))
    path = _save_upload(frozenset({".csv"}))
    try:
        substrate, velocity, inhibitor = read_kinetics_csv(path)
    finally:
        _remove(path)
    if len(substrate) > 5000:
        raise ValueError("浏览器酶动力学分析最多接受 5000 行")
    checks, warnings = [], []
    if mode == "mm":
        if inhibitor is not None:
            control = inhibitor == 0
            if not np.any(control):
                raise ValueError(
                    "Michaelis–Menten 拟合需要 [I] = 0 的对照，不能混合有抑制剂的数据"
                )
            substrate, velocity = substrate[control], velocity[control]
            warnings.append("检测到抑制剂列；本次只拟合 [I] = 0 的对照")
        fit = fit_michaelis_menten(substrate, velocity)
        checks = [validate_km_range(fit.km), validate_substrate_span(substrate, fit.km)]
        warnings.extend(fit.warnings)
        parameters = [
            {
                "parameter": name,
                "estimate": getattr(fit, name),
                "ci_low": ci[0] if ci else None,
                "ci_high": ci[1] if ci else None,
            }
            for name, ci in (("vmax", fit.vmax_ci), ("km", fit.km_ci))
        ]
        rows = [
            {"substrate": s, "observed": v, "fitted": v - r, "residual": r}
            for s, v, r in zip(substrate, velocity, fit.residuals, strict=True)
        ]
        plots = [
            figure(
                "Michaelis–Menten 拟合",
                plot_michaelis_menten(fit, substrate, velocity)[0],
                "enzyme-mm",
            )
        ]
        metrics = [
            ["Vmax", fit.vmax],
            ["Km", fit.km],
            ["R²", fit.r_squared],
            ["数据点", fit.n_points],
        ]
        tables = [table("参数与 95% 置信区间", parameters)]
        details = fit
    else:
        if inhibitor is None or len(np.unique(inhibitor)) < 2:
            raise ValueError(
                "全局比较需要 inhibitor_concentration 列及至少两个抑制剂浓度"
            )
        if len(np.unique(inhibitor)) > 24:
            raise ValueError("全局比较最多接受 24 组抑制剂浓度")
        datasets = [
            (float(i), substrate[inhibitor == i], velocity[inhibitor == i])
            for i in np.unique(inhibitor)
        ]
        comparison = compare_inhibition_models(datasets)
        best = comparison.best_fit
        ambiguous = (
            comparison.models[1].delta_aicc < 2
            or best.akaike_weight < 0.7
            or any(
                not m.converged or not math.isfinite(m.aicc) for m in comparison.models
            )
        )
        warnings.extend(
            [
                comparison.message,
                "Akaike 权重仅表示候选模型间相对支持；参数 SE 为标准误，不是置信区间。",
            ]
        )
        if ambiguous:
            warnings.append(
                "机制未决：候选模型证据接近或部分模型未充分收敛，请补充数据"
            )
        if not np.any(inhibitor == 0):
            warnings.append("缺少 [I] = 0 对照，未抑制条件的参数依赖外推")
        rows = [
            {
                "model": m.inhibition_type.value,
                "aicc": m.aicc,
                "delta_aicc": m.delta_aicc,
                "bic": m.bic,
                "akaike_weight": m.akaike_weight,
                "converged": m.converged,
                **m.parameters,
            }
            for m in comparison.models
        ]
        parameters = [
            {"parameter": k, "estimate": v, "standard_error": best.parameter_std.get(k)}
            for k, v in best.parameters.items()
        ]
        tables = [table("候选模型比较", rows), table("最优候选参数", parameters)]
        metrics = [
            ["当前候选", best.inhibition_type.value],
            ["机制判定", "未决" if ambiguous else "当前数据支持"],
            ["观测数", comparison.n_points],
            ["候选模型", len(rows)],
        ]
        plots = [
            figure(
                "全局抑制拟合与残差",
                plot_global_inhibition(best, datasets)[0],
                "enzyme-global",
            )
        ]
        details = comparison
    return jsonify(
        finite_json(
            dict(
                ok=True,
                metrics=metrics,
                effective_parameters={"mode": mode},
                tables=tables,
                rows=rows,
                details=details,
                validation=checks,
                warnings=warnings,
                plots=plots,
            )
        )
    )


@blueprint.post("/api/ferment")
def ferment():
    from bioeng_qpcr.fermentation_kinetics import (
        analyze_identifiability,
        fit_monod,
        read_fermentation_csv,
        solve_batch_fermentation,
        validate_data_sufficiency,
        validate_mu_max_range,
    )
    from bioeng_qpcr.plotting import plot_fermentation_kinetics

    _ensure_matplotlib()
    path = _save_upload(frozenset({".csv"}))
    try:
        time, biomass, substrate = read_fermentation_csv(path)
    finally:
        _remove(path)
    if len(time) > 500:
        raise ValueError("浏览器 Monod 拟合最多接受 500 个时间点")
    sufficient = validate_data_sufficiency(time, biomass, substrate)
    if not sufficient.passed:
        raise ValueError(sufficient.message)
    fit = fit_monod(time, biomass, substrate)
    ident = analyze_identifiability(fit, time, biomass, substrate)
    xfit, sfit = solve_batch_fermentation(
        time, fit.x0, fit.s0, fit.mu_max, fit.ks, fit.yxs
    )
    rows = [
        dict(
            time=t,
            biomass=x,
            biomass_fit=xf,
            biomass_residual=x - xf,
            substrate=s,
            substrate_fit=sf,
            substrate_residual=s - sf,
        )
        for t, x, xf, s, sf in zip(time, biomass, xfit, substrate, sfit, strict=True)
    ]
    parameters = [
        {
            "parameter": name,
            "estimate": getattr(fit, name),
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
        }
        for name, ci in (
            ("mu_max", fit.mu_max_ci),
            ("ks", fit.ks_ci),
            ("yxs", fit.yxs_ci),
        )
    ]
    warnings = list(fit.warnings)
    if not ident.is_identifiable:
        warnings.append("参数可辨识性不足：" + ident.message)
    checks = [sufficient, validate_mu_max_range(fit.mu_max)]
    correlations = [
        {"parameter": name, **dict(zip(ident.parameter_names, row, strict=True))}
        for name, row in zip(
            ident.parameter_names, ident.correlation_matrix, strict=True
        )
    ]
    return jsonify(
        finite_json(
            dict(
                ok=True,
                rows=rows,
                details={"fit": fit, "identifiability": ident},
                validation=checks,
                warnings=warnings,
                metrics=[
                    ["μmax (h⁻¹)", fit.mu_max],
                    ["Ks (g/L)", fit.ks],
                    ["Yxs (g/g)", fit.yxs],
                    ["参数可辨识", "是" if ident.is_identifiable else "否"],
                ],
                tables=[
                    table("参数与 95% 置信区间", parameters),
                    table("参数相关性", correlations),
                    table("观测、拟合与残差", rows),
                ],
                plots=[
                    figure(
                        "Monod 生长与底物曲线",
                        plot_fermentation_kinetics(fit, time, biomass, substrate)[0],
                        "fermentation",
                    )
                ],
            )
        )
    )


def scenario_bounds(raw, model):
    """Bounds are explicit changes to a fresh model, never edits to its source file."""
    changes = []
    seen = set()
    for line in raw.replace("\n", ";").split(";"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] not in model.reactions or parts[0] in seen:
            raise ValueError(
                "情景约束须为 reaction_id,lower,upper；反应必须存在且不能重复"
            )
        low, high = float(parts[1]), float(parts[2])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("情景边界必须有限，且下界不能大于上界")
        changes.append((parts[0], low, high))
        seen.add(parts[0])
        if len(changes) > 20:
            raise ValueError("一次情景比较最多修改 20 个反应")
    return changes


@blueprint.post("/api/fba")
def fba():
    from bioeng_qpcr.metabolic_flux import (
        find_blocked_reactions,
        find_dead_end_metabolites,
        load_model,
        run_fba,
        validate_mass_balance,
        validate_objective_feasibility,
    )
    from bioeng_qpcr.plotting import plot_flux_distribution

    _ensure_matplotlib()
    action = choice("action", "fba", ("fba", "blocked", "dead_ends"))
    direction = choice("direction", "max", ("max", "min"))
    objective = request.form.get("objective", "").strip() or None
    path = _save_upload(frozenset({".xml", ".sbml", ".json"}))
    try:
        try:
            model = load_model(path)
        except ImportError:
            raise
        except Exception as exc:
            raise ValueError(
                "模型无法解析，请检查 SBML / COBRA JSON 文件：" + str(exc)
            ) from exc
    finally:
        _remove(path)
    if len(model.reactions) > 2000 or len(model.metabolites) > 3000:
        raise ValueError(
            "浏览器分析限 2000 个反应、3000 个代谢物，请用桌面页处理更大的模型"
        )
    changes = scenario_bounds(request.form.get("bounds", ""), model)
    if changes and action != "fba":
        raise ValueError("情景边界仅用于 FBA 基线对比；网络检查请清空情景边界")
    checks, plots, warnings = (
        [],
        [],
        [
            "FBA 通量是在指定目标和边界下的一组最优解；"
            "替代最优解可能不同，通量差异不等同于唯一生物学响应。"
        ],
    )
    if action == "fba":
        baseline = run_fba(model, objective=objective, direction=direction)
        result = baseline
        if changes:
            for identifier, low, high in changes:
                model.reactions.get_by_id(identifier).bounds = (low, high)
            result = run_fba(model, objective=objective, direction=direction)
        checks = [
            validate_objective_feasibility(result),
            validate_mass_balance(model, result),
        ]
        if baseline.status != "optimal":
            warnings.append("基线未得到最优解：" + baseline.status)
        rows = [
            {
                "reaction": identifier,
                "baseline_status": baseline.status,
                "scenario_status": result.status,
                "baseline_flux": baseline.fluxes.get(identifier)
                if baseline.status == "optimal"
                else None,
                "scenario_flux": flux if result.status == "optimal" else None,
                "difference": flux - baseline.fluxes[identifier]
                if result.status == baseline.status == "optimal"
                else None,
            }
            for identifier, flux in result.fluxes.items()
        ]
        metrics = [
            ["基线状态", baseline.status],
            ["当前状态", result.status],
            ["基线目标值", baseline.objective_value],
            ["当前目标值", result.objective_value],
        ]
        details = {"baseline": baseline, "scenario": result, "bound_changes": changes}
        if changes and baseline.status == "optimal":
            plots.append(
                figure(
                    "基线通量分布",
                    plot_flux_distribution(baseline, top_n=15)[0],
                    "fba-baseline",
                )
            )
        if result.status == "optimal":
            plots.append(
                figure(
                    "当前条件通量分布",
                    plot_flux_distribution(result, top_n=15)[0],
                    "fba-fluxes",
                )
            )
    elif action == "blocked":
        if len(model.reactions) > 300:
            raise ValueError(
                "浏览器阻塞反应检查限 300 个反应，请在桌面页检查更大的模型"
            )
        result = find_blocked_reactions(model, processes=1)
        rows = [{"reaction": identifier} for identifier in result.blocked_reactions]
        metrics = [["反应总数", result.n_total], ["阻塞反应", result.n_blocked]]
        details = result
    else:
        result = find_dead_end_metabolites(model)
        rows = [
            {"metabolite": identifier, **result.details.get(identifier, {})}
            for identifier in result.dead_end_ids
        ]
        metrics = [["代谢物总数", result.n_total], ["死端代谢物", result.n_dead_ends]]
        details = result
    return jsonify(
        finite_json(
            dict(
                ok=True,
                rows=rows,
                tables=[table("完整分析结果", rows)],
                effective_parameters={
                    "action": action,
                    "direction": direction,
                    "objective": objective or "",
                    "bounds": request.form.get("bounds", ""),
                },
                details=details,
                metrics=metrics,
                validation=checks,
                warnings=warnings,
                plots=plots,
            )
        )
    )
