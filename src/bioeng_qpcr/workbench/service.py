"""Experiment transitions, reproducible examples and portable reports."""

from __future__ import annotations

import copy
import csv
import hashlib
import html
import io
import json
from dataclasses import asdict
from pathlib import Path

from bioeng_qpcr.provenance import ProvenanceManifest
from bioeng_qpcr.workbench.acquisition import validate_measurements
from bioeng_qpcr.workbench.analysis import calculate_curves, quality_control, quantify
from bioeng_qpcr.workbench.model import (
    SECTION_STEP,
    STEPS,
    AnalysisSettings,
    AssayRecord,
    CurveSettings,
    InstrumentProgram,
    PlateAssignment,
    Project,
    QCPolicy,
    ReactionRecipe,
    ReagentComponent,
    SampleRecord,
    ThermalStep,
    now,
    number,
    text,
    validate_design,
    validate_settings,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "web" / "examples" / "qpcr_workbench"
DEMOS = {
    "relative-ct": "脑 / 肾 ΔΔCt · 24 个 Ct 观测",
    "rdml-dilution": "DNA 标准曲线 · 原始扩增与熔解",
}


def create_project(
    name: str, demo: str | None = None, template: str | None = None
) -> Project:
    if demo and template:
        raise ValueError("公开数据示例与空白实验模板不能同时选择")
    if template:
        from bioeng_qpcr.workbench.templates import planned_project

        project = planned_project(template)
    elif demo:
        if demo not in DEMOS:
            raise ValueError("未知的公开示例")
        project = Project.from_dict(
            json.loads((EXAMPLES / f"{demo}.json").read_text(encoding="utf-8"))[
                "project"
            ]
        )
        project.demo_key = demo
    else:
        project = Project(name=text(name, "实验名称"))
    project.record(
        "create",
        f"公开示例: {demo}"
        if demo
        else f"实验规划模板: {template}"
        if template
        else "新实验",
    )
    return project


def patch_project(project: Project, changes: dict) -> None:
    if not isinstance(changes, dict) or not changes:
        raise ValueError("请提供要保存的字段")
    allowed = set(SECTION_STEP) | {"name", "plate_format"}
    if set(changes) - allowed:
        raise ValueError("不允许修改运行、来源、历史或服务端分析结果")
    shapes = {"samples": SampleRecord, "assays": AssayRecord, "plate": PlateAssignment}
    configs = {
        "recipe": ReactionRecipe,
        "program": InstrumentProgram,
        "curve_settings": CurveSettings,
        "qc_policy": QCPolicy,
        "analysis_settings": AnalysisSettings,
    }
    for key, value in changes.items():
        if key in shapes:
            if not isinstance(value, list) or len(value) > 1536:
                raise ValueError(f"{key} 必须为有界列表")
            for row in value:
                if not isinstance(row, dict):
                    raise ValueError(f"{key} 每行必须为对象")
                normalized = asdict(shapes[key](**row))
                for name, item in normalized.items():
                    if name in {"quantity", "expected_tm"}:
                        if item is not None:
                            number(item, name, 0, 1e30)
                    elif name == "reference":
                        if not isinstance(item, bool):
                            raise ValueError("内参标记必须为布尔值")
                    else:
                        text(item, name, required=False)
        if key in configs:
            if not isinstance(value, dict):
                raise ValueError(f"{key} 必须为对象")
            configs[key](**value)
        if key == "plate_format" and (type(value) is not int or value not in (96, 384)):
            raise ValueError("板型只支持整数 96 或 384")
    affected = []
    for key, value in changes.items():
        if getattr(project, key) == value:
            continue
        setattr(project, key, copy.deepcopy(value))
        if key != "name":
            affected.append("design" if key == "plate_format" else SECTION_STEP[key])
    text(project.name, "实验名称")
    # Drafts may be incomplete; structural/field validation still occurs when
    # advancing each stage. Settings are always valid before they are saved.
    validate_settings(project)
    recipe = ReactionRecipe(**project.recipe)
    if not isinstance(recipe.components, list) or len(recipe.components) > 100:
        raise ValueError("反应组分必须为有界列表")
    for component in recipe.components:
        item = ReagentComponent(**component)
        text(item.name, "组分名称", required=False)
        number(item.volume_ul, "组分体积", 0, 1000)
    program = InstrumentProgram(**project.program)
    if not isinstance(program.steps, list) or len(program.steps) > 100:
        raise ValueError("程序步骤必须为有界列表")
    for step in program.steps:
        ThermalStep(**step)
    if not isinstance(program.channels, list) or any(
        not isinstance(c, str) for c in program.channels
    ):
        raise ValueError("通道必须为文本列表")
    if not isinstance(program.melt, dict):
        raise ValueError("熔解配置必须为对象")
    if affected:
        project.invalidate(min(affected, key=STEPS.index))
    project.record("edit", ", ".join(changes))


def advance(project: Project, step: str) -> None:
    if step not in STEPS:
        raise ValueError("未知工作流步骤")
    previous = STEPS[: STEPS.index(step)]
    if any(key not in project.completed for key in previous):
        raise ValueError("请先完成上游步骤；修改后的步骤需要重新确认")
    if step in {"samples", "design", "program"}:
        validate_design(project, through=step)
    elif step == "acquisition":
        if not project.active_run:
            project.start_run()
        return
    elif step == "curves":
        project.invalidate("curves")
        project.outputs["curves"] = calculate_curves(project)
    elif step == "qc":
        project.invalidate("qc")
        project.outputs["qc"] = quality_control(project)
        if project.outputs["qc"]["blocked"]:
            _archive_output(project, step)
            project.record("qc_blocked", "质控存在 FAIL；可导出问题报告")
            return
    elif step == "analysis":
        project.invalidate("analysis")
        project.outputs["analysis"] = quantify(project)
    elif step == "report":
        project.outputs["report"] = {
            "generated_at": now(),
            "run_id": project.active_run_id,
            "analysis_settings": copy.deepcopy(project.analysis_settings),
            "presentation": copy.deepcopy(project.presentation),
        }
    _archive_output(project, step)
    project.finish(step)


def _archive_output(project: Project, step: str) -> None:
    if step in project.outputs and project.active_run:
        run = project.active_run
        provenance = ProvenanceManifest.capture_run(
            input_hash=hashlib.sha256(json.dumps(
                {"snapshot": run["snapshot"], "measurements": run["measurements"]},
                sort_keys=True, ensure_ascii=False, allow_nan=False,
            ).encode()).hexdigest(),
            raw_input_hash=run["source"].get("sha256"),
            parameters={
                "step": step,
                "curve_settings": copy.deepcopy(project.curve_settings),
                "qc_policy": copy.deepcopy(project.qc_policy),
                "analysis_settings": copy.deepcopy(project.analysis_settings),
            },
        )
        project.active_run.setdefault("analysis_versions", []).append(
            {
                "project_revision": project.revision + 1,
                "at": now(),
                "step": step,
                "result": copy.deepcopy(project.outputs[step]),
                "provenance": provenance,
            }
        )


def acquire(project: Project, rows: list[dict], source: dict, *, mode: str) -> None:
    if any(key not in project.completed for key in STEPS[:3]):
        raise ValueError("请先确认样品、板位和上机程序")
    if not project.active_run:
        project.start_run()
    run = project.active_run
    if run["status"] != "waiting_data":
        raise ValueError("当前运行已有采集数据；请创建新运行，保留旧运行快照")
    rows = validate_measurements(project, rows)
    run.update(
        measurements=copy.deepcopy(rows),
        source=copy.deepcopy(source),
        mode=mode,
        status="collected",
        finished_at=now(),
    )
    project.invalidate("acquisition")
    project.finish("acquisition")


def replay(project: Project) -> None:
    if project.demo_key not in DEMOS:
        raise ValueError("只有公开示例实验可以回放；真实实验请上传文件")
    example = json.loads(
        (EXAMPLES / f"{project.demo_key}.json").read_text(encoding="utf-8")
    )
    if any(
        getattr(project, key) != example["project"][key]
        for key in ("samples", "assays", "plate", "plate_format")
    ):
        raise ValueError(
            "示例设计已修改，不能把原始数据重新标记成另一个实验；请新建公开示例或上传相符数据"
        )
    acquire(
        project,
        example["measurements"],
        example["project"]["sources"][0],
        mode="public_replay",
    )


def csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.lstrip().startswith(
                ("=", "+", "-", "@")
            ):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return "\ufeff" + output.getvalue()


def report_html(project: Project) -> str:
    """Self-contained report, including blocked runs and missing evidence."""
    from bioeng_qpcr.workbench.templates import THEMES

    if not project.active_run:
        raise ValueError("请先开始运行；无运行快照时不能生成实验报告")

    def escape(value):
        return html.escape(str(value), quote=True)

    def cell(value):
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        return "<td>" + escape(value) + "</td>"

    def table(rows):
        if not rows:
            return "<p>尚未计算 / 未提供</p>"
        keys = list(dict.fromkeys(key for row in rows for key in row))
        head = "".join(f"<th>{escape(key)}</th>" for key in keys)
        body = "".join(
            "<tr>" + "".join(cell(row.get(key, "")) for key in keys) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def block(value):
        return (
            "<pre>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"
        )

    qc = project.outputs.get("qc", {})
    analysis = project.outputs.get("analysis", {})
    curves = project.outputs.get("curves", {})
    status = "请按实验室 SOP 审阅"
    if qc.get("blocked"):
        status = "质控阻断：不得作为通过的定量结论"
    elif not qc or any(
        qc.get("counts", {}).get(s, 0) for s in ("WARN", "NOT_ASSESSABLE")
    ):
        status = "含证据缺口 / 待复核"
    run = project.active_run
    snapshot = run["snapshot"]
    curve_columns = (
        "well",
        "assay_id",
        "cq",
        "instrument_cq",
        "tm",
        "peaks",
        "ct_note",
    )
    sections = [
        f"<h1>{escape(project.name)}</h1><p>{escape(status)}</p>",
        f"<p>生成时间（UTC）：{escape(now())} · 项目版本：{project.revision}</p>",
        "<p>研究与教学用途；质控通过不等于 MIQE 完整合规。",
        "缺少对照、熔解或原始曲线时，不推断其通过。</p>",
        "<h2>运行快照</h2>",
        table(
            [
                {
                    k: run[k]
                    for k in (
                        "id",
                        "started_at",
                        "finished_at",
                        "mode",
                        "status",
                        "design_sha256",
                    )
                }
            ]
        ),
        "<h2>样品</h2>",
        table(snapshot["samples"]),
        "<h2>引物与探针</h2>",
        table(snapshot["assays"]),
        "<h2>反应体系与板位</h2>",
        block(snapshot["recipe"]),
        table(snapshot["plate"]),
        "<h2>温度程序与通道</h2>",
        block(snapshot["program"]),
        "<h2>图表主题</h2>",
        block(project.presentation),
        "<h2>曲线与定量图</h2>",
        _report_figures(project),
        "<h2>Ct / 熔解结果与设置</h2>",
        block(curves.get("settings", {})),
        table([{k: row[k] for k in curve_columns} for row in curves.get("rows", [])]),
        "<h2>质控与阈值</h2>",
        block(qc.get("policy", {})),
        table(qc.get("checks", [])),
        "<h2>定量分析</h2>",
        block({k: v for k, v in analysis.items() if k != "rows"}),
        table(analysis.get("rows", [])),
        "<h2>来源与许可</h2>",
        table(project.sources + [run["source"]]),
        "<h2>审计记录</h2>",
        "<h3>各步骤的运行环境与输入摘要</h3>",
        block([
            {"step": v.get("step", "未知步骤"), "at": v.get("at", "未知时间"),
             "provenance": v.get("provenance", "旧记录未记录软件版本"),
             "imported_backup": v.get("imported_backup", False)}
            for v in run.get("analysis_versions", [])
        ]),
        table(project.audit),
    ]
    theme = THEMES[project.presentation["theme"]]
    themed_css = (
        f"body{{background:{theme['background']};color:{theme['ink']}}}"
        f"h1{{border-bottom:3px solid {theme['palette'][0]};padding-bottom:16px}}"
        f"h2{{color:{theme['palette'][0]}}}th,td{{border-color:{theme['grid']}}}"
        f"pre{{border-left:3px solid {theme['palette'][1]};padding:10px 15px}}"
    )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        "<title>qPCR 实验报告</title><style>"
        "body{font:14px/1.6 system-ui;margin:36px;color:#24352e}"
        "table{border-collapse:collapse;width:100%;font-size:12px;word-break:break-word}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{margin-top:30px}"
        "figure img{max-width:100%}figure{margin:18px 0}"
        "@media print{button{display:none}table{font-size:9px}}"
        + themed_css
        + '</style><button onclick="window.print()">打印 / 保存 PDF</button>'
        + "".join(sections)
        + "</html>"
    )


def _report_figures(project: Project) -> str:
    """The report and standalone downloads share the same plotting templates."""
    import base64

    from bioeng_qpcr.workbench.figures import figure_bytes

    figures = []
    available = []
    if any(m["amplification"] for m in project.active_run["measurements"]):
        available.append(("amplification", "原始扩增曲线"))
    if any(
        row["derivative"] for row in project.outputs.get("curves", {}).get("rows", [])
    ):
        available.append(("melting", "熔解导数；描述性峰分析，不是产物特异性认证"))
    analysis = project.outputs.get("analysis")
    if analysis:
        available.append(
            (
                "standard" if analysis["mode"] == "absolute" else "relative",
                "定量分析；技术重复汇总后的点估计",
            )
        )
    for kind, title in available:
        encoded = base64.b64encode(figure_bytes(project, kind, "svg")).decode("ascii")
        figures.append(
            '<figure><img alt="'
            + html.escape(title, quote=True)
            + '" src="data:image/svg+xml;base64,'
            + encoded
            + '"><figcaption>'
            + html.escape(title)
            + "</figcaption></figure>"
        )
    return "".join(figures) or "<p>尚无可绘制数据；不生成占位曲线。</p>"
