"""Flask routes for the persistent qPCR experiment workbench."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from bioeng_qpcr.workbench.acquisition import read_acquisition
from bioeng_qpcr.workbench.model import STEPS, now
from bioeng_qpcr.workbench.service import (
    DEMOS,
    acquire,
    advance,
    create_project,
    csv_text,
    patch_project,
    replay,
    report_html,
)
from bioeng_qpcr.workbench.store import ProjectStore, RevisionConflict

blueprint = Blueprint("qpcr_workbench", __name__)


def json_object() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("请求必须包含 JSON 对象")
    return payload


def store() -> ProjectStore:
    default = (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        / "bioeng-qpcr"
        / "workbench.sqlite3"
    )
    return ProjectStore(
        current_app.config.get("WORKBENCH_DB")
        or os.environ.get("BIOENG_QPCR_WORKBENCH_DB", str(default))
    )


def expected_revision(payload: dict) -> int:
    value = payload.get("revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("保存或执行操作必须携带有效 revision")
    return value


def save(project, expected: int):
    store().save(project, expected)
    return jsonify(ok=True, project=project.to_dict())


@blueprint.errorhandler(RevisionConflict)
def conflict(exc):
    return jsonify(ok=False, error=str(exc)), 409


@blueprint.errorhandler(ValueError)
@blueprint.errorhandler(TypeError)
def bad_input(exc):
    return jsonify(ok=False, error=str(exc)), 400


@blueprint.errorhandler(KeyError)
def not_found(exc):
    return jsonify(ok=False, error=str(exc)), 404


@blueprint.get("/workbench")
def page():
    return render_template("qpcr_workbench.html")


@blueprint.get("/api/qpcr/templates")
def templates():
    from bioeng_qpcr.workbench.templates import EXPERIMENT_TEMPLATES, THEMES

    return jsonify(ok=True, experiments=EXPERIMENT_TEMPLATES, themes=THEMES)


@blueprint.get("/api/qpcr/projects")
def projects():
    return jsonify(ok=True, projects=store().list(), demos=DEMOS)


@blueprint.post("/api/qpcr/projects")
def create():
    payload = json_object()
    project = create_project(
        payload.get("name", "未命名 qPCR 实验"),
        payload.get("demo"),
        payload.get("template"),
    )
    store().save(project)
    return jsonify(ok=True, project=project.to_dict()), 201


@blueprint.get("/api/qpcr/projects/<project_id>")
def get(project_id):
    return jsonify(ok=True, project=store().get(project_id).to_dict())


@blueprint.post("/api/qpcr/projects/import")
def restore_backup():
    from bioeng_qpcr.workbench.backup import restore_project

    data = json_object()
    project = restore_project(data)
    project.sources.append({
        "title": "用户导入的项目备份（非独立实验验证）",
        "sha256": hashlib.sha256(json.dumps(
            data, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode()).hexdigest(),
        "imported_at": now(),
    })
    store().save(project)
    return jsonify(ok=True, project=project.to_dict()), 201


@blueprint.patch("/api/qpcr/projects/<project_id>")
def patch(project_id):
    payload = json_object()
    expected = expected_revision(payload)
    project = store().get(project_id)
    patch_project(project, payload.get("changes"))
    return save(project, expected)


@blueprint.post("/api/qpcr/projects/<project_id>/steps/<step>")
def execute(project_id, step):
    payload = json_object()
    expected = expected_revision(payload)
    project = store().get(project_id)
    advance(project, step)
    return save(project, expected)


@blueprint.post("/api/qpcr/projects/<project_id>/replay")
def replay_public(project_id):
    payload = json_object()
    expected = expected_revision(payload)
    project = store().get(project_id)
    replay(project)
    return save(project, expected)


@blueprint.post("/api/qpcr/projects/<project_id>/new-run")
def new_run(project_id):
    payload = json_object()
    expected = expected_revision(payload)
    project = store().get(project_id)
    if any(step not in project.completed for step in STEPS[:3]):
        raise ValueError("请先确认当前设计")
    project.invalidate("acquisition")
    project.start_run()
    return save(project, expected)


@blueprint.post("/api/qpcr/projects/<project_id>/acquire")
def import_data(project_id):
    try:
        expected = int(request.form.get("revision", ""))
    except ValueError as exc:
        raise ValueError("上传必须携带 revision") from exc
    project = store().get(project_id)
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ValueError("请选择一个采集文件")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".csv", ".rdml", ".rdm", ".xml"}:
        raise ValueError("支持 tidy CSV、长表曲线 CSV 和 RDML")
    with tempfile.TemporaryDirectory(prefix="qpcr-acquisition-") as directory:
        path = Path(directory) / f"acquisition{suffix}"
        upload.save(path)
        rows = read_acquisition(path, project)
        source = {
            "title": Path(upload.filename).name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "imported_at": now(),
            "selection": "RDML 首个 experiment/run；不自动更换样品 / 板位"
            if suffix != ".csv"
            else "按孔位和检测项目严格匹配设计",
        }
    acquire(project, rows, source, mode="file_import")
    return save(project, expected)


@blueprint.get("/api/qpcr/projects/<project_id>/export/<kind>")
def export(project_id, kind):
    project = store().get(project_id)
    if kind == "project.json":
        content = json.dumps(
            asdict(project), ensure_ascii=False, indent=2, allow_nan=False
        )
        mime = "application/json"
    elif kind == "protocol.json":
        content = json.dumps(
            {
                "plate": project.plate,
                "recipe": project.recipe,
                "program": project.program,
                "note": "程序与板位设计文件；未向真实仪器下发命令",
            },
            ensure_ascii=False,
            indent=2,
        )
        mime = "application/json"
    elif kind == "plate.csv":
        content, mime = csv_text(project.plate), "text/csv"
    elif kind == "acquisition.csv":
        if not project.active_run:
            raise ValueError("当前没有运行")
        rows = []
        for measurement in project.active_run["measurements"]:
            base = {
                "well": measurement["well"],
                "assay_id": measurement["assay_id"],
                "instrument_cq": measurement["instrument_cq"],
            }
            if not measurement["amplification"] and not measurement["melting"]:
                rows.append({**base, "kind": "ct", "x": "", "fluorescence": ""})
            for field in ("amplification", "melting"):
                rows.extend(
                    {**base, "kind": field, "x": x, "fluorescence": y}
                    for x, y in measurement[field]
                )
        content, mime = csv_text(rows), "text/csv"
    elif kind == "results.csv":
        if "analysis" not in project.completed:
            raise ValueError("请先完成有效定量分析；质控失败时可导出问题报告")
        analysis = project.outputs["analysis"]
        metadata = {
            "run_id": project.active_run_id,
            "method": analysis["method"],
            "qc_warnings": " / ".join(analysis["warnings"]),
        }
        content, mime = (
            csv_text([{**metadata, **row} for row in analysis["rows"]]),
            "text/csv",
        )
    elif kind == "report.html":
        content, mime = report_html(project), "text/html"
    else:
        raise ValueError("未知导出类型")
    return Response(
        content,
        content_type=f"{mime}; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="qpcr-{project.id[:8]}-{kind}"'
            ),
            "Cache-Control": "no-store",
        },
    )


@blueprint.get("/api/qpcr/projects/<project_id>/figures/<kind>.<fmt>")
def figure_export(project_id, kind, fmt):
    from bioeng_qpcr.workbench.figures import figure_bytes

    project = store().get(project_id)
    content = figure_bytes(
        project,
        kind,
        fmt,
        well=request.args.get("well"),
        assay=request.args.get("assay"),
    )
    mime = {"svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf"}[fmt]
    return Response(
        content,
        content_type=mime,
        headers={
            "Content-Disposition": (
                f'attachment; filename="qpcr-{kind}-'
                f'{project.presentation["theme"]}.{fmt}"'
            ),
            "Cache-Control": "no-store",
        },
    )
