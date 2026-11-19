"""Local, immutable analysis archives shared by the browser tools."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from bioeng_qpcr.provenance import ProvenanceManifest
from bioeng_qpcr.web.workbench import json_object, store
from bioeng_qpcr.workbench.model import now

blueprint = Blueprint("analysis_records", __name__)
ENDPOINTS = {
    "/api/preview": "preview_csv",
    "/api/image": "preview_image",
    "/api/potency": "potency",
    "/api/dock": "docking",
    "/api/align": "sequence",
    "/api/enzyme": "enzyme",
    "/api/ferment": "ferment",
    "/api/fba": "fba",
}
MODULES = set(ENDPOINTS.values())


@blueprint.after_app_request
def archive_success(response):
    module = ENDPOINTS.get(request.path)
    if (
        not module
        or request.method != "POST"
        or response.status_code != 200
        or request.form.get("_archive") != "1"
    ):
        return response
    payload = response.get_json(silent=True)
    if not isinstance(payload, dict) or not payload.get("ok"):
        return response
    try:
        files = []
        for role, upload in request.files.items():
            upload.stream.seek(0)
            raw = upload.read()
            files.append(
                {
                    "role": role,
                    "name": Path(upload.filename).name,
                    "type": upload.mimetype or "application/octet-stream",
                    "data": base64.b64encode(raw).decode("ascii"),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        parameters = {
            **{k: v for k, v in request.form.items() if k != "_archive"},
            **payload.get("effective_parameters", {}),
        }
        provenance = ProvenanceManifest.capture_run(
            parameters=parameters,
            input_hash=hashlib.sha256(json.dumps(
                {f["role"]: f["sha256"] for f in files}, sort_keys=True,
            ).encode()).hexdigest(),
        )
        payload["provenance"] = provenance
        record = {
            "format": "biolab-analysis",
            "version": 1,
            "id": uuid.uuid4().hex,
            "module": module,
            "created_at": now(),
            "title": "、".join(f["name"] for f in files)[:200],
            "parameters": parameters,
            "provenance": provenance,
            "files": files,
            "result": payload,
            "imported": False,
        }
        store().save_record(record)
        payload["record_id"] = record["id"]
    except (OSError, sqlite3.Error, ValueError) as exc:
        current_app.logger.exception("Analysis archive failed")
        # Keep the computed result available even if the disk is full.
        payload["archive_error"] = f"本次结果尚未保存，请下载完整结果备份：{exc}"
    response.set_data(current_app.json.dumps(payload))
    return response


@blueprint.errorhandler(ValueError)
@blueprint.errorhandler(TypeError)
def invalid(exc):
    return jsonify(ok=False, error=str(exc)), 400


@blueprint.errorhandler(KeyError)
def missing(exc):
    return jsonify(ok=False, error=str(exc)), 404


@blueprint.get("/api/lab/records")
def records():
    module = request.args.get("module", "")
    if module not in MODULES | {"all"}:
        raise ValueError("请选择有效的分析模块")
    return jsonify(
        ok=True, records=store().list_records(None if module == "all" else module)
    )


@blueprint.get("/history")
def history_page():
    return render_template("history.html")


@blueprint.get("/api/lab/records/<record_id>/results.csv")
def results_csv(record_id):
    from bioeng_qpcr.workbench.service import csv_text

    record = store().get_record(record_id)
    if record["module"] not in {"enzyme", "ferment", "fba"}:
        raise ValueError("请使用此模块自己的完整结果导出")
    if record["imported"]:
        raise ValueError("外部备份未经复算，请运行分析后导出当前汇总")
    return Response(
        csv_text(record["result"]["rows"]), mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="analysis-results.csv"'},
    )


@blueprint.get("/api/lab/records/<record_id>")
def record(record_id):
    return jsonify(ok=True, record=store().get_record(record_id))


@blueprint.get("/api/lab/records/<record_id>/export")
def export(record_id):
    content = json.dumps(
        store().get_record(record_id), ensure_ascii=False, indent=2, allow_nan=False
    )
    return Response(
        content,
        mimetype="application/json",
        headers={"Content-Disposition": 'attachment; filename="biolab-analysis.json"'},
    )


@blueprint.post("/api/lab/records/import")
def restore():
    data = json_object()
    if (
        data.get("format") != "biolab-analysis"
        or type(data.get("version")) is not int
        or data["version"] != 1
        or data.get("module") not in MODULES
    ):
        raise ValueError("不是受支持的 BioLab 分析备份")
    files, parameters, result = (
        data.get("files"),
        data.get("parameters"),
        data.get("result"),
    )
    if not isinstance(files, list) or not 1 <= len(files) <= 2:
        raise ValueError("备份必须包含 1–2 个原始文件")
    if (
        not isinstance(parameters, dict)
        or len(parameters) > 100
        or any(not isinstance(v, str) for v in parameters.values())
        or not isinstance(result, dict)
    ):
        raise ValueError("备份的参数或结果格式无效")
    json.dumps(result, allow_nan=False)
    provenance = data.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("备份的版本记录格式无效")
    json.dumps(provenance, allow_nan=False)
    roles, normalized = set(), []
    for item in files:
        if (
            not isinstance(item, dict)
            or item.get("role")
            not in {"file", "receptor", "ligand", "query", "reference"}
            or item["role"] in roles
        ):
            raise ValueError("备份的文件角色无效或重复")
        roles.add(item["role"])
        name, mime = item.get("name"), item.get("type", "application/octet-stream")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 255
            or not isinstance(mime, str)
        ):
            raise ValueError("备份文件名称或类型无效")
        try:
            raw = base64.b64decode(item.get("data", ""), validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("原始文件编码无效") from exc
        if not raw or hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise ValueError("原始文件校验失败，备份可能已损坏")
        normalized.append(
            {
                "role": item["role"],
                "name": Path(name).name,
                "type": mime,
                "data": item["data"],
                "sha256": item["sha256"],
            }
        )
    required = {"docking": {"receptor"}, "sequence": {"query", "reference"}}.get(
        data["module"], {"file"}
    )
    if not required <= roles:
        raise ValueError("备份缺少此模块必需的原始文件")
    restored = {
        "format": "biolab-analysis",
        "version": 1,
        "id": uuid.uuid4().hex,
        "module": data["module"],
        "created_at": now(),
        "title": "导入备份 · " + "、".join(f["name"] for f in normalized)[:180],
        "parameters": parameters,
        "files": normalized,
        "result": result,
        "imported": True,
        # Imported provenance is a historical claim, never this machine's version.
        "provenance": provenance,
    }
    store().save_record(restored)
    return jsonify(ok=True, record=restored), 201
