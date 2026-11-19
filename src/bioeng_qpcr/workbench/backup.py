"""Restore portable project data without trusting imported computed conclusions."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, fields

from bioeng_qpcr.workbench.acquisition import validate_measurements
from bioeng_qpcr.workbench.model import Project, RunRecord, validate_design
from bioeng_qpcr.workbench.service import advance, patch_project

DESIGN_KEYS = ("samples", "assays", "plate", "plate_format", "recipe", "program")
SETTINGS_KEYS = ("curve_settings", "qc_policy", "analysis_settings", "presentation")


def restore_project(data: dict) -> Project:
    if not isinstance(data, dict):
        raise ValueError("项目备份必须为 JSON 对象")
    data = copy.deepcopy(data)
    data.pop("workflow", None)
    if set(data) - {f.name for f in fields(Project)}:
        raise ValueError("备份包含未知项目字段")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("不支持此项目备份版本")
    json.dumps(data, allow_nan=False)
    source = Project.from_dict(data)
    for collection in (source.sources, source.audit):
        if (
            not isinstance(collection, list)
            or len(collection) > 10000
            or any(not isinstance(row, dict) for row in collection)
        ):
            raise ValueError("备份来源与审计记录格式无效")
    if (
        not isinstance(source.runs, list)
        or len(source.runs) > 100
        or not isinstance(source.completed, list)
    ):
        raise ValueError("运行历史或完成状态格式无效")
    restored = Project()
    patch_project(
        restored,
        {key: getattr(source, key) for key in DESIGN_KEYS + SETTINGS_KEYS + ("name",)},
    )
    restored.name = source.name[:1900] + "（恢复）"
    restored.sources = copy.deepcopy(source.sources)
    restored.audit = copy.deepcopy(source.audit)
    # A backup may contain assumptions, but the user must confirm them again.
    restored.analysis_settings["efficiency_confirmed"] = False
    ids = set()
    for raw in source.runs:
        if not isinstance(raw, dict):
            raise ValueError("运行记录必须为对象")
        run = RunRecord(**raw)
        if (
            not isinstance(run.id, str)
            or not run.id
            or len(run.id) > 200
            or run.id in ids
            or run.status not in {"waiting_data", "collected"}
        ):
            raise ValueError("运行标识重复或状态无效")
        ids.add(run.id)
        if not isinstance(run.snapshot, dict) or set(run.snapshot) != set(DESIGN_KEYS):
            raise ValueError("运行设计快照字段不完整")
        snapshot = Project()
        patch_project(snapshot, run.snapshot)
        validate_design(snapshot)
        digest = hashlib.sha256(
            json.dumps(run.snapshot, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if digest != run.design_sha256:
            raise ValueError("运行设计快照校验失败")
        if not isinstance(run.measurements, list):
            raise ValueError("运行测量必须为列表")
        if run.status == "collected":
            run.measurements = validate_measurements(snapshot, run.measurements)
        elif run.measurements:
            raise ValueError("等待采集的运行不能带有测量结果")
        if (
            not isinstance(run.analysis_versions, list)
            or any(not isinstance(v, dict) for v in run.analysis_versions)
            or not isinstance(run.source, dict)
        ):
            raise ValueError("运行来源或历史分析格式无效")
        for key in (run.started_at, run.mode):
            if not isinstance(key, str):
                raise ValueError("运行时间与模式必须为文本")
        if run.finished_at is not None and not isinstance(run.finished_at, str):
            raise ValueError("运行完成时间必须为文本")
        for version in run.analysis_versions:
            version["imported_backup"] = True
        restored.runs.append(asdict(run))
    if source.active_run_id is not None and source.active_run_id not in ids:
        raise ValueError("当前运行不存在于备份中")
    restored.active_run_id = source.active_run_id
    if restored.active_run and restored.active_run["snapshot"] != restored.snapshot():
        raise ValueError("当前运行快照与当前设计不一致")
    if all(key in source.completed for key in ("samples", "design", "program")):
        for key in ("samples", "design", "program"):
            advance(restored, key)
        if restored.active_run and restored.active_run["status"] == "collected":
            restored.finish("acquisition")
            advance(restored, "curves")
            advance(restored, "qc")
    restored.record(
        "restore",
        "从用户备份恢复为新项目；历史分析标记为外部备份，"
        "当前曲线与质控重新计算；定量和报告需要重新确认",
    )
    return restored
