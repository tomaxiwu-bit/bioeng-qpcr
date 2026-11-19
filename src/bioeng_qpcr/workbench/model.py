"""Experiment design, acquisition snapshots and dependency-aware workflow state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

STEPS = (
    "samples",
    "design",
    "program",
    "acquisition",
    "curves",
    "qc",
    "analysis",
    "report",
)
STEP_LABELS = (
    "样品与引物 / 探针",
    "反应体系与板位",
    "温度循环与通道",
    "运行与数据采集",
    "Ct 与熔解曲线",
    "质量控制",
    "相对 / 绝对定量",
    "结果与报告",
)
SECTION_STEP = {
    "samples": "samples",
    "assays": "samples",
    "plate": "design",
    "recipe": "design",
    "program": "program",
    "curve_settings": "curves",
    "qc_policy": "qc",
    "analysis_settings": "analysis",
    "presentation": "report",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数值")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} 必须在 {low}–{high} 之间且有限")
    return float(value)


def text(value: Any, label: str, *, required: bool = True) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 500
        or (required and not value.strip())
    ):
        raise ValueError(
            f"{label} 必须是{'非空' if required else ''}文本（最多 500 字）"
        )
    return value.strip()


@dataclass
class SampleRecord:
    id: str
    name: str
    kind: str = "unknown"
    group: str = ""
    quantity: float | None = None
    unit: str = "copies/reaction"
    notes: str = ""


@dataclass
class AssayRecord:
    id: str
    name: str
    forward: str = ""
    reverse: str = ""
    probe: str = ""
    chemistry: str = "dye"
    channel: str = "SYBR/FAM"
    reference: bool = False
    expected_tm: float | None = None


@dataclass
class PlateAssignment:
    well: str
    sample_id: str
    assay_id: str


@dataclass
class ReagentComponent:
    name: str
    volume_ul: float


@dataclass
class ReactionRecipe:
    volume_ul: float = 20.0
    excess_percent: float = 10.0
    components: list[dict] = field(
        default_factory=lambda: [
            {"name": "2× Master Mix", "volume_ul": 10.0},
            {"name": "引物/探针混合液", "volume_ul": 2.0},
            {"name": "模板 / 无核酸水", "volume_ul": 2.0},
            {"name": "水", "volume_ul": 6.0},
        ]
    )
    provenance: str = "教学配置；请按试剂说明书和实验室 SOP 替换"


@dataclass
class ThermalStep:
    name: str
    temperature_c: float
    seconds: float
    cycles: int = 1
    acquire: bool = False


@dataclass
class InstrumentProgram:
    instrument: str = "未连接设备 · 文件导入模式"
    channels: list[str] = field(default_factory=lambda: ["SYBR/FAM"])
    steps: list[dict] = field(
        default_factory=lambda: [
            {
                "name": "预变性",
                "temperature_c": 95,
                "seconds": 180,
                "cycles": 1,
                "acquire": False,
            },
            {
                "name": "变性",
                "temperature_c": 95,
                "seconds": 15,
                "cycles": 40,
                "acquire": False,
            },
            {
                "name": "退火 / 延伸",
                "temperature_c": 60,
                "seconds": 30,
                "cycles": 40,
                "acquire": True,
            },
        ]
    )
    melt: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "start_c": 60.0,
            "end_c": 95.0,
            "step_c": 0.3,
        }
    )
    provenance: str = "教学程序；相邻且循环数相同的步骤组成循环块；请按真实仪器程序替换"


@dataclass
class CurveSettings:
    ct_method: str = "instrument"
    threshold: float = 1000.0
    baseline_start: int = 3
    baseline_end: int = 10
    melt_prominence: float = 0.15


@dataclass
class QCPolicy:
    ntc_max_cq: float = 35.0
    positive_min_cq: float = 5.0
    positive_max_cq: float = 35.0
    replicate_max_sd: float = 0.5
    tm_tolerance_c: float = 1.0
    provenance: str = "演示 SOP 阈值；不代表所有检测项目的通用界限"


@dataclass
class AnalysisSettings:
    mode: str = "relative"
    target_id: str = ""
    reference_id: str = ""
    calibrator_id: str = ""
    efficiency_confirmed: bool = False
    relative_method: str = "delta_delta_ct"
    reference_ids: list[str] = field(default_factory=list)
    efficiencies: dict[str, dict] = field(default_factory=dict)


@dataclass
class Measurement:
    well: str
    assay_id: str
    instrument_cq: float | None = None
    amplification: list[list[float]] = field(default_factory=list)
    melting: list[list[float]] = field(default_factory=list)


@dataclass
class RunRecord:
    id: str
    started_at: str
    snapshot: dict
    design_sha256: str
    status: str = "waiting_data"
    mode: str = "file_import"
    source: dict = field(default_factory=dict)
    measurements: list[dict] = field(default_factory=list)
    finished_at: str | None = None
    analysis_versions: list[dict] = field(default_factory=list)


@dataclass
class Project:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "未命名 qPCR 实验"
    revision: int = 0
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    schema_version: int = 1
    samples: list[dict] = field(default_factory=list)
    assays: list[dict] = field(default_factory=list)
    plate: list[dict] = field(default_factory=list)
    plate_format: int = 96
    recipe: dict = field(default_factory=lambda: asdict(ReactionRecipe()))
    program: dict = field(default_factory=lambda: asdict(InstrumentProgram()))
    curve_settings: dict = field(default_factory=lambda: asdict(CurveSettings()))
    qc_policy: dict = field(default_factory=lambda: asdict(QCPolicy()))
    analysis_settings: dict = field(default_factory=lambda: asdict(AnalysisSettings()))
    completed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    runs: list[dict] = field(default_factory=list)
    active_run_id: str | None = None
    sources: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    demo_key: str | None = None
    presentation: dict = field(
        default_factory=lambda: {
            "theme": "journal",
            "curve_scope": "selected",
            "relative_chart": "lollipop",
        }
    )

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        return cls(**copy.deepcopy(data))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["workflow"] = self.workflow()
        return result

    @property
    def active_run(self) -> dict | None:
        return next((run for run in self.runs if run["id"] == self.active_run_id), None)

    def workflow(self) -> list[dict]:
        nodes = []
        for index, key in enumerate(STEPS):
            ready = all(step in self.completed for step in STEPS[:index])
            status = (
                "complete" if key in self.completed else "ready" if ready else "pending"
            )
            if key in self.stale:
                status = "stale"
            if key == "qc" and self.outputs.get("qc", {}).get("blocked"):
                status = "blocked"
            if (
                key == "acquisition"
                and self.active_run
                and self.active_run["status"] == "waiting_data"
            ):
                status = "waiting_data"
            nodes.append({"key": key, "label": STEP_LABELS[index], "status": status})
        return nodes

    def invalidate(self, step: str) -> None:
        affected = STEPS[STEPS.index(step) :]
        for key in affected:
            if key in self.completed or key in self.outputs:
                if key not in self.stale:
                    self.stale.append(key)
            self.outputs.pop(key, None)
        self.completed = [key for key in self.completed if key not in affected]
        if STEPS.index(step) <= STEPS.index("program"):
            self.active_run_id = None

    def record(self, action: str, detail: str) -> None:
        self.updated_at = now()
        self.audit.append({"at": self.updated_at, "action": action, "detail": detail})

    def finish(self, step: str) -> None:
        if step not in self.completed:
            self.completed.append(step)
        self.stale = [key for key in self.stale if key != step]
        self.record("complete", step)

    def snapshot(self) -> dict:
        return copy.deepcopy(
            {
                key: getattr(self, key)
                for key in (
                    "samples",
                    "assays",
                    "plate",
                    "plate_format",
                    "recipe",
                    "program",
                )
            }
        )

    def start_run(self) -> None:
        if len(self.runs) >= 100:
            raise ValueError("单个项目最多保存 100 次运行，请导出并新建项目")
        snapshot = self.snapshot()
        digest = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        run = RunRecord(uuid.uuid4().hex, now(), snapshot, digest)
        self.runs.append(asdict(run))
        self.active_run_id = run.id
        self.record("start_run", run.id)


def _records(values: Any, cls: type, label: str) -> list[dict]:
    if not isinstance(values, list) or len(values) > 1536:
        raise ValueError(f"{label} 必须是有界列表")
    try:
        return [asdict(cls(**value)) for value in values]
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"{label} 字段不正确: {exc}") from exc


def validate_design(project: Project, *, through: str = "program") -> None:
    """Validate cross-record references before a design can become a run."""
    text(project.name, "实验名称")
    project.samples = _records(project.samples, SampleRecord, "样品")
    project.assays = _records(project.assays, AssayRecord, "检测项目")
    samples, assays = {}, {}
    for sample in project.samples:
        sid = text(sample["id"], "样品 ID")
        text(sample["name"], "样品名称")
        for key in ("group", "notes", "unit"):
            text(sample[key], key, required=False)
        if sid in samples or sample["kind"] not in {
            "unknown",
            "standard",
            "ntc",
            "positive",
            "negative",
            "nrt",
        }:
            raise ValueError("样品 ID 重复或样品类型无效")
        if sample["quantity"] is not None:
            number(sample["quantity"], "标准量", 1e-15, 1e30)
        if sample["kind"] == "standard" and (
            sample["quantity"] is None or not sample["unit"].strip()
        ):
            raise ValueError("标准品必须填写正标准量及单位")
        samples[sid] = sample
    for assay in project.assays:
        aid = text(assay["id"], "检测项目 ID")
        text(assay["name"], "检测项目名称")
        text(assay["channel"], "荧光通道")
        if (
            aid in assays
            or assay["chemistry"] not in {"dye", "probe"}
            or not isinstance(assay["reference"], bool)
        ):
            raise ValueError("检测项目 ID 重复、化学类型或内参标记无效")
        for key in ("forward", "reverse", "probe"):
            sequence = text(assay[key], key, required=False)
            if sequence and not re.fullmatch(
                "[ACGTRYSWKMBDHVNUacgtryswkmbdhvnu]+", sequence
            ):
                raise ValueError("引物 / 探针序列只能包含 IUPAC 核酸符号")
        if assay["expected_tm"] is not None:
            number(assay["expected_tm"], "预期 Tm", 0, 120)
        assays[aid] = assay
    if not samples or not assays:
        raise ValueError("至少添加一个样品和一个检测项目")
    if through == "samples":
        return
    project.plate = _records(project.plate, PlateAssignment, "板位")
    if project.plate_format not in (96, 384):
        raise ValueError("板型只支持 96 或 384 孔")
    rows, columns = (8, 12) if project.plate_format == 96 else (16, 24)
    seen, well_samples, well_channels = set(), {}, set()
    for assignment in project.plate:
        well, sid, aid = (
            assignment["well"],
            assignment["sample_id"],
            assignment["assay_id"],
        )
        match = re.fullmatch(r"([A-P])(\d{1,2})", str(well))
        if not match or ord(match[1]) - 65 >= rows or not 1 <= int(match[2]) <= columns:
            raise ValueError(f"无效孔位: {well}")
        assignment["well"] = well = f"{match[1]}{int(match[2]):02d}"
        if sid not in samples or aid not in assays:
            raise ValueError(f"{well} 引用了不存在的样品 / 检测项目")
        if (well, aid) in seen or (well in well_samples and well_samples[well] != sid):
            raise ValueError(f"{well} 有重复检测项目或不同样品冲突")
        channel_key = (well, assays[aid]["channel"])
        if channel_key in well_channels:
            raise ValueError(f"{well} 的同一荧光通道不能分配多个检测项目")
        seen.add((well, aid))
        well_channels.add(channel_key)
        well_samples[well] = sid
    if not seen:
        raise ValueError("至少分配一个板位")
    recipe = ReactionRecipe(**project.recipe)
    number(recipe.volume_ul, "每孔体积", 0.1, 1000)
    number(recipe.excess_percent, "配液余量", 0, 100)
    components = _records(recipe.components, ReagentComponent, "反应组分")
    for component in components:
        text(component["name"], "组分名称")
        number(component["volume_ul"], "组分体积", 0, 1000)
    if not math.isclose(
        sum(c["volume_ul"] for c in components), recipe.volume_ul, abs_tol=1e-6
    ):
        raise ValueError("组分体积之和必须等于每孔反应体积")
    if through == "design":
        return
    program = InstrumentProgram(**project.program)
    text(program.instrument, "仪器名称")
    if not isinstance(program.channels, list) or not 1 <= len(program.channels) <= 16:
        raise ValueError("至少选择一个荧光通道，最多 16 个")
    for channel in program.channels:
        text(channel, "通道")
    if len(set(program.channels)) != len(program.channels):
        raise ValueError("荧光通道不能重复")
    if any(a["channel"] not in program.channels for a in project.assays):
        raise ValueError("程序通道没有覆盖全部检测项目")
    steps = _records(program.steps, ThermalStep, "温度程序")
    if not steps or len(steps) > 100:
        raise ValueError("温度程序需要 1–100 个步骤")
    for step in steps:
        text(step["name"], "程序步骤")
        number(step["temperature_c"], "温度", 0, 110)
        number(step["seconds"], "持续时间", 0.1, 86400)
        number(step["cycles"], "循环次数", 1, 100)
        if not isinstance(step["cycles"], int) or not isinstance(step["acquire"], bool):
            raise ValueError("循环次数必须为整数，采集标记必须为布尔值")
    if not any(step["acquire"] for step in steps):
        raise ValueError("至少一个温度步骤必须启用荧光采集")
    melt = program.melt
    if not isinstance(melt.get("enabled"), bool):
        raise ValueError("熔解扫描开关必须为布尔值")
    if melt["enabled"]:
        number(melt["start_c"], "熔解起始温度", 0, 110)
        number(melt["end_c"], "熔解结束温度", 0, 110)
        number(melt["step_c"], "熔解温度步长", 0.01, 10)
        if melt["end_c"] <= melt["start_c"]:
            raise ValueError("熔解结束温度必须大于起始温度")


def validate_settings(project: Project) -> None:
    from bioeng_qpcr.workbench.templates import THEMES

    if not isinstance(project.presentation, dict) or set(project.presentation) != {
        "theme",
        "curve_scope",
        "relative_chart",
    }:
        raise ValueError("图表设置字段不正确")
    if project.presentation["theme"] not in THEMES:
        raise ValueError("未知图表主题")
    if project.presentation["curve_scope"] not in {"selected", "replicates", "all"}:
        raise ValueError("未知曲线显示范围")
    if project.presentation["relative_chart"] not in {"lollipop", "bar", "dot"}:
        raise ValueError("未知相对定量图模板")
    curves = CurveSettings(**project.curve_settings)
    if curves.ct_method not in {"instrument", "threshold"}:
        raise ValueError("Ct 方法只支持 instrument 或 threshold")
    number(curves.threshold, "荧光阈值", 1e-12, 1e30)
    for value in (curves.baseline_start, curves.baseline_end):
        number(value, "基线周期", 1, 1000)
        if not isinstance(value, int):
            raise ValueError("基线周期必须为整数")
    if curves.baseline_end <= curves.baseline_start:
        raise ValueError("基线终止周期必须大于起始周期")
    number(curves.melt_prominence, "熔解峰相对显著性", 0.01, 1)
    qc = QCPolicy(**project.qc_policy)
    for key in ("ntc_max_cq", "positive_min_cq", "positive_max_cq"):
        number(getattr(qc, key), key, 0, 1000)
    if qc.positive_max_cq <= qc.positive_min_cq:
        raise ValueError("阳性对照 Ct 范围无效")
    number(qc.replicate_max_sd, "重复孔 SD", 0.01, 10)
    number(qc.tm_tolerance_c, "Tm 容差", 0.01, 20)
    settings = AnalysisSettings(**project.analysis_settings)
    if settings.mode not in {"relative", "absolute"}:
        raise ValueError("定量模式只支持 relative 或 absolute")
    for value in (settings.target_id, settings.reference_id, settings.calibrator_id):
        text(value, "定量选择", required=False)
    if not isinstance(settings.efficiency_confirmed, bool):
        raise ValueError("效率假设确认必须为布尔值")
    if settings.relative_method not in {"delta_delta_ct", "efficiency_corrected"}:
        raise ValueError("未知的相对定量方法")
    if (
        not isinstance(settings.reference_ids, list)
        or len(settings.reference_ids) > 20
        or any(not isinstance(r, str) or not r for r in settings.reference_ids)
        or len(set(settings.reference_ids)) != len(settings.reference_ids)
    ):
        raise ValueError("内参必须为不重复的检测项目 ID 列表，最多 20 个")
    if not isinstance(settings.efficiencies, dict) or len(settings.efficiencies) > 1536:
        raise ValueError("效率证据必须为检测项目到数值与依据的映射")
    for aid, evidence in settings.efficiencies.items():
        text(aid, "效率检测项目")
        if not isinstance(evidence, dict) or set(evidence) != {"value", "source"}:
            raise ValueError("每项效率证据需要 value 和 source")
        if evidence["value"] is not None:
            number(evidence["value"], "效率（1 = 100%）", 0.001, 2)
        text(evidence["source"], "效率依据", required=False)
