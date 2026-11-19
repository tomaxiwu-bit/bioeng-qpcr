"""Reusable planning templates and presentation themes (never measured data)."""

from __future__ import annotations

from dataclasses import asdict

from bioeng_qpcr.workbench.model import (
    AnalysisSettings,
    AssayRecord,
    PlateAssignment,
    Project,
    SampleRecord,
)

THEMES = {
    "journal": {
        "name": "期刊简洁",
        "description": "白底细线 · 蓝绿配色 · 适合论文插图",
        "background": "#ffffff",
        "ink": "#293c4a",
        "grid": "#e4eaf0",
        "palette": ["#246b83", "#da8550", "#5b9c8a", "#8871ad", "#c3a448", "#6a8bac"],
    },
    "lab": {
        "name": "实验室汇报",
        "description": "暖白底 · 森林绿与金色 · 适合组会",
        "background": "#fcfcf8",
        "ink": "#294638",
        "grid": "#e3eadf",
        "palette": ["#347452", "#b28d45", "#738bac", "#aa738b", "#7e9b65", "#6fa9aa"],
    },
    "accessible": {
        "name": "色盲友好",
        "description": "Okabe–Ito 配色 · 线型与形状辅助辨识",
        "background": "#ffffff",
        "ink": "#26323f",
        "grid": "#e5e9ed",
        "palette": ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"],
    },
    "mono": {
        "name": "黑白印刷",
        "description": "灰阶层次 · 虚实线区分 · 适合打印",
        "background": "#ffffff",
        "ink": "#252525",
        "grid": "#e4e4e4",
        "palette": ["#252525", "#777777", "#494949", "#a0a0a0", "#606060", "#b5b5b5"],
    },
    "night": {
        "name": "深色展示",
        "description": "深海背景 · 明亮曲线 · 适合屏幕演示",
        "background": "#142932",
        "ink": "#e2edf1",
        "grid": "#2d4650",
        "palette": ["#71d5c3", "#f2b977", "#a4b5f5", "#ed9fbb", "#c2db84", "#83cee9"],
    },
    "pastel": {
        "name": "柔和配色",
        "description": "雾蓝与紫色 · 轻网格 · 适合教学与笔记",
        "background": "#fcfbff",
        "ink": "#4c4968",
        "grid": "#ece9f3",
        "palette": ["#738fc0", "#ad8ac0", "#74ac9b", "#d1a376", "#cd8f9f", "#96a66b"],
    },
}

EXPERIMENT_TEMPLATES = {
    "expression": {
        "name": "基因表达 · ΔΔCt",
        "category": "RELATIVE",
        "description": (
            "对照 / 处理各 3 个独立样品，双检测项目，3 个技术重复及阴阳性对照。"
        ),
        "wells": 48,
    },
    "standard": {
        "name": "绝对定量 · 标准曲线",
        "category": "ABSOLUTE",
        "description": (
            "5 个十倍稀释水平、2 个未知样品、阴阳性对照；标准量和单位需按真实材料校正。"
        ),
        "wells": 27,
    },
    "melt": {
        "name": "染料法 · 熔解验证",
        "category": "MELTING",
        "description": "单检测项目、3 个样品及对照；侧重主峰、多峰与重复孔一致性检查。",
        "wells": 15,
    },
    "multiplex": {
        "name": "多重探针 · 双通道",
        "category": "MULTIPLEX",
        "description": "FAM / HEX 两个检测项目共享物理孔；关闭非适用的熔解扫描。",
        "wells": 24,
    },
    "controls": {
        "name": "对照检查 · RT-qPCR",
        "category": "CONTROL PANEL",
        "description": "含无模板、无逆转录和阳性对照；用于建立或复核实验室质控 SOP。",
        "wells": 30,
    },
}


def planned_project(key: str) -> Project:
    """Prefill editable planning metadata; no raw data or results are created."""
    if key not in EXPERIMENT_TEMPLATES:
        raise ValueError("未知实验模板")
    template = EXPERIMENT_TEMPLATES[key]
    samples = [
        SampleRecord(id=f"control_{i}", name=f"Control {i}", group="control")
        for i in range(1, 4)
    ]
    samples += [
        SampleRecord(id=f"treated_{i}", name=f"Treated {i}", group="treated")
        for i in range(1, 4)
    ]
    assays = [
        AssayRecord(id="target", name="Target"),
        AssayRecord(id="reference", name="Reference", reference=True),
    ]
    mode = "relative"
    if key == "standard":
        samples = [
            SampleRecord(
                id=f"standard_{i}",
                name=f"Standard 10^{i}",
                kind="standard",
                quantity=10.0**i,
                unit="copies/reaction",
                notes="教学标准量；必须以真实标准品证书和稀释记录替换",
            )
            for i in range(2, 7)
        ]
        samples += [
            SampleRecord(id=f"unknown_{i}", name=f"Unknown {i}") for i in range(1, 3)
        ]
        assays = assays[:1]
        mode = "absolute"
    elif key == "melt":
        samples, assays = samples[:3], assays[:1]
    elif key == "controls":
        samples = samples[:2] + [SampleRecord(id="no_rt", name="No RT", kind="nrt")]
    if key == "multiplex":
        for assay, channel in zip(assays, ("FAM", "HEX"), strict=True):
            assay.channel, assay.chemistry = channel, "probe"
    samples += [
        SampleRecord(id="ntc", name="NTC", kind="ntc"),
        SampleRecord(id="positive", name="Positive control", kind="positive"),
    ]
    plate, position = [], 0
    for sample in samples:
        if key == "multiplex":
            for _ in range(3):
                well = f"{chr(65 + position // 12)}{position % 12 + 1:02d}"
                plate.extend(
                    asdict(PlateAssignment(well, sample.id, assay.id))
                    for assay in assays
                )
                position += 1
        else:
            for assay in assays:
                for _ in range(3):
                    well = f"{chr(65 + position // 12)}{position % 12 + 1:02d}"
                    plate.append(asdict(PlateAssignment(well, sample.id, assay.id)))
                    position += 1
    project = Project(
        name=template["name"] + " · 新实验",
        samples=[asdict(s) for s in samples],
        assays=[asdict(a) for a in assays],
        plate=plate,
        analysis_settings=asdict(
            AnalysisSettings(
                mode=mode,
                target_id="target",
                reference_id="reference" if len(assays) > 1 else "",
                calibrator_id=samples[0].id,
            )
        ),
    )
    if key == "multiplex":
        project.program["channels"] = ["FAM", "HEX"]
        project.program["melt"]["enabled"] = False
    project.sources = [
        {
            "title": f"本项目实验规划模板：{template['name']}",
            "url": "https://github.com/tomaxiwu-bit/bioeng-qpcr",
            "license": "MIT (planning configuration)",
            "selection": "仅预填可编辑的实验设计；没有测量数据",
            "limitations": (
                "引物与探针序列、真实标准量、试剂配方和仪器程序须按实际实验替换；"
                "不代表已执行或已验证的检测方法"
            ),
        }
    ]
    return project
