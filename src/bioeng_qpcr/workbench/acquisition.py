"""File acquisition adapter. REAL HARDWARE: integrate a vendor SDK here.

No instrument connection or live hardware execution is claimed. Replay loads
the exact bundled public observations; real exports must match the frozen
well/assay/sample design before they can enter the calculation pipeline.
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

from bioeng_qpcr.io import (
    NotAmplificationFile,
    read_amplification_csv,
    read_rdml,
    read_tidy_csv,
)
from bioeng_qpcr.io.rdml_reader import _unwrap_container
from bioeng_qpcr.workbench.model import Measurement, Project, number


def normalize_well(value: str) -> str:
    match = re.fullmatch(r"([A-Pa-p])(\d{1,2})", value)
    if not match:
        raise ValueError(f"孔位格式无效: {value}")
    return f"{match[1].upper()}{int(match[2]):02d}"


def validate_measurements(project: Project, rows: list[dict]) -> list[dict]:
    if not rows or len(rows) > 1536:
        raise ValueError("采集记录必须为非空列表，最多 1536 条")
    expected = {(a["well"], a["assay_id"]) for a in project.plate}
    seen, normalized, point_count = set(), [], 0
    for entry in rows:
        item = asdict(Measurement(**entry))
        item["well"] = normalize_well(item["well"])
        key = (item["well"], item["assay_id"])
        if key not in expected or key in seen:
            raise ValueError(f"{key} 不在当前板设计中，或存在重复测量")
        seen.add(key)
        if item["instrument_cq"] is not None:
            number(item["instrument_cq"], "仪器 Ct", 0, 1000)
        for field in ("amplification", "melting"):
            if not isinstance(item[field], list) or len(item[field]) > 5000:
                raise ValueError("每条曲线最多 5000 个点")
            for point in item[field]:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("曲线数据点必须为 [周期/温度, 荧光] 数值对")
                number(point[0], "周期 / 温度", 0, 10000)
                number(point[1], "荧光", -1e30, 1e30)
            point_count += len(item[field])
        if point_count > 1_000_000:
            raise ValueError("单次运行最多 100 万个曲线数据点")
        normalized.append(item)
    # Missing planned wells remain visible and become QC failures, rather than
    # disappearing through an implicit intersection with the imported table.
    return normalized


def read_acquisition(path: Path, project: Project) -> list[dict]:
    """Read tidy/trace CSV, workbench long CSV, or RDML's first run."""
    if path.suffix.lower() in {".rdml", ".rdm", ".xml"}:
        raw, _ = _unwrap_container(path.read_bytes())
        # ElementTree expands internal entities: reject declarations before the
        # second parse, including UTF-16/32 encodings with embedded nulls.
        declarations = raw.replace(b"\x00", b"").upper()
        if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
            raise ValueError("RDML 不允许 DOCTYPE / ENTITY 声明")
        experiment = read_rdml(path)
        root = ET.fromstring(raw)
        for node in root.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        melting = {}
        run = root.find("experiment/run")
        if run is not None:
            columns = int(run.findtext("pcrFormat/columns", "0"))
            for reaction in run.findall("react"):
                if columns <= 0:
                    continue
                index = int(reaction.get("id")) - 1
                well = f"{chr(65 + index // columns)}{index % columns + 1:02d}"
                for data in reaction.findall("data"):
                    melting[(well, data.find("tar").get("id"))] = [
                        [float(p.findtext("tmp")), float(p.findtext("fluor"))]
                        for p in data.findall("mdp")
                    ]
    else:
        raw_text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(raw_text))
        if {"well", "assay_id", "kind", "x", "fluorescence"} <= set(
            reader.fieldnames or []
        ):
            return _read_workbench_csv(reader, project)
        try:
            experiment = read_amplification_csv(path)
        except NotAmplificationFile:
            experiment = read_tidy_csv(path)
        melting = {}
    assays = {a["id"]: a["id"] for a in project.assays}
    for assay in project.assays:
        if assay["name"] not in assays:
            assays[assay["name"]] = assay["id"]
    samples = {s["id"]: {s["id"], s["name"]} for s in project.samples}
    assignments = {(a["well"], a["assay_id"]): a for a in project.plate}
    records = []
    for well in experiment.wells:
        # Existing RDML IO uses :target suffixes for multiplex measurements.
        position = normalize_well(well.position.split(":", 1)[0])
        aid = assays.get(well.target_name)
        assignment = assignments.get((position, aid))
        if not assignment or well.sample_name not in samples[assignment["sample_id"]]:
            raise ValueError(
                f"源文件 {position}/{well.target_name}/{well.sample_name} "
                "与设计不匹配；"
                "请先修正板位或导入相符文件"
            )
        trace = experiment.traces.get(well.position)
        amplification = (
            []
            if trace is None
            else list(
                zip(trace.cycles.tolist(), trace.fluorescence.tolist(), strict=True)
            )
        )
        records.append(
            asdict(
                Measurement(
                    position,
                    aid,
                    well.cq,
                    amplification,
                    melting.get((position, well.target_name), []),
                )
            )
        )
    return validate_measurements(project, records)


def _read_workbench_csv(reader, project: Project) -> list[dict]:
    grouped = {}
    for row in reader:
        key = (normalize_well(row["well"]), row["assay_id"])
        item = grouped.setdefault(key, asdict(Measurement(*key)))
        ct = row.get("instrument_cq", "").strip()
        if ct:
            value = float(ct)
            if item["instrument_cq"] is not None and value != item["instrument_cq"]:
                raise ValueError("同一孔 / 检测项目存在冲突 Ct")
            item["instrument_cq"] = value
        kind = row["kind"].strip()
        if kind in {"amplification", "melting"}:
            item[kind].append([float(row["x"]), float(row["fluorescence"])])
        elif kind != "ct":
            raise ValueError("kind 必须为 ct、amplification 或 melting")
    return validate_measurements(project, list(grouped.values()))
