"""Extract small, attributed workbench examples from pinned public data.

Run with --source-dir pointing at downloaded originals. No fabricated curves,
control wells, primer sequences or sample quantities are added.
REAL DATA: replace these examples through the acquisition upload API; do not
edit the bundled data to represent a real instrument run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from pathlib import Path

from bioeng_qpcr.workbench.model import (
    AnalysisSettings,
    AssayRecord,
    Measurement,
    PlateAssignment,
    Project,
    SampleRecord,
)

COMMIT = "fce7a44e8b47b90759c646b557de568f7af77e75"
RDML_URL = (
    f"https://raw.githubusercontent.com/RDML-consortium/rdmlpython/{COMMIT}/"
    "experiments/untergasser/large_DNA_dilutions.rdml"
)
CT_URL = (
    "https://raw.githubusercontent.com/cran/pcr/"
    "6510657abfc6c92a4bb44ad3e131d10acb6ae6ec/inst/extdata/ct1.csv"
)


def prepare(source_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = source_dir / "large_DNA_dilutions.rdml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "2529de8272eb75c7579239d7735aa1251383118988e6fa4a7c7ef69fb97ef53d":
        raise ValueError("Unexpected RDML source SHA-256")
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("rdml_data.xml"))
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    target = "FSTL_1_F_109"
    assay_node = next(
        node for node in root.findall("target") if node.get("id") == target
    )
    assay = AssayRecord(
        id=target,
        name=target,
        forward=assay_node.findtext("sequences/forwardPrimer/sequence", ""),
        reverse=assay_node.findtext("sequences/reversePrimer/sequence", ""),
        channel="SYBR/FAM",
    )
    sample_nodes = {node.get("id"): node for node in root.findall("sample")}
    selected = {f"Dil 1:10 - 10e{i}" for i in range(3, 9)} | {
        "Buccal Swap 1",
        "Buccal Swap 2",
        "genomic DNA 0ng",
    }
    samples, plate, measurements, original_ids = {}, [], [], {}
    run = root.find("experiment/run")
    for reaction in run.findall("react"):
        sid = reaction.find("sample").get("id")
        if sid not in selected:
            continue
        for data in reaction.findall("data"):
            if data.find("tar").get("id") != target:
                continue
            node = sample_nodes[sid]
            quantity = node.findtext("quantity/value")
            samples[sid] = asdict(
                SampleRecord(
                    id=sid,
                    name=sid,
                    kind={"std": "standard", "ntc": "ntc"}.get(
                        node.findtext("type"), "unknown"
                    ),
                    quantity=float(quantity) if quantity else None,
                    unit=node.findtext("quantity/unit", "cop"),
                    notes=(
                        "标准量与 cop 单位沿用源文件，未假定为每反应拷贝数或每微升浓度"
                    ),
                )
            )
            rid = int(reaction.get("id")) - 1
            well = f"{chr(65 + rid // 24)}{rid % 24 + 1:02d}"
            original_ids[well] = reaction.get("id")
            plate.append(asdict(PlateAssignment(well, sid, target)))
            cq = data.findtext("cq")
            measurements.append(
                asdict(
                    Measurement(
                        well,
                        target,
                        float(cq) if cq and float(cq) >= 0 else None,
                        [
                            [float(p.findtext("cyc")), float(p.findtext("fluor"))]
                            for p in data.findall("adp")
                        ],
                        [
                            [float(p.findtext("tmp")), float(p.findtext("fluor"))]
                            for p in data.findall("mdp")
                        ],
                    )
                )
            )
    project = Project(
        name="公开示例 · DNA 稀释标准曲线",
        plate_format=384,
        samples=list(samples.values()),
        assays=[asdict(assay)],
        plate=plate,
        analysis_settings=asdict(AnalysisSettings(mode="absolute", target_id=target)),
    )
    project.recipe["volume_ul"] = 10
    for component in project.recipe["components"]:
        component["volume_ul"] /= 2
    project.recipe["provenance"] = (
        "10 µL 总体积来自 RDML react/vol；各组分比例为教学配置，非原实验配方"
    )
    project.sources = [
        {
            "title": "RDML Consortium / Untergasser · large_DNA_dilutions",
            "url": RDML_URL,
            "license": "MIT",
            "sha256": digest,
            "selection": (
                f"首个运行 {run.get('id')}；单个检测项目 {target}；保留原始 384 孔地址"
            ),
            "original_reaction_ids": original_ids,
            "limitations": (
                "仪器 Ct、扩增、熔解点和标准量来自原文件；通道名称是显示映射；"
                "程序为教学配置；选取的子集中未提供阴阳性对照"
            ),
        }
    ]
    write_example(output / "rdml-dilution.json", project, measurements)
    shutil.copyfile(source_dir / "LICENSE", output / "RDML-MIT-LICENSE.txt")

    path = source_dir / "ct1.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "d3dcd92d2991126cf5442f894a971477da75a7b07ed42659924700839ffebbf0":
        raise ValueError("Unexpected ct1 source SHA-256")
    rows = list(csv.DictReader(path.read_text().splitlines()))
    samples = [
        asdict(SampleRecord(id=name, name=name, group=name))
        for name in ("brain", "kidney")
    ]
    assays = [
        asdict(AssayRecord(id=name, name=name, reference=name == "GAPDH"))
        for name in ("c_myc", "GAPDH")
    ]
    plate, measurements = [], []
    for index, row in enumerate(rows):
        sid = "brain" if index < 6 else "kidney"
        for offset, target in enumerate(("c_myc", "GAPDH")):
            position = index * 2 + offset
            well = f"{chr(65 + position // 12)}{position % 12 + 1:02d}"
            plate.append(asdict(PlateAssignment(well, sid, target)))
            measurements.append(asdict(Measurement(well, target, float(row[target]))))
    project = Project(
        name="公开示例 · 脑 / 肾 ΔΔCt",
        samples=samples,
        assays=assays,
        plate=plate,
        analysis_settings=asdict(
            AnalysisSettings(
                target_id="c_myc", reference_id="GAPDH", calibrator_id="brain"
            )
        ),
    )
    project.sources = [
        {
            "title": "CRAN pcr ct1 · Livak brain/kidney example",
            "url": CT_URL,
            "license": (
                "GPL-3.0-only (third-party example data, separate from project code)"
            ),
            "sha256": digest,
            "selection": (
                "12 行 × 2 基因；每个组织 6 个技术重复；"
                "演示板位为重新排版，不是原始仪器孔位"
            ),
            "limitations": (
                "只有 Ct；没有原始扩增、熔解、引物序列或对照，不能据此推断缺失证据；"
                "配液和程序是教学配置"
            ),
            "method_url": "https://doi.org/10.1006/meth.2001.1262",
        }
    ]
    write_example(output / "relative-ct.json", project, measurements)


def write_example(path: Path, project: Project, measurements: list[dict]) -> None:
    data = {"project": asdict(project), "measurements": measurements}
    # Runtime creates fresh project/run IDs and timestamps; fixtures are stable.
    for key in ("id", "created_at", "updated_at"):
        data["project"].pop(key)
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"{path.name}: {len(measurements)} measurements")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/bioeng_qpcr/web/examples/qpcr_workbench"),
    )
    args = parser.parse_args()
    prepare(args.source_dir, args.output)
