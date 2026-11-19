"""End-to-end tests for the combined research report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from bioeng_qpcr.io import read_tidy_csv
from bioeng_qpcr.reporting import build_research_report

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "researcher_example.csv"


def test_report_combines_readiness_analysis_and_provenance() -> None:
    experiment = read_tidy_csv(
        EXAMPLE,
        reference_targets=["GAPDH", "RPLP0"],
    )
    report = build_research_report(
        experiment,
        "IL6",
        ["GAPDH", "RPLP0"],
    )
    payload = json.loads(report.to_json())
    assert payload["report_type"] == "bioeng-qpcr researcher report"
    assert payload["experiment"]["raw_input_hash"]
    assert payload["analysis"]["summary"]["result_count"] == 4
    assert payload["analysis"]["summary"]["exclusion_count"] == 0
    assert payload["provenance"]["ruleset_versions"]["NTC-001"] == "2.0.0"
    assert "Alpha research software" in report.html
    assert "Efficiency-corrected normalized quantities" in report.html


def test_report_embeds_miqe2_checklist() -> None:
    experiment = read_tidy_csv(
        EXAMPLE,
        reference_targets=["GAPDH", "RPLP0"],
    )
    report = build_research_report(experiment, "IL6", ["GAPDH", "RPLP0"])
    payload = json.loads(report.to_json())

    miqe = payload["miqe2"]
    assert miqe["standard"] == "MIQE 2.0 (2025)"
    total = sum(miqe["summary"].values())
    assert total == len(miqe["items"])
    # the RDML-agnostic CSV example still asserts reference genes and NTC wells
    reported = {
        item["field_id"] for item in miqe["items"] if item["status"] == "reported"
    }
    assert "analysis_normalization" in reported
    assert "analysis_nc_frequency" in reported
    # declaring reference genes is not a validation *method* -> not overstated
    assert "analysis_reference_validation" not in reported
    assert miqe["evidence_gap_plan"]

    html = report.html
    assert "MIQE 2.0 (2025) reporting checklist" in html
    assert "Prioritized evidence-gap plan" in html
    assert "Data Analysis" in html
    assert "essential completeness" in html.lower()


def test_report_accepts_documented_miqe_evidence() -> None:
    experiment = read_tidy_csv(EXAMPLE, reference_targets=["GAPDH", "RPLP0"])
    report = build_research_report(
        experiment,
        "IL6",
        ["GAPDH", "RPLP0"],
        miqe_answers={"protocol_instrument": "Bio-Rad CFX96, serial CFX-001"},
    )
    payload = json.loads(report.to_json())
    miqe = payload["miqe2"]
    instrument = next(
        item for item in miqe["items"] if item["field_id"] == "protocol_instrument"
    )
    assert instrument["status"] == "reported"
    assert instrument["evidence"] == "Bio-Rad CFX96, serial CFX-001"
    evidence_hash = payload["provenance"]["parameters"]["miqe2_manual_evidence_sha256"]
    changed = build_research_report(
        experiment,
        "IL6",
        ["GAPDH", "RPLP0"],
        miqe_answers={"protocol_instrument": "QuantStudio 7 Pro"},
    )
    changed_hash = json.loads(changed.to_json())["provenance"]["parameters"][
        "miqe2_manual_evidence_sha256"
    ]
    assert evidence_hash != changed_hash


def test_report_writes_verifiable_research_package(tmp_path: Path) -> None:
    experiment = read_tidy_csv(EXAMPLE, reference_targets=["GAPDH", "RPLP0"])
    report = build_research_report(experiment, "IL6", ["GAPDH", "RPLP0"])
    package = report.write_package(tmp_path / "research.zip", source_file=EXAMPLE)

    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert {
            "report.html",
            "report.json",
            "miqe2.json",
            "provenance.json",
            "ro-crate-metadata.json",
            "checksums.sha256",
            f"input/{EXAMPLE.name}",
        } <= names
        for line in archive.read("checksums.sha256").decode("utf-8").splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
