"""End-to-end tests for the combined research report."""

from __future__ import annotations

import json
from pathlib import Path

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
