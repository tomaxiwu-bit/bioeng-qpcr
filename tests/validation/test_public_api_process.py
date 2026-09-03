"""Clean-process regression tests for the installed-style public API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_public_api_loads_builtin_rules_in_clean_process() -> None:
    """The documented import path must never yield a zero-rule pass."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    code = """
import json
from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.validation import ValidationEngine

report = ValidationEngine().run(Experiment(name="clean-process"))
print(json.dumps({
    "rules_run": len(report),
    "rule_ids": [result.rule_id for result in report.results],
    "overall_status": report.overall_status,
    "assessment_passed": report.assessment_passed,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "rules_run": 7,
        "rule_ids": [
            "ASSAY-001",
            "EFF-001",
            "INPUT-001",
            "NTC-001",
            "REF-001",
            "REF-002",
            "REP-001",
        ],
        "overall_status": "NOT_ASSESSABLE",
        "assessment_passed": False,
    }
