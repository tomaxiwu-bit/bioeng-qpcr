"""Tests for :class:`bioeng_qpcr.validation.report.ValidationReport`.

These tests exercise:

* Summary counts and status-based filtering properties.
* ``overall_status`` escalation logic (FAIL > WARN > NOT_ASSESSABLE > PASS).
* ``assessment_passed`` semantics (``True`` only when every result is PASS).
* ``to_dict`` structure.
* ``to_json`` round-trip.
* ``to_text`` formatting (header, summary, grouped details, footer).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from bioeng_qpcr.domain.enums import RuleStatus
from bioeng_qpcr.validation.report import ValidationReport
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata(rule_id: str = "EFF-001", name: str = "Test Rule") -> RuleMetadata:
    return RuleMetadata(
        rule_id=rule_id,
        name=name,
        description="A test rule.",
        miqe_section="MIQE §1.0",
    )


def _result(
    status: RuleStatus = RuleStatus.PASS,
    rule_id: str = "EFF-001",
    message: str = "ok",
    evidence: dict | None = None,
) -> RuleResult:
    return RuleResult(
        metadata=_metadata(rule_id=rule_id),
        status=status,
        message=message,
        evidence=evidence or {},
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_all_pass(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1"), _result(RuleStatus.PASS, "R2")],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.summary == {"PASS": 2, "WARN": 0, "FAIL": 0, "NOT_ASSESSABLE": 0}

    def test_mixed_statuses(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.WARN, "R2"),
                _result(RuleStatus.FAIL, "R3"),
                _result(RuleStatus.NOT_ASSESSABLE, "R4"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.summary == {"PASS": 1, "WARN": 1, "FAIL": 1, "NOT_ASSESSABLE": 1}

    def test_empty_results(self) -> None:
        with pytest.raises(ValueError, match="at least one executed rule"):
            ValidationReport(results=[], experiment_name="exp", experiment_hash="abc")


# ---------------------------------------------------------------------------
# Status-based filtering
# ---------------------------------------------------------------------------


class TestStatusFiltering:
    def test_passed(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.FAIL, "R2"),
                _result(RuleStatus.PASS, "R3"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert len(report.passed) == 2
        assert {r.rule_id for r in report.passed} == {"R1", "R3"}

    def test_warnings(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.WARN, "R1"),
                _result(RuleStatus.PASS, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert len(report.warnings) == 1
        assert report.warnings[0].rule_id == "R1"

    def test_failures(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.FAIL, "R1"),
                _result(RuleStatus.PASS, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert len(report.failures) == 1
        assert report.failures[0].rule_id == "R1"

    def test_not_assessable(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.NOT_ASSESSABLE, "R1"),
                _result(RuleStatus.PASS, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert len(report.not_assessable) == 1
        assert report.not_assessable[0].rule_id == "R1"

    def test_filtering_returns_new_lists(self) -> None:
        """Properties must return fresh lists, not internal references."""
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="abc",
        )
        lst1 = report.passed
        lst2 = report.passed
        assert lst1 == lst2
        assert lst1 is not lst2


# ---------------------------------------------------------------------------
# overall_status
# ---------------------------------------------------------------------------


class TestOverallStatus:
    def test_all_pass(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1"), _result(RuleStatus.PASS, "R2")],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.overall_status == "PASS"

    def test_fail_dominates(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.WARN, "R2"),
                _result(RuleStatus.FAIL, "R3"),
                _result(RuleStatus.NOT_ASSESSABLE, "R4"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.overall_status == "FAIL"

    def test_warn_over_not_assessable(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.WARN, "R1"),
                _result(RuleStatus.NOT_ASSESSABLE, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.overall_status == "WARN"

    def test_not_assessable_when_no_fail_no_warn(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.NOT_ASSESSABLE, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.overall_status == "NOT_ASSESSABLE"

    def test_empty_results_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one executed rule"):
            ValidationReport(results=[], experiment_name="exp", experiment_hash="abc")


# ---------------------------------------------------------------------------
# compliant
# ---------------------------------------------------------------------------


class TestCompliant:
    def test_all_pass_is_compliant(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.compliant is True

    def test_warn_not_compliant(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.WARN, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.compliant is False

    def test_fail_not_compliant(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.FAIL, "R1")],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.compliant is False

    def test_not_assessable_not_compliant(self) -> None:
        """NOT_ASSESSABLE means the experiment cannot be certified."""
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.NOT_ASSESSABLE, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.compliant is False

    def test_assessment_passed_alias(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="abc",
        )
        assert report.assessment_passed is True
        assert report.compliant is report.assessment_passed


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_structure(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="my_exp",
            experiment_hash="deadbeef",
        )
        d = report.to_dict()
        assert d["experiment_name"] == "my_exp"
        assert d["experiment_hash"] == "deadbeef"
        assert "summary" in d
        assert "overall_status" in d
        assert "assessment_passed" in d
        assert "rule_coverage" in d
        assert "compliant" in d
        assert "results" in d
        assert len(d["results"]) == 1

    def test_result_dict_fields(self) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = RuleResult(
            metadata=_metadata("R1"),
            status=RuleStatus.WARN,
            message="borderline",
            evidence={"value": 0.89},
            timestamp=ts,
        )
        report = ValidationReport(
            results=[result], experiment_name="exp", experiment_hash="h"
        )
        rd = report.to_dict()["results"][0]
        assert rd["rule_id"] == "R1"
        assert rd["name"] == "Test Rule"
        assert rd["status"] == "WARN"
        assert rd["message"] == "borderline"
        assert rd["evidence"] == {"value": 0.89}
        assert rd["miqe_section"] == "MIQE §1.0"
        assert rd["timestamp"] == ts.isoformat()

    def test_summary_in_dict(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.FAIL, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        d = report.to_dict()
        assert d["summary"]["PASS"] == 1
        assert d["summary"]["FAIL"] == 1


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


class TestToJson:
    def test_valid_json(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_json()
        data = json.loads(text)
        assert data["experiment_name"] == "exp"

    def test_indent(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="h",
        )
        text2 = report.to_json(indent=2)
        text0 = report.to_json(indent=0)
        # indent=2 should produce more whitespace than indent=0.
        assert len(text2) > len(text0)

    def test_round_trip_keys(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.FAIL, "R2", message="bad"),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        data = json.loads(report.to_json())
        assert data["experiment_name"] == "exp"
        assert data["experiment_hash"] == "h"
        assert data["overall_status"] == "FAIL"
        assert data["compliant"] is False
        assert len(data["results"]) == 2
        assert data["results"][0]["rule_id"] == "R1"
        assert data["results"][1]["rule_id"] == "R2"

    def test_evidence_serialised(self) -> None:
        report = ValidationReport(
            results=[
                _result(
                    RuleStatus.WARN,
                    "R1",
                    evidence={"values": [1, 2, 3], "name": "test"},
                ),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        data = json.loads(report.to_json())
        assert data["results"][0]["evidence"] == {"values": [1, 2, 3], "name": "test"}


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------


class TestToText:
    def test_contains_experiment_name(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="my_experiment",
            experiment_hash="abc123",
        )
        text = report.to_text()
        assert "my_experiment" in text

    def test_contains_hash(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="abc123",
        )
        text = report.to_text()
        assert "abc123" in text

    def test_contains_summary_section(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R1"),
                _result(RuleStatus.FAIL, "R2"),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        assert "SUMMARY" in text
        assert "PASS:" in text
        assert "FAIL:" in text

    def test_contains_overall_status(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.FAIL, "R1")],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        assert "Overall status: FAIL" in text

    def test_contains_assessment_result(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        assert "All assessed rules passed: True" in text
        assert "not a certification" in text

    def test_fail_group_appears_before_pass(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.PASS, "R-PASS", message="good"),
                _result(RuleStatus.FAIL, "R-FAIL", message="bad"),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        fail_pos = text.index("[FAIL]")
        pass_pos = text.index("[PASS]")
        assert fail_pos < pass_pos

    def test_warn_before_not_assessable(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.WARN, "R-WARN"),
                _result(RuleStatus.NOT_ASSESSABLE, "R-NA"),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        warn_pos = text.index("[WARN]")
        na_pos = text.index("[NOT_ASSESSABLE]")
        assert warn_pos < na_pos

    def test_empty_group_omitted(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        assert "[FAIL]" not in text
        assert "[WARN]" not in text
        assert "[NOT_ASSESSABLE]" not in text
        assert "[PASS]" in text

    def test_contains_rule_message(self) -> None:
        report = ValidationReport(
            results=[
                _result(RuleStatus.FAIL, "R1", message="Efficiency too low"),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        assert "Efficiency too low" in text

    def test_contains_evidence_when_present(self) -> None:
        report = ValidationReport(
            results=[
                _result(
                    RuleStatus.WARN,
                    "R1",
                    evidence={"efficiency": 0.89},
                ),
            ],
            experiment_name="exp",
            experiment_hash="h",
        )
        text = report.to_text()
        assert "0.89" in text


# ---------------------------------------------------------------------------
# len and repr
# ---------------------------------------------------------------------------


class TestDunders:
    def test_len(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1"), _result(RuleStatus.PASS, "R2")],
            experiment_name="exp",
            experiment_hash="h",
        )
        assert len(report) == 2

    def test_repr(self) -> None:
        report = ValidationReport(
            results=[_result(RuleStatus.PASS, "R1")],
            experiment_name="my_exp",
            experiment_hash="h",
        )
        r = repr(report)
        assert "my_exp" in r
        assert "PASS" in r
