"""Tests for :func:`bioeng_qpcr.validation.decorator.validation_rule`.

These tests exercise:

* Metadata attachment (``fn.metadata``) and identity preservation.
* The decorated function's callability with a single ``Experiment``.
* ``functools.wraps`` preservation of ``__name__``, ``__doc__``, etc.
* Metadata substitution: the wrapper guarantees the returned
  :class:`RuleResult` carries the decorator's canonical metadata, even when
  the inner function returns a result built with a different metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> RuleMetadata:
    defaults = dict(
        rule_id="EFF-001",
        name="Efficiency range check",
        description="Amplification efficiency must be within 90-110%.",
        miqe_section="MIQE 3.2",
    )
    defaults.update(overrides)
    return RuleMetadata(**defaults)


def _make_experiment(name: str = "test-exp") -> Experiment:
    return Experiment(name=name)


# ---------------------------------------------------------------------------
# Metadata attachment
# ---------------------------------------------------------------------------


class TestMetadataAttachment:
    def test_metadata_attribute_is_set(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        assert my_rule.metadata is md  # type: ignore[attr-defined]

    def test_metadata_attribute_identity(self) -> None:
        """fn.metadata must be the exact object passed to the decorator."""
        md = _make_metadata(rule_id="NTC-001")

        @validation_rule(md)
        def rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        assert rule.metadata is md  # type: ignore[attr-defined]
        assert rule.metadata.rule_id == "NTC-001"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# functools.wraps preservation
# ---------------------------------------------------------------------------


class TestWrapsPreservation:
    def test_preserves_function_name(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def check_efficiency(experiment: Experiment) -> RuleResult:
            """Check that efficiency is within range."""
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        assert check_efficiency.__name__ == "check_efficiency"

    def test_preserves_docstring(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def check_efficiency(experiment: Experiment) -> RuleResult:
            """Check that efficiency is within range."""
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        assert check_efficiency.__doc__ == "Check that efficiency is within range."

    def test_preserves_module(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def check_efficiency(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        assert check_efficiency.__module__ == __name__

    def test_preserves_qualname(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def check_efficiency(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        assert check_efficiency.__qualname__.endswith("check_efficiency")


# ---------------------------------------------------------------------------
# Callability and result
# ---------------------------------------------------------------------------


class TestCallable:
    def test_returns_rule_result(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        result = my_rule(_make_experiment())
        assert isinstance(result, RuleResult)

    def test_callable_with_single_experiment(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(
                metadata=md,
                status=RuleStatus.PASS,
                message=f"Checked {experiment.name}",
            )

        exp = _make_experiment(name="my-experiment")
        result = my_rule(exp)
        assert "my-experiment" in result.message

    def test_preserves_status_and_message(self) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(
                metadata=md,
                status=RuleStatus.WARN,
                message="Borderline efficiency.",
                evidence={"efficiency": 0.89},
            )

        result = my_rule(_make_experiment())
        assert result.status is RuleStatus.WARN
        assert result.message == "Borderline efficiency."
        assert result.evidence == {"efficiency": 0.89}

    @pytest.mark.parametrize("status", list(RuleStatus))
    def test_works_with_all_statuses(self, status: RuleStatus) -> None:
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=status, message="m")

        result = my_rule(_make_experiment())
        assert result.status is status


# ---------------------------------------------------------------------------
# Metadata substitution
# ---------------------------------------------------------------------------


class TestMetadataSubstitution:
    def test_result_metadata_is_canonical(self) -> None:
        """The returned result's metadata must be the decorator's object."""
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            # Inner function uses the closure metadata.
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        result = my_rule(_make_experiment())
        assert result.metadata is md

    def test_substitutes_wrong_metadata(self) -> None:
        """If the inner function returns a different metadata, the wrapper
        replaces it with the canonical one."""
        canonical = _make_metadata(rule_id="EFF-001")
        wrong = _make_metadata(rule_id="EFF-999")

        @validation_rule(canonical)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=wrong, status=RuleStatus.PASS, message="ok")

        result = my_rule(_make_experiment())
        assert result.metadata is canonical
        assert result.metadata is not wrong
        assert result.rule_id == "EFF-001"

    def test_substitution_preserves_other_fields(self) -> None:
        """When the wrapper substitutes metadata, status, message, evidence
        and timestamp must survive."""
        canonical = _make_metadata(rule_id="EFF-001")
        wrong = _make_metadata(rule_id="EFF-999")
        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)

        @validation_rule(canonical)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(
                metadata=wrong,
                status=RuleStatus.FAIL,
                message="failed",
                evidence={"value": 42},
                timestamp=ts,
            )

        result = my_rule(_make_experiment())
        assert result.metadata is canonical
        assert result.status is RuleStatus.FAIL
        assert result.message == "failed"
        assert result.evidence == {"value": 42}
        assert result.timestamp == ts

    def test_no_substitution_when_metadata_matches(self) -> None:
        """When the inner function already uses the canonical metadata, the
        wrapper returns the result as-is (same object identity)."""
        md = _make_metadata()

        @validation_rule(md)
        def my_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")

        # Call twice to ensure stable behaviour.
        r1 = my_rule(_make_experiment())
        r2 = my_rule(_make_experiment())
        assert r1.metadata is md
        assert r2.metadata is md


# ---------------------------------------------------------------------------
# Multiple independent rules
# ---------------------------------------------------------------------------


class TestMultipleRules:
    def test_two_rules_have_independent_metadata(self) -> None:
        md_a = _make_metadata(rule_id="EFF-001")
        md_b = _make_metadata(rule_id="NTC-001", name="NTC contamination check")

        @validation_rule(md_a)
        def rule_a(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md_a, status=RuleStatus.PASS, message="a")

        @validation_rule(md_b)
        def rule_b(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md_b, status=RuleStatus.FAIL, message="b")

        assert rule_a.metadata is md_a  # type: ignore[attr-defined]
        assert rule_b.metadata is md_b  # type: ignore[attr-defined]

        ra = rule_a(_make_experiment())
        rb = rule_b(_make_experiment())
        assert ra.metadata is md_a
        assert rb.metadata is md_b
        assert ra.rule_id == "EFF-001"
        assert rb.rule_id == "NTC-001"

    def test_rules_with_different_input_levels(self) -> None:
        md_a = _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A)
        md_c = _make_metadata(rule_id="R-C", min_input_level=InputLevel.LEVEL_C)

        @validation_rule(md_a)
        def rule_a(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md_a, status=RuleStatus.PASS, message="a")

        @validation_rule(md_c)
        def rule_c(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=md_c, status=RuleStatus.PASS, message="c")

        assert rule_a.metadata.min_input_level is InputLevel.LEVEL_A  # type: ignore[attr-defined]
        assert rule_c.metadata.min_input_level is InputLevel.LEVEL_C  # type: ignore[attr-defined]
