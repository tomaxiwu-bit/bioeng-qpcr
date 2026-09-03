"""Tests for :class:`bioeng_qpcr.validation.engine.ValidationEngine`.

These tests exercise:

* Running the engine against golden datasets and verifying the correct
  number of results and statuses.
* Input-level filtering (rules with a higher ``min_input_level`` than the
  experiment's level are excluded).
* Constructor ``input_level`` override.
* Strict exception handling (a software error raises rather than becoming
  scientific ``NOT_ASSESSABLE`` evidence).
* The ``register_rule`` proxy method.
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.engine import RuleExecutionError, ValidationEngine
from bioeng_qpcr.validation.registry import RuleRegistry
from bioeng_qpcr.validation.rules import register_all
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult
from tests.fixtures.golden_dataset import (
    make_golden_fail_eff,
    make_golden_minimal,
    make_golden_no_ntc,
    make_golden_no_ref,
    make_golden_ntc_amplification,
    make_golden_pass,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_and_register():
    """Clear the registry and re-register the built-in rules."""
    registry = RuleRegistry()
    registry.clear()
    register_all()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# Basic run
# ---------------------------------------------------------------------------


class TestEngineRun:
    def test_returns_validation_report(self) -> None:
        from bioeng_qpcr.validation.report import ValidationReport

        engine = ValidationEngine()
        report = engine.run(make_golden_pass())
        assert isinstance(report, ValidationReport)

    def test_seven_results_for_level_a(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_pass())
        assert len(report) == 7

    def test_golden_core_pass_with_readiness_gaps(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_pass())
        assert len(report.passed) == 4
        assert report.compliant is False
        assert report.overall_status == "WARN"

    def test_golden_fail_eff(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_fail_eff())
        assert len(report) == 7
        assert len(report.failures) == 1
        assert report.failures[0].rule_id == "EFF-001"
        assert report.compliant is False
        assert report.overall_status == "FAIL"

    def test_golden_no_ref(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_no_ref())
        assert len(report) == 7
        assert {result.rule_id for result in report.not_assessable} >= {
            "REF-001",
            "REF-002",
        }
        assert report.compliant is False
        assert report.overall_status == "WARN"

    def test_golden_no_ntc(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_no_ntc())
        assert len(report) == 7
        assert "NTC-001" in {result.rule_id for result in report.not_assessable}
        assert report.compliant is False

    def test_golden_ntc_amplification(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_ntc_amplification())
        assert len(report) == 7
        assert len(report.failures) == 1
        assert report.failures[0].rule_id == "NTC-001"
        assert report.compliant is False

    def test_golden_minimal_all_not_assessable(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_minimal())
        assert len(report) == 7
        assert len(report.not_assessable) == 5
        assert report.compliant is False
        assert report.overall_status == "WARN"

    def test_experiment_name_captured(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_pass())
        assert report.experiment_name == "golden_pass"

    def test_experiment_hash_captured(self) -> None:
        engine = ValidationEngine()
        exp = make_golden_pass()
        report = engine.run(exp)
        assert report.experiment_hash == exp.content_hash()

    def test_results_sorted_by_rule_id(self) -> None:
        engine = ValidationEngine()
        report = engine.run(make_golden_pass())
        rule_ids = [r.rule_id for r in report.results]
        assert rule_ids == sorted(rule_ids)


# ---------------------------------------------------------------------------
# Input-level filtering
# ---------------------------------------------------------------------------


class TestEngineLevelFiltering:
    def test_level_a_runs_only_level_a_rules(self) -> None:
        """A LEVEL_C rule should not run at LEVEL_A."""
        # Register a LEVEL_C rule.
        level_c_md = RuleMetadata(
            rule_id="LVL-C-001",
            name="Level C Only",
            description="Requires level C",
            miqe_section="TEST",
            min_input_level=InputLevel.LEVEL_C,
        )

        @validation_rule(level_c_md)
        def level_c_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=level_c_md, status=RuleStatus.PASS, message="ok")

        RuleRegistry().register(level_c_md, level_c_rule)

        engine = ValidationEngine()  # uses experiment.input_level
        report = engine.run(make_golden_pass())  # input_level = LEVEL_B
        rule_ids = [r.rule_id for r in report.results]
        assert "LVL-C-001" not in rule_ids
        assert len(report) == 7

    def test_level_c_runs_all_rules(self) -> None:
        """At LEVEL_C, both LEVEL_A and LEVEL_C rules should run."""
        level_c_md = RuleMetadata(
            rule_id="LVL-C-001",
            name="Level C Only",
            description="Requires level C",
            miqe_section="TEST",
            min_input_level=InputLevel.LEVEL_C,
        )

        @validation_rule(level_c_md)
        def level_c_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=level_c_md, status=RuleStatus.PASS, message="ok")

        RuleRegistry().register(level_c_md, level_c_rule)

        engine = ValidationEngine(input_level=InputLevel.LEVEL_C)
        report = engine.run(make_golden_pass())
        assert len(report) == 8
        rule_ids = [r.rule_id for r in report.results]
        assert "LVL-C-001" in rule_ids


# ---------------------------------------------------------------------------
# Constructor input_level override
# ---------------------------------------------------------------------------


class TestEngineInputLevelOverride:
    def test_constructor_overrides_experiment_level(self) -> None:
        """Engine input_level takes precedence over experiment.input_level."""
        level_c_md = RuleMetadata(
            rule_id="LVL-C-001",
            name="Level C Only",
            description="Requires level C",
            miqe_section="TEST",
            min_input_level=InputLevel.LEVEL_C,
        )

        @validation_rule(level_c_md)
        def level_c_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=level_c_md, status=RuleStatus.PASS, message="ok")

        RuleRegistry().register(level_c_md, level_c_rule)

        # Experiment is LEVEL_B, but engine forces LEVEL_C.
        exp = make_golden_pass()  # input_level = LEVEL_B
        assert exp.input_level is InputLevel.LEVEL_B

        engine = ValidationEngine(input_level=InputLevel.LEVEL_C)
        report = engine.run(exp)
        assert len(report) == 8  # 7 built-ins + 1 level-C rule

    def test_no_override_uses_experiment_level(self) -> None:
        exp = make_golden_pass()
        engine = ValidationEngine()  # no override
        report = engine.run(exp)
        assert len(report) == 7


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


class TestEngineExceptionHandling:
    def test_rule_exception_raises_in_strict_mode(self) -> None:
        error_md = RuleMetadata(
            rule_id="ERR-001",
            name="Error Rule",
            description="Always raises",
            miqe_section="TEST",
        )

        @validation_rule(error_md)
        def error_rule(experiment: Experiment) -> RuleResult:
            raise RuntimeError("Something went wrong")

        RuleRegistry().register(error_md, error_rule)

        engine = ValidationEngine()
        with pytest.raises(RuleExecutionError, match="ERR-001") as caught:
            engine.run(make_golden_pass())
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_other_rules_still_run_after_exception(self) -> None:
        error_md = RuleMetadata(
            rule_id="ERR-001",
            name="Error Rule",
            description="Always raises",
            miqe_section="TEST",
        )

        @validation_rule(error_md)
        def error_rule(experiment: Experiment) -> RuleResult:
            raise ValueError("boom")

        RuleRegistry().register(error_md, error_rule)

        engine = ValidationEngine(strict=False)
        report = engine.run(make_golden_pass())
        # The 3 built-in rules should still produce normal results.
        built_in_ids = {"EFF-001", "REF-001", "NTC-001"}
        built_in_results = [r for r in report.results if r.rule_id in built_in_ids]
        assert len(built_in_results) == 3
        # All built-in results should be PASS for golden_pass.
        assert all(r.is_pass for r in built_in_results)
        err_result = next(r for r in report.results if r.rule_id == "ERR-001")
        assert err_result.evidence["execution_error"] is True


class TestEngineRuleLoading:
    def test_constructor_restores_builtins_after_registry_clear(self) -> None:
        RuleRegistry().clear()
        report = ValidationEngine().run(make_golden_minimal())
        assert {result.rule_id for result in report.results} == {
            "ASSAY-001",
            "EFF-001",
            "INPUT-001",
            "NTC-001",
            "REF-001",
            "REF-002",
            "REP-001",
        }

    def test_report_records_rules_skipped_by_input_level(self) -> None:
        level_c_md = RuleMetadata(
            rule_id="LVL-C-001",
            name="Level C Only",
            description="Requires level C",
            miqe_section="TEST",
            min_input_level=InputLevel.LEVEL_C,
        )

        @validation_rule(level_c_md)
        def level_c_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(metadata=level_c_md, status=RuleStatus.PASS, message="ok")

        RuleRegistry().register(level_c_md, level_c_rule)
        report = ValidationEngine().run(make_golden_pass())
        assert report.rule_coverage == {
            "available": 8,
            "executed": 7,
            "skipped_input_level": 1,
            "skipped_rule_ids": ["LVL-C-001"],
        }


# ---------------------------------------------------------------------------
# register_rule proxy
# ---------------------------------------------------------------------------


class TestEngineRegisterRule:
    def test_register_rule_adds_to_registry(self) -> None:
        custom_md = RuleMetadata(
            rule_id="CUSTOM-001",
            name="Custom",
            description="Custom rule",
            miqe_section="TEST",
        )

        @validation_rule(custom_md)
        def custom_rule(experiment: Experiment) -> RuleResult:
            return RuleResult(
                metadata=custom_md, status=RuleStatus.PASS, message="custom"
            )

        engine = ValidationEngine()
        engine.register_rule(custom_md, custom_rule)

        assert "CUSTOM-001" in RuleRegistry()
        report = engine.run(make_golden_pass())
        assert len(report) == 8
