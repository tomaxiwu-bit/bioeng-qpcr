"""Tests for the built-in MIQE-readiness validation rules.

Each rule is tested against the golden datasets as well as targeted
edge-case experiments constructed inline.  The tests verify:

* Correct :class:`~bioeng_qpcr.domain.enums.RuleStatus` for each scenario.
* Meaningful human-readable messages.
* Structured evidence dictionaries with the expected keys and values.

A shared autouse fixture clears the global :class:`RuleRegistry` and
re-registers the built-in rules before every test so that test
isolation is guaranteed.
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Well
from bioeng_qpcr.domain.enums import RuleStatus
from bioeng_qpcr.validation.registry import RuleRegistry
from bioeng_qpcr.validation.rules import register_all
from bioeng_qpcr.validation.rules.assay_001 import check_assay_characterization
from bioeng_qpcr.validation.rules.eff_001 import check_efficiency
from bioeng_qpcr.validation.rules.ntc_001 import check_ntc
from bioeng_qpcr.validation.rules.ref_001 import check_reference_gene
from bioeng_qpcr.validation.rules.rep_001 import check_technical_replicates
from bioeng_qpcr.validation.types import RuleResult
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
# EFF-001
# ---------------------------------------------------------------------------


class TestEff001Pass:
    def test_status_is_pass(self) -> None:
        result = check_efficiency(make_golden_pass())
        assert result.status is RuleStatus.PASS

    def test_message_mentions_all_assays(self) -> None:
        result = check_efficiency(make_golden_pass())
        assert "2 assay(s)" in result.message

    def test_evidence_contains_efficiencies(self) -> None:
        result = check_efficiency(make_golden_pass())
        eff = result.evidence["efficiencies"]
        assert eff["GAPDH"] == pytest.approx(0.98)
        assert eff["TARGET"] == pytest.approx(1.02)

    def test_evidence_counts(self) -> None:
        result = check_efficiency(make_golden_pass())
        assert result.evidence["valid_count"] == 2
        assert result.evidence["invalid_count"] == 0
        assert result.evidence["missing_count"] == 0

    def test_returns_rule_result(self) -> None:
        result = check_efficiency(make_golden_pass())
        assert isinstance(result, RuleResult)
        assert result.rule_id == "EFF-001"


class TestEff001Fail:
    def test_status_is_fail(self) -> None:
        result = check_efficiency(make_golden_fail_eff())
        assert result.status is RuleStatus.FAIL

    def test_message_mentions_target(self) -> None:
        result = check_efficiency(make_golden_fail_eff())
        assert "TARGET" in result.message

    def test_evidence_invalid_count(self) -> None:
        result = check_efficiency(make_golden_fail_eff())
        assert result.evidence["invalid_count"] == 1
        assert result.evidence["valid_count"] == 1
        assert "TARGET" in result.evidence["invalid_assays"]

    def test_evidence_invalid_value(self) -> None:
        result = check_efficiency(make_golden_fail_eff())
        assert result.evidence["efficiencies"]["TARGET"] == pytest.approx(1.20)


class TestEff001NotAssessable:
    def test_status_is_not_assessable(self) -> None:
        result = check_efficiency(make_golden_minimal())
        assert result.status is RuleStatus.NOT_ASSESSABLE

    def test_message(self) -> None:
        result = check_efficiency(make_golden_minimal())
        assert "No efficiency data" in result.message

    def test_evidence_missing_count(self) -> None:
        result = check_efficiency(make_golden_minimal())
        assert result.evidence["missing_count"] == 2
        assert set(result.evidence["missing_assays"]) == {"GAPDH", "TARGET"}


class TestEff001WarnPartial:
    """Some assays have efficiency, some do not -> WARN."""

    @pytest.fixture
    def partial_experiment(self) -> Experiment:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=0.98),
            "TARGET": Assay(name="TARGET", efficiency=None),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.5),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.3),
        ]
        return Experiment(name="partial_eff", wells=wells, assays=assays)

    def test_status_is_warn(self, partial_experiment: Experiment) -> None:
        result = check_efficiency(partial_experiment)
        assert result.status is RuleStatus.WARN

    def test_message_lists_missing(self, partial_experiment: Experiment) -> None:
        result = check_efficiency(partial_experiment)
        assert "TARGET" in result.message

    def test_evidence_missing_assays(self, partial_experiment: Experiment) -> None:
        result = check_efficiency(partial_experiment)
        assert result.evidence["missing_count"] == 1
        assert result.evidence["missing_assays"] == ["TARGET"]
        assert result.evidence["valid_count"] == 1

    def test_invalid_value_dominates_missing_value(self) -> None:
        experiment = Experiment(
            name="partial_invalid_eff",
            assays={
                "BAD": Assay(name="BAD", efficiency=1.20),
                "MISSING": Assay(name="MISSING", efficiency=None),
            },
        )
        result = check_efficiency(experiment)
        assert result.status is RuleStatus.FAIL
        assert result.evidence["invalid_assays"] == ["BAD"]
        assert result.evidence["missing_assays"] == ["MISSING"]


class TestEff001EdgeCases:
    def test_empty_assays_is_not_assessable(self) -> None:
        exp = Experiment(name="empty")
        result = check_efficiency(exp)
        assert result.status is RuleStatus.NOT_ASSESSABLE

    def test_boundary_0_90_passes(self) -> None:
        assays = {"G": Assay(name="G", efficiency=0.90)}
        exp = Experiment(name="boundary_low", wells=[], assays=assays)
        result = check_efficiency(exp)
        assert result.status is RuleStatus.PASS

    def test_boundary_1_10_passes(self) -> None:
        assays = {"G": Assay(name="G", efficiency=1.10)}
        exp = Experiment(name="boundary_high", wells=[], assays=assays)
        result = check_efficiency(exp)
        assert result.status is RuleStatus.PASS

    def test_below_boundary_fails(self) -> None:
        assays = {"G": Assay(name="G", efficiency=0.89)}
        exp = Experiment(name="below", wells=[], assays=assays)
        result = check_efficiency(exp)
        assert result.status is RuleStatus.FAIL


# ---------------------------------------------------------------------------
# REF-001
# ---------------------------------------------------------------------------


class TestRef001Pass:
    def test_status_is_pass(self) -> None:
        result = check_reference_gene(make_golden_pass())
        assert result.status is RuleStatus.PASS

    def test_message_lists_reference(self) -> None:
        result = check_reference_gene(make_golden_pass())
        assert "GAPDH" in result.message

    def test_evidence_reference_targets(self) -> None:
        result = check_reference_gene(make_golden_pass())
        assert result.evidence["reference_targets"] == ["GAPDH"]
        assert result.evidence["total_assay_count"] == 2

    def test_rule_id(self) -> None:
        result = check_reference_gene(make_golden_pass())
        assert result.rule_id == "REF-001"


class TestRef001NotAssessable:
    def test_status_is_not_assessable(self) -> None:
        result = check_reference_gene(make_golden_no_ref())
        assert result.status is RuleStatus.NOT_ASSESSABLE

    def test_message(self) -> None:
        result = check_reference_gene(make_golden_no_ref())
        assert "No reference gene" in result.message

    def test_evidence_empty_reference_targets(self) -> None:
        result = check_reference_gene(make_golden_no_ref())
        assert result.evidence["reference_targets"] == []
        assert result.evidence["total_assay_count"] == 2

    def test_minimal_experiment_is_not_assessable(self) -> None:
        result = check_reference_gene(make_golden_minimal())
        assert result.status is RuleStatus.NOT_ASSESSABLE


class TestRef001MultipleReferences:
    def test_multiple_reference_targets(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=True),
            "ACTB": Assay(name="ACTB", efficiency=1.02, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        exp = Experiment(name="multi_ref", wells=[], assays=assays)
        result = check_reference_gene(exp)
        assert result.status is RuleStatus.PASS
        assert result.evidence["reference_targets"] == ["ACTB", "GAPDH"]
        assert result.evidence["total_assay_count"] == 3


# ---------------------------------------------------------------------------
# NTC-001
# ---------------------------------------------------------------------------


class TestNtc001Pass:
    def test_status_is_pass(self) -> None:
        result = check_ntc(make_golden_pass())
        assert result.status is RuleStatus.PASS

    def test_message(self) -> None:
        result = check_ntc(make_golden_pass())
        assert "no amplification" in result.message

    def test_evidence_ntc_wells(self) -> None:
        result = check_ntc(make_golden_pass())
        assert set(result.evidence["ntc_wells"]) == {"A05", "A06"}
        assert result.evidence["amplified_count"] == 0

    def test_evidence_cq_values_all_none(self) -> None:
        result = check_ntc(make_golden_pass())
        cq = result.evidence["cq_values"]
        assert all(v is None for v in cq.values())

    def test_rule_id(self) -> None:
        result = check_ntc(make_golden_pass())
        assert result.rule_id == "NTC-001"


class TestNtc001NotAssessable:
    def test_status_is_not_assessable(self) -> None:
        result = check_ntc(make_golden_no_ntc())
        assert result.status is RuleStatus.NOT_ASSESSABLE

    def test_message(self) -> None:
        result = check_ntc(make_golden_no_ntc())
        assert "No NTC wells" in result.message

    def test_evidence_empty(self) -> None:
        result = check_ntc(make_golden_no_ntc())
        assert result.evidence["ntc_wells"] == []

    def test_minimal_experiment_is_not_assessable(self) -> None:
        result = check_ntc(make_golden_minimal())
        assert result.status is RuleStatus.NOT_ASSESSABLE


class TestNtc001Fail:
    def test_status_is_fail(self) -> None:
        result = check_ntc(make_golden_ntc_amplification())
        assert result.status is RuleStatus.FAIL

    def test_message_mentions_contamination(self) -> None:
        result = check_ntc(make_golden_ntc_amplification())
        assert "contamination" in result.message.lower()

    def test_evidence_failing_wells(self) -> None:
        result = check_ntc(make_golden_ntc_amplification())
        assert result.evidence["amplified_count"] == 2
        failing = result.evidence["failing_wells"]
        assert len(failing) == 2
        assert all(cq < 35 for cq in failing.values())


class TestNtc001Warn:
    """NTC with late amplification (Cq >= 35) -> WARN."""

    @pytest.fixture
    def late_ntc_experiment(self) -> Experiment:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=True),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.5),
            Well(
                position="A05",
                sample_name="NTC",
                target_name="GAPDH",
                cq=36.0,
                cq_status="no_template",
            ),
        ]
        return Experiment(
            name="late_ntc",
            wells=wells,
            assays=assays,
            metadata={"ntc_fail_threshold": 35.0},
        )

    def test_status_is_warn(self, late_ntc_experiment: Experiment) -> None:
        result = check_ntc(late_ntc_experiment)
        assert result.status is RuleStatus.WARN

    def test_message_mentions_late(self, late_ntc_experiment: Experiment) -> None:
        result = check_ntc(late_ntc_experiment)
        assert "late" in result.message.lower()

    def test_evidence_warning_wells(self, late_ntc_experiment: Experiment) -> None:
        result = check_ntc(late_ntc_experiment)
        assert result.evidence["amplified_count"] == 1
        assert len(result.evidence["warning_wells"]) == 1
        assert len(result.evidence["failing_wells"]) == 0


class TestNtc001BoundaryCq35:
    """Cq exactly 35 should be WARN (35 >= 35)."""

    def test_cq_35_is_warn(self) -> None:
        wells = [
            Well(
                position="A05",
                sample_name="NTC",
                target_name="GAPDH",
                cq=35.0,
                cq_status="no_template",
            ),
        ]
        exp = Experiment(
            name="cq35",
            wells=wells,
            metadata={"ntc_fail_threshold": 35.0},
        )
        result = check_ntc(exp)
        assert result.status is RuleStatus.WARN

    def test_cq_34_9_is_fail(self) -> None:
        wells = [
            Well(
                position="A05",
                sample_name="NTC",
                target_name="GAPDH",
                cq=34.9,
                cq_status="no_template",
            ),
        ]
        exp = Experiment(
            name="cq349",
            wells=wells,
            metadata={"ntc_fail_threshold": 35.0},
        )
        result = check_ntc(exp)
        assert result.status is RuleStatus.FAIL


class TestNtc001MixedNtc:
    """Some NTC clean, some amplified -> FAIL if any Cq < 35."""

    def test_mixed_clean_and_fail(self) -> None:
        wells = [
            Well(
                position="A05",
                sample_name="NTC",
                target_name="GAPDH",
                cq=None,
                cq_status="no_template",
            ),
            Well(
                position="A06",
                sample_name="NTC",
                target_name="TARGET",
                cq=28.0,
                cq_status="no_template",
            ),
        ]
        exp = Experiment(
            name="mixed_ntc",
            wells=wells,
            metadata={"ntc_fail_threshold": 35.0},
        )
        result = check_ntc(exp)
        assert result.status is RuleStatus.FAIL
        assert result.evidence["amplified_count"] == 1


class TestAssayCharacterizationPolicy:
    @staticmethod
    def experiment(*, metadata: dict[str, object] | None = None) -> Experiment:
        assay = Assay(
            name="GOI",
            efficiency=0.95,
            r2=0.97,
            slope=-3.45,
            lod_cq=38.0,
            lloq_cq=35.0,
        )
        return Experiment(
            name="assay-policy",
            assays={"GOI": assay},
            metadata=metadata or {},
        )

    def test_low_r2_without_declared_policy_warns(self) -> None:
        result = check_assay_characterization(self.experiment())
        assert result.status is RuleStatus.WARN
        assert result.evidence["declared_minimum_r2"] is None

    def test_declared_r2_policy_can_fail(self) -> None:
        result = check_assay_characterization(
            self.experiment(metadata={"assay_min_r2": 0.98})
        )
        assert result.status is RuleStatus.FAIL
        assert result.evidence["declared_policy_failures"] == {"GOI": 0.97}

    def test_invalid_declared_policy_warns_instead_of_crashing(self) -> None:
        result = check_assay_characterization(
            self.experiment(metadata={"assay_min_r2": "not-a-number"})
        )
        assert result.status is RuleStatus.WARN
        assert result.evidence["invalid_declared_minimum_r2"] is True


def test_invalid_replicate_threshold_warns_instead_of_crashing() -> None:
    experiment = Experiment(
        name="invalid-replicate-policy",
        metadata={"technical_replicate_max_delta_cq": "bad"},
    )
    result = check_technical_replicates(experiment)
    assert result.status is RuleStatus.WARN
    assert result.evidence["threshold_delta_cq"] is None
