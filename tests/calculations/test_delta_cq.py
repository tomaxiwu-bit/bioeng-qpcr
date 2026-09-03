"""Tests for the ΔCq (delta-Cq) quantification method.

Uses synthetic experiments with known Cq values to verify:

* Basic ΔCq calculation.
* Mean Cq across technical replicates.
* Error when no reference gene is available.
* Error when Cq values are ``None`` (undetermined).
* The ``delta_cq_all`` batch function.
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.calculations.delta_cq import (
    DeltaCqResult,
    delta_cq,
    delta_cq_all,
)
from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.domain.enums import ReplicateType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _basic_assays() -> dict[str, Assay]:
    return {
        "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
        "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
    }


def _basic_samples() -> dict[str, Sample]:
    return {"S1": Sample(name="S1"), "S2": Sample(name="S2")}


# ---------------------------------------------------------------------------
# Basic ΔCq
# ---------------------------------------------------------------------------


class TestDeltaCqBasic:
    """Single-well-per-target ΔCq with known values."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.0),
        ]
        return Experiment(
            name="basic",
            wells=wells,
            samples=_basic_samples(),
            assays=_basic_assays(),
        )

    def test_returns_delta_cq_result(self, experiment: Experiment) -> None:
        result = delta_cq(experiment, "S1", "TARGET")
        assert isinstance(result, DeltaCqResult)

    def test_delta_cq_value(self, experiment: Experiment) -> None:
        # ΔCq = 18.0 − 20.0 = −2.0
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.delta_cq == pytest.approx(-2.0)

    def test_mean_cq_target(self, experiment: Experiment) -> None:
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.mean_cq_target == pytest.approx(18.0)

    def test_mean_cq_reference(self, experiment: Experiment) -> None:
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.mean_cq_reference == pytest.approx(20.0)

    def test_reference_target_auto_resolved(self, experiment: Experiment) -> None:
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.reference_target == "GAPDH"

    def test_sample_and_target_names_preserved(self, experiment: Experiment) -> None:
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.sample_name == "S1"
        assert result.target_name == "TARGET"

    def test_explicit_reference_target(self, experiment: Experiment) -> None:
        result = delta_cq(experiment, "S1", "TARGET", reference_target="GAPDH")
        assert result.delta_cq == pytest.approx(-2.0)
        assert result.reference_target == "GAPDH"


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------


class TestDeltaCqReplicates:
    """ΔCq with technical replicates -- mean Cq should be used."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        wells = [
            Well(
                position="A01",
                sample_name="S1",
                target_name="GAPDH",
                cq=20.0,
                replicate_type=ReplicateType.TECHNICAL,
            ),
            Well(
                position="A02",
                sample_name="S1",
                target_name="GAPDH",
                cq=20.2,
                replicate_type=ReplicateType.TECHNICAL,
            ),
            Well(
                position="A03",
                sample_name="S1",
                target_name="TARGET",
                cq=18.0,
                replicate_type=ReplicateType.TECHNICAL,
            ),
            Well(
                position="A04",
                sample_name="S1",
                target_name="TARGET",
                cq=18.2,
                replicate_type=ReplicateType.TECHNICAL,
            ),
        ]
        return Experiment(
            name="replicates",
            wells=wells,
            samples=_basic_samples(),
            assays=_basic_assays(),
        )

    def test_mean_cq_target_with_replicates(self, experiment: Experiment) -> None:
        # (18.0 + 18.2) / 2 = 18.1
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.mean_cq_target == pytest.approx(18.1)

    def test_mean_cq_reference_with_replicates(self, experiment: Experiment) -> None:
        # (20.0 + 20.2) / 2 = 20.1
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.mean_cq_reference == pytest.approx(20.1)

    def test_delta_cq_with_replicates(self, experiment: Experiment) -> None:
        # 18.1 − 20.1 = −2.0
        result = delta_cq(experiment, "S1", "TARGET")
        assert result.delta_cq == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestDeltaCqNoReference:
    """Experiment with no reference gene -- should raise ValueError."""

    def test_no_reference_gene_raises(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=False),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.0),
        ]
        exp = Experiment(
            name="no_ref", wells=wells, samples=_basic_samples(), assays=assays
        )
        with pytest.raises(ValueError, match="reference"):
            delta_cq(exp, "S1", "TARGET")

    def test_explicit_reference_works_without_is_reference(self) -> None:
        """Even without is_reference=True, explicit reference_target works."""
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=False),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.0),
        ]
        exp = Experiment(
            name="no_ref", wells=wells, samples=_basic_samples(), assays=assays
        )
        result = delta_cq(exp, "S1", "TARGET", reference_target="GAPDH")
        assert result.delta_cq == pytest.approx(-2.0)


class TestDeltaCqNoneCq:
    """Well with cq=None (undetermined) -- should raise ValueError."""

    def test_none_cq_target_raises(self) -> None:
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(
                position="A02",
                sample_name="S1",
                target_name="TARGET",
                cq=None,
                cq_status="undetermined",
            ),
        ]
        exp = Experiment(
            name="none_cq",
            wells=wells,
            samples=_basic_samples(),
            assays=_basic_assays(),
        )
        with pytest.raises(ValueError, match="None"):
            delta_cq(exp, "S1", "TARGET")

    def test_none_cq_reference_raises(self) -> None:
        wells = [
            Well(
                position="A01",
                sample_name="S1",
                target_name="GAPDH",
                cq=None,
                cq_status="undetermined",
            ),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.0),
        ]
        exp = Experiment(
            name="none_cq",
            wells=wells,
            samples=_basic_samples(),
            assays=_basic_assays(),
        )
        with pytest.raises(ValueError):
            delta_cq(exp, "S1", "TARGET")

    def test_target_not_found_raises(self) -> None:
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
        ]
        exp = Experiment(
            name="no_target",
            wells=wells,
            samples=_basic_samples(),
            assays=_basic_assays(),
        )
        with pytest.raises(ValueError, match="No wells found"):
            delta_cq(exp, "S1", "NONEXISTENT")


# ---------------------------------------------------------------------------
# delta_cq_all
# ---------------------------------------------------------------------------


class TestDeltaCqAll:
    """Batch ΔCq for all samples."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.0),
            Well(position="A03", sample_name="S2", target_name="GAPDH", cq=21.0),
            Well(position="A04", sample_name="S2", target_name="TARGET", cq=17.0),
        ]
        return Experiment(
            name="all",
            wells=wells,
            samples=_basic_samples(),
            assays=_basic_assays(),
        )

    def test_returns_list_of_results(self, experiment: Experiment) -> None:
        results = delta_cq_all(experiment, "TARGET")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, DeltaCqResult) for r in results)

    def test_s1_delta_cq(self, experiment: Experiment) -> None:
        results = delta_cq_all(experiment, "TARGET")
        s1 = next(r for r in results if r.sample_name == "S1")
        # 18.0 − 20.0 = −2.0
        assert s1.delta_cq == pytest.approx(-2.0)

    def test_s2_delta_cq(self, experiment: Experiment) -> None:
        results = delta_cq_all(experiment, "TARGET")
        s2 = next(r for r in results if r.sample_name == "S2")
        # 17.0 − 21.0 = −4.0
        assert s2.delta_cq == pytest.approx(-4.0)

    def test_reports_samples_without_target(self, experiment: Experiment) -> None:
        """Missing target data must not disappear silently."""
        with pytest.raises(ValueError, match="excluded 2 sample"):
            delta_cq_all(experiment, "NONEXISTENT")
        results = delta_cq_all(
            experiment,
            "NONEXISTENT",
            fail_on_exclusion=False,
        )
        assert results == []

    def test_skips_ntc_samples(self) -> None:
        """NTC wells should not appear in delta_cq_all results."""
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.0),
            Well(
                position="A03",
                sample_name="NTC",
                target_name="TARGET",
                cq=None,
                cq_status="no_template",
            ),
        ]
        samples = {"S1": Sample(name="S1"), "NTC": Sample(name="NTC")}
        exp = Experiment(
            name="with_ntc", wells=wells, samples=samples, assays=_basic_assays()
        )
        results = delta_cq_all(exp, "TARGET")
        assert len(results) == 1
        assert results[0].sample_name == "S1"
