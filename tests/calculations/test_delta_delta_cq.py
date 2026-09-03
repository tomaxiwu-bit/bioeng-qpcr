"""Tests for the ΔΔCq (Livak) quantification method.

Uses synthetic experiments with known Cq values to verify:

* ΔΔCq and fold-change calculation.
* The canonical example: ΔΔCq = −1 → fold change = 2.
* Control sample equals itself → fold change = 1.
* Error when the control sample has no data.
* The ``delta_delta_cq_all`` batch function.
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.calculations.delta_delta_cq import (
    DeltaDeltaCqResult,
    delta_delta_cq,
    delta_delta_cq_all,
)
from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _assays() -> dict[str, Assay]:
    return {
        "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
        "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
    }


def _samples() -> dict[str, Sample]:
    return {
        "CTRL": Sample(name="CTRL"),
        "S1": Sample(name="S1"),
        "S2": Sample(name="S2"),
    }


def _experiment(wells: list[Well]) -> Experiment:
    return Experiment(
        name="ddcq_test", wells=wells, samples=_samples(), assays=_assays()
    )


# ---------------------------------------------------------------------------
# Basic ΔΔCq
# ---------------------------------------------------------------------------


class TestDeltaDeltaCqBasic:
    """Known-answer: ΔCq_sample − ΔCq_control = −1 → fold change = 2."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        # CTRL: ΔCq = 20 − 20 = 0
        # S1:   ΔCq = 19 − 20 = −1
        # ΔΔCq  = −1 − 0 = −1
        # Fold  = 2^(−(−1)) = 2
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        return _experiment(wells)

    def test_returns_result_type(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert isinstance(result, DeltaDeltaCqResult)

    def test_delta_cq_of_sample(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert result.delta_cq == pytest.approx(-1.0)

    def test_delta_delta_cq_value(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert result.delta_delta_cq == pytest.approx(-1.0)

    def test_fold_change_is_two(self, experiment: Experiment) -> None:
        """Canonical: ΔΔCq = −1 → fold change = 2^1 = 2."""
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert result.fold_change == pytest.approx(2.0)

    def test_control_and_reference_recorded(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert result.control_sample == "CTRL"
        assert result.reference_target == "GAPDH"


# ---------------------------------------------------------------------------
# Control is itself
# ---------------------------------------------------------------------------


class TestDeltaDeltaCqControlSelf:
    """When sample == control, ΔΔCq = 0 and fold change = 1."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
        ]
        return _experiment(wells)

    def test_delta_delta_cq_zero(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "CTRL", "TARGET", "CTRL")
        assert result.delta_delta_cq == pytest.approx(0.0)

    def test_fold_change_one(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "CTRL", "TARGET", "CTRL")
        assert result.fold_change == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Down-regulation
# ---------------------------------------------------------------------------


class TestDeltaDeltaCqDownregulation:
    """ΔΔCq > 0 → fold change < 1 (down-regulation)."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        # CTRL: ΔCq = 20 − 20 = 0
        # S1:   ΔCq = 21 − 20 = 1   (target amplified later → less expressed)
        # ΔΔCq  = 1 − 0 = 1
        # Fold  = 2^(−1) = 0.5
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=21.0),
        ]
        return _experiment(wells)

    def test_delta_delta_cq_positive(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert result.delta_delta_cq == pytest.approx(1.0)

    def test_fold_change_half(self, experiment: Experiment) -> None:
        result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        assert result.fold_change == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestDeltaDeltaCqErrors:
    def test_control_missing_data_raises(self) -> None:
        """Control sample has no wells at all."""
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        exp = _experiment(wells)
        with pytest.raises(ValueError, match="Control sample"):
            delta_delta_cq(exp, "S1", "TARGET", "CTRL")

    def test_control_missing_target_raises(self) -> None:
        """Control sample has reference but not the target."""
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=19.0),
            Well(position="A03", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            # CTRL has no TARGET well
        ]
        exp = _experiment(wells)
        with pytest.raises(ValueError):
            delta_delta_cq(exp, "S1", "TARGET", "CTRL")

    def test_no_reference_gene_raises(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=False),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        exp = Experiment(name="no_ref", wells=wells, samples=_samples(), assays=assays)
        with pytest.raises(ValueError, match="reference"):
            delta_delta_cq(exp, "S1", "TARGET", "CTRL")


# ---------------------------------------------------------------------------
# delta_delta_cq_all
# ---------------------------------------------------------------------------


class TestDeltaDeltaCqAll:
    @pytest.fixture
    def experiment(self) -> Experiment:
        # CTRL: ΔCq = 0
        # S1:   ΔCq = −1 → ΔΔCq = −1 → fold = 2
        # S2:   ΔCq = −2 → ΔΔCq = −2 → fold = 4
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
            Well(position="A05", sample_name="S2", target_name="GAPDH", cq=20.0),
            Well(position="A06", sample_name="S2", target_name="TARGET", cq=18.0),
        ]
        return _experiment(wells)

    def test_returns_three_results(self, experiment: Experiment) -> None:
        results = delta_delta_cq_all(experiment, "TARGET", "CTRL")
        assert len(results) == 3

    def test_control_fold_change_one(self, experiment: Experiment) -> None:
        results = delta_delta_cq_all(experiment, "TARGET", "CTRL")
        ctrl = next(r for r in results if r.sample_name == "CTRL")
        assert ctrl.fold_change == pytest.approx(1.0)

    def test_s1_fold_change_two(self, experiment: Experiment) -> None:
        results = delta_delta_cq_all(experiment, "TARGET", "CTRL")
        s1 = next(r for r in results if r.sample_name == "S1")
        assert s1.fold_change == pytest.approx(2.0)

    def test_s2_fold_change_four(self, experiment: Experiment) -> None:
        results = delta_delta_cq_all(experiment, "TARGET", "CTRL")
        s2 = next(r for r in results if r.sample_name == "S2")
        assert s2.fold_change == pytest.approx(4.0)
