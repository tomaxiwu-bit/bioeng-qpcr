"""Tests for the Pfaffl efficiency-corrected quantification method.

Uses synthetic experiments with known Cq values and efficiencies to verify:

* Ratio calculation with known efficiencies.
* When efficiency = 1.0 for both genes, the Pfaffl ratio matches the
  ΔΔCq (Livak) fold change.
* Default efficiency (1.0 / 100 %) when assay efficiency is ``None``.
* Error when the control sample has no data.
* The ``pfaffl_all`` batch function.
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.calculations.delta_delta_cq import delta_delta_cq
from bioeng_qpcr.calculations.pfaffl import PfafflResult, pfaffl, pfaffl_all
from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _samples() -> dict[str, Sample]:
    return {
        "CTRL": Sample(name="CTRL"),
        "S1": Sample(name="S1"),
        "S2": Sample(name="S2"),
    }


def _experiment(wells: list[Well], assays: dict[str, Assay]) -> Experiment:
    return Experiment(
        name="pfaffl_test", wells=wells, samples=_samples(), assays=assays
    )


# ---------------------------------------------------------------------------
# Unit efficiency (matches ΔΔCq)
# ---------------------------------------------------------------------------


class TestPfafflUnitEfficiency:
    """When both efficiencies = 1.0, Pfaffl should match ΔΔCq."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        # CTRL: TARGET=20, GAPDH=20
        # S1:   TARGET=19, GAPDH=20
        # ΔCq_target    = 20 − 19 = 1
        # ΔCq_reference = 20 − 20 = 0
        # E_target = 2, E_ref = 2
        # Ratio = 2^1 / 2^0 = 2
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        return _experiment(wells, assays)

    def test_returns_pfaffl_result(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert isinstance(result, PfafflResult)

    def test_ratio_is_two(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.ratio == pytest.approx(2.0)

    def test_matches_delta_delta_cq(self, experiment: Experiment) -> None:
        """With unit efficiency, Pfaffl ratio == ΔΔCq fold change."""
        dd_result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        pf_result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert pf_result.ratio == pytest.approx(dd_result.fold_change)

    def test_delta_cq_target(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.delta_cq_target == pytest.approx(1.0)

    def test_delta_cq_reference(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.delta_cq_reference == pytest.approx(0.0)

    def test_efficiencies_recorded(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.target_efficiency == pytest.approx(1.0)
        assert result.reference_efficiency == pytest.approx(1.0)

    def test_control_and_reference_recorded(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.control_sample == "CTRL"
        assert result.reference_target == "GAPDH"


# ---------------------------------------------------------------------------
# Different efficiencies
# ---------------------------------------------------------------------------


class TestPfafflDifferentEfficiencies:
    """Pfaffl with asymmetric efficiencies should differ from ΔΔCq."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=0.95, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=1.05, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        return _experiment(wells, assays)

    def test_ratio_with_different_efficiencies(self, experiment: Experiment) -> None:
        # E_target = 2.05, E_ref = 1.95
        # ΔCq_target = 1, ΔCq_ref = 0
        # Ratio = 2.05^1 / 1.95^0 = 2.05
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.ratio == pytest.approx(2.05)

    def test_target_efficiency_recorded(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.target_efficiency == pytest.approx(1.05)

    def test_reference_efficiency_recorded(self, experiment: Experiment) -> None:
        result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        assert result.reference_efficiency == pytest.approx(0.95)

    def test_differs_from_delta_delta_cq(self, experiment: Experiment) -> None:
        """With different efficiencies, Pfaffl != ΔΔCq fold change."""
        dd_result = delta_delta_cq(experiment, "S1", "TARGET", "CTRL")
        pf_result = pfaffl(experiment, "S1", "TARGET", "CTRL")
        # ΔΔCq assumes 100% → fold = 2.0; Pfaffl = 2.05
        assert pf_result.ratio != pytest.approx(dd_result.fold_change)
        assert pf_result.ratio > dd_result.fold_change


# ---------------------------------------------------------------------------
# Default efficiency when None
# ---------------------------------------------------------------------------


class TestPfafflMissingEfficiency:
    """Pfaffl must not invent a 100% efficiency measurement."""

    def test_default_efficiency_used(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=None, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=None, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        exp = _experiment(wells, assays)
        with pytest.raises(ValueError, match="No measured amplification efficiency"):
            pfaffl(exp, "S1", "TARGET", "CTRL")

    def test_assay_not_in_dict_uses_default(self) -> None:
        """A missing target assay must produce an explicit error."""
        assays: dict[str, Assay] = {}
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        # Need at least GAPDH marked as reference for auto-resolution.
        assays = {"GAPDH": Assay(name="GAPDH", is_reference=True)}
        exp = _experiment(wells, assays)
        with pytest.raises(ValueError, match="No measured amplification efficiency"):
            pfaffl(exp, "S1", "TARGET", "CTRL")


# ---------------------------------------------------------------------------
# Reference gene also changes
# ---------------------------------------------------------------------------


class TestPfafflReferenceDeltaNonzero:
    """When the reference gene also changes between sample and control."""

    def test_both_target_and_reference_change(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        # CTRL: TARGET=20, GAPDH=20
        # S1:   TARGET=19, GAPDH=19
        # ΔCq_target    = 20 − 19 = 1
        # ΔCq_reference = 20 − 19 = 1
        # Ratio = 2^1 / 2^1 = 1.0
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=19.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        exp = _experiment(wells, assays)
        result = pfaffl(exp, "S1", "TARGET", "CTRL")
        assert result.delta_cq_target == pytest.approx(1.0)
        assert result.delta_cq_reference == pytest.approx(1.0)
        assert result.ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestPfafflErrors:
    def test_control_missing_data_raises(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=19.0),
            # CTRL has no wells
        ]
        exp = _experiment(wells, assays)
        with pytest.raises(ValueError, match="Control sample"):
            pfaffl(exp, "S1", "TARGET", "CTRL")

    def test_no_reference_gene_raises(self) -> None:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=False),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        wells = [
            Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="S1", target_name="TARGET", cq=19.0),
        ]
        exp = _experiment(wells, assays)
        with pytest.raises(ValueError, match="reference"):
            pfaffl(exp, "S1", "TARGET", "CTRL")


# ---------------------------------------------------------------------------
# pfaffl_all
# ---------------------------------------------------------------------------


class TestPfafflAll:
    @pytest.fixture
    def experiment(self) -> Experiment:
        assays = {
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
            "TARGET": Assay(name="TARGET", efficiency=1.0, is_reference=False),
        }
        # CTRL: ratio = 1
        # S1:   ratio = 2  (ΔCq_target=1, ΔCq_ref=0)
        # S2:   ratio = 4  (ΔCq_target=2, ΔCq_ref=0)
        wells = [
            Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
            Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=20.0),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S1", target_name="TARGET", cq=19.0),
            Well(position="A05", sample_name="S2", target_name="GAPDH", cq=20.0),
            Well(position="A06", sample_name="S2", target_name="TARGET", cq=18.0),
        ]
        return _experiment(wells, assays)

    def test_returns_three_results(self, experiment: Experiment) -> None:
        results = pfaffl_all(experiment, "TARGET", "CTRL")
        assert len(results) == 3
        assert all(isinstance(r, PfafflResult) for r in results)

    def test_control_ratio_one(self, experiment: Experiment) -> None:
        results = pfaffl_all(experiment, "TARGET", "CTRL")
        ctrl = next(r for r in results if r.sample_name == "CTRL")
        assert ctrl.ratio == pytest.approx(1.0)

    def test_s1_ratio_two(self, experiment: Experiment) -> None:
        results = pfaffl_all(experiment, "TARGET", "CTRL")
        s1 = next(r for r in results if r.sample_name == "S1")
        assert s1.ratio == pytest.approx(2.0)

    def test_s2_ratio_four(self, experiment: Experiment) -> None:
        results = pfaffl_all(experiment, "TARGET", "CTRL")
        s2 = next(r for r in results if r.sample_name == "S2")
        assert s2.ratio == pytest.approx(4.0)
