"""Tests for the amplification-curve analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.calculations.amplification_curve import (
    AmplificationAnalysisResult,
    AmplificationFitResult,
    AssayEffSummary,
    BaselineEstimate,
    _exponential_window_efficiency,
    analyze_amplification_curves,
    apply_assay_efficiency,
    baseline_correct,
    estimate_baseline,
    fit_amplification_curve,
)
from bioeng_qpcr.domain.entities import Assay, Experiment, RawTrace, Sample, Well
from bioeng_qpcr.domain.enums import ReplicateType


def _cycles(n: int = 40, start: int = 1) -> np.ndarray:
    return np.arange(start, np.float64(start) + n, dtype=float)


def _sigmoid_curve(
    cycles: np.ndarray,
    *,
    baseline: float = 2.0,
    plateau: float = 10.0,
    slope: float = 0.9,
    x0: float = 20.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    L = plateau - baseline
    sig = L / (1.0 + np.exp(-slope * (cycles - x0)))
    out = baseline + sig
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, noise, cycles.size)
    return out


def _flat_curve(cycles: np.ndarray, value: float = 2.0) -> np.ndarray:
    return np.full(cycles.size, value, dtype=float)


# ---------------------------------------------------------------------------
# Baseline estimation & correction
# ---------------------------------------------------------------------------


class TestEstimateBaseline:
    def test_constant_baseline_on_clean_sigmoid(self):
        cycles = _cycles()
        fluo = _sigmoid_curve(cycles)
        est = estimate_baseline(fluo)
        assert isinstance(est, BaselineEstimate)
        assert est.method == "constant"
        assert est.n_baseline_cycles > 0
        # Baseline window captures the flat background, not the plateau.
        assert est.value < fluo.mean()
        assert est.slope == 0.0
        assert est.takeoff_cycle is not None

    def test_linear_baseline_has_slope(self):
        cycles = _cycles()
        fluo = _sigmoid_curve(cycles)
        est = estimate_baseline(fluo, method="linear")
        assert est.method == "linear"
        assert isinstance(est.slope, float)

    def test_flat_trace_uses_whole_window_and_warns(self):
        fluo = _flat_curve(_cycles())
        est = estimate_baseline(fluo)
        assert est.takeoff_cycle is None
        assert any("No amplification" in w for w in est.warnings)
        assert est.n_baseline_cycles == fluo.size

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            estimate_baseline(np.array([1.0, 2.0]))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            estimate_baseline(_flat_curve(_cycles()), method="cubic")

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            estimate_baseline(np.array([1.0, np.nan, 3.0]))


class TestBaselineCorrect:
    def test_constant_correction_shifts_by_value(self):
        cycles = _cycles()
        fluo = _sigmoid_curve(cycles)
        est = estimate_baseline(fluo, method="constant")
        corrected, takeoff = baseline_correct(cycles, fluo, est)
        assert corrected.shape == fluo.shape
        assert takeoff == est.takeoff_cycle
        # Leading flat window corrects to ~zero.
        assert corrected[0] == pytest.approx(fluo[0] - est.value, abs=1e-9)

    def test_linear_correction_accounts_for_drift(self):
        cycles = _cycles()
        fluo = _sigmoid_curve(cycles)
        est = estimate_baseline(fluo, method="linear")
        corrected, _ = baseline_correct(cycles, fluo, est)
        assert np.all(np.isfinite(corrected))


# ---------------------------------------------------------------------------
# Per-well curve fitting
# ---------------------------------------------------------------------------


class TestFitAmplificationCurve:
    def test_recovers_cq_and_amplitude_on_clean_sigmoid(self):
        cycles = _cycles()
        x0, plateau, baseline = 20.0, 10.0, 2.0
        fluo = _sigmoid_curve(cycles, x0=x0, plateau=plateau, baseline=baseline)
        result = fit_amplification_curve(cycles, fluo, position="A01")
        assert isinstance(result, AmplificationFitResult)
        assert result.status == "ok"
        assert result.cq == pytest.approx(x0, abs=0.6)
        assert result.delta_rn == pytest.approx(plateau - baseline, rel=0.1)
        assert result.r_squared > 0.98
        assert result.cq_sdm is not None
        assert result.baseline_takeoff_cycle is not None

    def test_noisy_sigmoid_still_fits(self):
        cycles = _cycles()
        fluo = _sigmoid_curve(cycles, noise=0.02)
        result = fit_amplification_curve(cycles, fluo, position="A02")
        assert result.status == "ok"
        assert result.cq == pytest.approx(20.0, abs=1.0)
        assert result.r_squared > 0.98

    def test_efficiency_recovered_by_white_box_window(self):
        # The OLS on a clean exponential window recovers the true efficiency
        # exactly at the white-box level.
        cycles = np.arange(10.0, 16.0)
        eff_true = 0.90
        dR = 0.03 * (1.0 + eff_true) ** (cycles - cycles[0])
        eff, slope, n = _exponential_window_efficiency(cycles, dR, plateau=1.0)
        assert n >= 3
        assert eff == pytest.approx(eff_true, abs=0.05)
        assert slope == pytest.approx(np.log(1.0 + eff_true), abs=0.05)

    def test_flat_trace_is_non_amplifying(self):
        cycles = _cycles()
        result = fit_amplification_curve(cycles, _flat_curve(cycles), position="B01")
        assert result.status == "non_amplifying"
        assert result.cq is None
        assert result.eff is None

    def test_too_few_cycles_raises(self):
        with pytest.raises(ValueError):
            fit_amplification_curve(
                np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_amplification_curve(
                np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0])
            )

    def test_non_finite_raises(self):
        cycles = _cycles()
        fluo = np.full(cycles.size, np.nan)
        with pytest.raises(ValueError):
            fit_amplification_curve(cycles, fluo)

    def test_implausible_efficiency_is_degraded(self):
        # A very steep rise far beyond plausible per-cycle efficiency.
        cycles = _cycles()
        L, k, x0 = 8.0, 5.0, 15.0
        fluo = 2.0 + L / (1.0 + np.exp(-k * (cycles - x0)))
        result = fit_amplification_curve(cycles, fluo, position="D01")
        assert result.status == "degraded"
        assert any("efficiency" in w for w in result.warnings) or result.eff is None


# ---------------------------------------------------------------------------
# Whole-experiment analysis
# ---------------------------------------------------------------------------


def _make_experiment(n_cycles: int = 40) -> Experiment:
    cycles = _cycles(n_cycles)
    wells = [
        Well(position="A01", sample_name="S1", target_name="IL6", cq=20.1),
        Well(position="A02", sample_name="S1", target_name="GAPDH", cq=18.0),
        Well(position="A03", sample_name="NTC", target_name="IL6", cq=None),
    ]
    samples = {
        "S1": Sample(name="S1", group="control", replicate_type=ReplicateType.SINGLE),
        "NTC": Sample(name="NTC", replicate_type=ReplicateType.SINGLE),
    }
    assays = {
        "IL6": Assay(name="IL6", efficiency=1.0),
        "GAPDH": Assay(name="GAPDH", efficiency=1.0),
    }
    traces = {
        "A01": RawTrace(
            position="A01",
            cycles=cycles,
            fluorescence=_sigmoid_curve(cycles, x0=20.0),
        ),
        "A02": RawTrace(
            position="A02",
            cycles=cycles,
            fluorescence=_sigmoid_curve(cycles, x0=18.0, plateau=9.0),
        ),
        "A03": RawTrace(
            position="A03", cycles=cycles, fluorescence=_flat_curve(cycles)
        ),
        # A trace with no matching well to exercise retention.
        "Z99": RawTrace(
            position="Z99",
            cycles=cycles,
            fluorescence=_sigmoid_curve(cycles, x0=25.0),
        ),
    }
    return Experiment(
        name="amp-test",
        wells=wells,
        samples=samples,
        assays=assays,
        traces=traces,
    )


class TestAnalyzeAmplificationCurves:
    def test_links_and_aggregates(self):
        experiment = _make_experiment()
        result = analyze_amplification_curves(experiment)
        assert isinstance(result, AmplificationAnalysisResult)
        assert set(result.per_well) == {"A01", "A02", "A03"}
        assert result.trace_target["A01"] == "IL6"
        assert result.trace_target["A02"] == "GAPDH"
        # NTC is non-amplifying.
        assert "A03" in result.non_amplifying
        # Orphan trace is retained, not dropped.
        assert result.traces_without_well == ("Z99",)
        assert result.trace_cq["A01"] == pytest.approx(20.0, abs=0.6)
        assert result.trace_cq["A03"] is None

    def test_per_assay_median_efficiency(self):
        experiment = _make_experiment()
        result = analyze_amplification_curves(experiment)
        assert set(result.per_assay) == {"IL6", "GAPDH"}
        for summary in result.per_assay.values():
            assert isinstance(summary, AssayEffSummary)
            assert summary.n_wells == 1
            assert summary.median == pytest.approx(summary.values[0])

    def test_empty_traces_reports_warning_and_wells(self):
        experiment = _make_experiment()
        experiment.traces = {}
        result = analyze_amplification_curves(experiment)
        assert result.per_well == {}
        assert len(result.wells_without_trace) == 3
        assert any("No raw amplification traces" in w for w in result.warnings)

    def test_non_experiment_raises(self):
        with pytest.raises(TypeError):
            analyze_amplification_curves({"not": "experiment"})  # type: ignore[arg-type]


class TestApplyAssayEfficiency:
    def test_fills_missing_at_first(self):
        experiment = _make_experiment()
        experiment.assays["IL6"] = Assay(name="IL6", efficiency=None)
        result = analyze_amplification_curves(experiment)
        apply_assay_efficiency(experiment, result)
        assert experiment.assays["IL6"].efficiency is not None

    def test_does_not_overwrite_by_default(self):
        experiment = _make_experiment()
        kept = experiment.assays["IL6"].efficiency
        result = analyze_amplification_curves(experiment)
        apply_assay_efficiency(experiment, result)
        assert experiment.assays["IL6"].efficiency == kept

    def test_force_override(self):
        experiment = _make_experiment()
        result = analyze_amplification_curves(experiment)
        apply_assay_efficiency(experiment, result, replace=True)
        new_value = experiment.assays["IL6"].efficiency
        assert new_value is not None
        assert new_value != 1.0  # curve-derived median replaces the recorded 1.0