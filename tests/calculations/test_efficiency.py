"""Tests for efficiency fitting (standard curve and amplification trace).

Uses synthetic data with known parameters to verify:

* Standard-curve fitting with a known slope (−3.32 → ~100 % efficiency).
* R² for a perfect linear fit.
* Minimum of 3 data points.
* Error cases (mismatched lengths, too few points).
* 4-parameter logistic trace fitting returns a valid result.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.calculations.efficiency import (
    EfficiencyResult,
    fit_efficiency_from_standard_curve,
    fit_efficiency_from_trace,
)

# ---------------------------------------------------------------------------
# Standard-curve: known slope
# ---------------------------------------------------------------------------


class TestStandardCurveKnownSlope:
    """slope = −3.32 → efficiency ≈ 1.0 (100 %)."""

    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        log_conc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = -3.32
        intercept = 35.0
        cq = slope * log_conc + intercept
        return log_conc, cq

    def test_returns_efficiency_result(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        log_conc, cq = data
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert isinstance(result, EfficiencyResult)

    def test_slope_recovered(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        log_conc, cq = data
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.slope == pytest.approx(-3.32)

    def test_efficiency_approx_one(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        """slope = −3.32 → efficiency ≈ 1.0 (100 %)."""
        log_conc, cq = data
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.efficiency == pytest.approx(1.0, abs=0.01)

    def test_intercept_recovered(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        log_conc, cq = data
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.intercept == pytest.approx(35.0)

    def test_r_squared_perfect(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        log_conc, cq = data
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.r_squared == pytest.approx(1.0)

    def test_n_points(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        log_conc, cq = data
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.n_points == 5

    def test_target_name(self) -> None:
        log_conc = np.array([1.0, 2.0, 3.0])
        cq = -3.32 * log_conc + 35.0
        result = fit_efficiency_from_standard_curve(log_conc, cq, target_name="GAPDH")
        assert result.target_name == "GAPDH"


# ---------------------------------------------------------------------------
# Standard-curve: other slopes
# ---------------------------------------------------------------------------


class TestStandardCurveOtherSlopes:
    def test_slope_minus_3_1_higher_efficiency(self) -> None:
        # slope = −3.1 → efficiency > 1.0 (>100 %)
        log_conc = np.array([1.0, 2.0, 3.0, 4.0])
        cq = -3.1 * log_conc + 30.0
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.slope == pytest.approx(-3.1)
        assert result.efficiency > 1.0

    def test_slope_minus_3_6_lower_efficiency(self) -> None:
        # slope = −3.6 → efficiency < 1.0 (<100 %)
        log_conc = np.array([1.0, 2.0, 3.0, 4.0])
        cq = -3.6 * log_conc + 30.0
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.slope == pytest.approx(-3.6)
        assert result.efficiency < 1.0


# ---------------------------------------------------------------------------
# Standard-curve: noisy data
# ---------------------------------------------------------------------------


class TestStandardCurveNoisy:
    def test_r_squared_below_one(self) -> None:
        log_conc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Nearly linear but with small manual perturbations.
        cq = np.array([31.7, 28.3, 25.1, 21.6, 18.5])
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert 0.99 < result.r_squared < 1.0


# ---------------------------------------------------------------------------
# Standard-curve: minimum 3 points
# ---------------------------------------------------------------------------


class TestStandardCurveThreePoints:
    def test_three_points_succeeds(self) -> None:
        log_conc = np.array([1.0, 2.0, 3.0])
        cq = -3.32 * log_conc + 35.0
        result = fit_efficiency_from_standard_curve(log_conc, cq)
        assert result.n_points == 3
        assert result.efficiency == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Standard-curve: error cases
# ---------------------------------------------------------------------------


class TestStandardCurveErrors:
    def test_different_lengths_raises(self) -> None:
        log_conc = np.array([1.0, 2.0, 3.0])
        cq = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="same shape"):
            fit_efficiency_from_standard_curve(log_conc, cq)

    def test_two_points_raises(self) -> None:
        log_conc = np.array([1.0, 2.0])
        cq = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="3"):
            fit_efficiency_from_standard_curve(log_conc, cq)

    def test_one_point_raises(self) -> None:
        log_conc = np.array([1.0])
        cq = np.array([10.0])
        with pytest.raises(ValueError, match="3"):
            fit_efficiency_from_standard_curve(log_conc, cq)

    def test_empty_arrays_raise(self) -> None:
        log_conc = np.array([])
        cq = np.array([])
        with pytest.raises(ValueError, match="3"):
            fit_efficiency_from_standard_curve(log_conc, cq)

    def test_nonfinite_values_raise(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            fit_efficiency_from_standard_curve(
                np.array([1.0, 2.0, np.nan]),
                np.array([30.0, 27.0, 24.0]),
            )

    def test_positive_slope_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            fit_efficiency_from_standard_curve(
                np.array([1.0, 2.0, 3.0]),
                np.array([20.0, 23.0, 26.0]),
            )

    def test_flat_curve_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            fit_efficiency_from_standard_curve(
                np.array([1.0, 2.0, 3.0]),
                np.array([25.0, 25.0, 25.0]),
            )

    def test_requires_three_distinct_concentrations(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            fit_efficiency_from_standard_curve(
                np.array([1.0, 1.0, 2.0, 2.0]),
                np.array([30.0, 30.1, 26.7, 26.8]),
            )

    def test_requires_one_dimensional_arrays(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            fit_efficiency_from_standard_curve(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[31.0, 28.0], [25.0, 22.0]]),
            )


class TestStandardCurveUncertainty:
    def test_noisy_curve_returns_slope_and_efficiency_intervals(self) -> None:
        log_conc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cq = np.array([31.8, 28.1, 25.2, 21.5, 18.6])

        result = fit_efficiency_from_standard_curve(log_conc, cq)

        assert result.slope_se is not None
        assert result.slope_se > 0
        assert result.slope_ci is not None
        assert result.slope_ci[0] < result.slope < result.slope_ci[1]
        assert result.efficiency_ci is not None
        assert result.efficiency_ci[0] < result.efficiency < result.efficiency_ci[1]


# ---------------------------------------------------------------------------
# Trace fitting
# ---------------------------------------------------------------------------


class TestTraceFitting:
    """Fit a 4-parameter logistic to a synthetic amplification curve."""

    @pytest.fixture
    def trace(self) -> tuple[np.ndarray, np.ndarray]:
        cycles = np.arange(1, 46, dtype=float)
        L, k, x0, C = 100.0, 0.7, 25.0, 5.0
        fluorescence = L / (1.0 + np.exp(-k * (cycles - x0))) + C
        return cycles, fluorescence

    def test_returns_efficiency_result(
        self, trace: tuple[np.ndarray, np.ndarray]
    ) -> None:
        cycles, fluorescence = trace
        result = fit_efficiency_from_trace(cycles, fluorescence)
        assert isinstance(result, EfficiencyResult)

    def test_n_points(self, trace: tuple[np.ndarray, np.ndarray]) -> None:
        cycles, fluorescence = trace
        result = fit_efficiency_from_trace(cycles, fluorescence)
        assert result.n_points == len(cycles)

    def test_r_squared_high(self, trace: tuple[np.ndarray, np.ndarray]) -> None:
        """The fit should be near-perfect for noise-free data."""
        cycles, fluorescence = trace
        result = fit_efficiency_from_trace(cycles, fluorescence)
        assert result.r_squared > 0.99

    def test_efficiency_positive(self, trace: tuple[np.ndarray, np.ndarray]) -> None:
        cycles, fluorescence = trace
        result = fit_efficiency_from_trace(cycles, fluorescence)
        assert result.efficiency > 0.0

    def test_intercept_is_cq(self, trace: tuple[np.ndarray, np.ndarray]) -> None:
        """For trace fits, intercept holds the inflection-point Cq (x0)."""
        cycles, fluorescence = trace
        result = fit_efficiency_from_trace(cycles, fluorescence)
        # x0 = 25.0 in the synthetic data.
        assert result.intercept == pytest.approx(25.0, abs=1.0)

    def test_slope_is_k(self, trace: tuple[np.ndarray, np.ndarray]) -> None:
        """For trace fits, slope holds the logistic steepness k."""
        cycles, fluorescence = trace
        result = fit_efficiency_from_trace(cycles, fluorescence)
        # k = 0.7 in the synthetic data.
        assert result.slope == pytest.approx(0.7, abs=0.1)

    def test_efficiency_is_logistic_growth_factor(self) -> None:
        """The logistic efficiency factor is exp(k)-1 on the baseline-corrected
        amplitude, not a baseline-diluted 'relative growth at the inflection'."""
        cycles, fluorescence = self._trace(L=100.0, k=0.7, x0=25.0, C=5.0)
        result = fit_efficiency_from_trace(cycles, fluorescence)
        k = result.slope
        assert result.efficiency == pytest.approx(np.exp(k) - 1.0, rel=1e-6)

    def test_efficiency_is_invariant_to_baseline_offset(self) -> None:
        """A constant baseline C shifts the whole curve but not the per-cycle
        growth rate; the efficiency estimate must not change."""
        low_c = fit_efficiency_from_trace(
            *self._trace(L=100.0, k=0.7, x0=25.0, C=2.0)
        ).efficiency
        high_c = fit_efficiency_from_trace(
            *self._trace(L=100.0, k=0.7, x0=25.0, C=40.0)
        ).efficiency
        assert low_c == pytest.approx(high_c, rel=0.05)

    @staticmethod
    def _trace(
        L: float, k: float, x0: float, C: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cycles = np.arange(1, 46, dtype=float)
        fluorescence = L / (1.0 + np.exp(-k * (cycles - x0))) + C
        return cycles, fluorescence


# ---------------------------------------------------------------------------
# Trace fitting: error cases
# ---------------------------------------------------------------------------


class TestTraceFittingErrors:
    def test_different_lengths_raises(self) -> None:
        cycles = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fluorescence = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same shape"):
            fit_efficiency_from_trace(cycles, fluorescence)

    def test_too_few_points_raises(self) -> None:
        cycles = np.array([1.0, 2.0, 3.0])
        fluorescence = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="4"):
            fit_efficiency_from_trace(cycles, fluorescence)
