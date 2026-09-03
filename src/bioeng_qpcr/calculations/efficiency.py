"""Efficiency fitting from standard curves and amplification traces.

This module provides two ways to estimate qPCR amplification efficiency:

1. **Standard-curve fitting** -- linear regression of Cq vs log10(concentration).
   This is the gold-standard approach recommended by MIQE.

2. **Experimental trace-shape fitting** -- fitting a 4-parameter logistic
   (4PL) model to raw fluorescence curves.  This descriptive estimate has not
   been validated as an individual-curve amplification-efficiency method and
   must not be used as MIQE evidence without independent benchmarking.

Standard-curve formula
----------------------
    Cq = slope * log10(conc) + intercept
    Efficiency = 10^(-1/slope) - 1

A slope of −3.32 corresponds to an efficiency of about 1.0 (100 %) and an
amplification factor of about 2 per cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyResult:
    """Result of an efficiency fitting calculation.

    Attributes
        target_name:
            Name of the target assay.
        efficiency:
            Amplification efficiency (1.0 = 100 %).
        r_squared:
            Coefficient of determination (R²) of the fit.
        slope:
            Slope of the standard-curve regression, or the logistic
            steepness *k* for trace fits.
        intercept:
            Y-intercept of the standard-curve regression, or the inflection
            point (Cq) for trace fits.
        n_points:
            Number of data points used in the fit.
    """

    target_name: str
    efficiency: float
    r_squared: float
    slope: float
    intercept: float
    n_points: int
    slope_se: float | None = None
    slope_ci: tuple[float, float] | None = None
    efficiency_ci: tuple[float, float] | None = None


def _efficiency_from_slope(slope: float) -> float:
    """Convert a valid negative standard-curve slope to fractional efficiency."""
    if not np.isfinite(slope) or slope >= 0.0 or np.isclose(slope, 0.0):
        raise ValueError(
            "A standard curve must have a finite negative slope; "
            f"got {slope!r}."
        )
    try:
        efficiency = float(10.0 ** (-1.0 / slope) - 1.0)
    except OverflowError as exc:
        raise ValueError(
            f"Standard-curve slope {slope:g} produces a non-finite efficiency."
        ) from exc
    if not np.isfinite(efficiency):
        raise ValueError(
            f"Standard-curve slope {slope:g} produces a non-finite efficiency."
        )
    return efficiency


# ---------------------------------------------------------------------------
# 4-parameter logistic model
# ---------------------------------------------------------------------------


def _four_param_logistic(
    x: np.ndarray, L: float, k: float, x0: float, C: float
) -> np.ndarray:
    """4-parameter logistic function.

    .. math::
        f(x) = \\frac{L}{1 + e^{-k(x - x_0)}} + C

    Parameters
        x:
            Independent variable (cycle number).
        L:
            Amplitude (plateau − baseline).
        k:
            Steepness (growth rate).
        x0:
            Inflection point (midpoint cycle).
        C:
            Baseline (lower asymptote).
    """
    return L / (1.0 + np.exp(-k * (x - x0))) + C


# ---------------------------------------------------------------------------
# Standard-curve fitting
# ---------------------------------------------------------------------------


def fit_efficiency_from_standard_curve(
    log_concentrations: np.ndarray,
    cq_values: np.ndarray,
    target_name: str = "unknown",
) -> EfficiencyResult:
    """Fit amplification efficiency from a standard curve.

    Performs a linear regression of Cq values against log10(concentration)
    values.  The input *log_concentrations* is assumed to already be
    log10-transformed.

    Parameters
        log_concentrations:
            Array of log10(concentration) values.
        cq_values:
            Array of measured Cq values (one per concentration).
        target_name:
            Name of the target assay.

    Returns
        :class:`EfficiencyResult` with efficiency, R², slope, and intercept.

    Raises
        ValueError:
            If the two arrays have different lengths or fewer than 3 points
            are provided.
    """
    log_concentrations = np.asarray(log_concentrations, dtype=float)
    cq_values = np.asarray(cq_values, dtype=float)

    if log_concentrations.shape != cq_values.shape:
        raise ValueError(
            f"log_concentrations and cq_values must have the same shape, "
            f"got {log_concentrations.shape} vs {cq_values.shape}."
        )
    if log_concentrations.size < 3:
        raise ValueError(
            f"Need at least 3 points for standard-curve fitting, got "
            f"{log_concentrations.size}."
        )
    if log_concentrations.ndim != 1:
        raise ValueError("Standard-curve inputs must be one-dimensional arrays.")
    if not np.all(np.isfinite(log_concentrations)) or not np.all(
        np.isfinite(cq_values)
    ):
        raise ValueError("Standard-curve concentrations and Cq values must be finite.")
    n_levels = int(np.unique(log_concentrations).size)
    if n_levels < 3:
        raise ValueError(
            "Standard-curve fitting requires at least 3 distinct concentration "
            f"levels, got {n_levels}."
        )

    # Linear regression: Cq = slope * log10(conc) + intercept.
    coeffs = np.polyfit(log_concentrations, cq_values, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # R² of the linear fit.
    predicted = slope * log_concentrations + intercept
    ss_res = float(np.sum((cq_values - predicted) ** 2))
    ss_tot = float(np.sum((cq_values - np.mean(cq_values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # Efficiency = 10^(-1/slope) - 1. A non-negative slope contradicts the
    # expected dilution-series relationship and must not be converted into a
    # plausible-looking efficiency.
    efficiency = _efficiency_from_slope(slope)

    # OLS uncertainty for the slope, using a Student-t interval because
    # standard curves commonly contain few concentration levels. Replicate
    # points are retained as individual observations in the residual estimate.
    slope_se: float | None = None
    slope_ci: tuple[float, float] | None = None
    efficiency_ci: tuple[float, float] | None = None
    degrees_freedom = int(log_concentrations.size - 2)
    centered = log_concentrations - float(np.mean(log_concentrations))
    sxx = float(np.sum(centered**2))
    if degrees_freedom > 0 and sxx > 0:
        residual_variance = ss_res / degrees_freedom
        slope_se = float(np.sqrt(max(residual_variance, 0.0) / sxx))
        t_critical = float(student_t.ppf(0.975, degrees_freedom))
        slope_ci = (
            slope - t_critical * slope_se,
            slope + t_critical * slope_se,
        )
        if slope_ci[1] < 0.0 and not np.isclose(slope_ci[1], 0.0):
            efficiency_ci = (
                _efficiency_from_slope(slope_ci[0]),
                _efficiency_from_slope(slope_ci[1]),
            )

    return EfficiencyResult(
        target_name=target_name,
        efficiency=efficiency,
        r_squared=r_squared,
        slope=slope,
        intercept=intercept,
        n_points=int(log_concentrations.size),
        slope_se=slope_se,
        slope_ci=slope_ci,
        efficiency_ci=efficiency_ci,
    )


# ---------------------------------------------------------------------------
# Trace (amplification curve) fitting
# ---------------------------------------------------------------------------


def fit_logistic_trace_shape(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
) -> EfficiencyResult:
    """Fit an experimental 4PL trace-shape estimate.

    This is a simplified, unvalidated estimator.  It does not identify and
    fit the baseline-corrected log-linear exponential window used by
    established individual-curve workflows.  Its result is exploratory and
    must not be used as evidence of amplification efficiency in a readiness
    assessment.

    1. Fit the 4PL model ``f(x) = L / (1 + exp(-k*(x-x0))) + C`` to the raw
       fluorescence data using :func:`scipy.optimize.curve_fit`.
    2. Extract the Cq at the inflection point (``x0``).
    3. Estimate efficiency from the curve shape using the relative growth
       rate at the inflection point::

           max_derivative     = |L| * |k| / 4
           value_at_inflection = |L| / 2 + |C|
           relative_growth     = max_derivative / value_at_inflection
           efficiency          = exp(relative_growth) - 1

    Parameters
        cycles:
            Array of cycle numbers.
        fluorescence:
            Array of fluorescence readings (one per cycle).

    Returns
        :class:`EfficiencyResult` with estimated efficiency and fit quality.
        The ``slope`` field holds the logistic steepness *k*; the
        ``intercept`` field holds the inflection-point Cq (``x0``).

    Raises
        ValueError:
            If the arrays have different shapes, fewer than 4 points are
            provided, or the logistic fit fails to converge.
    """
    cycles = np.asarray(cycles, dtype=float)
    fluorescence = np.asarray(fluorescence, dtype=float)

    if cycles.shape != fluorescence.shape:
        raise ValueError(
            f"cycles and fluorescence must have the same shape, got "
            f"{cycles.shape} vs {fluorescence.shape}."
        )
    if cycles.size < 4:
        raise ValueError(
            f"Need at least 4 points for trace fitting, got {cycles.size}."
        )

    # Initial parameter estimates.
    c0 = float(np.min(fluorescence))
    l0 = float(np.max(fluorescence) - c0)
    x0_0 = float(cycles[len(cycles) // 2])
    k0 = 1.0

    try:
        popt, _ = curve_fit(
            _four_param_logistic,
            cycles,
            fluorescence,
            p0=[l0, k0, x0_0, c0],
            maxfev=10000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Failed to fit 4-parameter logistic to trace data: {exc}"
        ) from exc

    L, k, x0, C = (float(v) for v in popt)

    # Cq at the inflection point.
    cq = x0

    # Estimate efficiency from the relative growth rate at the inflection.
    max_derivative = abs(L) * abs(k) / 4.0
    value_at_inflection = abs(L) / 2.0 + abs(C)
    if value_at_inflection > 0:
        relative_growth = max_derivative / value_at_inflection
    else:
        relative_growth = abs(k) / 2.0
    efficiency = float(np.exp(relative_growth) - 1.0)

    # R² of the 4PL fit.
    predicted = _four_param_logistic(cycles, *popt)
    ss_res = float(np.sum((fluorescence - predicted) ** 2))
    ss_tot = float(np.sum((fluorescence - np.mean(fluorescence)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return EfficiencyResult(
        target_name="trace",
        efficiency=efficiency,
        r_squared=r_squared,
        slope=k,  # logistic steepness
        intercept=cq,  # inflection-point Cq
        n_points=int(cycles.size),
    )


def fit_efficiency_from_trace(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
) -> EfficiencyResult:
    """Compatibility alias for :func:`fit_logistic_trace_shape`.

    The name is retained for 0.1.x users.  The returned value is an
    experimental trace-shape estimate, not a validated efficiency method.
    """
    return fit_logistic_trace_shape(cycles, fluorescence)
