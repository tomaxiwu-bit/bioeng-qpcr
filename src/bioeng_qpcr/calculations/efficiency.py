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
    residual_standard_error: float | None = None
    mean_log_concentration: float | None = None
    sxx: float | None = None

    def predict_cq_interval(
        self,
        log_concentration: float,
        *,
        confidence: float = 0.95,
    ) -> tuple[float, float, float]:
        """Return fitted Cq and a two-sided prediction interval.

        The interval covers a future Cq observation at the supplied
        log10(concentration), including both regression and residual
        uncertainty. It is available only for standard-curve fits.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not np.isfinite(log_concentration):
            raise ValueError("log_concentration must be finite.")
        if (
            self.residual_standard_error is None
            or self.mean_log_concentration is None
            or self.sxx is None
            or self.sxx <= 0.0
            or self.n_points <= 2
        ):
            raise ValueError("Prediction intervals require a standard-curve fit.")

        estimate = self.slope * float(log_concentration) + self.intercept
        prediction_se = self.residual_standard_error * np.sqrt(
            1.0
            + 1.0 / self.n_points
            + (float(log_concentration) - self.mean_log_concentration) ** 2
            / self.sxx
        )
        critical = float(
            student_t.ppf(0.5 + confidence / 2.0, self.n_points - 2)
        )
        margin = critical * float(prediction_se)
        return float(estimate), float(estimate - margin), float(estimate + margin)


@dataclass(frozen=True)
class CalibrationLevel:
    """Detection and precision evidence for one standard concentration."""

    concentration: float
    n_total: int
    n_detected: int
    detection_rate: float
    mean_cq: float | None
    cq_sd: float | None
    lod_qualified: bool
    quantification_qualified: bool


@dataclass(frozen=True)
class StandardCurveRange:
    """Empirical LOD and quantifiable range from replicated standards."""

    levels: tuple[CalibrationLevel, ...]
    lod: float | None
    lloq: float | None
    uloq: float | None
    dynamic_range_log10: float | None
    min_detection_rate: float
    max_cq_sd: float
    min_replicates: int
    warnings: tuple[str, ...]


def assess_standard_curve_range(
    concentrations: np.ndarray,
    cq_values: np.ndarray,
    *,
    min_detection_rate: float = 0.95,
    max_cq_sd: float = 0.5,
    min_replicates: int = 3,
) -> StandardCurveRange:
    """Estimate empirical LOD/LLOQ/ULOQ from replicated standards.

    Missing/non-finite Cq values count as non-detections. LOD is the lowest
    concentration meeting the replicate-count and detection-rate criteria;
    LLOQ additionally requires the detected Cq standard deviation to be no
    greater than ``max_cq_sd``. These operational thresholds must be reported
    with the result and should be set from the laboratory's validated SOP.
    """
    concentrations = np.asarray(concentrations, dtype=float)
    cq_values = np.asarray(cq_values, dtype=float)
    if concentrations.shape != cq_values.shape or concentrations.ndim != 1:
        raise ValueError("concentrations and cq_values must be same-length 1D arrays.")
    if concentrations.size == 0:
        raise ValueError("At least one standard-curve observation is required.")
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0):
        raise ValueError("Standard concentrations must be finite and positive.")
    if not 0.0 < min_detection_rate <= 1.0:
        raise ValueError("min_detection_rate must be in (0, 1].")
    if not np.isfinite(max_cq_sd) or max_cq_sd < 0.0:
        raise ValueError("max_cq_sd must be finite and non-negative.")
    if min_replicates < 2:
        raise ValueError("min_replicates must be at least 2.")

    levels: list[CalibrationLevel] = []
    for concentration in sorted(np.unique(concentrations)):
        values = cq_values[concentrations == concentration]
        detected = values[np.isfinite(values)]
        n_total = int(values.size)
        n_detected = int(detected.size)
        rate = n_detected / n_total
        mean_cq = float(np.mean(detected)) if n_detected else None
        cq_sd = float(np.std(detected, ddof=1)) if n_detected >= 2 else None
        lod_ok = n_total >= min_replicates and rate >= min_detection_rate
        quant_ok = lod_ok and cq_sd is not None and cq_sd <= max_cq_sd
        levels.append(
            CalibrationLevel(
                concentration=float(concentration),
                n_total=n_total,
                n_detected=n_detected,
                detection_rate=rate,
                mean_cq=mean_cq,
                cq_sd=cq_sd,
                lod_qualified=lod_ok,
                quantification_qualified=quant_ok,
            )
        )

    lod_levels = [level.concentration for level in levels if level.lod_qualified]
    quant_levels = [
        level.concentration for level in levels if level.quantification_qualified
    ]
    lod = min(lod_levels) if lod_levels else None
    lloq = min(quant_levels) if quant_levels else None
    uloq = max(quant_levels) if quant_levels else None
    dynamic_range = (
        float(np.log10(uloq / lloq))
        if lloq is not None and uloq is not None
        else None
    )
    warnings: list[str] = []
    if lod is None:
        warnings.append("No tested concentration met the empirical LOD criteria.")
    if lloq is None:
        warnings.append("No tested concentration met the quantification criteria.")
    elif len(quant_levels) < 3:
        warnings.append(
            "Fewer than three concentration levels define the quantifiable range."
        )
    return StandardCurveRange(
        levels=tuple(levels),
        lod=lod,
        lloq=lloq,
        uloq=uloq,
        dynamic_range_log10=dynamic_range,
        min_detection_rate=min_detection_rate,
        max_cq_sd=max_cq_sd,
        min_replicates=min_replicates,
        warnings=tuple(warnings),
    )


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
    residual_standard_error: float | None = None
    degrees_freedom = int(log_concentrations.size - 2)
    centered = log_concentrations - float(np.mean(log_concentrations))
    sxx = float(np.sum(centered**2))
    if degrees_freedom > 0 and sxx > 0:
        residual_variance = ss_res / degrees_freedom
        residual_standard_error = float(np.sqrt(max(residual_variance, 0.0)))
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
        residual_standard_error=residual_standard_error,
        mean_log_concentration=float(np.mean(log_concentrations)),
        sxx=sxx,
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
    3. Estimate efficiency from the logistic growth factor on the
       *baseline-corrected* amplitude.  For a fitted 4PL the signal about the
       baseline is ``g(x) = L / (1 + exp(-k*(x-x0)))``, which rises
       exponentially like ``L·exp(k·(x-x0))`` early on, so ``log g`` is linear
       in cycle with slope ``k`` and the per-cycle amplification factor is
       ``exp(k)``::

           efficiency = exp(k) - 1

       The baseline offset ``C`` must not appear in the factor: it contributes
       a constant fluorescence that has no bearing on the per-cycle growth
       rate, and folding it into the denominator (as a "relative growth at the
       inflection") makes the estimate shrink with baseline height.

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

    # Logistic efficiency factor from the baseline-corrected log-linear rate.
    # The baseline offset C is deliberately excluded: it is a constant adding
    # nothing to the per-cycle growth rate, and including it would bias the
    # factor down by the ratio 2C/L.  With unit cycle spacing the per-cycle
    # amplification factor is exp(k), so the fractional efficiency is exp(k)-1.
    efficiency = float(np.exp(k) - 1.0)

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
