"""Validation rules for dose-response potency (IC50/EC50) determinations.

A fitted IC50 is a *number*, not a result.  It only becomes a result once the
assumptions behind it have been checked: that the curve actually reached both
plateaus, that the half-maximal concentration was measured rather than
extrapolated, that the Hill slope is physically plausible, and that the
residuals do not betray an asymmetric curve forced into a symmetric model.

Each rule below returns a :class:`PotencyValidationResult` with a stable
identifier, a pass/fail flag, a severity, and the structured evidence needed
to audit the decision.  :func:`run_potency_validation` runs the whole suite.

Every rule accepts either a :class:`~bioeng_qpcr.potency.PotencyResult` (the
full report) or a bare :class:`~bioeng_qpcr.potency.DoseResponseFit` (the
curve alone), so a caller who has only fitted a curve can still validate it.

===========  =====================================================
Rule         Checks
===========  =====================================================
POT-001      Relative potency lies inside the tested range
POT-002      Assay window relative to the control response
POT-003      Enough concentration levels and observations
POT-004      Hill slope magnitude is plausible
POT-005      Goodness of fit (R²)
POT-006      Concentration series spans at least two decades
POT-007      Confidence interval is not absurdly wide
POT-008      Both plateaus are supported by measured points
POT-009      Model adequacy: lack of fit vs the nested 5PL (AICc)
POT-010      Within-level replicate scatter relative to the window
POT-011      Monotonicity of the dose series (hook-effect screen)
===========  =====================================================

Two thresholds deserve a word, because the obvious definitions are wrong:

*POT-002* compares the fitted window with the **control response**, not with
the observed response span.  For any converged fit the two asymptotes bracket
the data almost by construction, so "window over observed span" is close to 1
for every curve -- including a 3 % effect, which it would happily pass.  The
control is the scale that matters: a 3 % window cannot support a claim about
a 50 % effect no matter how clean the fit looks.

*POT-010* compares within-level replicate scatter with the **window**, not
with the level mean.  A coefficient of variation explodes as the response
approaches zero, so on the plateau of an almost complete inhibition curve a
perfectly good 1-unit scatter reads as a 25 % CV and would be flagged.  What
actually limits the potency is whether the scatter is small compared with the
assay window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from bioeng_qpcr.potency.fitting import (
    CurveDirection,
    DoseResponseDataset,
    DoseResponseFit,
    PotencyResult,
    fit_dose_response_curve,
)
from bioeng_qpcr.potency.models import DoseResponseModel, evaluate_model

__all__ = [
    "PotencyValidationResult",
    "run_potency_validation",
    "validate_assay_window",
    "validate_ci_width",
    "validate_concentration_levels",
    "validate_concentration_span",
    "validate_fit_quality",
    "validate_hill_slope",
    "validate_model_adequacy",
    "validate_monotonicity",
    "validate_plateau_support",
    "validate_replicate_precision",
    "validate_within_tested_range",
]

#: Severity values, ordered from most to least serious.
SEVERITY_FAILURE = "failure"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: Default acceptance thresholds.  Callers may override any of them.
DEFAULT_MIN_LEVELS: int = 5
DEFAULT_MIN_POINTS: int = 6
DEFAULT_MIN_R_SQUARED: float = 0.95
DEFAULT_MIN_LOG_SPAN: float = 2.0
DEFAULT_HILL_RANGE: tuple[float, float] = (0.3, 5.0)
DEFAULT_PLATEAU_TOLERANCE: float = 0.15
DEFAULT_MAX_CI_FOLD: float = 10.0
DEFAULT_MAX_REPLICATE_SCATTER: float = 0.10
DEFAULT_MIN_WINDOW_FRACTION: float = 0.30

#: A rule input: either the full potency report or the fitted curve alone.
FitLike = PotencyResult | DoseResponseFit


@dataclass(frozen=True)
class PotencyValidationResult:
    """Outcome of a single potency validation check.

    Attributes
        rule_id:
            Stable rule identifier (``"POT-001"`` .. ``"POT-010"``).
        passed:
            Whether the check passed.
        message:
            Human-readable explanation including the observed values.
        severity:
            ``"failure"`` for a result that should not be reported as-is,
            ``"warning"`` for a caveat, ``"info"`` for context.
        evidence:
            Structured supporting data for auditing.
    """

    rule_id: str
    passed: bool
    message: str
    severity: str = SEVERITY_FAILURE
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_of(result: FitLike) -> DoseResponseFit:
    """Return the :class:`DoseResponseFit` behind *result*."""
    if isinstance(result, PotencyResult):
        return result.best_fit
    return result


def _confidence_of(result: FitLike) -> float:
    """Return the confidence level a result's intervals were built at."""
    if isinstance(result, PotencyResult):
        return result.confidence
    return 0.95


def _kind_of(result: FitLike) -> str:
    """Return ``"IC50"`` or ``"EC50"`` for *result*."""
    if isinstance(result, PotencyResult):
        return result.kind.value
    return "EC50" if _fit_of(result).direction is CurveDirection.INCREASING else "IC50"


def _result(
    rule_id: str,
    passed: bool,
    message: str,
    evidence: dict[str, Any],
    severity: str | None = None,
) -> PotencyValidationResult:
    """Build a result, defaulting the severity from *passed*."""
    if severity is None:
        severity = SEVERITY_INFO if passed else SEVERITY_FAILURE
    return PotencyValidationResult(
        rule_id=rule_id,
        passed=passed,
        message=message,
        severity=severity,
        evidence=evidence,
    )


def _tested_bounds(dataset: DoseResponseDataset) -> tuple[float, float]:
    """Return the ``(min, max)`` tested concentration."""
    concentration = np.asarray(dataset.concentration, dtype=float)
    concentration = concentration[np.isfinite(concentration)]
    if concentration.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.min(concentration)), float(np.max(concentration)))


def _control_response(dataset: DoseResponseDataset) -> float:
    """Return the mean response at the lowest tested concentration.

    That level is the vehicle/untreated control in every ordinary dose series,
    so it is the scale the assay window has to be judged against.
    """
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(response)
    concentration, response = concentration[finite], response[finite]
    if concentration.size == 0:
        return float("nan")
    lowest = float(np.min(concentration))
    return float(np.mean(response[np.isclose(concentration, lowest)]))


# ---------------------------------------------------------------------------
# POT-001: relative potency must be measured, not extrapolated
# ---------------------------------------------------------------------------


def validate_within_tested_range(
    result: FitLike, dataset: DoseResponseDataset
) -> PotencyValidationResult:
    """POT-001: the relative potency must lie inside the tested range.

    An IC50 reported from a curve whose inflection sits outside the tested
    concentrations is an extrapolation: the value is determined entirely by
    the assumed functional form, not by the measurements.

    Parameters
        result:
            Potency result or fitted curve to check.
        dataset:
            The dataset the result came from.

    Returns
        Passes when the half-maximal concentration falls within the tested
        concentrations.
    """
    low, high = _tested_bounds(dataset)
    value = _fit_of(result).half_max_concentration
    kind = _kind_of(result)
    evidence: dict[str, Any] = {
        "relative_value": value,
        "tested_low": low,
        "tested_high": high,
    }
    if not np.isfinite(value):
        return _result("POT-001", False, "Relative potency is not finite.", evidence)
    if low <= value <= high:
        return _result(
            "POT-001",
            True,
            f"Relative {kind} {value:.4g} lies inside the tested range "
            f"{low:.4g} .. {high:.4g}.",
            evidence,
        )
    return _result(
        "POT-001",
        False,
        f"Relative {kind} {value:.4g} lies outside the tested range "
        f"{low:.4g} .. {high:.4g}; report it as a limit, not a value.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-002: assay window
# ---------------------------------------------------------------------------


def validate_assay_window(
    result: FitLike,
    dataset: DoseResponseDataset,
    *,
    min_window_fraction: float = DEFAULT_MIN_WINDOW_FRACTION,
) -> PotencyValidationResult:
    """POT-002: the response window must be large relative to the control.

    The window is ``|top - bottom|`` expressed as a fraction of the control
    response (the mean signal of the lowest tested concentration).  Judging it
    against the control rather than against the observed span is deliberate:
    a converged fit's asymptotes bracket the data almost by construction, so
    the observed span makes the check vacuous.

    Parameters
        result:
            Potency result or fitted curve to check.
        dataset:
            The dataset the result came from.
        min_window_fraction:
            Minimum acceptable window as a fraction of the control response.

    Returns
        Passes when the window reaches *min_window_fraction* of the control.
    """
    fit = _fit_of(result)
    response = np.asarray(dataset.response, dtype=float)
    response = response[np.isfinite(response)]
    observed_span = (
        float(np.max(response) - np.min(response)) if response.size else 0.0
    )
    window = abs(fit.top - fit.bottom)
    control = _control_response(dataset)
    fraction = window / abs(control) if control else 0.0
    evidence = {
        "window": window,
        "control_response": control,
        "window_fraction_of_control": fraction,
        "observed_span": observed_span,
        "window_fraction_of_observed_span": (
            window / observed_span if observed_span else 0.0
        ),
        "min_window_fraction": min_window_fraction,
    }
    if window <= 0 or not np.isfinite(window):
        return _result("POT-002", False, "Fitted window is not positive.", evidence)
    if fraction >= min_window_fraction:
        return _result(
            "POT-002",
            True,
            f"Assay window {window:.4g} is {fraction:.0%} of the control "
            "response; large enough to define a half-maximal level.",
            evidence,
        )
    return _result(
        "POT-002",
        False,
        f"Assay window {window:.4g} is only {fraction:.0%} of the control "
        f"response (< {min_window_fraction:.0%}); the treatment barely moved "
        "the readout, so no potency can be claimed.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-003: experimental design
# ---------------------------------------------------------------------------


def validate_concentration_levels(
    dataset: DoseResponseDataset,
    *,
    min_levels: int = DEFAULT_MIN_LEVELS,
    min_points: int = DEFAULT_MIN_POINTS,
) -> PotencyValidationResult:
    """POT-003: enough distinct concentrations and observations.

    Fitting four or five parameters requires substantially more than four data
    points.  The classic failure mode is a "nice" curve fitted to five points,
    where the confidence interval is wider than the answer.

    Parameters
        dataset:
            Dataset to check.
        min_levels:
            Minimum number of distinct concentration levels.
        min_points:
            Minimum number of observations.

    Returns
        Passes when both minimums are met.
    """
    concentration = np.asarray(dataset.concentration, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(
        np.asarray(dataset.response, dtype=float)
    )
    concentration = concentration[finite]
    levels = int(np.unique(concentration).size)
    points = int(concentration.size)
    evidence = {
        "n_levels": levels,
        "n_points": points,
        "min_levels": min_levels,
        "min_points": min_points,
    }
    if levels >= min_levels and points >= min_points:
        return _result(
            "POT-003",
            True,
            f"{levels} concentration levels and {points} observations; "
            "adequate for a sigmoid fit.",
            evidence,
        )
    problems = []
    if levels < min_levels:
        problems.append(f"only {levels} concentration levels (< {min_levels})")
    if points < min_points:
        problems.append(f"only {points} observations (< {min_points})")
    return _result(
        "POT-003",
        False,
        "Experimental design is too sparse: " + " and ".join(problems) + ".",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-004: Hill slope
# ---------------------------------------------------------------------------


def validate_hill_slope(
    result: FitLike,
    *,
    hill_range: tuple[float, float] = DEFAULT_HILL_RANGE,
) -> PotencyValidationResult:
    """POT-004: the Hill slope magnitude must be physically plausible.

    Values well above the typical range flag steep, switch-like transitions
    that usually indicate non-specific effects (aggregation, cytotoxicity),
    while values far below it indicate a shallow curve whose IC50 is poorly
    localised.

    Parameters
        result:
            Potency result or fitted curve to check.
        hill_range:
            Acceptable ``(min, max)`` Hill slope magnitude.

    Returns
        Passes when the Hill slope magnitude is inside *hill_range*.
    """
    slope = _fit_of(result).hill_slope
    low, high = hill_range
    evidence = {"hill_slope": slope, "hill_range": list(hill_range)}
    if not np.isfinite(slope):
        return _result("POT-004", False, "Hill slope is not finite.", evidence)
    if low <= slope <= high:
        return _result(
            "POT-004",
            True,
            f"Hill slope magnitude {slope:.3g} is within the plausible range "
            f"[{low}, {high}].",
            evidence,
        )
    return _result(
        "POT-004",
        False,
        f"Hill slope magnitude {slope:.3g} is outside [{low}, {high}]; check "
        "for non-specific effects or an over-steep transition.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-005: goodness of fit
# ---------------------------------------------------------------------------


def validate_fit_quality(
    result: FitLike,
    *,
    min_r_squared: float = DEFAULT_MIN_R_SQUARED,
) -> PotencyValidationResult:
    """POT-005: the fitted curve must explain the data.

    Parameters
        result:
            Potency result or fitted curve to check.
        min_r_squared:
            Minimum acceptable coefficient of determination.

    Returns
        Passes when R² is finite and at least *min_r_squared*.
    """
    r_squared = _fit_of(result).r_squared
    evidence = {"r_squared": r_squared, "min_r_squared": min_r_squared}
    if not np.isfinite(r_squared):
        return _result("POT-005", False, "R² is not finite.", evidence)
    if r_squared >= min_r_squared:
        return _result(
            "POT-005", True, f"R² = {r_squared:.4f} meets the threshold.", evidence
        )
    return _result(
        "POT-005",
        False,
        f"R² = {r_squared:.4f} is below {min_r_squared}; the sigmoid does not "
        "describe the data and the potency is unreliable.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-006: concentration span
# ---------------------------------------------------------------------------


def validate_concentration_span(
    dataset: DoseResponseDataset,
    *,
    min_log_span: float = DEFAULT_MIN_LOG_SPAN,
) -> PotencyValidationResult:
    """POT-006: the series should span at least two log10 decades.

    A narrower series usually fails to resolve both plateaus and makes the
    half-maximal position sensitive to the assumed curve shape.

    Parameters
        dataset:
            Dataset to check.
        min_log_span:
            Minimum acceptable span in log10 decades.

    Returns
        Passes when the span is at least *min_log_span* decades.
    """
    low, high = _tested_bounds(dataset)
    evidence = {"tested_low": low, "tested_high": high, "min_log_span": min_log_span}
    if not (np.isfinite(low) and np.isfinite(high)) or low <= 0:
        return _result(
            "POT-006", False, "Tested concentration range is not usable.", evidence
        )
    span = float(np.log10(high / low))
    evidence["log_span"] = span
    if span >= min_log_span:
        return _result(
            "POT-006",
            True,
            f"Concentration series spans {span:.2f} decades.",
            evidence,
        )
    return _result(
        "POT-006",
        False,
        f"Concentration series spans only {span:.2f} decades "
        f"(< {min_log_span}); extend the range.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-007: confidence interval width
# ---------------------------------------------------------------------------


def validate_ci_width(
    result: FitLike,
    *,
    max_fold: float = DEFAULT_MAX_CI_FOLD,
) -> PotencyValidationResult:
    """POT-007: the confidence interval must not be implausibly wide.

    The interval is judged as a *fold* ratio (upper/lower) because potencies
    are log-distributed.  A ten-fold interval means the experiment has located
    the IC50 to within roughly one order of magnitude, which is rarely good
    enough to support a claim.

    Parameters
        result:
            Potency result or fitted curve to check.
        max_fold:
            Maximum acceptable ``upper / lower`` ratio.

    Returns
        Passes when the interval is finite and narrower than *max_fold*, and
        warns (rather than fails) when no interval could be computed at all.
    """
    fit = _fit_of(result)
    interval = fit.half_max_concentration_ci
    confidence = _confidence_of(result)
    evidence: dict[str, Any] = {
        "interval": list(interval) if interval else None,
        "max_fold": max_fold,
        "confidence": confidence,
    }
    if interval is None:
        return _result(
            "POT-007",
            False,
            "No confidence interval is available; report the point estimate "
            "only as provisional.",
            evidence,
            severity=SEVERITY_WARNING,
        )
    low, high = interval
    if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= low:
        return _result(
            "POT-007", False, "Confidence interval is degenerate.", evidence
        )
    fold = high / low
    evidence["fold"] = fold
    if fold <= max_fold:
        return _result(
            "POT-007",
            True,
            f"Confidence interval spans {fold:.2f}-fold ({confidence:.0%} level).",
            evidence,
        )
    return _result(
        "POT-007",
        False,
        f"Confidence interval spans {fold:.2f}-fold, wider than the "
        f"{max_fold}-fold limit; collect more replicates or a denser series.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-008: both plateaus supported by data
# ---------------------------------------------------------------------------


def validate_plateau_support(
    result: FitLike,
    dataset: DoseResponseDataset,
    *,
    tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
) -> PotencyValidationResult:
    """POT-008: measured points must actually define both plateaus.

    Each extreme of the concentration series is compared with the fitted
    asymptote it should approach.  If the mean response at the lowest (or
    highest) concentration is nowhere near the corresponding plateau, that
    plateau is an extrapolation of the model rather than an observation.

    Parameters
        result:
            Potency result or fitted curve to check.
        dataset:
            The dataset the result came from.
        tolerance:
            Allowed fractional deviation from the asymptote, expressed as a
            fraction of the fitted window.

    Returns
        Passes when both extremes sit within *tolerance* of their plateaus.
    """
    fit = _fit_of(result)
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(response)
    concentration, response = concentration[finite], response[finite]
    low, high = float(np.min(concentration)), float(np.max(concentration))
    window = abs(fit.top - fit.bottom)
    evidence: dict[str, Any] = {
        "tolerance": tolerance,
        "window": window,
        "low_concentration": low,
        "high_concentration": high,
    }
    if window <= 0:
        return _result("POT-008", False, "Fitted window is not positive.", evidence)

    def _fraction(value: float) -> float:
        return (value - fit.bottom) / (fit.top - fit.bottom)

    low_response = float(np.mean(response[np.isclose(concentration, low)]))
    high_response = float(np.mean(response[np.isclose(concentration, high)]))
    low_fraction = _fraction(low_response)
    high_fraction = _fraction(high_response)
    evidence.update(
        {
            "low_response": low_response,
            "high_response": high_response,
            "low_fraction": low_fraction,
            "high_fraction": high_fraction,
        }
    )

    if fit.direction is CurveDirection.INCREASING:
        expected_low, expected_high = 0.0, 1.0
    else:
        expected_low, expected_high = 1.0, 0.0
    low_ok = abs(low_fraction - expected_low) <= tolerance
    high_ok = abs(high_fraction - expected_high) <= tolerance
    if low_ok and high_ok:
        return _result(
            "POT-008",
            True,
            "Both plateaus are supported by measured points.",
            evidence,
        )
    missing = []
    if not low_ok:
        missing.append(
            f"lower end ({low_fraction:.0%} of the way to the far plateau)"
        )
    if not high_ok:
        missing.append(
            f"upper end ({high_fraction:.0%} of the way to the far plateau)"
        )
    return _result(
        "POT-008",
        False,
        "Plateau not reached at the " + " and ".join(missing) + "; extend the "
        "concentration range in that direction.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-009: residual pattern (symmetry check)
# ---------------------------------------------------------------------------


def validate_model_adequacy(
    result: FitLike,
    dataset: DoseResponseDataset,
    *,
    max_delta_aicc: float = 4.0,
    z_threshold: float = 1.96,
    rho_threshold: float = 0.6,
) -> PotencyValidationResult:
    """POT-009: the fitted model must actually describe the response.

    The primary test is a **lack-of-fit** comparison by AICc against the most
    flexible model in the family (the 5PL, which nests the 3PL and 4PL).  If
    the flexible model is strongly preferred, the reported potency is an
    artefact of an inadequate model assumption rather than a property of the
    data -- the classic case being an asymmetric curve forced into a 4PL.

    Two residual diagnostics are reported as supporting evidence:
    a Wald-Wolfowitz **runs test** on the residual signs and the **Spearman
    correlation** between the residuals and ``log10(concentration)``.  They
    catch shapes the 5PL cannot absorb either (the high-dose hook effect, a
    drifting baseline).  They are deliberately *not* the primary criterion:
    on the dozen points a dose-response assay provides, a smooth leftover
    trend barely perturbs the run count, so the runs test alone is far too
    underpowered to gate a report on.

    Parameters
        result:
            Potency result or fitted curve to check.
        dataset:
            The dataset the result came from.
        max_delta_aicc:
            AICc improvement over the fitted model that is large enough to
            declare the fit inadequate.  The conventional "strong evidence"
            threshold is 4-10; 4 is used so the screen errs towards warning.
        z_threshold:
            Absolute runs-test z above which the sign pattern is considered
            non-random.
        rho_threshold:
            Absolute Spearman correlation above which a monotone residual
            trend is declared.

    Returns
        Passes when the fitted model is adequate by AICc and neither residual
        diagnostic fires, and warns when there are too few residuals for the
        diagnostics to mean anything.
    """
    fit = _fit_of(result)
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(response)
    concentration, response = concentration[finite], response[finite]
    log_concentration = np.log10(concentration)
    order = np.argsort(log_concentration)
    x_sorted = log_concentration[order]
    predicted = evaluate_model(fit.model, fit.parameters, x_sorted)
    residuals = response[order] - predicted

    evidence: dict[str, Any] = {
        "model": fit.model.value,
        "model_aicc": fit.aicc,
        "max_delta_aicc": max_delta_aicc,
        "z_threshold": z_threshold,
        "rho_threshold": rho_threshold,
    }

    # --- Primary: lack of fit against the most flexible nested model ------
    delta_aicc = 0.0
    if fit.model is not DoseResponseModel.FIVE_PL:
        try:
            flexible = fit_dose_response_curve(
                dataset,
                DoseResponseModel.FIVE_PL,
                direction=fit.direction,
            )
        except ValueError:
            flexible = None
        if flexible is not None and np.isfinite(flexible.aicc):
            delta_aicc = float(fit.aicc - flexible.aicc)
            evidence["flexible_model"] = flexible.model.value
            evidence["flexible_aicc"] = flexible.aicc
    evidence["delta_aicc"] = delta_aicc

    # --- Supporting: residual sign pattern and monotone drift -------------
    signs = np.sign(residuals)
    signs = signs[signs != 0.0]
    n_positive = int(np.sum(signs > 0))
    n_negative = int(np.sum(signs < 0))
    total = n_positive + n_negative
    evidence["n_positive"] = n_positive
    evidence["n_negative"] = n_negative

    z_statistic = 0.0
    runs_enough = total >= 8 and n_positive > 0 and n_negative > 0
    if runs_enough:
        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        expected = 2.0 * n_positive * n_negative / total + 1.0
        variance = (
            2.0
            * n_positive
            * n_negative
            * (2.0 * n_positive * n_negative - total)
            / (total * total * (total - 1.0))
        )
        evidence["runs"] = runs
        evidence["expected_runs"] = expected
        if variance > 0:
            z_statistic = (runs - expected) / float(np.sqrt(variance))
    evidence["z"] = z_statistic

    rho = 0.0
    if residuals.size >= 3:
        rho = float(spearmanr(x_sorted, residuals).statistic)
        if not np.isfinite(rho):
            rho = 0.0
    evidence["rho"] = rho

    lack_of_fit = delta_aicc > max_delta_aicc
    runs_fired = runs_enough and abs(z_statistic) > z_threshold
    rho_fired = abs(rho) > rho_threshold

    if not (lack_of_fit or runs_fired or rho_fired):
        return _result(
            "POT-009",
            True,
            f"Model {fit.model.value} is adequate "
            f"(ΔAICc vs 5PL = {delta_aicc:.2f}, runs z = {z_statistic:.2f}, "
            f"Spearman ρ = {rho:.2f}).",
            evidence,
        )

    reasons: list[str] = []
    if lack_of_fit:
        reasons.append(
            f"the 5PL fits substantially better (ΔAICc = {delta_aicc:.2f} > "
            f"{max_delta_aicc})"
        )
    if runs_fired:
        reasons.append(
            f"the residual sign pattern is not random (z = {z_statistic:.2f})"
        )
    if rho_fired:
        reasons.append(f"residuals drift with concentration (ρ = {rho:.2f})")
    return _result(
        "POT-009",
        False,
        f"Model {fit.model.value} does not describe the response: "
        + "; ".join(reasons)
        + ". The reported potency depends on the model assumption -- refit "
        "with a flexible model and check the raw plate.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-011: monotonicity of the dose series
# ---------------------------------------------------------------------------


def validate_monotonicity(
    result: FitLike,
    dataset: DoseResponseDataset,
    *,
    tolerance_fraction: float = 0.10,
) -> PotencyValidationResult:
    """POT-011: the dose series must not reverse direction.

    A sigmoid is monotone by construction, so a response that goes down and
    then back up is evidence that the model is wrong for the data -- most
    often the **high-dose hook (prozone) effect**, where excess analyte
    saturates the capture reagent and the signal recovers.  A reversed
    dilution series, a mis-pipetted well, or a mis-assigned plate column look
    the same.

    This rule exists because the statistical screens are blind to it: with a
    dozen points, neither the AICc lack-of-fit test (both sigmoid models
    fail equally) nor a runs test (the deviation touches only the top two
    levels) has the power to object.  Comparing consecutive level means
    directly does, because monotonicity is an assumption of the assay rather
    than something to be inferred.

    Parameters
        result:
            Potency result or fitted curve, supplying the expected direction
            and the window against which a reversal is judged.
        dataset:
            The dataset to check.
        tolerance_fraction:
            Reversal size, as a fraction of the fitted window, below which
            the reversal is attributed to noise and ignored.

    Returns
        Passes when no level-to-level change opposes the expected direction
        by more than *tolerance_fraction* of the window.
    """
    fit = _fit_of(result)
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(response)
    concentration, response = concentration[finite], response[finite]
    levels = np.unique(concentration)
    if levels.size < 3:
        return _result(
            "POT-011",
            False,
            "Fewer than three concentration levels; monotonicity cannot be "
            "assessed.",
            {"n_levels": int(levels.size)},
            severity=SEVERITY_WARNING,
        )

    means = np.array(
        [float(np.mean(response[np.isclose(concentration, level)])) for level in levels]
    )
    window = abs(fit.top - fit.bottom)
    evidence: dict[str, Any] = {
        "window": window,
        "tolerance_fraction": tolerance_fraction,
        "levels": [float(level) for level in levels],
        "means": [float(value) for value in means],
    }
    if window <= 0:
        return _result(
            "POT-011",
            False,
            "Fitted window is not positive; reversals cannot be scaled.",
            evidence,
            severity=SEVERITY_WARNING,
        )

    expected_sign = -1.0 if fit.direction is CurveDirection.DECREASING else 1.0
    threshold = tolerance_fraction * window
    reversals: list[tuple[float, float, float]] = []
    for index in range(means.size - 1):
        change = float(means[index + 1] - means[index])
        if expected_sign * change < -threshold:
            reversals.append((float(levels[index]), float(levels[index + 1]), change))
    evidence["reversals"] = [
        {"from": start, "to": stop, "change": change}
        for start, stop, change in reversals
    ]
    if not reversals:
        return _result(
            "POT-011",
            True,
            f"Response is monotone across all {levels.size} levels, consistent "
            f"with a {'decreasing' if expected_sign < 0 else 'increasing'} curve.",
            evidence,
        )

    worst = min(reversals, key=lambda item: expected_sign * item[2])
    fraction = abs(worst[2]) / window
    return _result(
        "POT-011",
        False,
        f"Response reverses between {worst[0]:.4g} and {worst[1]:.4g} "
        f"({worst[2]:+.4g}, {fraction:.0%} of the window). A sigmoid cannot do "
        "this: suspect the high-dose hook (prozone) effect, a reversed "
        "dilution series, or a mis-pipetted well. An IC50 fitted across a "
        "reversal is not interpretable.",
        evidence,
    )


# ---------------------------------------------------------------------------
# POT-010: replicate scatter
# ---------------------------------------------------------------------------


def validate_replicate_precision(
    dataset: DoseResponseDataset,
    result: FitLike | None = None,
    *,
    max_scatter_fraction: float = DEFAULT_MAX_REPLICATE_SCATTER,
) -> PotencyValidationResult:
    """POT-010: within-level replicate scatter must be small vs the window.

    For each concentration level with more than one replicate the standard
    deviation of the per-replicate means is computed, and the largest such
    scatter is expressed as a fraction of the fitted assay window.  Scaling by
    the window rather than by the level mean avoids the classic trap where a
    level sitting on the bottom plateau reports a huge coefficient of
    variation for a perfectly acceptable absolute scatter.

    Parameters
        dataset:
            Dataset to check.
        result:
            Optional potency result or curve, used to obtain the window.  When
            omitted the observed response span is used instead.
        max_scatter_fraction:
            Maximum acceptable ratio of replicate scatter to the window.

    Returns
        Passes when no level exceeds *max_scatter_fraction*, and warns when
        replicate labels are unavailable.
    """
    groups = dataset.replicate_group
    if groups is None or len(groups) != np.size(dataset.concentration):
        return _result(
            "POT-010",
            False,
            "No per-observation replicate labels were supplied; within-level "
            "imprecision could not be assessed.",
            {"replicate_labels": False, "max_scatter_fraction": max_scatter_fraction},
            severity=SEVERITY_WARNING,
        )

    response = np.asarray(dataset.response, dtype=float)
    concentration = np.asarray(dataset.concentration, dtype=float)
    labels = np.asarray(groups, dtype=object)

    worst_scatter = 0.0
    worst_level: float | None = None
    worst_cv = 0.0
    n_multi = 0
    for level in np.unique(concentration):
        mask = np.isclose(concentration, level)
        values = response[mask]
        # Replicates are identified by the label AND the concentration, so a
        # label reused at another level is not pooled.
        distinct_labels = np.unique(labels[mask])
        if distinct_labels.size < 2 or values.size < 2:
            continue
        per_label = np.array(
            [
                float(np.mean(values[labels[mask] == label]))
                for label in distinct_labels
            ]
        )
        n_multi += 1
        scatter = float(np.std(per_label, ddof=1))
        mean = float(np.mean(per_label))
        if scatter > worst_scatter:
            worst_scatter, worst_level = scatter, float(level)
        if mean:
            worst_cv = max(worst_cv, scatter / abs(mean))

    reference = _reference_span(dataset, result)
    evidence: dict[str, Any] = {
        "max_scatter_fraction": max_scatter_fraction,
        "worst_scatter": worst_scatter,
        "worst_level": worst_level,
        "worst_cv": worst_cv,
        "reference_span": reference,
        "n_levels_with_replicates": n_multi,
    }
    if n_multi == 0:
        return _result(
            "POT-010",
            False,
            "No concentration level has more than one replicate; the assay "
            "precision is unknown.",
            evidence,
            severity=SEVERITY_WARNING,
        )
    if reference <= 0:
        return _result(
            "POT-010",
            False,
            "The assay window is not positive, so replicate scatter cannot be "
            "scaled.",
            evidence,
            severity=SEVERITY_WARNING,
        )
    fraction = worst_scatter / reference
    evidence["scatter_fraction"] = fraction
    if fraction <= max_scatter_fraction:
        return _result(
            "POT-010",
            True,
            f"Worst within-level scatter is {worst_scatter:.4g} "
            f"({fraction:.1%} of the window) across {n_multi} replicated levels.",
            evidence,
        )
    return _result(
        "POT-010",
        False,
        f"Within-level scatter reaches {worst_scatter:.4g} at concentration "
        f"{worst_level:.4g} ({fraction:.1%} of the window, limit "
        f"{max_scatter_fraction:.0%}); the assay precision limits the "
        "confidence in the potency.",
        evidence,
    )


def _reference_span(
    dataset: DoseResponseDataset, result: FitLike | None
) -> float:
    """Return the scale replicate scatter is compared against."""
    if result is not None:
        fit = _fit_of(result)
        window = abs(fit.top - fit.bottom)
        if window > 0:
            return window
    response = np.asarray(dataset.response, dtype=float)
    response = response[np.isfinite(response)]
    if response.size == 0:
        return 0.0
    return float(np.max(response) - np.min(response))


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


def run_potency_validation(
    result: FitLike,
    dataset: DoseResponseDataset,
    *,
    min_levels: int = DEFAULT_MIN_LEVELS,
    min_points: int = DEFAULT_MIN_POINTS,
    min_r_squared: float = DEFAULT_MIN_R_SQUARED,
    min_log_span: float = DEFAULT_MIN_LOG_SPAN,
    hill_range: tuple[float, float] = DEFAULT_HILL_RANGE,
    plateau_tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
    max_ci_fold: float = DEFAULT_MAX_CI_FOLD,
    max_replicate_scatter: float = DEFAULT_MAX_REPLICATE_SCATTER,
    min_window_fraction: float = DEFAULT_MIN_WINDOW_FRACTION,
) -> list[PotencyValidationResult]:
    """Run every potency validation rule against *result*.

    Parameters
        result:
            Potency result or fitted curve to validate.
        dataset:
            The dataset the result came from.
        min_levels, min_points, min_r_squared, min_log_span, hill_range,
        plateau_tolerance, max_ci_fold, max_replicate_scatter,
        min_window_fraction:
            Threshold overrides, documented on the individual rules.

    Returns
        Rule results in identifier order.  Every entry whose
        ``severity == "failure"`` should be resolved before the potency is
        reported.
    """
    return [
        validate_within_tested_range(result, dataset),
        validate_assay_window(
            result, dataset, min_window_fraction=min_window_fraction
        ),
        validate_concentration_levels(
            dataset, min_levels=min_levels, min_points=min_points
        ),
        validate_hill_slope(result, hill_range=hill_range),
        validate_fit_quality(result, min_r_squared=min_r_squared),
        validate_concentration_span(dataset, min_log_span=min_log_span),
        validate_ci_width(result, max_fold=max_ci_fold),
        validate_plateau_support(result, dataset, tolerance=plateau_tolerance),
        validate_model_adequacy(result, dataset),
        validate_replicate_precision(
            dataset, result, max_scatter_fraction=max_replicate_scatter
        ),
        validate_monotonicity(result, dataset),
    ]
