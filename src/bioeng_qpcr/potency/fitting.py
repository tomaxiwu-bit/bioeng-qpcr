"""Potency determination: fitting dose-response curves and deriving IC50/EC50.

This module turns a table of ``(concentration, response)`` measurements into
an auditable potency estimate.  It deliberately distinguishes three
quantities that are routinely (and incorrectly) conflated:

Relative potency
    The concentration at which the fitted curve is exactly halfway between
    its two asymptotes.  This is what "the IC50" almost always means in a
    paper, and it is what the inflection point of a 4PL gives you directly.
    It is *not* 50 % inhibition whenever the lower asymptote is non-zero.

Absolute potency
    The concentration that produces a response equal to a user-chosen
    fraction (by default 50 %) of the untreated control response.  This is
    the pharmacologically meaningful "50 % inhibition" and it requires
    inverting the fitted curve rather than reading off a parameter.

Reported parameter
    The fitted ``log_ec50``/``log_ic50`` coefficient.  For 3PL/4PL it equals
    the relative potency; for 5PL and Gompertz it does **not**, so this
    module never reports the raw parameter as a potency.

Confidence intervals come in two flavours.  The asymptotic interval uses the
covariance matrix returned by :func:`scipy.optimize.curve_fit` and propagates
it onto the half-maximal position with the delta method (so the interval is
correct even for models whose half-maximal position is not a parameter).
The bootstrap interval resamples observations and refits, which is more
honest for the small datasets typical of a dose-response experiment but
costs ``n_bootstrap`` extra fits.

Everything is computed on ``x = log10(concentration)``; see
:mod:`bioeng_qpcr.potency.models` for the sign convention of the Hill slope.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, unique
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import t as student_t

from bioeng_qpcr.potency.models import (
    MODEL_PARAMETERS,
    DoseResponseModel,
    evaluate_model,
    half_max_log_concentration,
    solve_log_concentration_for_response,
)

__all__ = [
    "CurveDirection",
    "DoseResponseDataset",
    "DoseResponseFit",
    "ModelComparison",
    "PotencyComparison",
    "PotencyKind",
    "PotencyResult",
    "compare_dose_response_models",
    "fit_dose_response",
    "fit_dose_response_batch",
    "fit_dose_response_curve",
    "relative_potency",
]

#: Hill magnitudes outside this window are flagged by the validation rules.
_MAX_HILL_MAGNITUDE: float = 10.0

#: Default candidate models, in increasing order of complexity.
_DEFAULT_MODELS: tuple[DoseResponseModel, ...] = (
    DoseResponseModel.FOUR_PL,
    DoseResponseModel.FIVE_PL,
    DoseResponseModel.GOMPERTZ,
)


def _linear_concentration_interval(
    log_interval: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Convert a log10 interval without leaking overflow to callers."""
    if log_interval is None:
        return None
    try:
        interval = (float(10.0 ** log_interval[0]), float(10.0 ** log_interval[1]))
    except OverflowError:
        return None
    if not (
        0.0 < interval[0] <= interval[1]
        and math.isfinite(interval[0])
        and math.isfinite(interval[1])
    ):
        return None
    return interval


@unique
class CurveDirection(str, Enum):
    """Direction of the dose-response relationship.

    Attributes
        AUTO:
            Infer the direction from the data (default).
        INCREASING:
            Response rises with concentration (agonist/activator assay).
        DECREASING:
            Response falls with concentration (inhibitor/cytotoxicity
            assay).
    """

    AUTO = "auto"
    INCREASING = "increasing"
    DECREASING = "decreasing"


@unique
class PotencyKind(str, Enum):
    """Which half-maximal quantity is being reported."""

    IC50 = "IC50"
    """Half-maximal *inhibitory* concentration (response falls)."""

    EC50 = "EC50"
    """Half-maximal *effective* concentration (response rises)."""


@dataclass(frozen=True)
class DoseResponseDataset:
    """A single compound's dose-response measurements.

    Attributes
        compound:
            Name of the compound, drug, or treatment.
        concentration:
            Concentrations as given by the user.  Must be strictly positive
            (the models are fitted in ``log10`` space).
        response:
            Measured responses, aligned with *concentration*.
        concentration_unit:
            Unit of *concentration* (e.g. ``"uM"``), used only for display.
        response_unit:
            Unit of *response* (e.g. ``"% inhibition"``), display only.
        direction:
            Expected curve direction; ``AUTO`` infers it from the data.
        replicate_group:
            Optional label per observation (e.g. well or replicate ID),
            preserved for auditing.
    """

    compound: str
    concentration: np.ndarray
    response: np.ndarray
    concentration_unit: str = ""
    response_unit: str = ""
    direction: CurveDirection = CurveDirection.AUTO
    replicate_group: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "concentration", np.asarray(self.concentration, dtype=float)
        )
        object.__setattr__(self, "response", np.asarray(self.response, dtype=float))

    def __len__(self) -> int:
        return int(self.concentration.size)


@dataclass(frozen=True)
class DoseResponseFit:
    """Result of fitting one model to one dataset.

    Attributes
        compound:
            Compound name, copied from the dataset.
        model:
            The fitted model.
        direction:
            Resolved curve direction.
        parameters:
            Fitted parameter values keyed by :data:`MODEL_PARAMETERS`.  The
            ``hill`` entry is **signed**.
        parameter_std:
            Standard error of each parameter from the covariance matrix
            (``nan`` when unavailable).
        parameter_ci:
            Two-sided confidence interval per parameter, ``None`` when the
            standard error is unavailable.
        half_max_log_concentration:
            ``log10`` of the relative half-maximal concentration -- the
            midpoint between the fitted asymptotes, correctly derived for
            asymmetric models.
        half_max_ci:
            Confidence interval for ``half_max_log_concentration``.
        hill_slope:
            Conventional positive Hill slope magnitude, ``|hill|``.
        bottom:
            Fitted lower asymptote.
        top:
            Fitted upper asymptote.
        n_points:
            Number of observations used.
        k_params:
            Number of fitted parameters.
        r_squared:
            Coefficient of determination on the fitted curve.
        r_squared_adjusted:
            Degrees-of-freedom-adjusted R².
        rmse:
            Residual standard error ``sqrt(RSS / (n - k))``.
        aic, aicc, bic:
            Information criteria (smaller is better).  ``aicc`` is ``inf``
            when ``n - k - 1 <= 0``.
        converged:
            Whether the nonlinear fit converged.
        warnings:
            Human-readable notes raised while fitting.
    """

    compound: str
    model: DoseResponseModel
    direction: CurveDirection
    parameters: dict[str, float]
    parameter_std: dict[str, float]
    parameter_ci: dict[str, tuple[float, float] | None]
    half_max_log_concentration: float
    half_max_ci: tuple[float, float] | None
    hill_slope: float
    bottom: float
    top: float
    n_points: int
    k_params: int
    r_squared: float
    r_squared_adjusted: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    converged: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def half_max_concentration(self) -> float:
        """Relative IC50/EC50 in the dataset's concentration units."""
        return float(10.0**self.half_max_log_concentration)

    @property
    def half_max_concentration_ci(self) -> tuple[float, float] | None:
        """Confidence interval of :attr:`half_max_concentration`."""
        return _linear_concentration_interval(self.half_max_ci)


@dataclass(frozen=True)
class ModelComparison:
    """Ranked candidate fits for one dataset.

    Attributes
        fits:
            All candidate fits, sorted by AICc (best first), or by AIC when
            every candidate's AICc is undefined for the available sample size.
        best:
            The winning fit.
        ambiguous:
            ``True`` when the runner-up is within 2 information-criterion
            units, i.e. the data do not clearly favour one model.
        message:
            Human-readable summary of the selection.
    """

    fits: list[DoseResponseFit] = field(default_factory=list)
    best: DoseResponseFit | None = None
    ambiguous: bool = False
    message: str = ""


@dataclass(frozen=True)
class PotencyResult:
    """Complete potency report for one compound.

    Attributes
        kind:
            Whether an IC50 or an EC50 is being reported.
        compound:
            Compound name.
        best_fit:
            Fitted curve used for the potency values.
        comparison:
            Ranked model comparison behind the choice of ``best_fit``.
        relative_value:
            Relative IC50/EC50 (midpoint of the fitted window).
        relative_ci:
            Confidence interval of :attr:`relative_value`.
        absolute_value:
            Absolute IC50/EC50 (response equal to
            ``control_fraction * control_response``); ``None`` when the
            target level is not reachable.
        absolute_ci:
            Confidence interval of :attr:`absolute_value`.
        control_response:
            The control response the absolute potency was referred to.
        control_fraction:
            Fraction of the control response used as the absolute target.
        ci_method:
            ``"asymptotic"`` or ``"bootstrap"``.
        confidence:
            Confidence level of every reported interval.
        extrapolated:
            ``True`` when the relative potency falls outside the tested
            concentration range.
        fold_range_covered:
            How many log10 decades of concentration were tested.
        warnings:
            Aggregated warnings from fitting and interpretation.
        message:
            One-line summary suitable for a report or a UI.
    """

    kind: PotencyKind
    compound: str
    best_fit: DoseResponseFit
    comparison: ModelComparison
    relative_value: float
    relative_ci: tuple[float, float] | None
    absolute_value: float | None
    absolute_ci: tuple[float, float] | None
    control_response: float | None
    control_fraction: float
    ci_method: str
    confidence: float
    extrapolated: bool
    fold_range_covered: float
    warnings: list[str] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class PotencyComparison:
    """Relative potency of compound A against compound B.

    Attributes
        kind:
            IC50 or EC50.
        compound_a, compound_b:
            The two compounds, A being the numerator of the ratio.
        ratio:
            ``value_a / value_b``.  A value below 1 means A is *more*
            potent than B.
        ci:
            Confidence interval of :attr:`ratio`.
        fold_difference:
            ``max(ratio, 1/ratio)`` -- how many fold the two potencies
            differ, independent of direction.
        confidence:
            Confidence level of :attr:`ci`.
        message:
            Human-readable summary.
    """

    kind: PotencyKind
    compound_a: str
    compound_b: str
    ratio: float
    ci: tuple[float, float] | None
    fold_difference: float
    confidence: float
    message: str = ""


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _clean_dataset(dataset: DoseResponseDataset, min_points: int) -> np.ndarray:
    """Validate a dataset and return ``log10(concentration)``.

    Parameters
        dataset:
            Dataset to validate.
        min_points:
            Minimum number of usable observations required.

    Returns
        ``log10`` of the finite, strictly positive concentrations.

    Raises
        ValueError:
            If the arrays are mismatched, non-finite, non-positive, or too
            short, or if fewer than three distinct concentrations remain.
    """
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    if concentration.shape != response.shape:
        raise ValueError(
            "concentration and response must have the same shape, got "
            f"{concentration.shape} and {response.shape}."
        )
    if concentration.ndim != 1:
        raise ValueError("concentration and response must be 1-dimensional.")
    finite = np.isfinite(concentration) & np.isfinite(response)
    concentration = concentration[finite]
    if concentration.size < min_points:
        raise ValueError(
            f"Need at least {min_points} finite observations, got {concentration.size}."
        )
    if np.any(concentration <= 0.0):
        raise ValueError(
            "All concentrations must be strictly positive; the curve is "
            "fitted on log10(concentration)."
        )
    log_concentration = np.log10(concentration)
    if np.unique(log_concentration).size < 3:
        raise ValueError(
            "At least 3 distinct concentration levels are required to fit a "
            "dose-response curve."
        )
    return log_concentration


def _resolve_direction(
    log_concentration: np.ndarray, response: np.ndarray, requested: CurveDirection
) -> CurveDirection:
    """Resolve ``AUTO`` into a concrete direction from the data trend."""
    if requested is not CurveDirection.AUTO:
        return requested
    centred_x = log_concentration - float(np.mean(log_concentration))
    centred_y = response - float(np.mean(response))
    denominator = float(np.sqrt(np.sum(centred_x**2) * np.sum(centred_y**2)))
    if denominator == 0.0:
        # A flat response is not a curve; default to the inhibition
        # convention and let the validation rules reject it.
        return CurveDirection.DECREASING
    slope = float(np.sum(centred_x * centred_y) / denominator)
    return CurveDirection.INCREASING if slope >= 0 else CurveDirection.DECREASING


def _initial_guesses(
    log_concentration: np.ndarray, response: np.ndarray, direction: CurveDirection
) -> list[list[float]]:
    """Build data-driven multi-start initial guesses for the 4PL shape.

    Several starting points are produced because the ``log_ec50`` sum of
    squares surface is notoriously multi-modal when there are few points on
    the plateaus; the caller keeps whichever start converges best.

    Parameters
        log_concentration:
            ``log10`` concentrations.
        response:
            Aligned responses.
        direction:
            Resolved curve direction.

    Returns
        A list of ``[bottom, top, log_ec50, hill]`` starting vectors.
    """
    order = np.argsort(log_concentration)
    x_sorted = log_concentration[order]
    y_sorted = response[order]
    n_low = max(1, int(round(0.25 * y_sorted.size)))
    low_response = float(np.mean(y_sorted[:n_low]))
    high_response = float(np.mean(y_sorted[-n_low:]))

    if direction is CurveDirection.INCREASING:
        bottom, top = low_response, high_response
        sign = 1.0
    else:
        bottom, top = high_response, low_response
        sign = -1.0

    midpoint = 0.5 * (bottom + top)
    span = float(np.max(y_sorted) - np.min(y_sorted)) or 1.0
    increasing = y_sorted if direction is CurveDirection.INCREASING else -y_sorted
    crossings = np.where(np.diff(np.sign(increasing - midpoint)) != 0)[0]
    if crossings.size:
        index = int(crossings[0])
        crossing = float(
            np.interp(
                midpoint,
                [increasing[index], increasing[index + 1]],
                [x_sorted[index], x_sorted[index + 1]],
            )
        )
    else:
        crossing = float(np.median(x_sorted))

    # Widen the asymptote guesses slightly beyond the observed responses so
    # the optimiser is free to place a plateau just outside the data.
    span_bottom = bottom - 0.05 * span
    span_top = top + 0.05 * span

    candidates = [crossing, float(np.median(x_sorted))]
    candidates.append(float(x_sorted[0]))
    candidates.append(float(x_sorted[-1]))

    guesses: list[list[float]] = []
    for centre in candidates:
        guesses.append([span_bottom, span_top, centre, sign * 1.0])
        guesses.append([span_bottom, span_top, centre, sign * 0.5])
        guesses.append([span_bottom, span_top, centre, sign * 2.0])
    return guesses


def _bounds(
    log_concentration: np.ndarray,
    response: np.ndarray,
    model: DoseResponseModel,
    direction: CurveDirection,
    allow_extrapolation: bool,
) -> tuple[list[float], list[float]]:
    """Return ``(lower, upper)`` parameter bounds for *model*."""
    span = float(np.max(response) - np.min(response)) or 1.0
    response_low = float(np.min(response)) - 5.0 * span
    response_high = float(np.max(response)) + 5.0 * span
    pad = 3.0 if allow_extrapolation else 0.0
    x_low = float(np.min(log_concentration)) - pad
    x_high = float(np.max(log_concentration)) + pad
    if direction is CurveDirection.INCREASING:
        hill_bounds = (1e-4, _MAX_HILL_MAGNITUDE)
    else:
        hill_bounds = (-_MAX_HILL_MAGNITUDE, -1e-4)

    names = MODEL_PARAMETERS[model]
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        if name == "bottom":
            lower.append(response_low)
            upper.append(response_high)
        elif name == "top":
            lower.append(response_low)
            upper.append(response_high)
        elif name == "log_ec50":
            lower.append(x_low)
            upper.append(x_high)
        elif name == "hill":
            lower.append(hill_bounds[0])
            upper.append(hill_bounds[1])
        elif name == "asym":
            lower.append(0.2)
            upper.append(5.0)
        else:  # pragma: no cover - guards against a registry typo
            raise ValueError(f"Unknown parameter {name!r} in model registry.")
    return lower, upper


def _information_criteria(rss: float, n: int, k: int) -> tuple[float, float, float]:
    """Return ``(AIC, AICc, BIC)`` for a least-squares fit.

    *k* counts fitted parameters plus one for the implicit noise scale.
    """
    if not math.isfinite(rss) or n <= 0:
        return (math.inf, math.inf, math.inf)
    tiny = np.finfo(float).tiny
    log_term = n * math.log(max(rss, tiny) / n)
    aic = log_term + 2 * k
    bic = log_term + k * math.log(n)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf
    return (float(aic), float(aicc), float(bic))


def _numerical_gradient(
    function: Callable[[dict[str, float]], float],
    parameters: dict[str, float],
    names: Sequence[str],
    step: float = 1e-6,
) -> np.ndarray:
    """Central-difference gradient of *function* with respect to *names*."""
    gradient = np.zeros(len(names), dtype=float)
    for index, name in enumerate(names):
        delta = step * max(1.0, abs(parameters[name]))
        plus = dict(parameters)
        minus = dict(parameters)
        plus[name] = parameters[name] + delta
        minus[name] = parameters[name] - delta
        gradient[index] = (function(plus) - function(minus)) / (2.0 * delta)
    return gradient


def _critical_value(confidence: float, degrees_of_freedom: int) -> float:
    """Two-sided Student-t critical value, falling back to the normal."""
    if degrees_of_freedom <= 0:
        return float("nan")
    return float(student_t.ppf(0.5 + 0.5 * confidence, degrees_of_freedom))


# ---------------------------------------------------------------------------
# Single-model fitting
# ---------------------------------------------------------------------------


def _fit_single_model(
    dataset: DoseResponseDataset,
    model: DoseResponseModel,
    log_concentration: np.ndarray,
    response: np.ndarray,
    direction: CurveDirection,
    *,
    confidence: float,
    allow_extrapolation: bool,
    min_points: int,
) -> DoseResponseFit:
    """Fit one model with multi-start initialisation.

    Raises
        ValueError:
            If no starting point converges to a finite solution.
    """
    names = MODEL_PARAMETERS[model]
    lower, upper = _bounds(
        log_concentration, response, model, direction, allow_extrapolation
    )

    def _model_function(x: np.ndarray, *values: float) -> np.ndarray:
        parameters = dict(zip(names, (float(v) for v in values), strict=True))
        return evaluate_model(model, parameters, x)

    base_guesses = _initial_guesses(log_concentration, response, direction)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for guess in base_guesses:
        p0 = _expand_guess(guess, model)
        try:
            with catch_warnings():
                # A singular covariance matrix is an expected outcome for
                # sparse data; it is reported through ``parameter_ci=None`` and
                # the warnings list, so SciPy's raw warning is suppressed to
                # keep the user-facing output readable.
                simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    _model_function,
                    log_concentration,
                    response,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=40000,
                    method="trf",
                )
        except (RuntimeError, ValueError):
            continue
        predicted = _model_function(log_concentration, *popt)
        rss = float(np.sum((response - predicted) ** 2))
        if math.isfinite(rss) and (best is None or rss < best[0]):
            best = (rss, popt, pcov)

    if best is None:
        raise ValueError(
            f"Model {model.value!r} did not converge for compound "
            f"{dataset.compound!r}. The data may be too sparse or flat for "
            "a sigmoid fit."
        )

    rss, popt, pcov = best
    parameters = {name: float(value) for name, value in zip(names, popt, strict=True)}
    standard_errors = np.sqrt(np.abs(np.diag(pcov)))
    parameter_std = {
        name: (float(se) if np.isfinite(se) else float("nan"))
        for name, se in zip(names, standard_errors, strict=True)
    }
    degrees_of_freedom = int(response.size - len(names))
    critical = _critical_value(confidence, degrees_of_freedom)
    parameter_ci: dict[str, tuple[float, float] | None] = {}
    for name in names:
        se = parameter_std[name]
        if math.isfinite(se) and math.isfinite(critical):
            parameter_ci[name] = (
                parameters[name] - critical * se,
                parameters[name] + critical * se,
            )
        else:
            parameter_ci[name] = None

    warnings: list[str] = []
    covariance_usable = math.isfinite(critical)
    if not covariance_usable:
        warnings.append(
            "Covariance matrix is not estimable (too few observations for "
            "the number of parameters); confidence intervals are omitted."
        )

    half_max = half_max_log_concentration(model, parameters)
    half_max_ci: tuple[float, float] | None = None
    if covariance_usable:
        gradient = _numerical_gradient(
            lambda values: half_max_log_concentration(model, values),
            parameters,
            names,
        )
        variance = float(gradient @ pcov @ gradient)
        if math.isfinite(variance) and variance >= 0.0:
            error = math.sqrt(variance)
            half_max_ci = (
                half_max - critical * error,
                half_max + critical * error,
            )
    if half_max_ci is None and "log_ec50" in parameter_ci:
        # Fall back to the raw parameter interval; exact for 3PL/4PL and a
        # documented approximation otherwise.
        raw = parameter_ci["log_ec50"]
        if raw is not None and model in (
            DoseResponseModel.THREE_PL,
            DoseResponseModel.FOUR_PL,
        ):
            half_max_ci = raw

    predicted = _model_function(log_concentration, *popt)
    mean_response = float(np.mean(response))
    total_sum_squares = float(np.sum((response - mean_response) ** 2))
    if total_sum_squares > 0.0:
        r_squared = 1.0 - rss / total_sum_squares
        r_squared_adjusted = (
            1.0
            - (1.0 - r_squared) * (response.size - 1) / (response.size - len(names) - 1)
            if response.size - len(names) - 1 > 0
            else float("nan")
        )
    else:
        r_squared = float("nan")
        r_squared_adjusted = float("nan")
    rmse = (
        math.sqrt(rss / degrees_of_freedom)
        if degrees_of_freedom > 0
        else math.sqrt(rss / response.size)
    )
    aic, aicc, bic = _information_criteria(rss, int(response.size), len(names) + 1)

    return DoseResponseFit(
        compound=dataset.compound,
        model=model,
        direction=direction,
        parameters=parameters,
        parameter_std=parameter_std,
        parameter_ci=parameter_ci,
        half_max_log_concentration=half_max,
        half_max_ci=half_max_ci,
        hill_slope=abs(parameters["hill"]),
        bottom=0.0 if model is DoseResponseModel.THREE_PL else parameters["bottom"],
        top=parameters["top"],
        n_points=int(response.size),
        k_params=len(names),
        r_squared=r_squared,
        r_squared_adjusted=r_squared_adjusted,
        rmse=rmse,
        aic=aic,
        aicc=aicc,
        bic=bic,
        converged=True,
        warnings=warnings,
    )


def _expand_guess(guess: list[float], model: DoseResponseModel) -> list[float]:
    """Expand a 4PL start vector to the parameter count of *model*."""
    if model is DoseResponseModel.THREE_PL:
        return [guess[1], guess[2], guess[3]]
    if model is DoseResponseModel.FIVE_PL:
        return [guess[0], guess[1], guess[2], guess[3], 1.0]
    return list(guess)


def fit_dose_response_curve(
    dataset: DoseResponseDataset,
    model: DoseResponseModel = DoseResponseModel.FOUR_PL,
    *,
    direction: CurveDirection | None = None,
    confidence: float = 0.95,
    allow_extrapolation: bool = True,
    min_points: int = 5,
) -> DoseResponseFit:
    """Fit a single dose-response model to a dataset.

    Parameters
        dataset:
            The compound's measurements.
        model:
            Model to fit.
        direction:
            Overrides ``dataset.direction`` when given.
        confidence:
            Confidence level for the reported intervals (0-1).
        allow_extrapolation:
            When ``True`` (the default) the fitted half-maximal
            concentration may fall slightly outside the tested range, and
            the caller is expected to flag it via the validation rules.
            When ``False`` the search is confined to the tested range.
        min_points:
            Minimum number of observations required.

    Returns
        The fitted :class:`DoseResponseFit`.

    Raises
        ValueError:
            If the dataset is unusable, the confidence level is out of
            range, or the model does not converge.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}.")
    log_concentration = _clean_dataset(dataset, min_points)
    response = np.asarray(dataset.response, dtype=float)[
        np.isfinite(dataset.concentration) & np.isfinite(dataset.response)
    ]
    requested = direction if direction is not None else dataset.direction
    resolved = _resolve_direction(log_concentration, response, requested)
    fit = _fit_single_model(
        dataset,
        model,
        log_concentration,
        response,
        resolved,
        confidence=confidence,
        allow_extrapolation=allow_extrapolation,
        min_points=min_points,
    )

    if requested is not CurveDirection.AUTO:
        inferred = _resolve_direction(log_concentration, response, CurveDirection.AUTO)
        if inferred is not requested:
            # The symmetric models can always mirror themselves by swapping
            # the two asymptotes, so a wrong direction flag still converges --
            # to a curve labelled backwards.  That is worth saying out loud.
            fit = replace(
                fit,
                warnings=[
                    *fit.warnings,
                    f"The requested curve direction ({requested.value}) "
                    f"contradicts the trend in the data ({inferred.value}); "
                    "the asymptotes are labelled as requested, so verify the "
                    "assay direction and the plate layout.",
                ],
            )
    return fit


def compare_dose_response_models(
    dataset: DoseResponseDataset,
    models: Iterable[DoseResponseModel] | None = None,
    *,
    direction: CurveDirection | None = None,
    confidence: float = 0.95,
    allow_extrapolation: bool = True,
    min_points: int = 5,
) -> ModelComparison:
    """Fit every candidate model and rank them by AICc, falling back to AIC.

    The 4PL is only a special case of the 5PL, so model selection is a
    genuine decision rather than a formality: whenever the extra asymmetry
    parameter is not supported by the data, AICc penalises it and the 4PL
    wins.  A runner-up within ΔAICc < 2 is reported as ambiguous so the
    caller can avoid over-claiming.

    Parameters
        dataset:
            The compound's measurements.
        models:
            Candidate models; defaults to 4PL, 5PL and Gompertz.
        direction:
            Overrides ``dataset.direction`` when given.
        confidence:
            Confidence level for the reported intervals.
        allow_extrapolation:
            Passed through to the single-model fit.
        min_points:
            Minimum number of observations required.

    Returns
        A :class:`ModelComparison` sorted by AICc, or AIC when AICc is
        undefined for every converged model.

    Raises
        ValueError:
            If no candidate model converges.
    """
    candidates = tuple(models) if models is not None else _DEFAULT_MODELS
    if not candidates:
        raise ValueError("At least one candidate model is required.")

    log_concentration = _clean_dataset(dataset, min_points)
    response = np.asarray(dataset.response, dtype=float)[
        np.isfinite(dataset.concentration) & np.isfinite(dataset.response)
    ]
    resolved = _resolve_direction(
        log_concentration,
        response,
        direction if direction is not None else dataset.direction,
    )

    fits: list[DoseResponseFit] = []
    for model in candidates:
        try:
            fits.append(
                _fit_single_model(
                    dataset,
                    model,
                    log_concentration,
                    response,
                    resolved,
                    confidence=confidence,
                    allow_extrapolation=allow_extrapolation,
                    min_points=min_points,
                )
            )
        except ValueError:
            continue
    if not fits:
        raise ValueError(
            f"No candidate dose-response model converged for compound "
            f"{dataset.compound!r}."
        )

    use_aicc = all(math.isfinite(fit.aicc) for fit in fits)
    criterion = (lambda fit: fit.aicc) if use_aicc else (lambda fit: fit.aic)
    label = "AICc" if use_aicc else "AIC"
    fits.sort(key=lambda fit: (criterion(fit), fit.aic))
    best = fits[0]
    delta = criterion(fits[1]) - criterion(best) if len(fits) > 1 else math.inf
    ambiguous = delta < 2.0
    message = (
        f"Best model for {dataset.compound!r}: {best.model.value} "
        f"({label}={criterion(best):.2f}, R²={best.r_squared:.4f}, "
        f"n={best.n_points})."
    )
    if not use_aicc:
        message += (
            " AICc is undefined at this sample size, so models were ranked by "
            "AIC; treat the choice as provisional."
        )
    if ambiguous:
        message += (
            f" Ambiguous: {fits[1].model.value} is within "
            f"Δ{label}={delta:.2f} (< 2); prefer the simpler "
            "model unless there is a mechanistic reason for asymmetry."
        )
    return ModelComparison(fits=fits, best=best, ambiguous=ambiguous, message=message)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _bootstrap_half_max(
    dataset: DoseResponseDataset,
    model: DoseResponseModel,
    direction: CurveDirection,
    *,
    confidence: float,
    n_bootstrap: int,
    seed: int,
    allow_extrapolation: bool,
) -> tuple[tuple[float, float] | None, list[str]]:
    """Resample the experimental design, refit, and return a percentile interval.

    Repeated biological-replicate identifiers are resampled as whole
    trajectories.  Otherwise observations are resampled within concentration
    strata so every bootstrap dataset preserves the tested dose levels.

    Returns
        ``(interval, warnings)`` where *interval* is ``None`` when too few
        bootstrap replicates converged to be trustworthy.
    """
    warnings: list[str] = []
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    finite = np.isfinite(concentration) & np.isfinite(response)
    concentration = concentration[finite]
    response = response[finite]
    groups: np.ndarray | None = None
    if (
        dataset.replicate_group is not None
        and len(dataset.replicate_group) == finite.size
    ):
        groups = np.asarray(dataset.replicate_group, dtype=object)[finite]
    group_labels = list(dict.fromkeys(groups.tolist())) if groups is not None else []
    cluster_replicates = bool(group_labels) and any(
        np.unique(concentration[groups == label]).size > 1 for label in group_labels
    )
    generator = np.random.default_rng(seed)

    estimates: list[float] = []
    for _ in range(int(n_bootstrap)):
        resampled_groups: tuple[str, ...] | None = None
        if cluster_replicates:
            chosen = generator.choice(group_labels, len(group_labels), replace=True)
            parts: list[np.ndarray] = []
            labels: list[str] = []
            for draw, label in enumerate(chosen):
                part = np.flatnonzero(groups == label)
                parts.append(part)
                labels.extend([f"{label}#{draw + 1}"] * part.size)
            index = np.concatenate(parts)
            resampled_groups = tuple(labels)
        else:
            parts = []
            for dose in np.unique(concentration):
                members = np.flatnonzero(concentration == dose)
                parts.append(generator.choice(members, members.size, replace=True))
            index = np.concatenate(parts)
            if groups is not None:
                resampled_groups = tuple(str(value) for value in groups[index])
        resampled = replace(
            dataset,
            concentration=concentration[index],
            response=response[index],
            replicate_group=resampled_groups,
        )
        if np.unique(resampled.concentration).size < 3:
            continue
        try:
            fit = fit_dose_response_curve(
                resampled,
                model,
                direction=direction,
                allow_extrapolation=allow_extrapolation,
                min_points=4,
            )
        except ValueError:
            continue
        estimates.append(fit.half_max_log_concentration)

    required = max(20, int(0.5 * n_bootstrap))
    if len(estimates) < required:
        warnings.append(
            f"Only {len(estimates)} of {n_bootstrap} bootstrap replicates "
            "converged; the bootstrap interval was discarded. Report the "
            "asymptotic interval instead and treat the data as sparse."
        )
        return (None, warnings)

    alpha = 0.5 * (1.0 - confidence)
    lower = float(np.quantile(estimates, alpha))
    upper = float(np.quantile(estimates, 1.0 - alpha))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        warnings.append(
            "The bootstrap distribution of the half-maximal concentration "
            "is degenerate; the interval was discarded."
        )
        return (None, warnings)
    return ((lower, upper), warnings)


# ---------------------------------------------------------------------------
# High-level potency
# ---------------------------------------------------------------------------


def fit_dose_response(
    dataset: DoseResponseDataset,
    *,
    kind: PotencyKind | None = None,
    models: Iterable[DoseResponseModel] | None = None,
    direction: CurveDirection | None = None,
    ci_method: str = "asymptotic",
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    seed: int = 0,
    allow_extrapolation: bool = True,
    control_fraction: float = 0.5,
    control_response: float | None = None,
    min_points: int = 5,
) -> PotencyResult:
    """Fit a dose-response curve and report IC50 and/or EC50.

    Parameters
        dataset:
            The compound's measurements.
        kind:
            ``IC50`` or ``EC50``.  When omitted it follows the resolved
            curve direction (a falling curve yields an IC50, a rising curve
            an EC50).
        models:
            Candidate models for selection; defaults to 4PL/5PL/Gompertz.
        direction:
            Overrides ``dataset.direction`` when given.
        ci_method:
            ``"asymptotic"`` (delta method on the covariance matrix) or
            ``"bootstrap"`` (percentile interval from resampled refits).
        confidence:
            Confidence level for every reported interval.
        n_bootstrap:
            Number of bootstrap replicates when ``ci_method="bootstrap"``.
        seed:
            Seed for the bootstrap resampling, so results are reproducible.
        allow_extrapolation:
            Permit the half-maximal concentration to fall outside the
            tested range (flagged via :attr:`PotencyResult.extrapolated`).
        control_fraction:
            Fraction of the control response defining the *absolute*
            potency (default 0.5, i.e. 50 % inhibition).
        control_response:
            Explicit control response.  When omitted it is taken as the
            mean observed response at the lowest tested concentration.
        min_points:
            Minimum number of observations required.

    Returns
        A :class:`PotencyResult` carrying both the relative and the
        absolute potency, the fitted curve, and the model comparison.

    Raises
        ValueError:
            If ``ci_method`` is unknown, the dataset is unusable, or no
            model converges.
    """
    if ci_method not in {"asymptotic", "bootstrap"}:
        raise ValueError(
            f"ci_method must be 'asymptotic' or 'bootstrap', got {ci_method!r}."
        )
    comparison = compare_dose_response_models(
        dataset,
        models,
        direction=direction,
        confidence=confidence,
        allow_extrapolation=allow_extrapolation,
        min_points=min_points,
    )
    best = comparison.best
    assert best is not None  # guaranteed by compare_dose_response_models

    resolved_kind = kind
    if resolved_kind is None:
        resolved_kind = (
            PotencyKind.EC50
            if best.direction is CurveDirection.INCREASING
            else PotencyKind.IC50
        )

    warnings = list(best.warnings)
    half_max_ci = best.half_max_ci
    if ci_method == "bootstrap":
        bootstrap_ci, bootstrap_warnings = _bootstrap_half_max(
            dataset,
            best.model,
            best.direction,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=seed,
            allow_extrapolation=allow_extrapolation,
        )
        warnings.extend(bootstrap_warnings)
        if bootstrap_ci is not None:
            half_max_ci = bootstrap_ci

    concentration = np.asarray(dataset.concentration, dtype=float)
    concentration = concentration[np.isfinite(concentration)]
    tested_low = float(np.min(concentration))
    tested_high = float(np.max(concentration))
    relative_value = float(10.0**best.half_max_log_concentration)
    extrapolated = not (tested_low <= relative_value <= tested_high)
    if extrapolated:
        warnings.append(
            f"The relative {resolved_kind.value} ({relative_value:.4g}) falls "
            f"outside the tested concentration range "
            f"({tested_low:.4g} .. {tested_high:.4g}); the estimate is an "
            "extrapolation and should be re-measured with a wider range."
        )

    relative_ci = _linear_concentration_interval(half_max_ci)
    if half_max_ci is not None and relative_ci is None:
        warnings.append(
            "The confidence interval is numerically unbounded in linear "
            "concentration space and has been omitted; add measurements on "
            "both response plateaus."
        )

    if control_response is None:
        lowest = np.isclose(concentration, tested_low)
        reference_response = float(
            np.mean(np.asarray(dataset.response, dtype=float)[lowest])
        )
        warnings.append(
            "No untreated/vehicle control response was supplied; the mean "
            "response at the lowest tested concentration was used instead."
        )
    else:
        reference_response = float(control_response)
    target = control_fraction * reference_response
    absolute_value: float | None = None
    absolute_ci: tuple[float, float] | None = None
    try:
        absolute_log = solve_log_concentration_for_response(
            best.model, best.parameters, target
        )
    except (ValueError, KeyError) as exc:
        warnings.append(f"Absolute {resolved_kind.value} is not reported: {exc}")
    else:
        absolute_value = float(10.0**absolute_log)
        if half_max_ci is not None and best.half_max_ci is not None:
            # Re-map the delta-method interval that was computed for the
            # half-maximal response onto the requested response level.  The
            # transformation is exact for the symmetric models and a
            # documented first-order approximation otherwise.
            if best.model in (
                DoseResponseModel.THREE_PL,
                DoseResponseModel.FOUR_PL,
            ):
                shift = absolute_log - best.half_max_log_concentration
                absolute_ci = _linear_concentration_interval(
                    (half_max_ci[0] + shift, half_max_ci[1] + shift)
                )
                if absolute_ci is None:
                    warnings.append(
                        f"The absolute {resolved_kind.value} confidence "
                        "interval is numerically unbounded and has been omitted."
                    )
            else:
                warnings.append(
                    "The confidence interval of the absolute "
                    f"{resolved_kind.value} is a first-order approximation "
                    "because the fitted model is asymmetric."
                )
        if not (tested_low <= absolute_value <= tested_high):
            warnings.append(
                f"The absolute {resolved_kind.value} ({absolute_value:.4g}) "
                "falls outside the tested concentration range."
            )

    fold_range = float(np.log10(tested_high / tested_low) if tested_low > 0 else 0.0)
    window = abs(best.top - best.bottom)
    message = (
        f"{resolved_kind.value} for {dataset.compound!r}: relative={relative_value:.4g}"
    )
    if absolute_value is not None:
        message += f", absolute({control_fraction:.0%} of control)={absolute_value:.4g}"
    message += (
        f" ({best.model.value}, R²={best.r_squared:.4f}, "
        f"n={best.n_points}, range={fold_range:.1f} log10, "
        f"window={window:.4g})."
    )
    if extrapolated:
        message += " NOTE: extrapolated beyond the tested range."

    return PotencyResult(
        kind=resolved_kind,
        compound=dataset.compound,
        best_fit=best,
        comparison=comparison,
        relative_value=relative_value,
        relative_ci=relative_ci,
        absolute_value=absolute_value,
        absolute_ci=absolute_ci,
        control_response=reference_response,
        control_fraction=control_fraction,
        ci_method=ci_method,
        confidence=confidence,
        extrapolated=extrapolated,
        fold_range_covered=fold_range,
        warnings=warnings,
        message=message,
    )


def fit_dose_response_batch(
    datasets: Iterable[DoseResponseDataset], **kwargs
) -> list[PotencyResult]:
    """Fit several compounds, skipping the ones that cannot be fitted.

    Compounds that raise :class:`ValueError` are represented by a synthetic
    result whose ``message`` records the failure, so a batch run never
    aborts on one bad curve.

    Parameters
        datasets:
            Datasets to fit, typically one per compound.
        **kwargs:
            Forwarded verbatim to :func:`fit_dose_response`.

    Returns
        One :class:`PotencyResult` per input, in input order.
    """
    results: list[PotencyResult] = []
    for dataset in datasets:
        try:
            results.append(fit_dose_response(dataset, **kwargs))
        except ValueError as exc:
            results.append(_failed_result(dataset, exc))
    return results


def _failed_result(dataset: DoseResponseDataset, exc: Exception) -> PotencyResult:
    """Build an empty placeholder result describing why fitting failed."""
    placeholder = DoseResponseFit(
        compound=dataset.compound,
        model=DoseResponseModel.FOUR_PL,
        direction=CurveDirection.DECREASING,
        parameters={},
        parameter_std={},
        parameter_ci={},
        half_max_log_concentration=float("nan"),
        half_max_ci=None,
        hill_slope=float("nan"),
        bottom=float("nan"),
        top=float("nan"),
        n_points=int(np.size(dataset.concentration)),
        k_params=0,
        r_squared=float("nan"),
        r_squared_adjusted=float("nan"),
        rmse=float("nan"),
        aic=float("inf"),
        aicc=float("inf"),
        bic=float("inf"),
        converged=False,
        warnings=[str(exc)],
    )
    return PotencyResult(
        kind=PotencyKind.IC50,
        compound=dataset.compound,
        best_fit=placeholder,
        comparison=ModelComparison(fits=[], best=placeholder, ambiguous=False),
        relative_value=float("nan"),
        relative_ci=None,
        absolute_value=None,
        absolute_ci=None,
        control_response=None,
        control_fraction=0.5,
        ci_method="asymptotic",
        confidence=0.95,
        extrapolated=False,
        fold_range_covered=float("nan"),
        warnings=[str(exc)],
        message=f"Potency could not be determined for {dataset.compound!r}: {exc}",
    )


def relative_potency(
    result_a: PotencyResult,
    result_b: PotencyResult,
    *,
    quantity: str = "relative",
    confidence: float | None = None,
) -> PotencyComparison:
    """Compare the potency of two compounds.

    Parameters
        result_a:
            Potency of the numerator compound.
        result_b:
            Potency of the denominator compound.
        quantity:
            ``"relative"`` (default) or ``"absolute"``.
        confidence:
            Compatibility check for the already-computed intervals. It must
            equal the confidence level used to fit both inputs; intervals
            cannot be relabelled without refitting.

    Returns
        A :class:`PotencyComparison` with the ratio and its interval.

    Raises
        ValueError:
            If ``quantity`` is unknown, the two results report different
            potency kinds, or either value is unavailable.
    """
    if quantity not in {"relative", "absolute"}:
        raise ValueError(
            f"quantity must be 'relative' or 'absolute', got {quantity!r}."
        )
    if result_a.kind is not result_b.kind:
        raise ValueError(
            f"Cannot compare {result_a.kind.value} with {result_b.kind.value}."
        )
    if quantity == "relative":
        value_a, value_b = result_a.relative_value, result_b.relative_value
        ci_a, ci_b = result_a.relative_ci, result_b.relative_ci
    else:
        if result_a.absolute_value is None or result_b.absolute_value is None:
            raise ValueError(
                "Absolute potency is unavailable for at least one compound."
            )
        value_a, value_b = result_a.absolute_value, result_b.absolute_value
        ci_a, ci_b = result_a.absolute_ci, result_b.absolute_ci

    if not math.isclose(result_a.confidence, result_b.confidence):
        raise ValueError(
            "Relative potency requires both results to use the same confidence "
            "level; refit them with matching settings."
        )
    level = result_a.confidence
    if confidence is not None and not math.isclose(confidence, level):
        raise ValueError(
            "confidence cannot relabel existing intervals; refit both results "
            f"at confidence={confidence}."
        )
    if not (math.isfinite(value_a) and math.isfinite(value_b)) or value_b == 0.0:
        raise ValueError("Both potency values must be finite and non-zero.")

    ratio = value_a / value_b
    interval: tuple[float, float] | None = None
    if ci_a is not None and ci_b is not None:
        # Combine the two independent log-scale half-spans in quadrature.
        # Opposite sides pair because A/B decreases with A and increases as B
        # decreases.  This retains asymmetry from bootstrap intervals.
        log_ratio = math.log10(ratio)
        log_a = math.log10(value_a)
        log_b = math.log10(value_b)
        lower_span = math.hypot(
            log_a - math.log10(ci_a[0]),
            math.log10(ci_b[1]) - log_b,
        )
        upper_span = math.hypot(
            math.log10(ci_a[1]) - log_a,
            log_b - math.log10(ci_b[0]),
        )
        interval = (
            float(10.0 ** (log_ratio - lower_span)),
            float(10.0 ** (log_ratio + upper_span)),
        )

    fold = max(ratio, 1.0 / ratio)
    direction = "more" if ratio < 1.0 else "less"
    message = (
        f"{result_a.compound!r} is {fold:.2f}-fold {direction} potent than "
        f"{result_b.compound!r} ({result_a.kind.value}, {quantity})."
    )
    return PotencyComparison(
        kind=result_a.kind,
        compound_a=result_a.compound,
        compound_b=result_b.compound,
        ratio=float(ratio),
        ci=interval,
        fold_difference=float(fold),
        confidence=level,
        message=message,
    )
