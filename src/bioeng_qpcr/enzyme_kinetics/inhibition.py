"""Enzyme inhibition type classification.

Classifies the inhibition mechanism (competitive, non-competitive,
uncompetitive, mixed, or none) from a set of Michaelis-Menten curves
measured at different inhibitor concentrations.

Classification logic
--------------------
For each inhibitor concentration ``[I]``, a Michaelis-Menten fit yields
apparent parameters ``(Km_app, Vmax_app)``.  The trends of these
parameters with increasing ``[I]`` identify the mechanism:

============  ============  ============  ============================
Type          Km trend      Vmax trend    Lineweaver-Burk
============  ============  ============  ============================
Competitive   increases     unchanged     lines intersect on y-axis
Non-comp.     unchanged     decreases     lines intersect on x-axis
Uncompet.     decreases     decreases     parallel lines
Mixed         changes       changes       lines intersect in Q-II
None          unchanged     unchanged     lines overlap
============  ============  ============  ============================

The coefficient of variation (CV) of Km_app and Vmax_app across [I]
values is used as the quantitative threshold (default 5%).

:func:`compare_inhibition_models` provides the rigorous alternative:
it globally fits the full rate equation of every mechanism to all
datasets simultaneously and ranks them by AICc/AIC/BIC with Akaike
weights, replacing the trend heuristic with statistical model selection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum, unique

import numpy as np
from scipy.optimize import curve_fit

from bioeng_qpcr.enzyme_kinetics.michaelis_menten import (
    LineweaverBurkResult,
    MichaelisMentenResult,
    fit_lineweaver_burk,
    fit_michaelis_menten,
)

__all__ = [
    "GlobalInhibitionFit",
    "InhibitionDataset",
    "InhibitionModelComparison",
    "InhibitionResult",
    "InhibitionType",
    "classify_inhibition",
    "compare_inhibition_models",
]


@unique
class InhibitionType(str, Enum):
    """Enumerates the four classical reversible inhibition mechanisms."""

    COMPETITIVE = "competitive"
    """Competitive: inhibitor binds free enzyme; Km increases, Vmax unchanged."""

    NON_COMPETITIVE = "non_competitive"
    """Non-competitive: inhibitor binds E and ES equally.

    Vmax decreases, Km unchanged.
    """

    UNCOMPETITIVE = "uncompetitive"
    """Uncompetitive: inhibitor binds ES only; both Km and Vmax decrease."""

    MIXED = "mixed"
    """Mixed: inhibitor binds E and ES with different affinities; both change."""

    NONE = "none"
    """No detectable inhibition across the tested [I] range."""


@dataclass(frozen=True)
class InhibitionDataset:
    """A single Michaelis-Menten curve at one inhibitor concentration.

    Attributes
        inhibitor_concentration:
            Concentration of the inhibitor (0 for the uninhibited control).
        substrate:
            Array of substrate concentrations.
        velocity:
            Array of measured velocities.
        mm_result:
            Fitted :class:`MichaelisMentenResult` for this dataset.
        lb_result:
            Fitted :class:`LineweaverBurkResult` for this dataset.
    """

    inhibitor_concentration: float
    substrate: np.ndarray
    velocity: np.ndarray
    mm_result: MichaelisMentenResult
    lb_result: LineweaverBurkResult


@dataclass(frozen=True)
class InhibitionResult:
    """Outcome of inhibition type classification.

    Attributes
        inhibition_type:
            The classified :class:`InhibitionType`.
        km_cv:
            Coefficient of variation of Km_app across [I] values.
        vmax_cv:
            Coefficient of variation of Vmax_app across [I] values.
        datasets:
            List of per-[I] :class:`InhibitionDataset` objects.
        confidence:
            Heuristic confidence in the classification (``"high"`` when
            one CV is clearly above threshold and the other clearly below;
            ``"medium"`` for borderline cases).
        message:
            Human-readable explanation of the classification.
    """

    inhibition_type: InhibitionType
    km_cv: float
    vmax_cv: float
    datasets: list[InhibitionDataset] = field(default_factory=list)
    confidence: str = "medium"
    message: str = ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: CV threshold below which a parameter is considered "unchanged".
_CV_THRESHOLD: float = 0.05


def _coefficient_of_variation(values: list[float]) -> float:
    """Compute the coefficient of variation (std / mean).

    Returns 0.0 when all values are identical or the list is empty.
    """
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if mean == 0:
        return 0.0
    return float(np.std(arr, ddof=1) / mean) if len(arr) > 1 else 0.0


def classify_inhibition(
    datasets: list[tuple[float, np.ndarray, np.ndarray]],
) -> InhibitionResult:
    """Classify the inhibition mechanism from multi-[I] kinetics data.

    Parameters
        datasets:
            List of ``(inhibitor_concentration, substrate, velocity)``
            tuples.  Must contain at least 2 datasets (typically the
            uninhibited control plus one or more inhibitor concentrations).

    Returns
        :class:`InhibitionResult` with the classified mechanism, CV
        statistics, per-dataset fit results, and a confidence rating.

    Raises
        ValueError:
            If fewer than 2 datasets are provided, or if any individual
            Michaelis-Menten fit fails.
    """
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 datasets for inhibition classification, "
            f"got {len(datasets)}."
        )

    # Sort by inhibitor concentration so trends are meaningful.
    sorted_datasets = sorted(datasets, key=lambda d: d[0])

    # Fit each dataset.
    fitted: list[InhibitionDataset] = []
    for inhibitor_conc, substrate, velocity in sorted_datasets:
        mm = fit_michaelis_menten(substrate, velocity)
        lb = fit_lineweaver_burk(substrate, velocity)
        fitted.append(
            InhibitionDataset(
                inhibitor_concentration=float(inhibitor_conc),
                substrate=np.asarray(substrate, dtype=float),
                velocity=np.asarray(velocity, dtype=float),
                mm_result=mm,
                lb_result=lb,
            )
        )

    # Extract apparent parameters.
    km_apps = [d.mm_result.km for d in fitted]
    vmax_apps = [d.mm_result.vmax for d in fitted]
    inhibitor_concs = [d.inhibitor_concentration for d in fitted]

    km_cv = _coefficient_of_variation(km_apps)
    vmax_cv = _coefficient_of_variation(vmax_apps)

    km_changes = km_cv > _CV_THRESHOLD
    vmax_changes = vmax_cv > _CV_THRESHOLD
    km_increasing = _is_increasing(km_apps, inhibitor_concs)
    vmax_decreasing = _is_decreasing(vmax_apps, inhibitor_concs)

    # Classify.
    if not km_changes and not vmax_changes:
        inh_type = InhibitionType.NONE
        confidence = "high"
        message = (
            f"No inhibition detected (Km CV={km_cv:.1%}, "
            f"Vmax CV={vmax_cv:.1%})."
        )
    elif km_changes and not vmax_changes and km_increasing:
        inh_type = InhibitionType.COMPETITIVE
        confidence = "high"
        message = (
            f"Competitive inhibition: Km increases with [I] "
            f"(CV={km_cv:.1%}), Vmax unchanged (CV={vmax_cv:.1%})."
        )
    elif not km_changes and vmax_changes and vmax_decreasing:
        inh_type = InhibitionType.NON_COMPETITIVE
        confidence = "high"
        message = (
            f"Non-competitive inhibition: Vmax decreases with [I] "
            f"(CV={vmax_cv:.1%}), Km unchanged (CV={km_cv:.1%})."
        )
    elif km_changes and vmax_changes:
        # Both change — distinguish uncompetitive from mixed.
        # Uncompetitive: both Km and Vmax decrease proportionally
        # (the ratio Km/Vmax stays constant → LB slope unchanged).
        lb_slopes = [d.lb_result.slope for d in fitted]
        lb_slope_cv = _coefficient_of_variation(lb_slopes)

        # Check whether both parameters decrease (not one increasing).
        km_decreasing = _is_decreasing(km_apps, inhibitor_concs)
        vmax_decreasing = _is_decreasing(vmax_apps, inhibitor_concs)

        if lb_slope_cv < _CV_THRESHOLD and km_decreasing and vmax_decreasing:
            inh_type = InhibitionType.UNCOMPETITIVE
            confidence = "medium"
            message = (
                f"Uncompetitive inhibition: both Km and Vmax decrease "
                f"proportionally (LB slope CV={lb_slope_cv:.1%})."
            )
        else:
            inh_type = InhibitionType.MIXED
            confidence = "medium"
            message = (
                f"Mixed inhibition: both Km (CV={km_cv:.1%}) and Vmax "
                f"(CV={vmax_cv:.1%}) change non-proportionally "
                f"(LB slope CV={lb_slope_cv:.1%})."
            )
    else:
        # A substantial change in only one apparent parameter, but in the
        # opposite direction from its classical inhibition mechanism, is not
        # evidence for competitive or non-competitive inhibition.
        inh_type = InhibitionType.MIXED
        confidence = "low"
        message = (
            "Apparent parameter changes do not match a classical inhibition "
            "direction; report as indeterminate/mixed and inspect the raw "
            f"curves (Km CV={km_cv:.1%}, Vmax CV={vmax_cv:.1%})."
        )

    return InhibitionResult(
        inhibition_type=inh_type,
        km_cv=km_cv,
        vmax_cv=vmax_cv,
        datasets=fitted,
        confidence=confidence,
        message=message,
    )


def _is_decreasing(values: list[float], x: list[float]) -> bool:
    """Check whether *values* decrease as *x* increases.

    Uses a simple sign test on the Pearson correlation.  Returns
    ``True`` when the correlation is negative.
    """
    if len(values) < 2:
        return False
    arr_v = np.asarray(values, dtype=float)
    arr_x = np.asarray(x, dtype=float)
    # Pearson correlation.
    vx = arr_x - np.mean(arr_x)
    vy = arr_v - np.mean(arr_v)
    denom = float(np.sqrt(np.sum(vx**2) * np.sum(vy**2)))
    if denom == 0:
        return False
    corr = float(np.sum(vx * vy) / denom)
    return corr < 0


def _is_increasing(values: list[float], x: list[float]) -> bool:
    """Return whether *values* increase with *x* by Pearson trend sign."""
    if len(values) < 2:
        return False
    arr_v = np.asarray(values, dtype=float)
    arr_x = np.asarray(x, dtype=float)
    vx = arr_x - np.mean(arr_x)
    vy = arr_v - np.mean(arr_v)
    denom = float(np.sqrt(np.sum(vx**2) * np.sum(vy**2)))
    if denom == 0:
        return False
    corr = float(np.sum(vx * vy) / denom)
    return corr > 0


# ---------------------------------------------------------------------------
# Global multi-model fitting with AICc/AIC/BIC model selection
# ---------------------------------------------------------------------------


def _rate_no_inhibition(substrate, inhibitor, vmax, km):
    """Plain Michaelis-Menten: inhibitor has no effect."""
    return vmax * substrate / (km + substrate)


def _rate_competitive(substrate, inhibitor, vmax, km, ki):
    """Competitive: inhibitor binds free enzyme only."""
    return vmax * substrate / (km * (1.0 + inhibitor / ki) + substrate)


def _rate_noncompetitive(substrate, inhibitor, vmax, km, ki):
    """Pure non-competitive: inhibitor binds E and ES with equal affinity."""
    return vmax * substrate / ((km + substrate) * (1.0 + inhibitor / ki))


def _rate_uncompetitive(substrate, inhibitor, vmax, km, ki):
    """Uncompetitive: inhibitor binds the ES complex only."""
    return vmax * substrate / (km + substrate * (1.0 + inhibitor / ki))


def _rate_mixed(substrate, inhibitor, vmax, km, kic, kiu):
    """Mixed: distinct affinities for free enzyme (Kic) and ES (Kiu)."""
    return vmax * substrate / (
        km * (1.0 + inhibitor / kic) + substrate * (1.0 + inhibitor / kiu)
    )


@dataclass(frozen=True)
class GlobalInhibitionFit:
    """Global fit of one inhibition model to all datasets jointly.

    Attributes
        inhibition_type:
            Mechanism this fit assumed.
        parameters:
            Fitted parameter values by name (``vmax``, ``km``, ``ki`` —
            or ``kic``/``kiu`` for the mixed model).
        parameter_std:
            Standard error of each parameter from the covariance matrix.
            Entries are ``nan`` when the covariance could not be estimated.
        rss:
            Residual sum of squares across all pooled points.
        n_points:
            Number of pooled observations.
        k_params:
            Number of fitted parameters (excluding the implicit noise term).
        aic:
            Akaike information criterion (smaller is better).
        aicc:
            Small-sample-corrected AIC; ``inf`` when not estimable.
        bic:
            Bayesian information criterion (smaller is better).
        delta_aicc:
            Difference between this model's AICc and the best model's AICc.
        akaike_weight:
            Normalized Akaike weight over the candidate set (sums to 1).
        converged:
            Whether the nonlinear fit converged.
    """

    inhibition_type: InhibitionType
    parameters: dict[str, float]
    parameter_std: dict[str, float]
    rss: float
    n_points: int
    k_params: int
    aic: float
    aicc: float
    bic: float
    delta_aicc: float
    akaike_weight: float
    converged: bool


@dataclass(frozen=True)
class InhibitionModelComparison:
    """Outcome of global inhibition model selection.

    Attributes
        models:
            All candidate fits sorted by AICc (best first).
        best_type:
            Mechanism with the lowest AICc.
        best_fit:
            The winning :class:`GlobalInhibitionFit`.
        n_points:
            Total pooled observations.
        message:
            Human-readable summary including model-selection caveats.
    """

    models: list[GlobalInhibitionFit] = field(default_factory=list)
    best_type: InhibitionType = InhibitionType.NONE
    best_fit: GlobalInhibitionFit | None = None
    n_points: int = 0
    message: str = ""


def _stack_datasets(
    datasets: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate (substrate, inhibitor, velocity) columns across datasets."""
    substrates: list[np.ndarray] = []
    inhibitors: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for inhibitor_conc, substrate, velocity in datasets:
        substrate_arr = np.asarray(substrate, dtype=float)
        velocity_arr = np.asarray(velocity, dtype=float)
        if substrate_arr.shape != velocity_arr.shape:
            raise ValueError(
                "substrate and velocity arrays must have the same shape."
            )
        substrates.append(substrate_arr)
        inhibitors.append(np.full_like(substrate_arr, float(inhibitor_conc)))
        velocities.append(velocity_arr)
    return (
        np.concatenate(substrates),
        np.concatenate(inhibitors),
        np.concatenate(velocities),
    )


def _information_criteria(rss: float, n: int, k: int) -> tuple[float, float, float]:
    """Return (AIC, AICc, BIC) for a least-squares fit.

    *k* counts fitted parameters plus one for the implicit noise scale.
    """
    eps = np.finfo(float).tiny
    log_term = n * np.log(max(rss, eps) / n)
    aic = log_term + 2 * k
    bic = log_term + k * np.log(n)
    if n - k - 1 > 0:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = np.inf
    return float(aic), float(aicc), float(bic)


_MODEL_REGISTRY: list[
    tuple[InhibitionType, list[str], Callable[..., np.ndarray]]
] = [
    (InhibitionType.COMPETITIVE, ["vmax", "km", "ki"], _rate_competitive),
    (InhibitionType.NON_COMPETITIVE, ["vmax", "km", "ki"], _rate_noncompetitive),
    (InhibitionType.UNCOMPETITIVE, ["vmax", "km", "ki"], _rate_uncompetitive),
    (InhibitionType.MIXED, ["vmax", "km", "kic", "kiu"], _rate_mixed),
    (InhibitionType.NONE, ["vmax", "km"], _rate_no_inhibition),
]


def _make_model(rate: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Wrap a rate equation into the signature curve_fit expects."""

    def _model(x: np.ndarray, *params: float) -> np.ndarray:
        return rate(x[0], x[1], *params)

    return _model


def compare_inhibition_models(
    datasets: list[tuple[float, np.ndarray, np.ndarray]],
) -> InhibitionModelComparison:
    """Globally fit all inhibition mechanisms and select by AICc.

    Every candidate rate equation (competitive, non-competitive,
    uncompetitive, mixed, and a no-inhibition control) is fitted to the
    pooled data from all inhibitor concentrations at once.  Models are
    ranked by small-sample-corrected AIC (AICc); Akaike weights quantify
    the relative support for each mechanism.  This is the rigorous
    counterpart to the trend-based :func:`classify_inhibition` screen.

    Parameters
        datasets:
            List of ``(inhibitor_concentration, substrate, velocity)``
            tuples.  Requires at least two datasets spanning at least two
            distinct inhibitor concentrations, and at least one positive
            inhibitor concentration.

    Returns
        :class:`InhibitionModelComparison` with per-model fits sorted by
        AICc and a message flagging ambiguous selections (ΔAICc < 2).

    Raises
        ValueError:
            If fewer than 2 datasets are given, the inhibitor
            concentrations span fewer than 2 distinct values, or no
            positive inhibitor concentration is present.
    """
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 datasets for model comparison, "
            f"got {len(datasets)}."
        )

    substrate_all, inhibitor_all, velocity_all = _stack_datasets(datasets)

    distinct_inhibitors = {float(i) for i in np.unique(inhibitor_all)}
    if not any(i > 0 for i in distinct_inhibitors):
        raise ValueError(
            "All inhibitor concentrations are zero; there is no inhibition "
            "to model."
        )
    if len(distinct_inhibitors) < 2:
        raise ValueError(
            "Model comparison needs at least 2 distinct inhibitor "
            f"concentrations, got {sorted(distinct_inhibitors)}."
        )

    n_points = int(velocity_all.size)
    vmax0 = 1.2 * float(np.max(velocity_all))
    km0 = float(np.median(substrate_all))
    positive_inhibitors = inhibitor_all[inhibitor_all > 0]
    ki0 = float(np.max(positive_inhibitors)) if positive_inhibitors.size else 1.0

    fits: list[GlobalInhibitionFit] = []
    for inhibition_type, param_names, rate in _MODEL_REGISTRY:
        p0 = [vmax0, km0] + [ki0] * (len(param_names) - 2)

        _model = _make_model(rate)
        xdata = np.vstack([substrate_all, inhibitor_all])
        try:
            popt, pcov = curve_fit(
                _model,
                xdata,
                velocity_all,
                p0=p0,
                bounds=(np.finfo(float).eps, np.inf),
                maxfev=20000,
            )
            predicted = rate(substrate_all, inhibitor_all, *popt)
            rss = float(np.sum((velocity_all - predicted) ** 2))
            stds = np.sqrt(np.abs(np.diag(pcov)))
            parameter_std = {
                name: (float(s) if np.isfinite(s) else float("nan"))
                for name, s in zip(param_names, stds, strict=True)
            }
            converged = True
        except (RuntimeError, ValueError):
            popt = [float("nan")] * len(param_names)
            parameter_std = {name: float("nan") for name in param_names}
            rss = float("inf")
            converged = False

        k = len(param_names) + 1
        aic, aicc, bic = _information_criteria(rss, n_points, k)
        fits.append(
            GlobalInhibitionFit(
                inhibition_type=inhibition_type,
                parameters={
                    name: float(v)
                    for name, v in zip(param_names, popt, strict=True)
                },
                parameter_std=parameter_std,
                rss=rss,
                n_points=n_points,
                k_params=len(param_names),
                aic=aic,
                aicc=aicc,
                bic=bic,
                delta_aicc=0.0,
                akaike_weight=0.0,
                converged=converged,
            )
        )

    finite_fits = [f for f in fits if np.isfinite(f.aicc)]
    if not finite_fits:
        raise ValueError("No inhibition model converged to a finite solution.")

    best_aicc = min(f.aicc for f in finite_fits)
    weight_denominator = sum(
        np.exp(-(f.aicc - best_aicc) / 2.0) for f in finite_fits
    )
    ranked: list[GlobalInhibitionFit] = []
    for fit in fits:
        if np.isfinite(fit.aicc):
            delta = fit.aicc - best_aicc
            weight = float(np.exp(-delta / 2.0) / weight_denominator)
            ranked.append(replace(fit, delta_aicc=delta, akaike_weight=weight))
        else:
            ranked.append(fit)

    ranked.sort(key=lambda f: (f.aicc, f.aic))

    best = ranked[0]
    runner_up = ranked[1]
    message = (
        f"Best model: {best.inhibition_type.value} "
        f"(AICc={best.aicc:.2f}, Akaike weight={best.akaike_weight:.3f}, "
        f"RSS={best.rss:.4g} over {best.n_points} points)."
    )
    if runner_up is not None and runner_up.delta_aicc < 2.0:
        message += (
            f" Ambiguous: {runner_up.inhibition_type.value} is within "
            f"ΔAICc={runner_up.delta_aicc:.2f} (< 2); treat the mechanism "
            "call as unresolved and collect more data."
        )
    elif best.akaike_weight < 0.7:
        message += (
            f" Weak support (weight < 0.7); runner-up "
            f"{runner_up.inhibition_type.value} has "
            f"ΔAICc={runner_up.delta_aicc:.2f}."
        )

    return InhibitionModelComparison(
        models=ranked,
        best_type=best.inhibition_type,
        best_fit=best,
        n_points=n_points,
        message=message,
    )
