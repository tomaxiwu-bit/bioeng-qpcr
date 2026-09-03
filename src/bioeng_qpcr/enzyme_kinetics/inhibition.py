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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique

import numpy as np

from bioeng_qpcr.enzyme_kinetics.michaelis_menten import (
    LineweaverBurkResult,
    MichaelisMentenResult,
    fit_lineweaver_burk,
    fit_michaelis_menten,
)

__all__ = [
    "InhibitionType",
    "InhibitionDataset",
    "InhibitionResult",
    "classify_inhibition",
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
