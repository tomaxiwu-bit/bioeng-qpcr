"""Plotting functions for dose-response and potency analysis.

Provides:

* :func:`plot_dose_response` -- the standard log-concentration sigmoid plot
  with the observations, the fitted curve, the half-maximal level and its
  confidence interval marked.
* :func:`plot_dose_response_residuals` -- residuals against log concentration,
  ordered so the runs test described in the validation rules is visible by
  eye rather than only reported as a statistic.
* :func:`plot_potency_comparison` -- a log-scaled potency chart for ranking
  several compounds in one figure.

Every function accepts an optional :class:`~bioeng_qpcr.plotting.PlotStyle`
so the whole suite can be themed consistently.

Axis text is deliberately in English, matching the rest of the plotting
package: matplotlib's bundled DejaVu Sans has no CJK glyphs, so Chinese labels
would render as empty boxes in every saved figure.  The Chinese-facing text
lives in the CLI, the GUI and the web UI, where the font stack handles it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bioeng_qpcr.plotting.style import (
    PlotStyle,
    apply_style,
    create_figure,
)
from bioeng_qpcr.potency.fitting import PotencyResult
from bioeng_qpcr.potency.models import evaluate_model

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from bioeng_qpcr.potency.fitting import DoseResponseDataset, DoseResponseFit

__all__ = [
    "plot_dose_response",
    "plot_dose_response_residuals",
    "plot_potency_comparison",
]


def _kind_label(kind: str | None, fit: DoseResponseFit) -> str:
    """Return the potency label, inferring it from the curve when absent."""
    if kind is not None:
        return kind
    return "EC50" if fit.direction.value == "increasing" else "IC50"


def plot_dose_response(
    fit: DoseResponseFit,
    dataset: DoseResponseDataset,
    *,
    kind: str | None = None,
    style: PlotStyle | None = None,
    show_interval: bool = True,
) -> tuple[Figure, Axes]:
    """Render a fitted dose-response curve with its observations.

    The x-axis stays in the data's own concentration units but is drawn on a
    log10 scale, which is how every dose-response figure is read.  The fitted
    curve is evaluated over the tested range only -- extending it outside
    would draw a confident-looking line through a region where the model is
    pure extrapolation.

    Parameters
        fit:
            Fitted curve.
        dataset:
            The observations the curve was fitted to.
        kind:
            ``"IC50"`` or ``"EC50"`` for the annotation; inferred from the
            curve direction when omitted.
        style:
            Optional shared theme.
        show_interval:
            Shade the confidence interval of the half-maximal concentration.

    Returns
        ``(figure, axes)`` tuple.

    Raises
        ValueError:
            If the dataset holds no strictly positive concentration.
    """
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    if concentration.size == 0 or np.any(concentration <= 0):
        raise ValueError(
            "Dose-response plots need strictly positive concentrations "
            "(the x-axis is logarithmic)."
        )

    figure, ax = create_figure(style)

    tested_low = float(np.min(concentration))
    tested_high = float(np.max(concentration))
    log_grid = np.linspace(
        np.log10(tested_low), np.log10(tested_high), 300
    )
    curve = evaluate_model(fit.model, fit.parameters, log_grid)

    ax.scatter(
        concentration,
        response,
        s=42,
        zorder=5,
        label=f"{dataset.compound} observed",
        edgecolors="white",
        linewidths=0.6,
    )

    label = (
        f"{fit.model.value} fit "
        f"(R²={fit.r_squared:.4f}, n={fit.n_points})"
        if np.isfinite(fit.r_squared)
        else f"{fit.model.value} fit"
    )
    ax.plot(10.0**log_grid, curve, linewidth=2.0, label=label, zorder=4)

    midpoint = 0.5 * (fit.bottom + fit.top)
    ax.axhline(
        midpoint,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=f"half-maximal response = {midpoint:.4g}",
    )
    half_max = fit.half_max_concentration
    ax.axvline(half_max, linestyle=":", linewidth=1.2, alpha=0.75)

    if show_interval and fit.half_max_concentration_ci is not None:
        low, high = fit.half_max_concentration_ci
        if np.isfinite(low) and np.isfinite(high) and low < high:
            ax.axvspan(low, high, alpha=0.12, zorder=0)
            ax.annotate(
                f"{fit.half_max_concentration_ci[0]:.3g} – "
                f"{fit.half_max_concentration_ci[1]:.3g}",
                xy=(np.sqrt(low * high), midpoint),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.85,
            )

    label_kind = _kind_label(kind, fit)
    ax.annotate(
        f"{label_kind} ≈ {half_max:.4g}",
        xy=(half_max, midpoint),
        xytext=(12, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "alpha": 0.8},
        fontsize=9,
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        f"concentration ({dataset.concentration_unit or 'arbitrary units'}, "
        "log scale)"
    )
    ax.set_ylabel(f"response ({dataset.response_unit or 'arbitrary units'})")
    ax.set_title(f"{dataset.compound} dose-response")
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax


def plot_dose_response_residuals(
    fit: DoseResponseFit,
    dataset: DoseResponseDataset,
    *,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Render fitted residuals against log10 concentration.

    A random cloud is what a correct symmetric model looks like.  A visible
    S-shape means the data are asymmetric and the half-maximal position has
    been biased -- the situation the ``POT-009`` runs test reports numerically.

    Parameters
        fit:
            Fitted curve.
        dataset:
            The observations the curve was fitted to.
        style:
            Optional shared theme.

    Returns
        ``(figure, axes)`` tuple.
    """
    concentration = np.asarray(dataset.concentration, dtype=float)
    response = np.asarray(dataset.response, dtype=float)
    log_concentration = np.log10(concentration)
    order = np.argsort(log_concentration)
    predicted = evaluate_model(
        fit.model, fit.parameters, log_concentration[order]
    )
    residuals = response[order] - predicted

    figure, ax = create_figure(style)
    ax.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax.scatter(
        log_concentration[order],
        residuals,
        s=34,
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
    )
    # A smooth guide line makes a systematic trend obvious.
    if residuals.size >= 3:
        smoothed = np.convolve(
            residuals, np.ones(3) / 3, mode="same"
        )
        ax.plot(
            log_concentration[order],
            smoothed,
            linewidth=1.4,
            alpha=0.8,
            label="3-point moving average",
        )
    ax.set_xlabel("log10(concentration)")
    ax.set_ylabel("residual (observed − fitted)")
    ax.set_title(f"{dataset.compound} residual diagnostics ({fit.model.value})")
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax


def plot_potency_comparison(
    results: list[PotencyResult],
    *,
    quantity: str = "relative",
    kind: str | None = None,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Render a log-scaled potency chart comparing several compounds.

    Potencies span orders of magnitude, so the value axis is logarithmic and
    the confidence intervals are drawn on that same scale -- which is also
    why they come out asymmetric, as they mathematically should.

    Parameters
        results:
            Potency results to compare; non-finite entries are skipped.
        quantity:
            ``"relative"`` (default) or ``"absolute"``.
        kind:
            Label for the axis (``"IC50"``/``"EC50"``); taken from the first
            result when omitted.
        style:
            Optional shared theme.

    Returns
        ``(figure, axes)`` tuple.

    Raises
        ValueError:
            If *quantity* is unknown or no result carries a usable value.
    """
    if quantity not in ("relative", "absolute"):
        raise ValueError(
            f"quantity must be 'relative' or 'absolute', got {quantity!r}."
        )

    usable = [result for result in results if result.best_fit.converged]
    if quantity == "absolute":
        usable = [result for result in usable if result.absolute_value is not None]

    labels: list[str] = []
    values: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for result in usable:
        if quantity == "relative":
            value, interval = result.relative_value, result.relative_ci
        else:
            value, interval = result.absolute_value, result.absolute_ci
        if value is None or not np.isfinite(value) or value <= 0:
            continue
        labels.append(result.compound)
        values.append(float(value))
        if interval is None:
            lower_errors.append(0.0)
            upper_errors.append(0.0)
        else:
            lower_errors.append(max(float(value) - interval[0], 0.0))
            upper_errors.append(max(interval[1] - float(value), 0.0))

    if not values:
        raise ValueError(
            "No compound has a finite potency value to plot; check the fit "
            "warnings before drawing a comparison figure."
        )

    label_kind = kind or (usable[0].kind.value if usable else "IC50")
    positions = np.arange(len(values))
    figure, ax = create_figure(style)
    ax.barh(
        positions,
        values,
        xerr=np.vstack([lower_errors, upper_errors]),
        height=0.6,
        error_kw={"linewidth": 1.0, "capsize": 3},
        zorder=3,
    )
    ax.set_yticks(positions, labels)
    ax.set_xscale("log")
    ax.set_xlabel(
        f"{'absolute' if quantity == 'absolute' else 'relative'} {label_kind} "
        "(log scale)"
    )
    ax.set_title(f"{label_kind} comparison ({len(values)} compounds)")
    ax.invert_yaxis()
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax
