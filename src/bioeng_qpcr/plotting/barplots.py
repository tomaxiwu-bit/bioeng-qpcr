"""Fold-change bar plots for relative-quantification results.

The functions here are *duck typed*: any result object exposing a
``sample_name`` attribute together with either a ``fold_change``
(:class:`~bioeng_qpcr.calculations.delta_delta_cq.DeltaDeltaCqResult`) or a
``ratio`` (:class:`~bioeng_qpcr.calculations.pfaffl.PfafflResult`) numeric
attribute can be plotted.  This keeps the plotting layer decoupled from the
calculations layer.
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bioeng_qpcr.plotting.style import PlotStyle, apply_style, create_figure

# Semantic colours for up/down regulation.  Green = up-regulated (>=1),
# red = down-regulated (<1).
_UP_COLOR = "#4CAF50"
_DOWN_COLOR = "#E53935"


def _extract_value(result: Any) -> float:
    """Return the quantitative value from a result object.

    Prefers ``fold_change`` (ΔΔCq results) and falls back to ``ratio``
    (Pfaffl results).  Raises :class:`AttributeError` if neither exists.
    """
    value = getattr(result, "fold_change", None)
    if value is None:
        value = getattr(result, "ratio", None)
    if value is None:
        raise AttributeError(
            "Result object must expose a numeric 'fold_change' or 'ratio' attribute."
        )
    return float(value)


def _bar_colors(values: list[float]) -> list[str]:
    """Colour each bar by the sign of the fold change."""
    return [_UP_COLOR if v >= 1.0 else _DOWN_COLOR for v in values]


def _annotate_bars(ax: Axes, bars, values: list[float], font_size: int) -> None:
    """Write the numeric value above/below each bar."""
    for bar, val in zip(bars, values, strict=True):
        height = bar.get_height()
        offset = max(abs(val) * 0.01, 0.05)
        if val >= 1.0:
            y = height + offset
            va = "bottom"
        else:
            y = height - offset
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{val:.2f}",
            ha="center",
            va=va,
            fontsize=font_size,
        )


def plot_fold_change(
    results: list,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Bar plot of fold-change (or Pfaffl ratio) per sample.

    Bars are coloured by direction: green for up-regulation (>= 1), red for
    down-regulation (< 1).  A dashed reference line is drawn at ``y = 1.0``
    (no change) and each bar is annotated with its value.

    Parameters
        results:
            List of result objects (duck typed, see module docstring).
        style:
            Optional :class:`PlotStyle`.

    Returns
        A ``(Figure, Axes)`` tuple.

    Raises
        ValueError:
            If *results* is empty.
    """
    if not results:
        raise ValueError("Cannot plot fold change: results list is empty.")

    s = style if style is not None else PlotStyle()
    fig, ax = create_figure(style)

    samples = [r.sample_name for r in results]
    values = [_extract_value(r) for r in results]
    colors = _bar_colors(values)

    bars = ax.bar(
        samples,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="No change (fold=1)",
    )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Fold Change")
    apply_style(ax, style)
    _annotate_bars(ax, bars, values, s.font_size)
    return fig, ax


def plot_fold_change_with_ci(
    results: list,
    lower: list[float],
    upper: list[float],
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Bar plot of fold change with confidence-interval error bars.

    Identical to :func:`plot_fold_change` but each bar carries an asymmetric
    error bar spanning ``[lower_i, upper_i]``.

    Parameters
        results:
            List of result objects (duck typed).
        lower:
            Lower CI bound for each result, same length as *results*.
        upper:
            Upper CI bound for each result, same length as *results*.
        style:
            Optional :class:`PlotStyle`.

    Returns
        A ``(Figure, Axes)`` tuple.

    Raises
        ValueError:
            If *results* is empty, or the three lists differ in length.
    """
    if not results:
        raise ValueError("Cannot plot fold change: results list is empty.")
    if not (len(results) == len(lower) == len(upper)):
        raise ValueError(
            "results, lower and upper must all have the same length: "
            f"got {len(results)}, {len(lower)}, {len(upper)}."
        )

    s = style if style is not None else PlotStyle()
    fig, ax = create_figure(style)

    samples = [r.sample_name for r in results]
    values = [_extract_value(r) for r in results]
    colors = _bar_colors(values)

    # Asymmetric error: distance below and above the bar top.
    yerr_lower = [v - lo for v, lo in zip(values, lower, strict=True)]
    yerr_upper = [hi - v for v, hi in zip(values, upper, strict=True)]

    bars = ax.bar(
        samples,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        yerr=[yerr_lower, yerr_upper],
        capsize=4,
        error_kw={"elinewidth": 1.0, "ecolor": "black"},
    )
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="No change (fold=1)",
    )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Fold Change")
    apply_style(ax, style)
    _annotate_bars(ax, bars, values, s.font_size)
    return fig, ax
