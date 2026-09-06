"""Composable style/theme system for bioeng-qpcr plots.

This module provides a lightweight, dataclass-based styling layer that sits on
top of matplotlib.  The design is *composable*: every public plotting function
accepts an optional :class:`PlotStyle` so callers can share a single theme
across amplification curves, bar plots and heatmaps without reaching into
matplotlib's rcParams.

Public API
----------
* :class:`PlotStyle` -- the immutable theme descriptor.
* :func:`apply_style` -- apply a theme to an existing :class:`~matplotlib.axes.Axes`.
* :func:`create_figure` -- create a pre-styled ``(Figure, Axes)`` pair.
* :func:`save_figure` -- persist a figure honouring the theme's DPI.
* :func:`styled_plot` -- context manager that creates, styles and cleans up.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Warm editorial palette adapted from the Claude Design Library.
# Terracotta-forward, colour-blind friendly, and consistent with the
# desktop application's "Warm Editorial Science" theme.
_DEFAULT_PALETTE: list[str] = [
    "#d97757",  # terracotta (brand primary)
    "#8ca06f",  # sage green
    "#b05730",  # deep terracotta
    "#9c87f5",  # muted violet
    "#c4933a",  # warm ochre
    "#5b8a99",  # slate teal
]


@dataclass
class PlotStyle:
    """A composable plotting theme.

    Attributes
        figure_size:
            Default figure dimensions in inches ``(width, height)``.
        dpi:
            Resolution used both for on-screen figures and when saving.
        color_palette:
            Ordered list of hex colour strings cycled through by category.
        font_size:
            Base font size for axis labels and tick labels.
        title_size:
            Font size for the axes title.
        grid:
            Whether to draw a background grid.
        grid_alpha:
            Opacity of the grid lines (0-1).
        legend:
            Whether to render a legend when labelled artists are present.
    """

    figure_size: tuple[float, float] = (8, 6)
    dpi: int = 150
    color_palette: list[str] = field(default_factory=lambda: list(_DEFAULT_PALETTE))
    font_size: int = 12
    title_size: int = 14
    grid: bool = True
    grid_alpha: float = 0.3
    legend: bool = True


# A single shared default instance so that ``None`` can be used as a sentinel
# without paying for a fresh allocation on every call.
_DEFAULT_STYLE = PlotStyle()


def _resolve(style: PlotStyle | None) -> PlotStyle:
    """Return *style* or the shared default when *style* is ``None``."""
    return style if style is not None else _DEFAULT_STYLE


def apply_style(ax: Axes, style: PlotStyle | None = None) -> None:
    """Apply a :class:`PlotStyle` to *ax*.

    Configures the grid, tick/label/title font sizes and the legend.  When
    ``style.legend`` is ``True`` and the axes contains labelled artists a
    *de-duplicated* legend is drawn (repeated labels, e.g. several wells
    sharing one target, collapse to a single entry).  When ``style.legend``
    is ``False`` any existing legend is removed.
    """
    s = _resolve(style)

    if s.grid:
        ax.grid(True, alpha=s.grid_alpha)
    else:
        ax.grid(False)

    ax.tick_params(labelsize=s.font_size)
    ax.xaxis.label.set_fontsize(s.font_size)
    ax.yaxis.label.set_fontsize(s.font_size)
    ax.title.set_fontsize(s.title_size)

    handles, labels = ax.get_legend_handles_labels()
    if s.legend and handles:
        # Collapse duplicate labels while preserving first-seen order.
        seen: set[str] = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels, strict=True):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        ax.legend(unique_handles, unique_labels, fontsize=s.font_size)
    elif not s.legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def create_figure(style: PlotStyle | None = None) -> tuple[Figure, Axes]:
    """Create a ``(Figure, Axes)`` pair sized and resolved per *style*."""
    s = _resolve(style)
    fig, ax = plt.subplots(figsize=s.figure_size, dpi=s.dpi)
    return fig, ax


def save_figure(fig: Figure, path: str, style: PlotStyle | None = None) -> None:
    """Save *fig* to *path* using the DPI from *style*.

    The output is cropped tightly to the artists (``bbox_inches="tight"``)
    so legends and labels are never clipped.
    """
    s = _resolve(style)
    fig.savefig(path, dpi=s.dpi, bbox_inches="tight")


@contextmanager
def styled_plot(
    style: PlotStyle | None = None,
) -> Iterator[tuple[Figure, Axes]]:
    """Context manager that yields a styled ``(fig, ax)`` and cleans up.

    Usage::

        with styled_plot(my_style) as (fig, ax):
            ax.plot(x, y, label="trace")
            ax.set_xlabel("Cycle")
        # Style is applied and the figure is closed automatically here.

    The style is applied *after* the ``with`` block body runs so that any
    artists added by the caller (and therefore any legend handles) are taken
    into account.  The figure is always closed, even if the body raises.
    """
    fig, ax = create_figure(style)
    try:
        yield fig, ax
        apply_style(ax, style)
    finally:
        plt.close(fig)
