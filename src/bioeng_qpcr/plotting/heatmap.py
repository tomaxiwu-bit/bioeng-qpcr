"""Cq heatmap visualisation for qPCR experiments.

Renders a samples x targets matrix of quantification-cycle (Cq) values.
Low Cq corresponds to high expression and is drawn *bright*; high Cq
corresponds to low expression and is drawn *dark*.  Missing Cq values
(undetermined wells) are masked and annotated as ``"N/A"``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.plotting.style import PlotStyle, apply_style, create_figure

# Light warm grey used for masked (missing) cells so they read as "no data".
_BAD_COLOR = "#e3e0d4"

# Custom terracotta colormap: low Cq (high expression) = bright cream,
# high Cq (low expression) = deep terracotta. Aligned with the Warm
# Editorial Science palette from the Claude Design Library.
_TERRACOTTA_CMAP = LinearSegmentedColormap.from_list(
    "terracotta",
    [("#f8e3d8"), ("#e8a98f"), ("#d97757"), ("#b05730"), ("#4d3528")],
)


def plot_cq_heatmap(
    experiment: Experiment,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of mean Cq values (rows = samples, columns = targets).

    For each sample/target pair the mean Cq of all matching wells is used.
    Pairs with no determined Cq are masked and labelled ``"N/A"``.

    Parameters
        experiment:
            The experiment whose wells define the matrix.
        style:
            Optional :class:`PlotStyle`.

    Returns
        A ``(Figure, Axes)`` tuple.  The axes carries the image and a
        colourbar is attached to the figure.
    """
    s = style if style is not None else PlotStyle()

    # sample_names excludes NTC wells; both lists are sorted for determinism.
    samples = experiment.sample_names
    targets = experiment.target_names
    n_samples = len(samples)
    n_targets = len(targets)

    fig, ax = create_figure(style)

    # Degenerate case: nothing to draw.  Return an empty (but labelled)
    # axes so callers still receive a valid Figure/Axes pair.
    if n_samples == 0 or n_targets == 0:
        ax.set_xlabel("Target")
        ax.set_ylabel("Sample")
        apply_style(ax, style)
        return fig, ax

    # Build the Cq matrix (NaN where no determined Cq is available).
    matrix = np.full((n_samples, n_targets), np.nan)
    for i, sample in enumerate(samples):
        for j, target in enumerate(targets):
            wells = experiment.wells_for(sample, target)
            cqs = [w.cq for w in wells if w.cq is not None]
            if cqs:
                matrix[i, j] = float(np.mean(cqs))

    masked = np.ma.masked_invalid(matrix)

    # Terracotta colormap: low Cq (high expression) = bright cream,
    # high Cq (low expression) = deep terracotta.
    cmap = _TERRACOTTA_CMAP.with_extremes(bad=_BAD_COLOR)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cq")
    cbar.ax.tick_params(labelsize=s.font_size)

    ax.set_xticks(np.arange(n_targets))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_samples))
    ax.set_yticklabels(samples)

    # Annotate every cell with its Cq value (or "N/A" when masked).
    for i in range(n_samples):
        for j in range(n_targets):
            value = matrix[i, j]
            label = "N/A" if np.isnan(value) else f"{value:.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=s.font_size,
            )

    ax.set_xlabel("Target")
    ax.set_ylabel("Sample")
    apply_style(ax, style)
    return fig, ax
