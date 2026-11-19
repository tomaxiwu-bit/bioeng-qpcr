"""Plotting functions for molecular docking results.

Provides:

* :func:`plot_pose_ranking` -- total score per pose, with each scoring term
  stacked so a pose's ranking can be attributed to the physics rather than
  taken on trust.
* :func:`plot_score_terms` -- the energy decomposition of a single pose.

Scores are signed potentials, so terms can be positive (repulsive) as well as
negative (attractive); the stacked bars therefore keep the sign of every term
instead of taking absolute values, which would be meaningless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bioeng_qpcr.plotting.style import PlotStyle, apply_style, create_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from bioeng_qpcr.docking.scoring import ScoreBreakdown
    from bioeng_qpcr.docking.search import Pose

__all__ = ["plot_pose_ranking", "plot_score_terms"]

#: Contribution names in stacking order, with the breakdown attribute names.
_TERMS: tuple[tuple[str, str], ...] = (
    ("van der Waals", "vdw"),
    ("electrostatic", "electrostatic"),
    ("hydrogen bond", "hbond"),
    ("hydrophobic", "hydrophobic"),
    ("desolvation", "desolvation"),
)


def plot_pose_ranking(
    poses: list[Pose] | tuple[Pose, ...],
    *,
    style: PlotStyle | None = None,
    top_n: int | None = None,
) -> tuple[Figure, Axes]:
    """Render a stacked breakdown of the scoring terms for each pose.

    Parameters
        poses:
            Poses to plot, best-first.
        style:
            Optional shared theme.
        top_n:
            Limit the number of poses drawn (default: all of them).

    Returns
        ``(figure, axes)`` tuple.

    Raises
        ValueError:
            If *poses* is empty.
    """
    selected = list(poses)[:top_n] if top_n else list(poses)
    if not selected:
        raise ValueError("At least one pose is required to draw a ranking plot.")

    labels = [f"pose {index + 1}" for index in range(len(selected))]
    positions = np.arange(len(selected))
    figure, ax = create_figure(style)
    left_negative = np.zeros(len(selected))
    left_positive = np.zeros(len(selected))
    for term_label, attribute in _TERMS:
        values = np.array(
            [float(getattr(pose.breakdown, attribute)) for pose in selected]
        )
        negative = np.minimum(values, 0.0)
        positive = np.maximum(values, 0.0)
        if np.any(negative):
            ax.barh(
                positions,
                negative,
                left=left_negative,
                height=0.6,
                label=f"{term_label} (<0)",
            )
            left_negative = left_negative + negative
        if np.any(positive):
            ax.barh(
                positions,
                positive,
                left=left_positive,
                height=0.6,
                label=f"{term_label} (>0)",
            )
            left_positive = left_positive + positive

    totals = [pose.score for pose in selected]
    for position, total in zip(positions, totals, strict=True):
        ax.annotate(
            f"{total:.2f}",
            xy=(total, position),
            xytext=(6 if total >= 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if total >= 0 else "right",
            fontsize=8,
        )

    ax.set_yticks(positions, labels)
    ax.margins(x=0.15)
    ax.set_xlabel("Empirical score contribution (arbitrary units)")
    ax.set_title("Pose ranking and score components")
    ax.axvline(0.0, linewidth=0.8, color="#6e6d68")
    ax.invert_yaxis()
    apply_style(ax, style)
    legend = ax.get_legend()
    if legend is not None:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.12),
            ncol=2,
            fontsize=8,
            frameon=False,
            borderaxespad=0,
        )
    figure.tight_layout()
    return figure, ax


def plot_score_terms(
    breakdown: ScoreBreakdown, *, style: PlotStyle | None = None
) -> tuple[Figure, Axes]:
    """Render the per-term decomposition of a single pose.

    Parameters
        breakdown:
            Breakdown returned with a pose.
        style:
            Optional shared theme.

    Returns
        ``(figure, axes)`` tuple.
    """
    labels = [label for label, _ in _TERMS]
    values = [float(getattr(breakdown, attribute)) for _, attribute in _TERMS]

    figure, ax = create_figure(style)
    positions = np.arange(len(labels))
    ax.barh(positions, values, height=0.6)
    ax.axvline(0.0, linewidth=0.8, color="#6e6d68")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Empirical score contribution (arbitrary units)")
    ax.set_title(f"Score breakdown (total = {breakdown.total:.3f})")
    ax.invert_yaxis()
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax
