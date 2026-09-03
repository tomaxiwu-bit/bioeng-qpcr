"""Plotting functions for metabolic flux analysis.

Provides:
* :func:`plot_flux_distribution` — horizontal bar chart of the top-N
  active reactions by absolute flux magnitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from bioeng_qpcr.metabolic_flux.fba import FBAResult

__all__ = ["plot_flux_distribution"]


def plot_flux_distribution(
    result: FBAResult,
    *,
    top_n: int = 20,
) -> tuple[Figure, object]:
    """Render a flux distribution bar chart.

    Plots the top-N reactions by absolute flux magnitude as a
    horizontal bar chart.  Positive fluxes (forward direction) are
    coloured blue; negative fluxes (reverse direction) are red.

    Parameters
        result:
            A :class:`~FBAResult` from :func:`run_fba`.
        top_n:
            Maximum number of reactions to display.

    Returns
        ``(figure, axes)`` tuple.
    """
    # Sort by absolute flux and take top-N.
    sorted_fluxes = sorted(
        result.fluxes.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    if not sorted_fluxes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No active reactions", ha="center", va="center")
        ax.set_title("Flux Distribution (empty)")
        return fig, ax

    # Reverse for top-to-bottom display.
    labels = [r[0] for r in reversed(sorted_fluxes)]
    values = [r[1] for r in reversed(sorted_fluxes)]
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.35)))
    ax.barh(range(len(labels)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Flux (mmol/gDW/h)")
    ax.set_title(
        f"Top {len(labels)} Fluxes "
        f"(objective={result.objective_value:.4f})"
    )
    ax.axvline(0, color="#666666", linewidth=0.8)
    fig.tight_layout()
    return fig, ax
