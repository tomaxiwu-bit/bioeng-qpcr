"""Plotting functions for fermentation kinetics analysis.

Provides:
* :func:`plot_fermentation_kinetics` — dual y-axis plot showing
  biomass growth and substrate consumption with fitted curves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from bioeng_qpcr.fermentation_kinetics.monod import MonodFitResult

__all__ = ["plot_fermentation_kinetics"]


def plot_fermentation_kinetics(
    result: MonodFitResult,
    time: np.ndarray,
    biomass: np.ndarray,
    substrate: np.ndarray,
) -> tuple[Figure, object]:
    """Render a fermentation kinetics plot with dual y-axis.

    Left axis shows biomass (observed scatter + fitted curve);
    right axis shows substrate (observed scatter + fitted curve).

    Parameters
        result:
            Fitted :class:`~MonodFitResult`.
        time:
            Original time points (h).
        biomass:
            Observed biomass concentrations (g/L).
        substrate:
            Observed substrate concentrations (g/L).

    Returns
        ``(figure, axes)`` tuple where *axes* is the left (biomass) axis.
    """
    from bioeng_qpcr.fermentation_kinetics.monod import (
        solve_batch_fermentation,
    )

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Fitted curves (smooth).
    t_smooth = np.linspace(float(time[0]), float(time[-1]), 200)
    x_fit, s_fit = solve_batch_fermentation(
        t_smooth,
        result.x0,
        result.s0,
        result.mu_max,
        result.ks,
        result.yxs,
    )

    # Biomass (left axis).
    color1 = "#2563eb"
    ax1.scatter(time, biomass, color=color1, zorder=5, label="Biomass (obs)")
    ax1.plot(
        t_smooth,
        x_fit,
        color=color1,
        linewidth=2,
        label=f"Biomass fit (μmax={result.mu_max:.3f} h⁻¹)",
    )
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Biomass (g/L)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Substrate (right axis).
    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.scatter(
        time,
        substrate,
        color=color2,
        zorder=5,
        marker="s",
        label="Substrate (obs)",
    )
    ax2.plot(
        t_smooth,
        s_fit,
        color=color2,
        linewidth=2,
        linestyle="--",
        label=f"Substrate fit (Ks={result.ks:.3f} g/L)",
    )
    ax2.set_ylabel("Substrate (g/L)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Batch Fermentation Kinetics")

    # Combined legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()
    return fig, ax1
