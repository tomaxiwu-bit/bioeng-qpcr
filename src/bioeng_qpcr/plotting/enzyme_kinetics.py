"""Plotting functions for enzyme kinetics analysis.

Provides:
* :func:`plot_michaelis_menten` — scatter of observed data with fitted
  Michaelis-Menten curve.
* :func:`plot_lineweaver_burk` — double-reciprocal plot with one line
  per inhibitor concentration for inhibition diagnosis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from bioeng_qpcr.enzyme_kinetics.inhibition import InhibitionDataset
    from bioeng_qpcr.enzyme_kinetics.michaelis_menten import MichaelisMentenResult

__all__ = ["plot_lineweaver_burk", "plot_michaelis_menten"]


def plot_michaelis_menten(
    result: MichaelisMentenResult,
    substrate: np.ndarray,
    velocity: np.ndarray,
) -> tuple[Figure, object]:
    """Render a Michaelis-Menten plot (observed data + fitted curve).

    Parameters
        result:
            Fitted :class:`~bioeng_qpcr.enzyme_kinetics.MichaelisMentenResult`.
        substrate:
            Original substrate concentrations.
        velocity:
            Original measured velocities.

    Returns
        ``(figure, axes)`` tuple.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Scatter observed data.
    ax.scatter(substrate, velocity, color="#2563eb", zorder=5, label="Observed")

    # Fitted curve.
    s_smooth = np.linspace(
        float(np.min(substrate)), float(np.max(substrate)) * 1.2, 200
    )
    v_smooth = result.vmax * s_smooth / (result.km + s_smooth)
    ax.plot(
        s_smooth,
        v_smooth,
        color="#dc2626",
        linewidth=2,
        label=f"Fit (Vmax={result.vmax:.2f}, Km={result.km:.2f})",
    )

    # Mark Km on the curve.
    ax.axhline(
        result.vmax / 2,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"½Vmax = {result.vmax / 2:.2f}",
    )
    ax.axvline(result.km, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("[S] (substrate concentration)")
    ax.set_ylabel("V (velocity)")
    ax.set_title("Michaelis-Menten Fit")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    return fig, ax


def plot_lineweaver_burk(
    datasets: list[InhibitionDataset],
) -> tuple[Figure, object]:
    """Render a Lineweaver-Burk (double-reciprocal) plot.

    Each inhibitor concentration produces one line.  The intersection
    pattern of the lines reveals the inhibition mechanism.

    Parameters
        datasets:
            List of :class:`~bioeng_qpcr.enzyme_kinetics.InhibitionDataset`,
            one per inhibitor concentration.

    Returns
        ``(figure, axes)`` tuple.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(datasets)))

    for color, ds in zip(colors, datasets, strict=False):
        inv_s = 1.0 / ds.substrate
        inv_v = 1.0 / ds.velocity
        label = f"[I] = {ds.inhibitor_concentration:g}"
        ax.scatter(inv_s, inv_v, color=color, zorder=5, s=30)
        # Fitted line.
        s_fit = np.linspace(
            float(np.min(inv_s)) * 0.8,
            float(np.max(inv_s)) * 1.2,
            100,
        )
        v_fit = ds.lb_result.slope * s_fit + ds.lb_result.intercept
        ax.plot(s_fit, v_fit, color=color, linewidth=1.5, label=label)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("1/[S]")
    ax.set_ylabel("1/V")
    ax.set_title("Lineweaver-Burk Plot")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    return fig, ax
