"""Plotting module: matplotlib-based composable visualisation for qPCR data.

This module provides lightweight, composable plotting helpers that operate
on the :mod:`bioeng_qpcr.domain` model.  Every public function accepts an
optional :class:`PlotStyle` so a single theme can be shared across all plot
types.

Submodules
----------
* :mod:`bioeng_qpcr.plotting.style` -- composable style/theme system.
* :mod:`bioeng_qpcr.plotting.amplification` -- amplification curve plots.
* :mod:`bioeng_qpcr.plotting.barplots` -- fold-change bar plots.
* :mod:`bioeng_qpcr.plotting.heatmap` -- Cq heatmap.
* :mod:`bioeng_qpcr.plotting.enzyme_kinetics` -- MM and LB plots (Phase 2).
* :mod:`bioeng_qpcr.plotting.fermentation_kinetics` -- growth curves (Phase 3).
* :mod:`bioeng_qpcr.plotting.metabolic_flux` -- flux distribution (Phase 4).
"""

from __future__ import annotations

from bioeng_qpcr.plotting.amplification import (
    plot_amplification_by_target,
    plot_amplification_curves,
)
from bioeng_qpcr.plotting.barplots import (
    plot_fold_change,
    plot_fold_change_with_ci,
)
from bioeng_qpcr.plotting.enzyme_kinetics import (
    plot_lineweaver_burk,
    plot_michaelis_menten,
)
from bioeng_qpcr.plotting.fermentation_kinetics import (
    plot_fermentation_kinetics,
)
from bioeng_qpcr.plotting.heatmap import plot_cq_heatmap
from bioeng_qpcr.plotting.metabolic_flux import plot_flux_distribution
from bioeng_qpcr.plotting.style import (
    PlotStyle,
    apply_style,
    create_figure,
    save_figure,
    styled_plot,
)

__all__ = [
    # style
    "PlotStyle",
    "apply_style",
    "create_figure",
    "save_figure",
    "styled_plot",
    # amplification
    "plot_amplification_curves",
    "plot_amplification_by_target",
    # barplots
    "plot_fold_change",
    "plot_fold_change_with_ci",
    # heatmap
    "plot_cq_heatmap",
    # enzyme kinetics
    "plot_michaelis_menten",
    "plot_lineweaver_burk",
    # fermentation kinetics
    "plot_fermentation_kinetics",
    # metabolic flux
    "plot_flux_distribution",
]
