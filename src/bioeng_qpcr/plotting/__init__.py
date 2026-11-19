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
* :mod:`bioeng_qpcr.plotting.dose_response` -- IC50/EC50 curves (Phase 5).
* :mod:`bioeng_qpcr.plotting.docking` -- pose ranking and energy terms.
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
from bioeng_qpcr.plotting.docking import (
    plot_pose_ranking,
    plot_score_terms,
)
from bioeng_qpcr.plotting.dose_response import (
    plot_dose_response,
    plot_dose_response_residuals,
    plot_potency_comparison,
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
from bioeng_qpcr.plotting.sequence import (
    plot_alignment_coverage,
    plot_read_qualities,
    plot_variant_map,
)
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
    "plot_amplification_by_target",
    # amplification
    "plot_amplification_curves",
    "plot_alignment_coverage",
    # heatmap
    "plot_cq_heatmap",
    # dose_response
    "plot_dose_response",
    "plot_dose_response_residuals",
    # fermentation kinetics
    "plot_fermentation_kinetics",
    # metabolic flux
    "plot_flux_distribution",
    # barplots
    "plot_fold_change",
    "plot_fold_change_with_ci",
    "plot_lineweaver_burk",
    # enzyme kinetics
    "plot_michaelis_menten",
    # docking
    "plot_pose_ranking",
    "plot_potency_comparison",
    "plot_read_qualities",
    "plot_variant_map",
    "plot_score_terms",
    "save_figure",
    "styled_plot",
]
