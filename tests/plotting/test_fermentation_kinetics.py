"""Tests for fermentation kinetics plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bioeng_qpcr.fermentation_kinetics.monod import fit_monod
from bioeng_qpcr.plotting.fermentation_kinetics import (
    plot_fermentation_kinetics,
)
from tests.fermentation_kinetics.conftest import (
    make_fermentation_data,
)


class TestPlotFermentationKinetics:
    def test_returns_figure_and_axes(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        fig, ax = plot_fermentation_kinetics(result, time, biomass, substrate)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_figure_has_title(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        fig, ax = plot_fermentation_kinetics(result, time, biomass, substrate)
        assert "Fermentation" in ax.get_title()
        plt.close(fig)

    def test_biomass_scatter_present(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        fig, ax = plot_fermentation_kinetics(result, time, biomass, substrate)
        # Check that at least one PathCollection (scatter) exists.
        collections = ax.collections
        assert len(collections) > 0
        plt.close(fig)

    def test_dual_y_axes(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        fig, ax = plot_fermentation_kinetics(result, time, biomass, substrate)
        # The twin axis should exist.
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_legend_present(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        fig, ax = plot_fermentation_kinetics(result, time, biomass, substrate)
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)
