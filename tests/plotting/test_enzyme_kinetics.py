"""Tests for enzyme kinetics plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bioeng_qpcr.enzyme_kinetics.inhibition import classify_inhibition
from bioeng_qpcr.enzyme_kinetics.michaelis_menten import fit_michaelis_menten
from bioeng_qpcr.plotting.enzyme_kinetics import (
    plot_lineweaver_burk,
    plot_michaelis_menten,
)
from tests.enzyme_kinetics.conftest import (
    make_competitive_inhibition_data,
    make_mm_data,
)


class TestPlotMichaelisMenten:
    def test_returns_figure(self):
        s, v = make_mm_data(vmax=100.0, km=10.0)
        result = fit_michaelis_menten(s, v)
        fig, ax = plot_michaelis_menten(result, s, v)
        assert fig is not None
        assert ax is not None
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_has_scatter(self):
        s, v = make_mm_data(vmax=100.0, km=10.0)
        result = fit_michaelis_menten(s, v)
        fig, ax = plot_michaelis_menten(result, s, v)
        collections = ax.collections
        assert len(collections) > 0
        plt.close(fig)

    def test_title_set(self):
        s, v = make_mm_data()
        result = fit_michaelis_menten(s, v)
        fig, ax = plot_michaelis_menten(result, s, v)
        assert "Michaelis-Menten" in ax.get_title()
        plt.close(fig)


class TestPlotLineweaverBurk:
    def test_returns_figure(self):
        datasets_data = make_competitive_inhibition_data()
        result = classify_inhibition(datasets_data)
        fig, ax = plot_lineweaver_burk(result.datasets)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_multiple_lines(self):
        datasets_data = make_competitive_inhibition_data()
        result = classify_inhibition(datasets_data)
        fig, ax = plot_lineweaver_burk(result.datasets)
        # One line per dataset (4 datasets → 4 lines + axes lines)
        lines = ax.get_lines()
        assert len(lines) >= 4
        plt.close(fig)

    def test_title_set(self):
        datasets_data = make_competitive_inhibition_data()
        result = classify_inhibition(datasets_data)
        fig, ax = plot_lineweaver_burk(result.datasets)
        assert "Lineweaver-Burk" in ax.get_title()
        plt.close(fig)
