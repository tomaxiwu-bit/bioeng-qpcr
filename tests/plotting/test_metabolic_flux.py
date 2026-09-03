"""Tests for metabolic flux plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

try:
    import cobra  # noqa: F401

    COBRA_AVAILABLE = True
except ImportError:
    COBRA_AVAILABLE = False

skip_if_no_cobra = pytest.mark.skipif(
    not COBRA_AVAILABLE,
    reason="COBRApy not installed; skipping FBA tests.",
)


@pytest.fixture(scope="module")
def textbook_model():
    if not COBRA_AVAILABLE:
        pytest.skip("COBRApy not installed")
    from cobra.io import load_model

    return load_model("textbook")


@skip_if_no_cobra
class TestPlotFluxDistribution:
    def test_returns_figure_and_axes(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.plotting.metabolic_flux import (
            plot_flux_distribution,
        )

        result = run_fba(textbook_model)
        fig, ax = plot_flux_distribution(result)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_title_contains_objective(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.plotting.metabolic_flux import (
            plot_flux_distribution,
        )

        result = run_fba(textbook_model)
        fig, ax = plot_flux_distribution(result)
        title = ax.get_title()
        assert "Flux" in title or "objective" in title.lower()
        plt.close(fig)

    def test_top_n_limits_bars(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.plotting.metabolic_flux import (
            plot_flux_distribution,
        )

        result = run_fba(textbook_model)
        fig, ax = plot_flux_distribution(result, top_n=5)
        # Count the number of bar patches.
        n_bars = len(ax.patches)
        assert n_bars <= 5
        plt.close(fig)

    def test_empty_fluxes(self):
        from bioeng_qpcr.metabolic_flux.fba import FBAResult
        from bioeng_qpcr.plotting.metabolic_flux import (
            plot_flux_distribution,
        )

        result = FBAResult(objective_value=0.0, status="optimal")
        fig, ax = plot_flux_distribution(result)
        assert "empty" in ax.get_title().lower()
        plt.close(fig)
