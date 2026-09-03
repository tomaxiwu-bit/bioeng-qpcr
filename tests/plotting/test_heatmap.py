"""Tests for the Cq-heatmap plotting helper.

Builds an :class:`Experiment` with known Cq values (including one
undetermined well) and verifies the matrix layout, imshow presence, cell
annotations (with ``"N/A"`` for missing data), axis labels, colourbar,
style application and file saving.

All tests run headless via the ``Agg`` backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bioeng_qpcr.domain.entities import Experiment, Well
from bioeng_qpcr.plotting.heatmap import plot_cq_heatmap
from bioeng_qpcr.plotting.style import PlotStyle, save_figure

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _experiment_with_cq() -> Experiment:
    """Three samples x two targets; S2/TARGET is undetermined (Cq=None).

    Expected mean-Cq matrix (rows = samples, cols = targets):

        CTRL   GAPDH=20.00  TARGET=22.00
        S1     GAPDH=19.50  TARGET=18.00
        S2     GAPDH=20.50  TARGET=N/A
    """
    wells = [
        Well(position="A01", sample_name="CTRL", target_name="GAPDH", cq=20.0),
        Well(position="A02", sample_name="CTRL", target_name="TARGET", cq=22.0),
        Well(position="A03", sample_name="S1", target_name="GAPDH", cq=19.5),
        Well(position="A04", sample_name="S1", target_name="TARGET", cq=18.0),
        Well(position="A05", sample_name="S2", target_name="GAPDH", cq=20.5),
        # Undetermined: well exists but Cq is None -> should render as "N/A".
        Well(position="A06", sample_name="S2", target_name="TARGET", cq=None),
    ]
    return Experiment(name="heatmap_test", wells=wells)


def _cell_texts(ax: Axes) -> dict[tuple[int, int], str]:
    """Map (col, row) -> annotation text drawn on *ax*."""
    out: dict[tuple[int, int], str] = {}
    for text in ax.texts:
        x, y = text.get_position()
        out[(round(x), round(y))] = text.get_text()
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCqHeatmap:
    @pytest.fixture
    def experiment(self) -> Experiment:
        return _experiment_with_cq()

    def test_returns_figure_and_axes(self, experiment: Experiment) -> None:
        fig, ax = plot_cq_heatmap(experiment)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_imshow_present(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        assert len(ax.images) == 1

    def test_axis_labels(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        assert ax.get_xlabel() == "Target"
        assert ax.get_ylabel() == "Sample"

    def test_tick_labels_match_matrix(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert xlabels == ["GAPDH", "TARGET"]
        assert ylabels == ["CTRL", "S1", "S2"]

    def test_annotation_count(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        # 3 samples x 2 targets = 6 annotated cells.
        assert len(ax.texts) == 6

    def test_missing_cq_shows_na(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        texts = _cell_texts(ax)
        # S2 is row 2, TARGET is column 1.
        assert texts[(1, 2)] == "N/A"

    def test_data_values_match_input(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        texts = _cell_texts(ax)
        assert texts[(0, 0)] == "20.00"  # CTRL / GAPDH
        assert texts[(1, 0)] == "22.00"  # CTRL / TARGET
        assert texts[(0, 1)] == "19.50"  # S1 / GAPDH
        assert texts[(1, 1)] == "18.00"  # S1 / TARGET
        assert texts[(0, 2)] == "20.50"  # S2 / GAPDH

    def test_image_array_has_nan_for_missing(self, experiment: Experiment) -> None:
        _, ax = plot_cq_heatmap(experiment)
        arr = ax.images[0].get_array()
        # CTRL/GAPDH present, S2/TARGET masked.
        assert float(arr[0, 0]) == pytest.approx(20.0)
        assert np.ma.is_masked(arr[2, 1])

    def test_colorbar_present(self, experiment: Experiment) -> None:
        fig, ax = plot_cq_heatmap(experiment)
        # Main axes plus one colorbar axes.
        assert len(fig.axes) == 2
        # The image carries a reference to its colorbar.
        assert ax.images[0].colorbar is not None

    def test_custom_style_applied(self, experiment: Experiment) -> None:
        style = PlotStyle(grid=False, legend=False, font_size=8)
        _, ax = plot_cq_heatmap(experiment, style)
        assert not ax.xaxis.get_gridlines()[0].get_visible()

    def test_save_figure_creates_file(
        self, experiment: Experiment, tmp_path: Path
    ) -> None:
        fig, _ = plot_cq_heatmap(experiment)
        out = tmp_path / "heatmap.png"
        save_figure(fig, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_experiment_returns_axes_without_image(self) -> None:
        exp = Experiment(name="empty", wells=[], traces={})
        fig, ax = plot_cq_heatmap(exp)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.images) == 0
        assert ax.get_xlabel() == "Target"
        assert ax.get_ylabel() == "Sample"
