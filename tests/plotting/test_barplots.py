"""Tests for the fold-change bar-plot helpers.

Uses duck-typed result objects (``SimpleNamespace``) that expose either a
``fold_change`` (ΔΔCq-style) or ``ratio`` (Pfaffl-style) attribute.  Verifies
figure/axes creation, bar counts, axis labels, bar heights matching the
input, the no-change reference line, error cases, confidence-interval error
bars, style application and file saving.

All tests run headless via the ``Agg`` backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bioeng_qpcr.plotting.barplots import (
    plot_fold_change,
    plot_fold_change_with_ci,
)
from bioeng_qpcr.plotting.style import PlotStyle, save_figure

# ---------------------------------------------------------------------------
# Synthetic result helpers
# ---------------------------------------------------------------------------


def _ddcq_results() -> list:
    """ΔΔCq-style results: control + up + down regulation."""
    return [
        SimpleNamespace(sample_name="CTRL", fold_change=1.0),
        SimpleNamespace(sample_name="S1", fold_change=2.0),
        SimpleNamespace(sample_name="S2", fold_change=0.5),
    ]


def _pfaffl_results() -> list:
    """Pfaffl-style results exposing ``ratio`` instead of ``fold_change``."""
    return [
        SimpleNamespace(sample_name="S1", ratio=3.0),
        SimpleNamespace(sample_name="S2", ratio=0.25),
    ]


def _bar_heights(ax: Axes) -> list[float]:
    """Return the height (value) of every Rectangle bar on *ax*."""
    return [float(patch.get_height()) for patch in ax.patches]


def _has_horizontal_line_at(ax: Axes, y: float) -> bool:
    """True if any Line2D on *ax* sits at constant height *y*."""
    for line in ax.get_lines():
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size and np.allclose(ydata, y):
            return True
    return False


# ---------------------------------------------------------------------------
# plot_fold_change
# ---------------------------------------------------------------------------


class TestFoldChange:
    @pytest.fixture
    def results(self) -> list:
        return _ddcq_results()

    def test_returns_figure_and_axes(self, results: list) -> None:
        fig, ax = plot_fold_change(results)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_one_bar_per_result(self, results: list) -> None:
        _, ax = plot_fold_change(results)
        assert len(ax.patches) == len(results)

    def test_axis_labels(self, results: list) -> None:
        _, ax = plot_fold_change(results)
        assert ax.get_xlabel() == "Sample"
        assert ax.get_ylabel() == "Fold Change"

    def test_bar_heights_match_values(self, results: list) -> None:
        _, ax = plot_fold_change(results)
        heights = sorted(_bar_heights(ax))
        expected = sorted([1.0, 2.0, 0.5])
        assert heights == pytest.approx(expected)

    def test_reference_line_at_one(self, results: list) -> None:
        _, ax = plot_fold_change(results)
        assert _has_horizontal_line_at(ax, 1.0)

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="results list is empty"):
            plot_fold_change([])

    def test_pfaffl_ratio_supported(self) -> None:
        results = _pfaffl_results()
        _, ax = plot_fold_change(results)
        heights = sorted(_bar_heights(ax))
        assert heights == pytest.approx([0.25, 3.0])

    def test_bars_coloured_by_sign(self, results: list) -> None:
        _, ax = plot_fold_change(results)
        # CTRL (1.0) and S1 (2.0) -> up/sage; S2 (0.5) -> down/warm-red.
        colors = [patch.get_facecolor() for patch in ax.patches]
        # At least one warm-red and at least one sage bar face colour present.
        import matplotlib.colors as mcolors

        hex_colors = {mcolors.to_hex(c) for c in colors}
        assert "#8ca06f" in hex_colors  # up
        assert "#c43a39" in hex_colors  # down

    def test_custom_style_applied(self, results: list) -> None:
        style = PlotStyle(grid=False, legend=False, font_size=10)
        _, ax = plot_fold_change(results, style)
        assert not ax.xaxis.get_gridlines()[0].get_visible()
        assert ax.get_legend() is None

    def test_save_figure_creates_file(self, results: list, tmp_path: Path) -> None:
        fig, _ = plot_fold_change(results)
        out = tmp_path / "fold_change.png"
        save_figure(fig, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_fold_change_with_ci
# ---------------------------------------------------------------------------


class TestFoldChangeWithCi:
    @pytest.fixture
    def results(self) -> list:
        return _ddcq_results()

    @pytest.fixture
    def bounds(self) -> tuple[list[float], list[float]]:
        # Asymmetric CI around each fold change.
        lower = [0.9, 1.6, 0.3]
        upper = [1.1, 2.6, 0.8]
        return lower, upper

    def test_returns_figure_and_axes(self, results: list, bounds: tuple) -> None:
        lower, upper = bounds
        fig, ax = plot_fold_change_with_ci(results, lower, upper)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_one_bar_per_result(self, results: list, bounds: tuple) -> None:
        lower, upper = bounds
        _, ax = plot_fold_change_with_ci(results, lower, upper)
        assert len(ax.patches) == len(results)

    def test_bar_heights_match_values(self, results: list, bounds: tuple) -> None:
        lower, upper = bounds
        _, ax = plot_fold_change_with_ci(results, lower, upper)
        heights = sorted(_bar_heights(ax))
        assert heights == pytest.approx(sorted([1.0, 2.0, 0.5]))

    def test_error_bars_present(self, results: list, bounds: tuple) -> None:
        """The CI version draws more Line2D artists than the plain version."""
        lower, upper = bounds
        _, ax_ci = plot_fold_change_with_ci(results, lower, upper)
        _, ax_plain = plot_fold_change(results)
        # CI adds cap lines on top of the shared reference line.
        assert len(ax_ci.get_lines()) > len(ax_plain.get_lines())

    def test_reference_line_at_one(self, results: list, bounds: tuple) -> None:
        lower, upper = bounds
        _, ax = plot_fold_change_with_ci(results, lower, upper)
        assert _has_horizontal_line_at(ax, 1.0)

    def test_mismatched_lengths_raise(self, results: list, bounds: tuple) -> None:
        lower, _ = bounds
        with pytest.raises(ValueError, match="same length"):
            plot_fold_change_with_ci(results, lower, [1.0])

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="results list is empty"):
            plot_fold_change_with_ci([], [], [])

    def test_save_figure_creates_file(
        self, results: list, bounds: tuple, tmp_path: Path
    ) -> None:
        lower, upper = bounds
        fig, _ = plot_fold_change_with_ci(results, lower, upper)
        out = tmp_path / "fold_change_ci.png"
        save_figure(fig, str(out))
        assert out.exists()
        assert out.stat().st_size > 0
