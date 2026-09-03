"""Tests for the amplification-curve plotting helpers.

Verifies that figures/axes are created, the correct number of traces are
drawn, axis labels and data values match the input, error cases raise, the
style system is honoured, and ``save_figure`` writes a file to disk.

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

from bioeng_qpcr.domain.entities import Experiment, RawTrace, Well
from bioeng_qpcr.plotting.amplification import (
    plot_amplification_by_target,
    plot_amplification_curves,
)
from bioeng_qpcr.plotting.style import PlotStyle, apply_style, save_figure

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _sigmoid(
    cycles: np.ndarray, midpoint: float, steepness: float, max_val: float
) -> np.ndarray:
    """Classic S-shaped amplification curve."""
    return max_val / (1.0 + np.exp(-steepness * (cycles - midpoint)))


def _make_trace(
    position: str,
    n_cycles: int = 40,
    midpoint: float = 25.0,
    steepness: float = 0.7,
    max_val: float = 10.0,
) -> RawTrace:
    cycles = np.arange(1, n_cycles + 1, dtype=float)
    fluorescence = _sigmoid(cycles, midpoint, steepness, max_val)
    return RawTrace(position=position, cycles=cycles, fluorescence=fluorescence)


def _experiment_with_traces() -> Experiment:
    """Two samples x two targets, each well carrying a sigmoidal trace."""
    wells = [
        Well(position="A01", sample_name="S1", target_name="GAPDH"),
        Well(position="A02", sample_name="S1", target_name="TARGET"),
        Well(position="A03", sample_name="S2", target_name="GAPDH"),
        Well(position="A04", sample_name="S2", target_name="TARGET"),
    ]
    traces = {
        "A01": _make_trace("A01", midpoint=22.0),
        "A02": _make_trace("A02", midpoint=25.0),
        "A03": _make_trace("A03", midpoint=21.0),
        "A04": _make_trace("A04", midpoint=28.0),
    }
    return Experiment(name="amp_test", wells=wells, traces=traces)


# ---------------------------------------------------------------------------
# plot_amplification_curves
# ---------------------------------------------------------------------------


class TestAmplificationCurves:
    @pytest.fixture
    def experiment(self) -> Experiment:
        return _experiment_with_traces()

    def test_returns_figure_and_axes(self, experiment: Experiment) -> None:
        fig, ax = plot_amplification_curves(experiment)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_one_line_per_trace(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_curves(experiment)
        # One Line2D per trace, nothing else drawn as a line.
        assert len(ax.get_lines()) == len(experiment.traces)

    def test_axis_labels(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_curves(experiment)
        assert ax.get_xlabel() == "Cycle"
        assert ax.get_ylabel() == "Fluorescence (\u0394Rn)"

    def test_data_values_match_traces(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_curves(experiment)
        # The first plotted line corresponds to the first trace (A01) because
        # dict insertion order is preserved and matches well order.
        line = ax.get_lines()[0]
        trace = experiment.traces["A01"]
        np.testing.assert_allclose(line.get_xdata(), trace.cycles)
        np.testing.assert_allclose(line.get_ydata(), trace.fluorescence)

    def test_legend_has_one_entry_per_target(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_curves(experiment)
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert set(labels) == {"GAPDH", "TARGET"}
        # Duplicate labels collapsed: 2 targets, not 4.
        assert len(labels) == 2

    def test_no_traces_raises_value_error(self) -> None:
        exp = Experiment(name="empty", wells=[], traces={})
        with pytest.raises(ValueError, match="no amplification traces"):
            plot_amplification_curves(exp)

    def test_custom_style_applied(self, experiment: Experiment) -> None:
        style = PlotStyle(grid=False, legend=False, font_size=9)
        _, ax = plot_amplification_curves(experiment, style)
        # grid disabled -> no gridlines visible.
        assert not ax.xaxis.get_gridlines()[0].get_visible()
        # legend disabled.
        assert ax.get_legend() is None

    def test_save_figure_creates_file(
        self, experiment: Experiment, tmp_path: Path
    ) -> None:
        fig, _ = plot_amplification_curves(experiment)
        out = tmp_path / "amp.png"
        save_figure(fig, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_amplification_by_target
# ---------------------------------------------------------------------------


class TestAmplificationByTarget:
    @pytest.fixture
    def experiment(self) -> Experiment:
        return _experiment_with_traces()

    def test_returns_figure_and_axes(self, experiment: Experiment) -> None:
        fig, ax = plot_amplification_by_target(experiment, "TARGET")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_only_target_wells_plotted(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_by_target(experiment, "TARGET")
        # Two wells carry the TARGET assay (A02, A04).
        assert len(ax.get_lines()) == 2

    def test_coloured_by_sample(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_by_target(experiment, "TARGET")
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert set(labels) == {"S1", "S2"}

    def test_axis_labels(self, experiment: Experiment) -> None:
        _, ax = plot_amplification_by_target(experiment, "TARGET")
        assert ax.get_xlabel() == "Cycle"
        assert ax.get_ylabel() == "Fluorescence (\u0394Rn)"
        assert "TARGET" in ax.get_title()

    def test_unknown_target_raises(self, experiment: Experiment) -> None:
        with pytest.raises(ValueError, match="No wells found for target"):
            plot_amplification_by_target(experiment, "NOPE")

    def test_no_traces_raises(self) -> None:
        exp = Experiment(name="empty", wells=[], traces={})
        with pytest.raises(ValueError, match="no amplification traces"):
            plot_amplification_by_target(exp, "TARGET")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


class TestStyleHelpers:
    def test_apply_style_sets_grid_and_fonts(self) -> None:
        from bioeng_qpcr.plotting.style import create_figure

        style = PlotStyle(grid=True, grid_alpha=0.5, font_size=13, title_size=16)
        fig, ax = create_figure(style)
        ax.plot([0, 1], [0, 1], label="t")
        ax.set_xlabel("x")
        ax.set_title("title")
        apply_style(ax, style)
        assert ax.xaxis.label.get_fontsize() == 13
        assert ax.title.get_fontsize() == 16
        assert ax.get_legend() is not None
        # cleanup handled by matplotlib; explicitly close to avoid resource leak.
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_apply_style_legend_false_removes_legend(self) -> None:
        from bioeng_qpcr.plotting.style import create_figure

        style = PlotStyle(legend=True)
        fig, ax = create_figure(style)
        ax.plot([0, 1], [0, 1], label="t")
        apply_style(ax, style)
        assert ax.get_legend() is not None
        # Now disable legend -> existing legend removed.
        apply_style(ax, PlotStyle(legend=False))
        assert ax.get_legend() is None
        import matplotlib.pyplot as plt

        plt.close(fig)
