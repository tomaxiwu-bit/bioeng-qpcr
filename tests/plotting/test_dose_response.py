"""Tests for dose-response and potency plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bioeng_qpcr.plotting import (  # noqa: E402
    plot_dose_response,
    plot_dose_response_residuals,
    plot_potency_comparison,
)
from bioeng_qpcr.plotting.style import PlotStyle  # noqa: E402
from bioeng_qpcr.potency import (  # noqa: E402
    DoseResponseDataset,
    fit_dose_response,
    fit_dose_response_batch,
    fit_dose_response_curve,
)

CONCENTRATIONS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


def _broken_result():
    """Return the placeholder ``fit_dose_response_batch`` produces on failure.

    That placeholder is the documented representation of "this compound could
    not be fitted", which is exactly what a plotting call has to tolerate.
    """
    return fit_dose_response_batch(
        [
            DoseResponseDataset(
                compound="broken",
                concentration=np.array([1.0, 2.0, 3.0]),
                response=np.array([1.0, 2.0, 3.0]),
            )
        ]
    )[0]


def _dataset(
    compound: str = "CompoundA",
    log_ic50: float = float(np.log10(3.2)),
    bottom: float = 4.0,
    top: float = 100.0,
    noise: float = 1.0,
    seed: int = 3,
) -> DoseResponseDataset:
    generator = np.random.default_rng(seed)
    responses = []
    for concentration in CONCENTRATIONS:
        log_value = float(np.log10(concentration))
        predicted = bottom + (top - bottom) / (
            1.0 + 10.0 ** ((log_ic50 - log_value) * -1.1)
        )
        responses.append(predicted + float(generator.normal(0.0, noise)))
    return DoseResponseDataset(
        compound=compound,
        concentration=CONCENTRATIONS.copy(),
        response=np.array(responses),
        concentration_unit="uM",
        response_unit="% activity",
    )


class TestPlotDoseResponse:
    def test_returns_figure_and_axes(self):
        dataset = _dataset()
        fit = fit_dose_response_curve(dataset)
        figure, ax = plot_dose_response(fit, dataset)
        try:
            assert ax.get_xscale() == "log"
            assert figure.axes
            assert "IC50" in ax.get_title() or dataset.compound in ax.get_title()
        finally:
            plt.close(figure)

    def test_marks_the_half_maximal_concentration(self):
        dataset = _dataset()
        result = fit_dose_response(dataset)
        figure, ax = plot_dose_response(result.best_fit, dataset)
        try:
            verticals = list(ax.get_lines())
            xs = [
                line.get_xdata()[0]
                for line in verticals
                if len(line.get_xdata()) == 2
            ]
            assert any(
                np.isclose(float(x), result.relative_value, rtol=1e-6) for x in xs
            )
        finally:
            plt.close(figure)

    def test_interval_span_can_be_disabled(self):
        dataset = _dataset()
        fit = fit_dose_response_curve(dataset)
        with_span, ax_with = plot_dose_response(fit, dataset, show_interval=True)
        without_span, ax_without = plot_dose_response(fit, dataset, show_interval=False)
        try:
            assert len(ax_with.patches) > len(ax_without.patches)
        finally:
            plt.close(with_span)
            plt.close(without_span)

    def test_curve_is_only_drawn_over_the_tested_range(self):
        dataset = _dataset()
        fit = fit_dose_response_curve(dataset)
        figure, ax = plot_dose_response(fit, dataset)
        try:
            curve = max(
                (line for line in ax.get_lines()),
                key=lambda line: len(line.get_xdata()),
            )
            xs = np.asarray(curve.get_xdata(), dtype=float)
            assert xs.min() >= CONCENTRATIONS.min() * 0.999
            assert xs.max() <= CONCENTRATIONS.max() * 1.001
        finally:
            plt.close(figure)

    def test_non_positive_concentrations_are_rejected(self):
        dataset = DoseResponseDataset(
            compound="bad",
            concentration=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            response=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        fit = fit_dose_response_curve(
            DoseResponseDataset(
                compound="ok",
                concentration=CONCENTRATIONS.copy(),
                response=np.linspace(100.0, 5.0, CONCENTRATIONS.size),
            )
        )
        with pytest.raises(ValueError, match="strictly positive"):
            plot_dose_response(fit, dataset)

    def test_style_is_honoured(self):
        dataset = _dataset()
        fit = fit_dose_response_curve(dataset)
        figure, ax = plot_dose_response(
            fit, dataset, style=PlotStyle(grid=False, legend=False)
        )
        try:
            assert not any(line.get_visible() for line in ax.get_xgridlines())
            assert ax.get_legend() is None
        finally:
            plt.close(figure)


class TestPlotResiduals:
    def test_returns_a_residual_cloud(self):
        dataset = _dataset(noise=2.0)
        fit = fit_dose_response_curve(dataset)
        figure, ax = plot_dose_response_residuals(fit, dataset)
        try:
            offsets = [
                collection.get_offsets()
                for collection in ax.collections
                if hasattr(collection, "get_offsets")
            ]
            assert any(np.asarray(offsets_array).shape[0] == dataset.response.size
                       for offsets_array in offsets)
        finally:
            plt.close(figure)

    def test_zero_line_is_drawn(self):
        dataset = _dataset(noise=2.0)
        fit = fit_dose_response_curve(dataset)
        figure, ax = plot_dose_response_residuals(fit, dataset)
        try:
            assert any(
                len(line.get_xdata()) == 2 and not np.any(line.get_ydata())
                for line in ax.get_lines()
            )
        finally:
            plt.close(figure)


class TestPlotPotencyComparison:
    def _results(self):
        return [
            fit_dose_response(_dataset(compound="Strong", log_ic50=np.log10(1.0))),
            fit_dose_response(_dataset(compound="Weak", log_ic50=np.log10(100.0))),
        ]

    def test_draws_one_bar_per_compound(self):
        results = self._results()
        figure, ax = plot_potency_comparison(results)
        try:
            assert ax.get_xscale() == "log"
            labels = [tick.get_text() for tick in ax.get_yticklabels()]
            assert labels == ["Strong", "Weak"]
        finally:
            plt.close(figure)

    def test_absolute_quantity_is_supported(self):
        results = self._results()
        figure, ax = plot_potency_comparison(results, quantity="absolute")
        try:
            assert "absolute" in ax.get_xlabel()
        finally:
            plt.close(figure)

    def test_unfittable_results_are_skipped(self):
        good = fit_dose_response(_dataset())
        broken = _broken_result()
        figure, ax = plot_potency_comparison([good, broken])
        try:
            assert [tick.get_text() for tick in ax.get_yticklabels()] == ["CompoundA"]
        finally:
            plt.close(figure)

    def test_no_usable_values_is_an_error(self):
        with pytest.raises(ValueError, match="No compound has a finite potency"):
            plot_potency_comparison([_broken_result()])

    def test_unknown_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="quantity"):
            plot_potency_comparison(self._results(), quantity="derived")
