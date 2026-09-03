"""Tests for the bioeng-qpcr ipywidgets QPCRAnalyzer."""

from __future__ import annotations

import pytest

ipywidgets = pytest.importorskip("ipywidgets")

from bioeng_qpcr.widgets import QPCRAnalyzer  # noqa: E402

from .conftest import EXAMPLE_CSV  # noqa: E402


class TestQPCRAnalyzerInit:
    def test_instantiation(self):
        analyzer = QPCRAnalyzer()
        assert analyzer is not None
        assert analyzer.experiment is None

    def test_has_ui_components(self):
        analyzer = QPCRAnalyzer()
        assert analyzer._file_text is not None
        assert analyzer._target_dropdown is not None
        assert analyzer._reference_select is not None
        assert analyzer._method_dropdown is not None
        assert analyzer._control_text is not None
        assert analyzer._load_btn is not None
        assert analyzer._validate_btn is not None
        assert analyzer._analyze_btn is not None
        assert analyzer._plot_btn is not None
        assert analyzer._output is not None

    def test_method_options(self):
        assert len(QPCRAnalyzer._METHOD_OPTIONS) == 3
        values = [v for _, v in QPCRAnalyzer._METHOD_OPTIONS]
        assert "normalized_quantity" in values
        assert "pfaffl" in values
        assert "delta_delta_cq" in values

    def test_control_hidden_by_default(self):
        analyzer = QPCRAnalyzer()
        assert analyzer._control_box.layout.display == "none"

    def test_control_visible_for_pfaffl(self):
        analyzer = QPCRAnalyzer()
        analyzer._method_dropdown.value = "pfaffl"
        assert analyzer._control_box.layout.display != "none"

    def test_control_visible_for_ddcq(self):
        analyzer = QPCRAnalyzer()
        analyzer._method_dropdown.value = "delta_delta_cq"
        assert analyzer._control_box.layout.display != "none"

    def test_control_hidden_when_back_to_normalized(self):
        analyzer = QPCRAnalyzer()
        analyzer._method_dropdown.value = "pfaffl"
        analyzer._method_dropdown.value = "normalized_quantity"
        assert analyzer._control_box.layout.display == "none"


class TestQPCRAnalyzerLoadData:
    def test_load_data_success(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        assert analyzer.experiment is not None
        assert len(analyzer.experiment.wells) > 0
        assert len(analyzer._target_dropdown.options) > 0
        assert analyzer._target_dropdown.disabled is False
        assert analyzer._reference_select.disabled is False

    def test_load_data_empty_path(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = ""
        analyzer._on_load_data(None)
        assert analyzer.experiment is None

    def test_load_data_nonexistent_file(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = "nonexistent.csv"
        analyzer._on_load_data(None)
        assert analyzer.experiment is None

    def test_populate_choices_sets_default_target(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        # Default target should be a non-reference gene (IL6)
        assert analyzer._target_dropdown.value == "IL6"

    def test_populate_choices_sets_references(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        ref_options = list(analyzer._reference_select.options)
        assert "GAPDH" in ref_options
        assert "RPLP0" in ref_options


class TestQPCRAnalyzerValidate:
    def test_validate_without_data_shows_error(self):
        analyzer = QPCRAnalyzer()
        analyzer._on_validate(None)
        # No crash; experiment is still None
        assert analyzer.experiment is None

    def test_validate_with_data(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        analyzer._on_validate(None)
        assert analyzer._last_report is not None


class TestQPCRAnalyzerAnalyze:
    def test_analyze_normalized_quantity(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        analyzer._method_dropdown.value = "normalized_quantity"
        analyzer._on_analyze(None)
        assert analyzer._last_batch is not None
        assert analyzer._last_batch.parameters["method"] == "normalized_quantity"

    def test_analyze_pfaffl(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        analyzer._method_dropdown.value = "pfaffl"
        analyzer._control_text.value = "Control_1"
        analyzer._on_analyze(None)
        assert analyzer._last_batch is not None
        assert analyzer._last_batch.parameters["method"] == "pfaffl"

    def test_analyze_delta_delta_cq(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        analyzer._method_dropdown.value = "delta_delta_cq"
        analyzer._control_text.value = "Control_1"
        analyzer._on_analyze(None)
        assert analyzer._last_batch is not None
        assert analyzer._last_batch.parameters["method"] == "delta_delta_cq"

    def test_analyze_pfaffl_without_control(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        analyzer._method_dropdown.value = "pfaffl"
        analyzer._control_text.value = ""
        analyzer._on_analyze(None)
        # Should not crash; batch should not be set
        assert analyzer._last_batch is None


class TestQPCRAnalyzerPlot:
    def test_plot_heatmap_without_analysis(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        # No analysis done yet, so it should fall back to heatmap
        analyzer._on_plot(None)
        # No crash

    def test_plot_fold_change_after_ddcq(self):
        analyzer = QPCRAnalyzer()
        analyzer._file_text.value = EXAMPLE_CSV
        analyzer._on_load_data(None)
        analyzer._method_dropdown.value = "delta_delta_cq"
        analyzer._control_text.value = "Control_1"
        analyzer._on_analyze(None)
        # After ddCq analysis, plot should show fold change
        analyzer._on_plot(None)
        # No crash


class TestResultsPlottable:
    def test_plottable_with_fold_change(self):
        from bioeng_qpcr.calculations import delta_delta_cq_batch
        from bioeng_qpcr.io import read_tidy_csv

        experiment = read_tidy_csv(EXAMPLE_CSV, reference_targets=["GAPDH"])
        batch = delta_delta_cq_batch(
            experiment, "IL6", "Control_1", reference_target="GAPDH"
        )
        assert QPCRAnalyzer._results_are_plottable(batch.results) is True

    def test_not_plottable_with_normalized_quantity(self):
        from bioeng_qpcr.calculations import normalized_quantity_all
        from bioeng_qpcr.io import read_tidy_csv

        experiment = read_tidy_csv(
            EXAMPLE_CSV, reference_targets=["GAPDH", "RPLP0"]
        )
        batch = normalized_quantity_all(
            experiment, "IL6", reference_targets=["GAPDH", "RPLP0"]
        )
        assert QPCRAnalyzer._results_are_plottable(batch.results) is False

    def test_not_plottable_empty(self):
        assert QPCRAnalyzer._results_are_plottable([]) is False
