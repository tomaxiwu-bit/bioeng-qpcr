"""Interactive ipywidgets-based qPCR analysis widget for Jupyter notebooks.

This module provides :class:`QPCRAnalyzer`, a self-contained interactive
widget that wraps the bioeng-qpcr engine (IO, validation, calculations and
plotting) behind a point-and-click interface for use inside Jupyter /
JupyterLab notebooks.

``ipywidgets`` is an optional dependency checked at import time; without it
this module still imports, but instantiating :class:`QPCRAnalyzer` raises a
clear, actionable :class:`ImportError`.  ``matplotlib`` is imported lazily
only when a plot is requested, so validation and analysis work without a
matplotlib installation.
"""

from __future__ import annotations

import html
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

try:
    from ipywidgets import (
        HTML,
        Button,
        Dropdown,
        HBox,
        Layout,
        Output,
        SelectMultiple,
        Text,
        VBox,
    )
except ImportError as _exc:  # pragma: no cover
    Button = Dropdown = HBox = HTML = Layout = Output = None
    SelectMultiple = Text = VBox = None
    _IPYWIDGETS_IMPORT_ERROR: ImportError | None = _exc
else:
    _IPYWIDGETS_IMPORT_ERROR = None

from bioeng_qpcr.calculations import (
    delta_delta_cq_batch,
    normalized_quantity_all,
    pfaffl_batch,
)
from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.io import CSVValidationError, read_tidy_csv
from bioeng_qpcr.validation import ValidationEngine

__all__ = ["QPCRAnalyzer"]


class QPCRAnalyzer:
    """Interactive qPCR analysis widget for Jupyter notebooks.

    Wraps the bioeng-qpcr engine behind a vertical layout of ipywidgets:
    a file input, parameter controls, action buttons and a shared output
    area that renders text tables and matplotlib figures.

    Usage:
        analyzer = QPCRAnalyzer()
        analyzer.display()  # show inside a Jupyter cell
    """

    # (label, value) pairs for the method dropdown.
    _METHOD_OPTIONS: list[tuple[str, str]] = [
        ("Normalized Quantity (efficiency-corrected)", "normalized_quantity"),
        ("Pfaffl (efficiency-corrected ratio)", "pfaffl"),
        ("\u0394\u0394Cq (Livak, assumes 100% efficiency)", "delta_delta_cq"),
    ]

    # Key columns surfaced per method in the results table.
    _RESULT_COLUMNS: dict[str, list[str]] = {
        "normalized_quantity": [
            "sample_name",
            "mean_cq_target",
            "normalized_quantity",
            "log2_normalized_quantity",
        ],
        "pfaffl": [
            "sample_name",
            "ratio",
            "delta_cq_target",
            "delta_cq_reference",
            "reference_target",
        ],
        "delta_delta_cq": [
            "sample_name",
            "fold_change",
            "delta_cq",
            "delta_delta_cq",
            "reference_target",
        ],
    }

    def __init__(self) -> None:
        if _IPYWIDGETS_IMPORT_ERROR is not None:
            raise ImportError(
                "ipywidgets is required for QPCRAnalyzer. Install it with:\n"
                "    pip install ipywidgets\n\n"
                f"Original error: {_IPYWIDGETS_IMPORT_ERROR}"
            ) from _IPYWIDGETS_IMPORT_ERROR
        self._experiment: Experiment | None = None
        self._csv_path: str | None = None
        self._last_batch: Any = None
        self._last_report: Any = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._title = HTML(self._title_html())

        self._file_text = Text(
            value="",
            placeholder="path/to/experiment.csv",
            description="CSV file:",
            layout=Layout(width="70%"),
        )

        self._target_dropdown = Dropdown(
            description="Target:",
            options=[],
            value=None,
            disabled=True,
            layout=Layout(width="45%"),
        )
        self._reference_select = SelectMultiple(
            description="References:",
            options=[],
            value=(),
            disabled=True,
            layout=Layout(width="45%"),
            rows=5,
        )
        self._method_dropdown = Dropdown(
            description="Method:",
            options=self._METHOD_OPTIONS,
            value="normalized_quantity",
            layout=Layout(width="55%"),
        )
        self._control_text = Text(
            value="",
            placeholder="control / calibrator sample name",
            description="Control:",
            layout=Layout(width="45%"),
        )
        self._control_box = HBox([self._control_text])
        # normalized_quantity (the default) does not need a control sample.
        self._control_box.layout.display = "none"

        self._load_btn = Button(
            description="Load Data",
            tooltip="Read the tidy CSV and populate target/reference choices.",
            button_style="primary",
        )
        self._validate_btn = Button(
            description="Validate",
            tooltip="Run the MIQE readiness assessment.",
        )
        self._analyze_btn = Button(
            description="Analyze",
            tooltip="Run the selected relative-quantification method.",
            button_style="success",
        )
        self._plot_btn = Button(
            description="Plot",
            tooltip="Render a fold-change bar plot or a Cq heatmap.",
        )

        self._load_btn.on_click(self._on_load_data)
        self._validate_btn.on_click(self._on_validate)
        self._analyze_btn.on_click(self._on_analyze)
        self._plot_btn.on_click(self._on_plot)
        self._method_dropdown.observe(self._on_method_change, names="value")

        self._output = Output()
        self._output.layout = Layout(
            min_height="320px",
            border="1px solid #d0d7de",
            padding="8px",
            overflow="auto",
        )

        params = VBox(
            [
                HBox([self._target_dropdown, self._reference_select]),
                HBox([self._method_dropdown, self._control_box]),
            ]
        )
        buttons = HBox(
            [
                self._load_btn,
                self._validate_btn,
                self._analyze_btn,
                self._plot_btn,
            ]
        )
        self._container = VBox(
            [
                self._title,
                self._file_text,
                params,
                buttons,
                self._output,
            ]
        )

    @staticmethod
    def _title_html() -> str:
        style = (
            "background:linear-gradient(90deg,#1f3a5f,#2c5282);"
            "color:#fff;padding:10px 14px;border-radius:6px;"
            "font-size:16px;font-weight:600;"
        )
        return f'<div style="{style}">bioeng-qpcr Interactive Analyzer</div>'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the widget in the current Jupyter / IPython cell."""
        try:
            from IPython.display import display as ipy_display
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "QPCRAnalyzer.display() requires a Jupyter / IPython "
                "environment (IPython.display is unavailable)."
            ) from exc
        ipy_display(self._container)

    @property
    def experiment(self) -> Experiment | None:
        """The currently loaded experiment, or ``None``."""
        return self._experiment

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_load_data(self, _button: Any) -> None:
        path = self._file_text.value.strip()
        if not path:
            self._show_error("Please enter a CSV file path first.")
            return
        try:
            experiment = read_tidy_csv(path, strict=True)
        except (FileNotFoundError, CSVValidationError, ValueError) as exc:
            self._show_error(f"Failed to load CSV:\n{exc}")
            return
        self._experiment = experiment
        self._csv_path = path
        self._last_batch = None
        self._last_report = None
        self._populate_choices(experiment)
        refs = experiment.reference_targets
        with self._output:
            self._output.clear_output()
            print(f"Loaded experiment '{experiment.name}' from:\n  {path}")
            print(
                f"  Wells: {len(experiment.wells)}  |  "
                f"Samples: {len(experiment.sample_names)}  |  "
                f"Targets: {len(experiment.target_names)}"
            )
            if refs:
                print(f"  Reference genes (marked): {', '.join(refs)}")
            else:
                print("  No reference genes marked in the file;")
                print("  select reference gene(s) below before analyzing.")

    def _on_validate(self, _button: Any) -> None:
        if self._experiment is None:
            self._show_error("Load data first.")
            return
        engine = ValidationEngine()
        try:
            report = engine.run(self._experiment)
        except Exception as exc:  # noqa: BLE001 -- surface to the user
            self._show_error(f"Validation failed:\n{exc}")
            return
        self._last_report = report
        with self._output:
            self._output.clear_output()
            print(report.to_text())

    def _on_analyze(self, _button: Any) -> None:
        if self._experiment is None:
            self._show_error("Load data first.")
            return
        experiment = self._experiment
        target = self._target_dropdown.value
        method = self._method_dropdown.value
        references = list(self._reference_select.value)
        if target is None:
            self._show_error("Select a target gene of interest.")
            return
        try:
            batch = self._run_analysis(experiment, target, method, references)
        except Exception as exc:  # noqa: BLE001 -- surface to the user
            self._show_error(f"Analysis failed:\n{exc}")
            return
        self._last_batch = batch
        self._render_batch(batch)

    def _on_plot(self, _button: Any) -> None:
        if self._experiment is None:
            self._show_error("Load data first.")
            return
        try:
            import matplotlib.pyplot as plt

            from bioeng_qpcr.plotting import plot_cq_heatmap, plot_fold_change
        except ImportError as exc:
            self._show_error(
                "matplotlib is required for plotting. Install it with:\n"
                "    pip install matplotlib\n\n"
                f"Original error: {exc}"
            )
            return
        batch = self._last_batch
        plottable = batch is not None and self._results_are_plottable(batch.results)
        try:
            if plottable:
                fig, _ax = plot_fold_change(batch.results)
                title = "Fold change per sample"
            else:
                fig, _ax = plot_cq_heatmap(self._experiment)
                title = "Cq heatmap"
        except Exception as exc:  # noqa: BLE001 -- surface to the user
            self._show_error(f"Plotting failed:\n{exc}")
            return
        with self._output:
            self._output.clear_output()
            print(title)
            plt.show()

    def _on_method_change(self, _change: Any) -> None:
        needs_control = self._method_dropdown.value in (
            "pfaffl",
            "delta_delta_cq",
        )
        self._control_box.layout.display = "" if needs_control else "none"

    # ------------------------------------------------------------------
    # Analysis dispatch
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        experiment: Experiment,
        target: str,
        method: str,
        references: list[str],
    ) -> Any:
        if method == "normalized_quantity":
            return normalized_quantity_all(
                experiment, target, reference_targets=references or None
            )
        # pfaffl and delta_delta_cq require a control sample and accept
        # exactly one reference gene.
        control = self._control_text.value.strip()
        if not control:
            raise ValueError(
                f"A control (calibrator) sample name is required for method '{method}'."
            )
        if len(references) > 1:
            raise ValueError(
                f"Method '{method}' accepts exactly one reference gene, "
                f"but {len(references)} were selected: "
                f"{', '.join(references)}."
            )
        reference_target = references[0] if references else None
        if method == "pfaffl":
            return pfaffl_batch(
                experiment, target, control, reference_target=reference_target
            )
        return delta_delta_cq_batch(
            experiment, target, control, reference_target=reference_target
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_batch(self, batch: Any) -> None:
        from IPython.display import display as ipy_display

        method = batch.parameters.get("method", "")
        columns = self._RESULT_COLUMNS.get(method, [])
        rows = [asdict(r) for r in batch.results if is_dataclass(r)]
        with self._output:
            self._output.clear_output()
            print(f"Method: {method}")
            param_str = ", ".join(
                f"{k}={v}" for k, v in batch.parameters.items() if k != "method"
            )
            if param_str:
                print(f"Parameters: {param_str}")
            print(f"Results: {len(batch.results)}   Excluded: {len(batch.exclusions)}")
            print()
            if rows:
                df = pd.DataFrame(rows)
                if columns:
                    present = [c for c in columns if c in df.columns]
                    if present:
                        df = df[present]
                ipy_display(df)
            else:
                print("(no results)")
            for warning in batch.warnings:
                print(f"\nWarning: {warning}")
            for exclusion in batch.exclusions:
                print(f"Excluded {exclusion.sample_name}: {exclusion.reason}")

    def _show_error(self, message: str) -> None:
        from IPython.display import HTML
        from IPython.display import display as ipy_display

        escaped = html.escape(message)
        self._output.clear_output()
        with self._output:
            ipy_display(
                HTML(
                    '<pre style="color:#c0392b;white-space:pre-wrap;margin:0">'
                    f"{escaped}</pre>"
                )
            )

    # ------------------------------------------------------------------
    # Choice population
    # ------------------------------------------------------------------

    def _populate_choices(self, experiment: Experiment) -> None:
        all_targets = experiment.target_names
        ref_targets = experiment.reference_targets

        # Target dropdown: all targets; default to the first non-reference.
        self._target_dropdown.options = all_targets
        default_target = None
        non_ref = [t for t in all_targets if t not in ref_targets]
        if non_ref:
            default_target = non_ref[0]
        elif all_targets:
            default_target = all_targets[0]
        self._target_dropdown.value = default_target
        self._target_dropdown.disabled = False

        # Reference select: marked references, else fall back to all targets
        # so the researcher can designate references manually.
        ref_options = ref_targets if ref_targets else all_targets
        self._reference_select.options = ref_options
        self._reference_select.value = tuple(ref_options[:1])
        self._reference_select.disabled = False

        # Control sample default: first non-NTC sample.
        samples = experiment.sample_names
        if samples:
            self._control_text.value = samples[0]

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _results_are_plottable(results: list[Any]) -> bool:
        """True when results carry a ``fold_change`` or ``ratio`` attribute."""
        if not results:
            return False
        first = results[0]
        return hasattr(first, "fold_change") or hasattr(first, "ratio")
