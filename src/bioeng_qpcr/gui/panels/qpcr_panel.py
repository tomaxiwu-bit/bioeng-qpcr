"""qPCR analysis panel for the bioeng-qpcr desktop application.

The panel wires the researcher-facing analysis workflow into the GUI:

* CSV ingestion via :class:`FileSelector` (strict or lenient mode).
* MIQE readiness assessment via :class:`ValidationEngine`.
* Three relative-quantification methods -- normalized quantity, Pfaffl and
  ΔΔCq -- exposed through a single method selector.
* Explicit, opt-in group comparison for normalized-quantity runs.
* Embedded visualisation (fold-change bar plot or Cq heatmap) via
  :class:`PlotCanvas`.
* One-click self-contained HTML research-report export.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bioeng_qpcr.calculations import (
    compare_groups,
    delta_delta_cq_batch,
    normalized_quantity_all,
    pfaffl_batch,
)
from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.io import CSVValidationError, read_tidy_csv
from bioeng_qpcr.plotting import plot_cq_heatmap, plot_fold_change
from bioeng_qpcr.reporting import build_research_report
from bioeng_qpcr.validation import RuleExecutionError, ValidationEngine

__all__ = ["QpcrPanel"]


class QpcrPanel(BasePanel):
    """qPCR analysis panel: MIQE validation, relative quantification, reporting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wire_signals()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "qPCR 分析"

    def _panel_subtitle(self) -> str:
        return "MIQE 验证 · 相对定量 · 分组比较"

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        super()._build_ui()
        self._build_controls()
        self._build_plot_area()

    def _build_controls(self) -> None:
        layout = self._controls_layout

        # --- Data input ---
        input_card = Card("数据输入")
        self._file_selector = FileSelector(
            "选择 CSV 数据文件",
            "CSV files (*.csv);;All files (*.*)",
        )
        input_card.addWidget(self._file_selector)

        self._lenient_check = QCheckBox("宽松模式（忽略非致命校验问题）")
        self._lenient_check.setToolTip(
            "勾选后以宽松模式读取 CSV：可恢复的非法值将被强制转换，"
            "所有问题记录在实验元数据中而非抛出异常。"
        )
        input_card.addWidget(self._lenient_check)
        layout.addWidget(input_card)

        # --- Analysis parameters ---
        param_card = Card("分析参数")

        self._method_combo = QComboBox()
        self._method_combo.addItem(
            "归一化量 (normalized_quantity)", "normalized_quantity"
        )
        self._method_combo.addItem("Pfaffl (效率校正)", "pfaffl")
        self._method_combo.addItem("ΔΔCq (Livak)", "delta_delta_cq")
        param_card.addLayout(self._labelled_row("分析方法", self._method_combo))

        self._target_edit = QLineEdit()
        self._target_edit.setPlaceholderText("例如: GAPDH")
        param_card.addLayout(self._labelled_row("目标基因", self._target_edit))

        self._control_edit = QLineEdit()
        self._control_edit.setPlaceholderText("ΔΔCq / Pfaffl 需要，例如: CTRL")
        param_card.addLayout(self._labelled_row("对照样本", self._control_edit))

        self._group_compare_check = QCheckBox("执行两组比较（需明确选择分组）")
        self._group_compare_check.setToolTip(
            "仅用于归一化量分析。软件不会猜测对照组，也不会自动忽略其他分组。"
        )
        self._group_compare_check.setEnabled(False)
        param_card.addWidget(self._group_compare_check)

        self._control_group_combo = QComboBox()
        self._control_group_combo.addItem("请选择对照组…", None)
        self._control_group_combo.setEnabled(False)
        param_card.addLayout(
            self._labelled_row("分组比较 · 对照组", self._control_group_combo)
        )

        self._treated_group_combo = QComboBox()
        self._treated_group_combo.addItem("请选择处理组…", None)
        self._treated_group_combo.setEnabled(False)
        param_card.addLayout(
            self._labelled_row("分组比较 · 处理组", self._treated_group_combo)
        )

        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("逗号分隔，可选，例如: ACTB, RPLP0")
        param_card.addLayout(self._labelled_row("参考基因", self._ref_edit))

        layout.addWidget(param_card)

        # --- Actions ---
        action_card = Card("开始分析")

        self._validate_btn = QPushButton("MIQE 验证")
        self._validate_btn.setObjectName("SecondaryButton")
        self._run_btn = QPushButton("运行分析")
        self._run_btn.setObjectName("PrimaryButton")
        self._report_btn = QPushButton("生成报告")
        self._report_btn.setObjectName("QuietButton")

        action_card.addWidget(self._run_btn)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)
        secondary_row.addWidget(self._validate_btn)
        secondary_row.addWidget(self._report_btn)
        action_card.addLayout(secondary_row)
        layout.addWidget(action_card)

        layout.addStretch()

    def _build_plot_area(self) -> None:
        """Append the plot canvas below the result display on the right pane."""
        self._plot_canvas = PlotCanvas(self)
        self._plot_canvas.setMinimumHeight(230)
        self._add_plot_widget(self._plot_canvas, "数据可视化")

    def _wire_signals(self) -> None:
        self._file_selector.file_selected.connect(self._on_file_selected)
        self._method_combo.currentIndexChanged.connect(
            self._update_group_controls_enabled
        )
        self._group_compare_check.toggled.connect(
            self._update_group_controls_enabled
        )
        self._validate_btn.clicked.connect(self._on_validate)
        self._run_btn.clicked.connect(self._on_run_analysis)
        self._report_btn.clicked.connect(self._on_generate_report)

    def _labelled_row(self, label_text: str, widget: QWidget) -> QVBoxLayout:
        row = QVBoxLayout()
        row.setSpacing(6)
        label = QLabel(label_text)
        label.setObjectName("FieldLabel")
        row.addWidget(label)
        row.addWidget(widget)
        return row

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_file_selected(self, path: str) -> None:
        self._emit_status(f"已选择文件：{Path(path).name}")
        # Populate selectors early enough for the user to make an explicit
        # choice before running the analysis. Parsing errors are still shown
        # through the normal analysis/validation path.
        try:
            experiment = read_tidy_csv(
                path,
                reference_targets=self._parse_reference_targets(),
                strict=not self._lenient_check.isChecked(),
            )
        except Exception:  # preflight only; normal load path reports the error
            self._set_available_groups(())
            return
        self._set_available_groups(experiment)

    def _set_available_groups(self, experiment_or_groups) -> None:
        """Populate group selectors without making a scientific choice."""
        if hasattr(experiment_or_groups, "samples"):
            groups = sorted(
                {
                    sample.group
                    for sample in experiment_or_groups.samples.values()
                    if sample.group
                }
            )
        else:
            groups = sorted({group for group in experiment_or_groups if group})

        for combo, placeholder in (
            (self._control_group_combo, "请选择对照组…"),
            (self._treated_group_combo, "请选择处理组…"),
        ):
            previous = combo.currentData()
            combo.clear()
            combo.addItem(placeholder, None)
            for group in groups:
                combo.addItem(group, group)
            previous_index = combo.findData(previous)
            combo.setCurrentIndex(previous_index if previous_index >= 0 else 0)

        self._update_group_controls_enabled()

    def _update_group_controls_enabled(self, *_args) -> None:
        is_normalized = self._method_combo.currentData() == "normalized_quantity"
        has_two_groups = self._control_group_combo.count() >= 3
        comparison_available = is_normalized and has_two_groups
        self._group_compare_check.setEnabled(comparison_available)
        if not comparison_available:
            self._group_compare_check.setChecked(False)
        selection_enabled = (
            comparison_available and self._group_compare_check.isChecked()
        )
        self._control_group_combo.setEnabled(selection_enabled)
        self._treated_group_combo.setEnabled(selection_enabled)

    def _parse_reference_targets(self) -> list[str] | None:
        text = self._ref_edit.text().strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return parts or None

    def _load_experiment(self):
        """Read the selected CSV; show an error and return None on failure."""
        path = self._file_selector.get_path()
        if not path:
            self._show_error("请先选择 CSV 数据文件。")
            return None
        reference_targets = self._parse_reference_targets()
        strict = not self._lenient_check.isChecked()
        try:
            return read_tidy_csv(
                path, reference_targets=reference_targets, strict=strict
            )
        except CSVValidationError as exc:
            self._show_error(f"CSV 解析失败：\n{exc}")
        except OSError as exc:
            self._show_error(f"无法读取文件：{exc}")
        except Exception as exc:  # defensive: never crash the panel
            self._show_error(f"读取文件时发生意外错误：{exc}")
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_validate(self) -> None:
        experiment = self._load_experiment()
        if experiment is None:
            return
        self._set_available_groups(experiment)

        self._emit_status("正在执行 MIQE 验证...")
        try:
            report = ValidationEngine().run(experiment)
        except RuleExecutionError as exc:
            self._show_error(f"验证规则执行出错：{exc}")
            return
        except Exception as exc:  # defensive: never crash the panel
            self._show_error(f"验证失败：{exc}")
            return

        summary = report.summary
        header = (
            "MIQE 就绪性评估\n"
            f"实验: {report.experiment_name}\n"
            f"总体状态: {report.overall_status}\n"
            f"通过 {summary['PASS']} · 警告 {summary['WARN']} · "
            f"失败 {summary['FAIL']} · 无法评估 {summary['NOT_ASSESSABLE']}\n"
        )
        self._result_display.set_text(header + "\n" + report.to_text())
        self._emit_status(f"MIQE 验证完成：{report.overall_status}")

    def _on_run_analysis(self) -> None:
        experiment = self._load_experiment()
        if experiment is None:
            return
        self._set_available_groups(experiment)

        target = self._target_edit.text().strip()
        if not target:
            self._show_error("请输入目标基因名称。")
            return

        method = self._method_combo.currentData()
        reference_targets = self._parse_reference_targets()
        control = self._control_edit.text().strip()

        if method in ("delta_delta_cq", "pfaffl") and not control:
            label = "ΔΔCq" if method == "delta_delta_cq" else "Pfaffl"
            self._show_error(f"{label} 方法需要指定对照样本。")
            return

        self._emit_status("正在运行分析...")
        self._plot_canvas.clear()
        plot_ok = False

        try:
            if method == "normalized_quantity":
                batch = normalized_quantity_all(experiment, target, reference_targets)
                text = self._format_normalized_quantity(
                    batch, experiment, target, reference_targets
                )
                self._result_display.set_text(text)
                plot_ok = self._draw_heatmap(experiment)

            elif method == "delta_delta_cq":
                ref = reference_targets[0] if reference_targets else None
                batch = delta_delta_cq_batch(experiment, target, control, ref)
                text = self._format_delta_delta_cq(batch, target, control, ref)
                self._result_display.set_text(text)
                plot_ok = self._draw_fold_change(batch.results)

            elif method == "pfaffl":
                ref = reference_targets[0] if reference_targets else None
                batch = pfaffl_batch(experiment, target, control, ref)
                text = self._format_pfaffl(batch, target, control, ref)
                self._result_display.set_text(text)
                plot_ok = self._draw_fold_change(batch.results)

            else:
                self._show_error(f"未知的分析方法：{method}")
                return
        except ValueError as exc:
            self._show_error(f"分析失败：{exc}")
            return
        except Exception as exc:  # defensive: never crash the panel
            self._show_error(f"分析时发生意外错误：{exc}")
            return

        if plot_ok:
            self._emit_status("分析完成")
        else:
            self._emit_status("分析完成（图表绘制失败，请查看结果文本）")

    def _on_generate_report(self) -> None:
        experiment = self._load_experiment()
        if experiment is None:
            return

        target = self._target_edit.text().strip()
        if not target:
            self._show_error("请输入目标基因名称。")
            return
        reference_targets = self._parse_reference_targets()

        self._emit_status("正在生成研究报告...")
        try:
            report = build_research_report(experiment, target, reference_targets)
        except Exception as exc:  # defensive: never crash the panel
            self._show_error(f"生成报告失败：{exc}")
            return

        default_name = f"{experiment.name}_report.html"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存研究报告",
            default_name,
            "HTML files (*.html);;All files (*.*)",
        )
        if not path:
            self._emit_status("已取消保存报告。")
            return

        try:
            Path(path).write_text(report.html, encoding="utf-8")
        except OSError as exc:
            self._show_error(f"写入文件失败：{exc}")
            return

        refs_text = (
            ", ".join(reference_targets) if reference_targets else "（实验默认）"
        )
        self._result_display.set_text(
            "研究报告已保存\n"
            f"{'=' * 60}\n"
            f"保存路径:   {path}\n"
            f"实验名称:   {experiment.name}\n"
            f"目标基因:   {target}\n"
            f"参考基因:   {refs_text}\n"
            f"报告大小:   {len(report.html)} 字符"
        )
        self._emit_status("报告已生成")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _draw_fold_change(self, results) -> bool:
        if not results:
            self._plot_canvas.clear()
            return False
        try:
            fig, _ax = plot_fold_change(results)
            self._plot_canvas.set_figure(fig)
            self._plot_canvas.updateGeometry()
            return True
        except Exception:
            self._plot_canvas.clear()
            return False

    def _draw_heatmap(self, experiment) -> bool:
        try:
            fig, _ax = plot_cq_heatmap(experiment)
            self._plot_canvas.set_figure(fig)
            self._plot_canvas.updateGeometry()
            return True
        except Exception:
            self._plot_canvas.clear()
            return False

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    def _format_normalized_quantity(self, batch, experiment, target, reference_targets):
        refs = reference_targets or list(experiment.reference_targets)
        lines = [
            "相对定量分析 — 归一化量 (normalized_quantity)",
            "=" * 60,
            f"目标基因:    {target}",
            f"参考基因:    {', '.join(refs) if refs else '（未指定）'}",
            f"样本总数:    {len(experiment.sample_names)}    "
            f"有效结果: {len(batch.results)}    排除: {len(batch.exclusions)}",
            "",
        ]
        if batch.results:
            headers = ["样本", "归一化量", "log2(NQ)"]
            rows = [
                [
                    r.sample_name,
                    f"{r.normalized_quantity:.6g}",
                    f"{r.log2_normalized_quantity:.4f}",
                ]
                for r in batch.results
            ]
            lines.append(self._table(headers, rows))
        else:
            lines.append("（无有效结果）")
        lines.append("")
        lines.extend(self._format_exclusions(batch.exclusions))
        lines.extend(self._format_warnings(batch.warnings))

        comparison = self._selected_group_comparison(
            experiment, target, reference_targets
        )
        if comparison is not None:
            lines.extend(self._format_group_comparison(comparison))
        return "\n".join(lines)

    def _format_delta_delta_cq(self, batch, target, control, ref):
        lines = [
            "相对定量分析 — ΔΔCq (Livak)",
            "=" * 60,
            f"目标基因:    {target}",
            f"参考基因:    {ref or '（实验默认）'}",
            f"对照样本:    {control}",
            f"有效结果: {len(batch.results)}    排除: {len(batch.exclusions)}",
            "",
        ]
        if batch.results:
            headers = ["样本", "ΔCq", "ΔΔCq", "Fold Change"]
            rows = [
                [
                    r.sample_name,
                    f"{r.delta_cq:.4f}",
                    f"{r.delta_delta_cq:.4f}",
                    f"{r.fold_change:.4f}",
                ]
                for r in batch.results
            ]
            lines.append(self._table(headers, rows))
        else:
            lines.append("（无有效结果）")
        lines.append("")
        lines.extend(self._format_exclusions(batch.exclusions))
        lines.extend(self._format_warnings(batch.warnings))
        return "\n".join(lines)

    def _format_pfaffl(self, batch, target, control, ref):
        target_eff = batch.results[0].target_efficiency if batch.results else None
        ref_eff = batch.results[0].reference_efficiency if batch.results else None
        lines = [
            "相对定量分析 — Pfaffl (效率校正)",
            "=" * 60,
            f"目标基因:    {target}"
            + (f" (E={target_eff:.4f})" if target_eff is not None else ""),
            f"参考基因:    {ref or '（实验默认）'}"
            + (f" (E={ref_eff:.4f})" if ref_eff is not None else ""),
            f"对照样本:    {control}",
            f"有效结果: {len(batch.results)}    排除: {len(batch.exclusions)}",
            "",
        ]
        if batch.results:
            headers = ["样本", "ΔCq(target)", "ΔCq(ref)", "Ratio"]
            rows = [
                [
                    r.sample_name,
                    f"{r.delta_cq_target:.4f}",
                    f"{r.delta_cq_reference:.4f}",
                    f"{r.ratio:.4f}",
                ]
                for r in batch.results
            ]
            lines.append(self._table(headers, rows))
        else:
            lines.append("（无有效结果）")
        lines.append("")
        lines.extend(self._format_exclusions(batch.exclusions))
        lines.extend(self._format_warnings(batch.warnings))
        return "\n".join(lines)

    def _selected_group_comparison(self, experiment, target, reference_targets):
        """Run only the two-group comparison explicitly requested by the user."""
        if not self._group_compare_check.isChecked():
            return None

        control_group = self._control_group_combo.currentData()
        treated_group = self._treated_group_combo.currentData()
        if not control_group or not treated_group:
            raise ValueError("执行分组比较前，请明确选择对照组和处理组。")
        if control_group == treated_group:
            raise ValueError("对照组和处理组必须是不同的分组。")

        groups = {
            sample.group
            for sample in experiment.samples.values()
            if sample.group
        }
        missing = [
            group for group in (control_group, treated_group) if group not in groups
        ]
        if missing:
            raise ValueError(f"所选分组不存在于当前数据：{', '.join(missing)}")

        comparison, _batch = compare_groups(
            experiment, target, treated_group, control_group, reference_targets
        )
        return comparison

    def _format_group_comparison(self, comparison):
        lines = [
            "分组比较 (基于归一化量)",
            "-" * 60,
            f"  处理组:     {comparison.treated_group} "
            f"(n={len(comparison.treated_values)})",
            f"  对照组:     {comparison.control_group} "
            f"(n={len(comparison.control_values)})",
            f"  处理组均值: {comparison.treated_mean:.6g}",
            f"  对照组均值: {comparison.control_mean:.6g}",
            f"  倍数差异:   {comparison.fold_difference:.4f}",
            f"  log2 倍数:  {comparison.log2_fold_difference:.4f}",
        ]
        if (
            comparison.welch_t_statistic is not None
            and comparison.p_value is not None
        ):
            lines.append(f"  Welch t:    {comparison.welch_t_statistic:.4f}")
            lines.append(f"  p 值:       {comparison.p_value:.4f}")
        else:
            lines.append("  Welch t:    （样本数不足，未执行检验）")
        lines.append("")
        return lines

    @staticmethod
    def _format_exclusions(exclusions) -> list[str]:
        if not exclusions:
            return []
        lines = ["排除的样本:"]
        lines.extend(
            f"  • {item.sample_name} [{item.code}]: {item.reason}"
            for item in exclusions
        )
        lines.append("")
        return lines

    @staticmethod
    def _format_warnings(warnings) -> list[str]:
        if not warnings:
            return []
        lines = ["警告:"]
        lines.extend(f"  • {warning}" for warning in warnings)
        lines.append("")
        return lines

    @staticmethod
    def _table(headers: list[str], rows: list[list]) -> str:
        """Render a monospace-aligned text table from headers and rows."""
        str_rows = [[str(cell) for cell in row] for row in rows]
        widths = [len(h) for h in headers]
        for row in str_rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))

        def render(cells: list[str]) -> str:
            parts = []
            for i in range(len(headers)):
                cell = cells[i] if i < len(cells) else ""
                parts.append(cell.ljust(widths[i]))
            return "  ".join(parts)

        sep = "  ".join("-" * w for w in widths)
        lines = [render(headers), sep]
        lines.extend(render(row) for row in str_rows)
        return "\n".join(lines)
