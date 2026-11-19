"""Potency (IC50/EC50) analysis panel.

Fits a dose-response curve, reports both the relative and the absolute
potency, and runs the POT-001..POT-011 audit rules against the result.  The
panel is deliberately explicit about the difference between the two values:
they coincide only when the response returns to zero, and reporting the
midpoint of the fitted window as "the IC50" is one of the most common
mistakes in the literature.
"""

from __future__ import annotations

import contextlib
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.io import read_dose_response_csv
from bioeng_qpcr.plotting import plot_dose_response, plot_potency_comparison
from bioeng_qpcr.potency import (
    DoseResponseModel,
    PotencyKind,
    fit_dose_response,
    relative_potency,
    run_potency_validation,
)

#: Display label for each supported response model.
_MODEL_LABELS: dict[str, str] = {
    "自动选择（4PL/5PL/Gompertz）": "auto",
    "4 参数 Logistic": DoseResponseModel.FOUR_PL.value,
    "5 参数 Logistic（非对称）": DoseResponseModel.FIVE_PL.value,
    "Gompertz": DoseResponseModel.GOMPERTZ.value,
    "3 参数 Logistic（下平台为 0）": DoseResponseModel.THREE_PL.value,
}


def _labelled(text: str, widget: QWidget) -> QWidget:
    """Wrap *widget* with a field label styled like the rest of the suite."""
    from PySide6.QtWidgets import QVBoxLayout

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    label = QLabel(text)
    label.setObjectName("FieldLabel")
    layout.addWidget(label)
    layout.addWidget(widget)
    return container


class PotencyPanel(BasePanel):
    """Panel for IC50/EC50 determination and dose-response model selection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_controls()
        self._install_plot_canvas()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "效价分析"

    def _panel_subtitle(self) -> str:
        return "IC50 / EC50 · 剂量-反应曲线 · 模型选择"

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        file_card = Card("数据文件")
        self._file_selector = FileSelector(
            label="选择剂量-反应 CSV 文件",
            file_filter="CSV files (*.csv);;All files (*.*)",
        )
        self._file_selector.file_selected.connect(self._on_file_selected)
        file_card.addWidget(self._file_selector)
        hint = QLabel(
            "长格式：compound, concentration, response[, replicate]\n"
            "矩阵格式：第一列为 concentration，其后每列一个化合物"
        )
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        file_card.addWidget(hint)
        self._controls_layout.addWidget(file_card)

        settings = Card("分析设置")

        self._kind_combo = QComboBox()
        self._kind_combo.addItems(["自动（按曲线方向）", "IC50", "EC50"])
        settings.addWidget(_labelled("报告量", self._kind_combo))

        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["自动识别", "长格式", "矩阵格式"])
        settings.addWidget(_labelled("数据布局", self._layout_combo))

        self._normalize_combo = QComboBox()
        self._normalize_combo.addItems(["保持原始响应", "转换为对照百分比"])
        settings.addWidget(_labelled("归一化", self._normalize_combo))

        self._model_combo = QComboBox()
        self._model_combo.addItems(list(_MODEL_LABELS))
        settings.addWidget(_labelled("模型", self._model_combo))

        self._ci_combo = QComboBox()
        self._ci_combo.addItems(["渐近（协方差矩阵）", "Bootstrap（重采样）"])
        settings.addWidget(_labelled("置信区间", self._ci_combo))

        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.50, 0.999)
        self._confidence_spin.setSingleStep(0.01)
        self._confidence_spin.setDecimals(3)
        self._confidence_spin.setValue(0.95)
        settings.addWidget(_labelled("置信水平", self._confidence_spin))

        self._bootstrap_spin = QSpinBox()
        self._bootstrap_spin.setRange(50, 5000)
        self._bootstrap_spin.setSingleStep(50)
        self._bootstrap_spin.setValue(500)
        settings.addWidget(_labelled("Bootstrap 次数", self._bootstrap_spin))

        self._control_fraction_spin = QDoubleSpinBox()
        self._control_fraction_spin.setRange(0.05, 0.95)
        self._control_fraction_spin.setSingleStep(0.05)
        self._control_fraction_spin.setDecimals(2)
        self._control_fraction_spin.setValue(0.50)
        settings.addWidget(
            _labelled("绝对效价的对照比例", self._control_fraction_spin)
        )

        self._explicit_control_check = QCheckBox("使用实测未处理/溶剂对照响应")
        settings.addWidget(self._explicit_control_check)
        self._control_response_spin = QDoubleSpinBox()
        self._control_response_spin.setRange(-1e12, 1e12)
        self._control_response_spin.setDecimals(6)
        self._control_response_spin.setValue(100.0)
        settings.addWidget(_labelled("对照响应", self._control_response_spin))

        self._validate_check = QCheckBox("运行 POT-001 ~ POT-011 审计规则")
        self._validate_check.setChecked(True)
        settings.addWidget(self._validate_check)

        self._compare_check = QCheckBox("多化合物时输出相对效价比较")
        self._compare_check.setChecked(True)
        settings.addWidget(self._compare_check)

        self._controls_layout.addWidget(settings)

        action_card = Card("执行")
        self._btn_fit = QPushButton("拟合剂量-反应曲线")
        self._btn_fit.setObjectName("PrimaryButton")
        self._btn_fit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_fit.clicked.connect(self._on_fit)
        action_card.addWidget(self._btn_fit)
        self._controls_layout.addWidget(action_card)
        self._controls_layout.addStretch()

    def _install_plot_canvas(self) -> None:
        self._plot_canvas = PlotCanvas(self, width=8, height=5, dpi=100)
        self._add_plot_widget(self._plot_canvas, "曲线与效价")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_file_selected(self, path: str) -> None:
        self._emit_status(f"已选择文件: {path}")

    def _render_figure(self, fig: Any) -> None:
        if self._plot_canvas is None:
            return
        previous = self._plot_canvas.figure
        self._plot_canvas.set_figure(fig)
        if previous is not fig:
            with contextlib.suppress(Exception):
                import matplotlib.pyplot as plt

                plt.close(previous)

    @staticmethod
    def _format_ci(ci: tuple[float, float] | None) -> str:
        if ci is None:
            return "（不可用）"
        return f"（{ci[0]:.4g} ~ {ci[1]:.4g}）"

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _on_fit(self) -> None:
        path = self._file_selector.get_path()
        if not path:
            self._show_error("请先选择剂量-反应 CSV 文件。")
            return

        kind_text = self._kind_combo.currentText()
        kind = None
        if kind_text == "IC50":
            kind = PotencyKind.IC50
        elif kind_text == "EC50":
            kind = PotencyKind.EC50

        layout = {
            "自动识别": "auto",
            "长格式": "long",
            "矩阵格式": "matrix",
        }[self._layout_combo.currentText()]
        normalize = (
            "percent_of_control"
            if self._normalize_combo.currentIndex() == 1
            else "none"
        )
        ci_method = (
            "bootstrap" if self._ci_combo.currentIndex() == 1 else "asymptotic"
        )
        model_key = _MODEL_LABELS[self._model_combo.currentText()]
        models = None if model_key == "auto" else [DoseResponseModel(model_key)]

        self._emit_status("正在拟合剂量-反应曲线...")

        try:
            datasets = read_dose_response_csv(
                path, layout=layout, normalize=normalize
            )
        except Exception as exc:
            self._show_error(f"读取 CSV 失败: {exc}")
            return

        results = []
        for dataset in datasets:
            try:
                results.append(
                    fit_dose_response(
                        dataset,
                        kind=kind,
                        models=models,
                        ci_method=ci_method,
                        confidence=self._confidence_spin.value(),
                        n_bootstrap=self._bootstrap_spin.value(),
                        control_fraction=self._control_fraction_spin.value(),
                        control_response=(
                            self._control_response_spin.value()
                            if self._explicit_control_check.isChecked()
                            else None
                        ),
                    )
                )
            except Exception as exc:
                self._show_error(f"{dataset.compound} 拟合失败: {exc}")
                return

        lines: list[str] = []
        failures = 0
        for result, dataset in zip(results, datasets, strict=True):
            fit = result.best_fit
            lines.append(f"{result.compound}  —  {result.kind.value}")
            lines.append("=" * 46)
            lines.append(f"  相对 {result.kind.value}: {result.relative_value:.4g}")
            lines.append(
                f"    {result.confidence:.0%} CI "
                f"{self._format_ci(result.relative_ci)}"
            )
            if result.absolute_value is not None:
                lines.append(
                    f"  绝对 {result.kind.value}"
                    f"（对照的 {result.control_fraction:.0%}）: "
                    f"{result.absolute_value:.4g}"
                )
                lines.append(
                    f"    {result.confidence:.0%} CI "
                    f"{self._format_ci(result.absolute_ci)}"
                )
            else:
                lines.append(
                    f"  绝对 {result.kind.value}: 不可达（曲线未覆盖该响应水平）"
                )
            lines.append(
                f"  模型 {fit.model.value}   Hill {fit.hill_slope:.3g}   "
                f"R² {fit.r_squared:.5f}   n {fit.n_points}"
            )
            lines.append(
                f"  窗口 {fit.bottom:.4g} → {fit.top:.4g}   "
                f"跨度 {result.fold_range_covered:.2f} 个数量级"
            )
            if result.extrapolated:
                lines.append("  [!] 效价落在测试浓度范围之外（外推结果）")
            for warning in result.warnings:
                lines.append(f"  [!] {warning}")
            lines.append(f"  {result.comparison.message}")
            lines.append("")

            if self._validate_check.isChecked():
                for check in run_potency_validation(result, dataset):
                    if check.passed:
                        continue
                    if check.severity == "failure":
                        failures += 1
                    tag = "[错误]" if check.severity == "failure" else "[警告]"
                    lines.append(f"  {tag} {check.rule_id}: {check.message}")
                lines.append("")

        if self._compare_check.isChecked() and len(results) > 1:
            lines.append("相对效价比较")
            lines.append("-" * 46)
            reference = results[0]
            for other in results[1:]:
                try:
                    comparison = relative_potency(reference, other)
                except ValueError:
                    continue
                lines.append(f"  {comparison.message}")
                if comparison.ci is not None:
                    lines.append(
                        f"    比值 {comparison.confidence:.0%} CI "
                        f"{comparison.ci[0]:.4g} ~ {comparison.ci[1]:.4g}"
                    )

        self._result_display.set_text("\n".join(lines).rstrip("\n"))

        try:
            if len(results) == 1:
                fig, _ax = plot_dose_response(results[0].best_fit, datasets[0])
            else:
                fig, _ax = plot_potency_comparison(results)
        except Exception as exc:
            self._show_error(f"绘图失败: {exc}")
            return
        self._render_figure(fig)

        if failures:
            self._emit_status(f"完成，但有 {failures} 项审计规则未通过。")
        else:
            self._emit_status("剂量-反应拟合完成。")
