"""Enzyme kinetics analysis panel.

Provides a GUI for:

* Fitting the Michaelis-Menten equation ``V = Vmax * [S] / (Km + [S])`` to
  substrate-velocity data, with Km-range and substrate-span validation.
* Classifying the reversible inhibition mechanism (competitive,
  non-competitive, uncompetitive, mixed, none) from multi-inhibitor
  datasets via a Lineweaver-Burk plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QWidget

from bioeng_qpcr.enzyme_kinetics import (
    InhibitionType,
    classify_inhibition,
    fit_michaelis_menten,
    read_kinetics_csv,
    validate_km_range,
    validate_substrate_span,
)
from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.plotting import plot_lineweaver_burk, plot_michaelis_menten

#: Human-readable Chinese labels for each inhibition mechanism.
_INHIBITION_LABELS: dict[InhibitionType, str] = {
    InhibitionType.COMPETITIVE: "竞争性抑制",
    InhibitionType.NON_COMPETITIVE: "非竞争性抑制",
    InhibitionType.UNCOMPETITIVE: "反竞争性抑制",
    InhibitionType.MIXED: "混合型抑制",
    InhibitionType.NONE: "无明显抑制",
}


class EnzymePanel(BasePanel):
    """Panel for Michaelis-Menten fitting and inhibition classification."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_controls()
        self._install_plot_canvas()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "酶动力学"

    def _panel_subtitle(self) -> str:
        return "Michaelis-Menten · 抑制判别"

    # ------------------------------------------------------------------
    # Controls (left column)
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        # --- Data file card ---
        file_card = Card("数据文件")
        self._file_selector = FileSelector(
            label="选择酶动力学 CSV 文件",
            file_filter="CSV files (*.csv);;All files (*.*)",
        )
        self._file_selector.file_selected.connect(self._on_file_selected)
        file_card.addWidget(self._file_selector)
        self._controls_layout.addWidget(file_card)

        # --- Actions card ---
        action_card = Card("选择分析任务")

        self._btn_fit_mm = QPushButton("拟合 Michaelis-Menten")
        self._btn_fit_mm.setObjectName("PrimaryButton")
        self._btn_fit_mm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_fit_mm.clicked.connect(self._on_fit_michaelis_menten)
        action_card.addWidget(self._btn_fit_mm)

        self._btn_classify = QPushButton("抑制类型判别")
        self._btn_classify.setObjectName("SecondaryButton")
        self._btn_classify.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_classify.clicked.connect(self._on_classify_inhibition)
        action_card.addWidget(self._btn_classify)

        self._controls_layout.addWidget(action_card)
        self._controls_layout.addStretch()

    def _install_plot_canvas(self) -> None:
        """Create the plot canvas below the standardized results card."""
        self._plot_canvas = PlotCanvas(self, width=8, height=5, dpi=100)
        self._add_plot_widget(self._plot_canvas, "动力学曲线")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _on_file_selected(self, path: str) -> None:
        self._emit_status(f"已选择文件: {path}")

    def _get_csv_path(self) -> str | None:
        path = self._file_selector.get_path()
        if not path:
            self._show_error("请先选择 CSV 数据文件。")
            return None
        return path

    def _render_figure(self, fig: Any) -> None:
        """Display *fig* on the plot canvas, releasing the previous figure."""
        if self._plot_canvas is None:
            return
        previous = self._plot_canvas.figure
        self._plot_canvas.set_figure(fig)
        # Avoid leaking pyplot-managed figures across repeated analyses.
        if previous is not fig:
            try:
                import matplotlib.pyplot as plt

                plt.close(previous)
            except Exception:
                pass

    @staticmethod
    def _format_ci(ci: tuple[float, float] | None) -> str:
        if ci is None:
            return ""
        return f"  (95% CI: {ci[0]:.3g} ~ {ci[1]:.3g})"

    # ------------------------------------------------------------------
    # Action: Michaelis-Menten fitting
    # ------------------------------------------------------------------
    def _on_fit_michaelis_menten(self) -> None:
        path = self._get_csv_path()
        if path is None:
            return

        self._emit_status("正在拟合 Michaelis-Menten 方程...")

        try:
            substrate, velocity, inhibitor = read_kinetics_csv(path)
        except Exception as exc:  # noqa: BLE001 - report any read error to UI
            self._show_error(f"读取 CSV 失败: {exc}")
            return

        try:
            # When inhibitor data is present, fit only the uninhibited
            # control ([I] == 0); otherwise fit the whole series.
            note = ""
            if inhibitor is not None:
                control_mask = np.isclose(inhibitor, 0.0)
                if control_mask.any():
                    s_fit = substrate[control_mask]
                    v_fit = velocity[control_mask]
                    note = "使用对照组 ([I] = 0) 数据进行拟合。"
                else:
                    s_fit = substrate
                    v_fit = velocity
                    note = "未检测到 [I] = 0 的对照组，使用全部数据进行拟合。"
            else:
                s_fit = substrate
                v_fit = velocity

            result = fit_michaelis_menten(s_fit, v_fit)

            # Validation rules.
            km_check = validate_km_range(result.km)
            span_check = validate_substrate_span(s_fit, result.km)

            lines: list[str] = []
            lines.append("Michaelis-Menten 拟合结果")
            lines.append("=" * 42)
            lines.append(f"数据点数: {result.n_points}")
            lines.append(
                f"Vmax = {result.vmax:.4g}{self._format_ci(result.vmax_ci)}"
            )
            lines.append(
                f"Km   = {result.km:.4g}{self._format_ci(result.km_ci)}"
            )
            r_squared = (
                f"{result.r_squared:.4f}"
                if result.r_squared is not None
                else "未定义（观测值无方差）"
            )
            lines.append(f"R²   = {r_squared}")
            for warning in result.warnings:
                lines.append(f"[警告] {warning}")
            if note:
                lines.append("")
                lines.append(f"[数据说明] {note}")
            lines.append("")
            lines.append("验证")
            lines.append("-" * 42)
            km_status = "通过" if km_check.passed else "未通过"
            lines.append(f"ENZ-001 (Km 范围): {km_status} — {km_check.message}")
            span_status = "通过" if span_check.passed else "未通过"
            lines.append(
                f"ENZ-002 (底物跨度): {span_status} — {span_check.message}"
            )

            self._result_display.set_text("\n".join(lines))

            # Plot observed data + fitted curve.
            fig, _ax = plot_michaelis_menten(result, s_fit, v_fit)
            self._render_figure(fig)

            self._emit_status("Michaelis-Menten 拟合完成。")
        except Exception as exc:  # noqa: BLE001 - report fit error to UI
            self._show_error(f"拟合失败: {exc}")

    # ------------------------------------------------------------------
    # Action: Inhibition classification
    # ------------------------------------------------------------------
    def _on_classify_inhibition(self) -> None:
        path = self._get_csv_path()
        if path is None:
            return

        self._emit_status("正在判别抑制类型...")

        try:
            substrate, velocity, inhibitor = read_kinetics_csv(path)
        except Exception as exc:  # noqa: BLE001 - report any read error to UI
            self._show_error(f"读取 CSV 失败: {exc}")
            return

        if inhibitor is None:
            self._show_error(
                "CSV 文件缺少 inhibitor_concentration 列，"
                "无法进行抑制类型判别。"
            )
            return

        try:
            # Group rows by inhibitor concentration.
            unique_concs = np.unique(inhibitor)
            if unique_concs.size < 2:
                self._show_error(
                    "抑制判别需要至少 2 个不同抑制剂浓度，"
                    f"当前仅检测到 {unique_concs.size} 个。"
                )
                return

            datasets: list[tuple[float, np.ndarray, np.ndarray]] = []
            for conc in unique_concs:
                mask = inhibitor == conc
                datasets.append(
                    (float(conc), substrate[mask], velocity[mask])
                )

            result = classify_inhibition(datasets)

            label = _INHIBITION_LABELS.get(
                result.inhibition_type, str(result.inhibition_type.value)
            )

            lines: list[str] = []
            lines.append("抑制类型判别结果")
            lines.append("=" * 42)
            lines.append(f"抑制类型: {label}")
            lines.append(f"置信度:   {result.confidence}")
            lines.append(f"Km 变异系数 (CV):   {result.km_cv:.1%}")
            lines.append(f"Vmax 变异系数 (CV): {result.vmax_cv:.1%}")
            lines.append("")
            lines.append("说明: " + result.message)
            lines.append("")
            lines.append("各抑制剂浓度下的表观参数")
            lines.append("-" * 42)
            for ds in result.datasets:
                mm = ds.mm_result
                r_squared = (
                    f"{mm.r_squared:.4f}"
                    if mm.r_squared is not None
                    else "未定义"
                )
                lines.append(
                    f"[I] = {ds.inhibitor_concentration:g}:  "
                    f"Km_app = {mm.km:.4g},  "
                    f"Vmax_app = {mm.vmax:.4g},  "
                    f"R² = {r_squared}"
                )

            self._result_display.set_text("\n".join(lines))

            # Lineweaver-Burk plot (one line per inhibitor concentration).
            fig, _ax = plot_lineweaver_burk(result.datasets)
            self._render_figure(fig)

            self._emit_status(f"抑制判别完成: {label}。")
        except Exception as exc:  # noqa: BLE001 - report classification error to UI
            self._show_error(f"抑制判别失败: {exc}")
