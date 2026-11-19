"""Enzyme kinetics analysis panel.

Provides a GUI for:

* Fitting the Michaelis-Menten equation ``V = Vmax * [S] / (Km + [S])`` to
  substrate-velocity data, with Km-range and substrate-span validation.
* Classifying the reversible inhibition mechanism (competitive,
  non-competitive, uncompetitive, mixed, none) from multi-inhibitor
  datasets via a Lineweaver-Burk plot.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QWidget

from bioeng_qpcr.enzyme_kinetics import (
    InhibitionType,
    classify_inhibition,
    compare_inhibition_models,
    fit_michaelis_menten,
    read_kinetics_csv,
    validate_km_range,
    validate_substrate_span,
)
from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.plotting import plot_lineweaver_burk, plot_michaelis_menten
from bioeng_qpcr.plotting.enzyme_kinetics import plot_global_inhibition

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

        self._btn_global = QPushButton("全局拟合与模型比较")
        self._btn_global.setObjectName("PrimaryButton")
        self._btn_global.clicked.connect(self._on_global_inhibition)
        action_card.addWidget(self._btn_global)

        self._btn_classify = QPushButton("参数趋势判别（辅助）")
        self._btn_classify.setObjectName("SecondaryButton")
        self._btn_classify.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_classify.clicked.connect(self._on_classify_inhibition)
        action_card.addWidget(self._btn_classify)

        self._btn_cancel = QPushButton("取消当前任务")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setToolTip("当前模型比较结束后停止，结果不会写入界面")
        self._btn_cancel.clicked.connect(self._cancel_background_task)
        action_card.addWidget(self._btn_cancel)

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
            with contextlib.suppress(Exception):
                import matplotlib.pyplot as plt

                plt.close(previous)

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
        except Exception as exc:
            self._show_error(f"读取 CSV 失败: {exc}")
            return

        try:
            # When inhibitor data is present, fit only the uninhibited
            # control ([I] == 0); otherwise fit the whole series.
            note = ""
            if inhibitor is not None:
                control_mask = inhibitor == 0.0
                if control_mask.any():
                    s_fit = substrate[control_mask]
                    v_fit = velocity[control_mask]
                    note = "使用对照组 ([I] = 0) 数据进行拟合。"
                else:
                    if np.unique(inhibitor).size > 1:
                        raise ValueError(
                            "存在多个抑制剂浓度但缺少 [I] = 0 对照，"
                            "不能混合拟合一条曲线。"
                            "请按抑制剂浓度分别导入，并补充 [I] = 0 对照。"
                        )
                    s_fit = substrate
                    v_fit = velocity
                    note = (
                        f"仅拟合 [I] = {float(inhibitor[0]):g} 组的表观参数，"
                        "不代表无抑制剂的 Km / Vmax。"
                    )
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
            lines.append(f"Vmax = {result.vmax:.4g}{self._format_ci(result.vmax_ci)}")
            lines.append(f"Km   = {result.km:.4g}{self._format_ci(result.km_ci)}")
            r_squared = (
                f"{result.r_squared:.4f}"
                if result.r_squared is not None
                else "未定义（观测值无方差）"
            )
            lines.append(f"R²   = {r_squared}")
            lines.extend(f"[警告] {warning}" for warning in result.warnings)
            if note:
                lines.append("")
                lines.append(f"[数据说明] {note}")
            lines.append("")
            lines.append("验证")
            lines.append("-" * 42)
            km_status = "通过" if km_check.passed else "未通过"
            lines.append(f"ENZ-001 (Km 范围): {km_status} — {km_check.message}")
            span_status = "通过" if span_check.passed else "未通过"
            lines.append(f"ENZ-002 (底物跨度): {span_status} — {span_check.message}")

            self._result_display.set_text("\n".join(lines))

            # Plot observed data + fitted curve.
            fig, _ax = plot_michaelis_menten(result, s_fit, v_fit)
            self._render_figure(fig)

            self._emit_status("Michaelis-Menten 拟合完成。")
        except Exception as exc:
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
        except Exception as exc:
            self._show_error(f"读取 CSV 失败: {exc}")
            return

        if inhibitor is None:
            self._show_error(
                "CSV 文件缺少 inhibitor_concentration 列，无法进行抑制类型判别。"
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
                datasets.append((float(conc), substrate[mask], velocity[mask]))

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
                    f"{mm.r_squared:.4f}" if mm.r_squared is not None else "未定义"
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
        except Exception as exc:
            self._show_error(f"抑制判别失败: {exc}")

    def _on_global_inhibition(self) -> None:
        path = self._get_csv_path()
        if path is None:
            return

        def operation(token):
            substrate, velocity, inhibitor = read_kinetics_csv(path)
            if inhibitor is None:
                raise ValueError("需要 inhibitor_concentration 列")
            datasets = [
                (float(c), substrate[inhibitor == c], velocity[inhibitor == c])
                for c in np.unique(inhibitor)
            ]
            token.raise_if_cancelled()
            result = compare_inhibition_models(datasets)
            token.raise_if_cancelled()
            return result, datasets

        self._start_background_task(
            operation,
            self._complete_global_inhibition,
            busy_widgets=(
                self._file_selector,
                self._btn_fit_mm,
                self._btn_classify,
                self._btn_global,
            ),
            cancel_widget=self._btn_cancel,
            status="正在进行全局抑制模型比较…",
            error_prefix="全局拟合失败",
        )

    def _complete_global_inhibition(self, payload) -> None:
        result, datasets = payload
        best = result.best_fit
        ambiguous = (
            result.models[1].delta_aicc < 2
            or best.akaike_weight < 0.7
            or any(not m.converged or not np.isfinite(m.aicc) for m in result.models)
        )
        lines = [
            "全局抑制模型比较",
            "=" * 60,
            "机制未决：候选模型证据接近，请补充数据。"
            if ambiguous
            else "当前数据支持的候选机制：" + _INHIBITION_LABELS[result.best_type],
            "Akaike 权重表示候选模型间相对支持，不是机制正确的概率。",
            "模型          AICc       ΔAICc      BIC       权重    收敛",
        ]
        for fit in result.models:
            lines.append(
                f"{fit.inhibition_type.value:16} {fit.aicc:9.3g} "
                f"{fit.delta_aicc:9.3g} {fit.bic:9.3g} "
                f"{fit.akaike_weight:7.3f} {fit.converged}"
            )
        lines.extend(["", "最优候选参数（标准误不等于置信区间）"])
        for name, value in best.parameters.items():
            error = best.parameter_std[name]
            lines.append(
                f"{name} = {value:.6g}; SE = "
                + (f"{error:.4g}" if np.isfinite(error) else "不可估计")
            )
        if not any(c == 0 for c, _, _ in datasets):
            lines.append("[警告] 缺少 [I] = 0 对照，基础参数解释需额外审阅。")
        lines.extend(["", result.message])
        self._result_display.set_text("\n".join(lines))
        figure, _ = plot_global_inhibition(best, datasets)
        self._render_figure(figure)
        self._emit_status("全局模型比较完成；请审阅参数、残差与模型证据")
