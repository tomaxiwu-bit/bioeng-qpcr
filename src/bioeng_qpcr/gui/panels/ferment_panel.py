"""Fermentation kinetics analysis panel (Monod fitting + identifiability).

Provides a GUI panel for:
* Loading batch-fermentation CSV data (time, biomass, substrate).
* Fitting Monod parameters (mu_max, Ks, Yxs) with validation.
* Plotting observed vs. fitted growth / consumption curves.
* Analysing practical parameter identifiability via the Fisher
  Information Matrix.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QWidget

from bioeng_qpcr.fermentation_kinetics import (
    FermentationValidationResult,
    IdentifiabilityResult,
    MonodFitResult,
    analyze_identifiability,
    fit_monod,
    read_fermentation_csv,
    validate_data_sufficiency,
    validate_mu_max_range,
)
from bioeng_qpcr.gui.components.background_task import CancellationToken
from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.plotting import plot_fermentation_kinetics

__all__ = ["FermentPanel"]


class FermentPanel(BasePanel):
    """发酵动力学分析面板：Monod 参数拟合与可辨识性分析。

    The left column holds a data-input card (CSV file selector) and an
    action card with two primary buttons.  The right column shows a
    read-only text result display and, below it, an embedded matplotlib
    canvas with the dual y-axis growth / consumption plot.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # State shared between the fit and identifiability actions.
        self._last_fit_result: MonodFitResult | None = None
        self._last_time = None
        self._last_biomass = None
        self._last_substrate = None

    # ------------------------------------------------------------------
    # BasePanel overrides
    # ------------------------------------------------------------------

    def _panel_title(self) -> str:
        return "发酵动力学"

    def _panel_subtitle(self) -> str:
        return "Monod 拟合 · 可辨识性"

    def _build_ui(self) -> None:
        super()._build_ui()

        self._plot_canvas = PlotCanvas(width=7, height=5)
        self._add_plot_widget(self._plot_canvas, "生长与底物曲线")

        # --- Left controls: data input card ---
        input_card = Card("数据输入")
        self._file_selector = FileSelector(label="选择发酵动力学 CSV 文件")
        input_card.addWidget(self._file_selector)
        self._controls_layout.addWidget(input_card)

        # --- Left controls: action buttons card ---
        action_card = Card("选择分析任务")

        self._fit_button = QPushButton("拟合 Monod 参数")
        self._fit_button.setObjectName("PrimaryButton")
        self._fit_button.setToolTip(
            "从 CSV 数据拟合 μmax、Ks、Yxs 参数并绘制拟合曲线"
        )
        self._fit_button.clicked.connect(self._on_fit_monod)
        action_card.addWidget(self._fit_button)

        self._ident_button = QPushButton("可辨识性分析")
        self._ident_button.setObjectName("SecondaryButton")
        self._ident_button.setToolTip(
            "基于 Fisher 信息矩阵分析参数可辨识性（需先拟合）"
        )
        self._ident_button.clicked.connect(self._on_identifiability)
        action_card.addWidget(self._ident_button)

        self._cancel_button = QPushButton("取消当前任务")
        self._cancel_button.setObjectName("QuietButton")
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip(
            "发出取消请求；正在执行的数值求解步骤结束后停止并丢弃结果"
        )
        self._cancel_button.clicked.connect(self._cancel_background_task)
        action_card.addWidget(self._cancel_button)

        self._controls_layout.addWidget(action_card)
        self._controls_layout.addStretch()

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_fit_monod(self) -> None:
        """Start CSV validation and Monod fitting in a worker thread."""
        csv_path = self._file_selector.get_path()
        if not csv_path:
            self._show_error("请先选择 CSV 数据文件。")
            return

        # Do not let a failed replacement fit leave an older dataset looking
        # current and available for identifiability analysis.
        self._last_fit_result = None
        self._last_time = None
        self._last_biomass = None
        self._last_substrate = None

        def operation(token: CancellationToken):
            token.report("正在读取发酵动力学数据...")
            time, biomass, substrate = read_fermentation_csv(csv_path)
            token.raise_if_cancelled()
            ds_check = validate_data_sufficiency(time, biomass, substrate)
            if not ds_check.passed:
                raise ValueError(
                    f"数据校验失败 ({ds_check.rule_id}): {ds_check.message}"
                )
            token.report("正在拟合 Monod 参数...")
            result = fit_monod(time, biomass, substrate)
            token.raise_if_cancelled()
            mu_check = validate_mu_max_range(result.mu_max)
            return result, time, biomass, substrate, mu_check, ds_check

        self._start_background_task(
            operation,
            self._complete_monod_fit,
            busy_widgets=(
                self._file_selector,
                self._fit_button,
                self._ident_button,
            ),
            cancel_widget=self._cancel_button,
            status="正在准备 Monod 拟合...",
            error_prefix="Monod 拟合失败",
        )

    def _complete_monod_fit(self, payload) -> None:
        """Store and render a successful worker-thread Monod fit."""
        result, time, biomass, substrate, mu_check, ds_check = payload

        self._last_fit_result = result
        self._last_time = time
        self._last_biomass = biomass
        self._last_substrate = substrate
        self._result_display.set_text(
            self._format_fit_result(result, mu_check, ds_check)
        )

        try:
            fig, _ = plot_fermentation_kinetics(
                result, time, biomass, substrate
            )
            self._plot_canvas.set_figure(fig)
        except Exception as exc:
            self._append_result(f"\n[警告] 绘图失败: {exc}")
        self._emit_status("Monod 拟合完成")

    def _on_identifiability(self) -> None:
        """Analyse parameter identifiability from the last fit result."""
        if self._last_fit_result is None:
            self._show_error("请先执行 Monod 参数拟合。")
            return

        fit_result = self._last_fit_result
        time = self._last_time
        biomass = self._last_biomass
        substrate = self._last_substrate

        def operation(token: CancellationToken):
            token.report("正在计算 Fisher 信息矩阵...")
            result = analyze_identifiability(
                fit_result,
                time,
                biomass,
                substrate,
            )
            token.raise_if_cancelled()
            return result

        self._start_background_task(
            operation,
            self._complete_identifiability,
            busy_widgets=(
                self._file_selector,
                self._fit_button,
                self._ident_button,
            ),
            cancel_widget=self._cancel_button,
            status="正在进行可辨识性分析...",
            error_prefix="可辨识性分析失败",
        )

    def _complete_identifiability(self, result: IdentifiabilityResult) -> None:
        self._result_display.set_text(
            self._format_identifiability_result(result)
        )
        self._emit_status("可辨识性分析完成")

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_fit_result(
        result: MonodFitResult,
        mu_check: FermentationValidationResult,
        ds_check: FermentationValidationResult,
    ) -> str:
        """Format Monod fit results and validation checks for display."""
        sep = "=" * 54
        lines: list[str] = [
            sep,
            "  Monod 参数拟合结果",
            sep,
            "",
            f"  μmax  = {result.mu_max:.4f} h⁻¹",
        ]
        if result.mu_max_ci:
            lines.append(
                f"        95% CI: [{result.mu_max_ci[0]:.4f}, "
                f"{result.mu_max_ci[1]:.4f}]"
            )
        lines.append(f"  Ks    = {result.ks:.4f} g/L")
        if result.ks_ci:
            lines.append(
                f"        95% CI: [{result.ks_ci[0]:.4f}, "
                f"{result.ks_ci[1]:.4f}]"
            )
        lines.append(f"  Yxs   = {result.yxs:.4f} g/g")
        if result.yxs_ci:
            lines.append(
                f"        95% CI: [{result.yxs_ci[0]:.4f}, "
                f"{result.yxs_ci[1]:.4f}]"
            )
        lines.extend(
            [
                "  R²    = "
                + (
                    f"{result.r_squared:.4f}"
                    if result.r_squared is not None
                    else "未定义（观测响应无方差）"
                ),
                f"  数据点数 = {result.n_points}",
                f"  X0    = {result.x0:.4f} g/L",
                f"  S0    = {result.s0:.4f} g/L",
                "",
                "-" * 54,
                "  校验结果",
                "-" * 54,
            ]
        )
        lines.extend(f"  [警告] {warning}" for warning in result.warnings)
        mu_status = "通过" if mu_check.passed else "未通过"
        lines.append(f"  {mu_check.rule_id} (μmax 范围) [{mu_status}]")
        lines.append(f"    {mu_check.message}")
        ds_status = "通过" if ds_check.passed else "未通过"
        lines.append(f"  {ds_check.rule_id} (数据充分性) [{ds_status}]")
        lines.append(f"    {ds_check.message}")
        return "\n".join(lines)

    @staticmethod
    def _format_identifiability_result(
        id_result: IdentifiabilityResult,
    ) -> str:
        """Format identifiability analysis results for display."""
        sep = "=" * 54
        lines: list[str] = [
            sep,
            "  参数可辨识性分析",
            sep,
            "",
        ]
        ident_label = "可辨识" if id_result.is_identifiable else "不可辨识"
        lines.append(f"  结论: {ident_label} (置信度: {id_result.confidence})")
        lines.append(f"  条件数: {id_result.condition_number:.2f}")
        lines.append("")
        lines.append(f"  {id_result.message}")
        lines.append("")

        # Correlation matrix
        lines.extend(
            [
                "-" * 54,
                "  参数相关性矩阵",
                "-" * 54,
            ]
        )
        names = id_result.parameter_names
        # Align header with the row layout: "  {name:<8s} " = 11 chars.
        header = f"  {'':<8s} " + "  ".join(f"{n:>10s}" for n in names)
        lines.append(header)
        corr = id_result.correlation_matrix
        for i, name in enumerate(names):
            row_vals = "  ".join(
                f"{corr[i][j]:>10.4f}" for j in range(len(names))
            )
            lines.append(f"  {name:<8s} {row_vals}")
        lines.append("")

        # FIM eigenvalues
        lines.extend(
            [
                "-" * 54,
                "  FIM 特征值",
                "-" * 54,
            ]
        )
        for idx, ev in enumerate(id_result.eigenvalues):
            lines.append(f"  λ{idx + 1} = {ev:.6e}")

        return "\n".join(lines)
