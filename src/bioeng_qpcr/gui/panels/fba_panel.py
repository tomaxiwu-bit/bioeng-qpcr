"""FBA metabolic flux analysis panel.

Provides a GUI for Flux Balance Analysis (FBA), blocked-reaction
detection, and dead-end metabolite identification using the
:mod:`bioeng_qpcr.metabolic_flux` module, which depends on the optional
``cobra`` (COBRApy) package.

When COBRApy is not installed the panel still loads (so the rest of the
application remains usable) and shows an informative warning whenever an
action button is clicked.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from bioeng_qpcr.gui.components.background_task import CancellationToken
from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas

# ---------------------------------------------------------------------------
# Optional cobra-dependent imports
# ---------------------------------------------------------------------------
# COBRApy is imported lazily inside the metabolic_flux functions, but we
# probe for it at module level so we can warn the user *before* attempting
# any computation.  The whole import block is wrapped in try/except so the
# panel can always be instantiated even when cobra is missing.
try:
    import cobra  # noqa: F401  -- presence probe only

    from bioeng_qpcr.metabolic_flux import (
        find_blocked_reactions,
        find_dead_end_metabolites,
        load_model,
        run_fba,
        validate_mass_balance,
        validate_objective_feasibility,
    )
    from bioeng_qpcr.plotting import plot_flux_distribution

    _COBRA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without cobra
    _COBRA_AVAILABLE = False


# Number of top-flux reactions to show in text and plot.
_TOP_N = 15


class FbaPanel(BasePanel):
    """Panel for Flux Balance Analysis and metabolic network analysis.

    The left side holds model-source, optimisation-parameter and action
    controls.  The right side shows a textual result display with a
    matplotlib canvas beneath it for the flux-distribution bar chart.
    """

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "FBA 代谢通量"

    def _panel_subtitle(self) -> str:
        return "FBA 求解 · 网络分析"

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Cached state -- initialised before super()._build_ui() so they
        # always exist regardless of import outcome.
        self._model = None
        self._loaded_source: str | None = None
        self._last_fba_result = None

        super()._build_ui()
        self._build_controls()
        self._attach_plot_canvas()

    def _build_controls(self) -> None:
        """Populate the left-hand controls column with cards."""
        layout = self._controls_layout

        # --- Card 1: model source ------------------------------------
        source_card = Card("模型来源")

        source_lbl = QLabel("模型文件或内置名称")
        source_lbl.setObjectName("FieldLabel")
        source_card.addWidget(source_lbl)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText(
            "输入 SBML/XML 文件路径或内置模型名（如 textbook）"
        )
        self._browse_btn = QPushButton("选择文件")
        self._browse_btn.setObjectName("SecondaryButton")
        self._browse_btn.setCursor(_pointing_hand_cursor())
        self._browse_btn.clicked.connect(self._on_browse)
        source_row.addWidget(self._source_edit)
        source_row.addWidget(self._browse_btn)
        source_card.addLayout(source_row)

        layout.addWidget(source_card)

        # --- Card 2: optimisation parameters --------------------------
        param_card = Card("优化参数")

        obj_lbl = QLabel("目标反应（可选）")
        obj_lbl.setObjectName("FieldLabel")
        param_card.addWidget(obj_lbl)

        self._objective_edit = QLineEdit()
        self._objective_edit.setPlaceholderText(
            "目标反应 ID（留空使用模型默认）"
        )
        param_card.addWidget(self._objective_edit)

        dir_lbl = QLabel("优化方向")
        dir_lbl.setObjectName("FieldLabel")
        param_card.addWidget(dir_lbl)

        self._direction_combo = QComboBox()
        self._direction_combo.addItem("maximize", "max")
        self._direction_combo.addItem("minimize", "min")
        param_card.addWidget(self._direction_combo)

        layout.addWidget(param_card)

        # --- Card 3: actions ------------------------------------------
        action_card = Card("网络分析")

        self._run_fba_btn = QPushButton("运行 FBA")
        self._run_fba_btn.setObjectName("PrimaryButton")
        self._run_fba_btn.setCursor(_pointing_hand_cursor())
        self._run_fba_btn.clicked.connect(self._on_run_fba)
        action_card.addWidget(self._run_fba_btn)

        self._blocked_btn = QPushButton("检测阻塞反应")
        self._blocked_btn.setObjectName("SecondaryButton")
        self._blocked_btn.setCursor(_pointing_hand_cursor())
        self._blocked_btn.clicked.connect(self._on_blocked)
        action_card.addWidget(self._blocked_btn)

        self._dead_end_btn = QPushButton("检测死端代谢物")
        self._dead_end_btn.setObjectName("SecondaryButton")
        self._dead_end_btn.setCursor(_pointing_hand_cursor())
        self._dead_end_btn.clicked.connect(self._on_dead_end)
        action_card.addWidget(self._dead_end_btn)

        self._cancel_btn = QPushButton("取消当前任务")
        self._cancel_btn.setObjectName("QuietButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setToolTip(
            "发出取消请求；当前求解步骤结束后停止并丢弃结果"
        )
        self._cancel_btn.clicked.connect(self._cancel_background_task)
        action_card.addWidget(self._cancel_btn)

        layout.addWidget(action_card)
        layout.addStretch()

    def _attach_plot_canvas(self) -> None:
        """Create the plot canvas and add it below the result display."""
        self._plot_canvas = PlotCanvas(self, width=8, height=5)
        self._add_plot_widget(self._plot_canvas, "通量分布")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        """Open a file dialog to pick an SBML model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "SBML files (*.xml *.sbml);;All files (*.*)",
        )
        if path:
            self._source_edit.setText(path)

    def _on_run_fba(self) -> None:
        """Load and solve the model without blocking the Qt event loop."""
        if not _COBRA_AVAILABLE:
            self._show_cobra_warning()
            return
        source = self._source_edit.text().strip()
        if not source:
            self._show_error("请输入模型来源（文件路径或内置模型名）")
            return
        objective = self._objective_edit.text().strip() or None
        direction = self._direction_combo.currentData()
        cached_model = self._model if self._loaded_source == source else None

        def operation(token: CancellationToken):
            model = cached_model
            if model is None:
                token.report(f"正在加载模型: {source} ...")
                model = load_model(source)
            token.report("正在运行 FBA 求解...")
            result = run_fba(
                model, objective=objective, direction=direction
            )
            token.raise_if_cancelled()
            token.report("正在验证 FBA 解...")
            feasibility = validate_objective_feasibility(result)
            mass_balance = validate_mass_balance(model, result)
            token.raise_if_cancelled()
            return model, source, result, feasibility, mass_balance

        self._start_fba_background_task(
            operation,
            self._complete_fba,
            status="正在准备 FBA 求解...",
            error_prefix="FBA 求解失败",
        )

    def _complete_fba(self, payload) -> None:
        model, source, result, feasibility, mass_balance = payload
        self._cache_model(model, source)
        self._last_fba_result = result
        self._display_fba_result(result, feasibility, mass_balance)
        self._plot_fba(result)
        self._emit_status("FBA 求解完成")

    def _on_blocked(self) -> None:
        """Detect and display blocked reactions."""
        if not _COBRA_AVAILABLE:
            self._show_cobra_warning()
            return
        source = self._source_edit.text().strip()
        if not source:
            self._show_error("请输入模型来源（文件路径或内置模型名）")
            return
        cached_model = self._model if self._loaded_source == source else None

        def operation(token: CancellationToken):
            model = cached_model
            if model is None:
                token.report(f"正在加载模型: {source} ...")
                model = load_model(source)
            token.report("正在检测阻塞反应...")
            result = find_blocked_reactions(model)
            token.raise_if_cancelled()
            return model, source, result

        self._start_fba_background_task(
            operation,
            self._complete_blocked,
            status="正在准备阻塞反应检测...",
            error_prefix="阻塞反应检测失败",
        )

    def _complete_blocked(self, payload) -> None:
        model, source, result = payload
        self._cache_model(model, source)
        lines = [
            "=" * 55,
            "阻塞反应检测结果",
            "=" * 55,
            f"总反应数:     {result.n_total}",
            f"阻塞反应数:   {result.n_blocked}",
            f"阻塞比例:     {result.fraction_blocked:.2%}",
            "",
        ]
        if result.blocked_reactions:
            lines.append("阻塞反应列表:")
            lines.extend(f"  - {rxn_id}" for rxn_id in result.blocked_reactions)
        else:
            lines.append("未发现阻塞反应。")
        self._result_display.set_text("\n".join(lines))
        self._clear_plot()
        self._emit_status("阻塞反应检测完成")

    def _on_dead_end(self) -> None:
        """Detect and display dead-end metabolites."""
        if not _COBRA_AVAILABLE:
            self._show_cobra_warning()
            return
        source = self._source_edit.text().strip()
        if not source:
            self._show_error("请输入模型来源（文件路径或内置模型名）")
            return
        cached_model = self._model if self._loaded_source == source else None

        def operation(token: CancellationToken):
            model = cached_model
            if model is None:
                token.report(f"正在加载模型: {source} ...")
                model = load_model(source)
            token.report("正在检测死端代谢物...")
            result = find_dead_end_metabolites(model)
            token.raise_if_cancelled()
            return model, source, result

        self._start_fba_background_task(
            operation,
            self._complete_dead_end,
            status="正在准备死端代谢物检测...",
            error_prefix="死端代谢物检测失败",
        )

    def _complete_dead_end(self, payload) -> None:
        model, source, result = payload
        self._cache_model(model, source)
        sinks = [
            mid
            for mid, info in result.details.items()
            if info.get("type") == "sink"
        ]
        sources = [
            mid
            for mid, info in result.details.items()
            if info.get("type") == "source"
        ]
        isolated = [
            mid
            for mid, info in result.details.items()
            if info.get("type") == "isolated"
        ]
        lines = [
            "=" * 55,
            "死端代谢物检测结果",
            "=" * 55,
            f"总代谢物数:       {result.n_total}",
            f"死端代谢物数:     {result.n_dead_ends}",
            f"  Sink（仅消耗）:  {len(sinks)}",
            f"  Source（仅生成）:{len(sources)}",
            f"  孤立代谢物:      {len(isolated)}",
        ]
        for heading, identifiers in (
            ("Sink 代谢物（仅被消耗，无生成反应）:", sinks),
            ("Source 代谢物（仅被生成，无消耗反应）:", sources),
            ("孤立代谢物（无内部反应参与）:", isolated),
        ):
            if identifiers:
                lines.extend(["", heading])
                lines.extend(f"  - {identifier}" for identifier in identifiers)
        if not result.dead_end_ids:
            lines.extend(["", "未发现死端代谢物。"])
        self._result_display.set_text("\n".join(lines))
        self._clear_plot()
        self._emit_status("死端代谢物检测完成")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cache_model(self, model, source: str) -> None:
        """Publish a worker-loaded model after successful task completion."""
        if self._loaded_source != source:
            self._last_fba_result = None
        self._model = model
        self._loaded_source = source

    def _start_fba_background_task(
        self,
        operation,
        on_success,
        *,
        status: str,
        error_prefix: str,
    ) -> None:
        self._start_background_task(
            operation,
            on_success,
            busy_widgets=(
                self._source_edit,
                self._browse_btn,
                self._objective_edit,
                self._direction_combo,
                self._run_fba_btn,
                self._blocked_btn,
                self._dead_end_btn,
            ),
            cancel_widget=self._cancel_btn,
            status=status,
            error_prefix=error_prefix,
        )

    def _show_cobra_warning(self) -> None:
        """Inform the user that COBRApy is required."""
        message = (
            "COBRApy 未安装，无法进行代谢通量分析。\n\n"
            "请安装可选依赖：\n"
            "    pip install cobra\n\n"
            "安装后重启应用程序即可使用 FBA 相关功能。"
        )
        self._result_display.set_text(message)
        self._clear_plot()
        self._emit_status("警告: COBRApy 未安装")

    def _display_fba_result(self, result, feasibility, mass_balance) -> None:
        """Render the FBA solution, validation and top fluxes as text."""
        lines: list[str] = []
        lines.append("=" * 55)
        lines.append("FBA 求解结果")
        lines.append("=" * 55)
        lines.append(f"求解状态:     {result.status}")
        lines.append(f"目标函数值:   {result.objective_value:.6f}")

        # Model summary ------------------------------------------------
        summary = result.model_summary
        if summary is not None:
            compartments = (
                ", ".join(summary.compartments) if summary.compartments else "无"
            )
            lines.append("")
            lines.append("-" * 55)
            lines.append("模型概览")
            lines.append("-" * 55)
            lines.append(f"模型 ID:      {summary.model_id}")
            lines.append(f"反应数:       {summary.n_reactions}")
            lines.append(f"代谢物数:     {summary.n_metabolites}")
            lines.append(f"基因数:       {summary.n_genes}")
            lines.append(f"区室:         {compartments}")
            lines.append(f"优化方向:     {summary.objective_direction}")

        # Active reactions & top fluxes --------------------------------
        lines.append("")
        lines.append("-" * 55)
        lines.append(f"活跃反应: {result.n_active} / {result.n_total}")
        lines.append("-" * 55)

        sorted_fluxes = sorted(
            result.fluxes.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        top = sorted_fluxes[:_TOP_N]
        lines.append("")
        lines.append(f"通量排名前 {_TOP_N} 的反应:")
        for rxn_id, flux in top:
            lines.append(f"  {rxn_id:<32s} {flux:>12.6f}")

        # Validation ---------------------------------------------------
        lines.append("")
        lines.append("-" * 55)
        lines.append("验证")
        lines.append("-" * 55)

        feas_flag = "通过" if feasibility.passed else "未通过"
        lines.append(f"[{feasibility.rule_id}] 目标可行性: {feas_flag}")
        lines.append(f"  {feasibility.message}")

        mb_flag = "通过" if mass_balance.passed else "未通过"
        lines.append(f"[{mass_balance.rule_id}] 质量平衡: {mb_flag}")
        lines.append(f"  {mass_balance.message}")

        self._result_display.set_text("\n".join(lines))

    def _plot_fba(self, result) -> None:
        """Draw the flux-distribution bar chart on the canvas."""
        if self._plot_canvas is None:
            return
        fig, _ax = plot_flux_distribution(result, top_n=_TOP_N)
        self._plot_canvas.set_figure(fig)

    def _clear_plot(self) -> None:
        """Clear the plot canvas if it exists."""
        if self._plot_canvas is not None:
            self._plot_canvas.clear()


# ---------------------------------------------------------------------------
# Small utility helpers (kept module-private)
# ---------------------------------------------------------------------------
def _pointing_hand_cursor():
    """Return the pointing-hand cursor enum value (avoids a verbose import)."""
    from PySide6.QtCore import Qt

    return Qt.CursorShape.PointingHandCursor
