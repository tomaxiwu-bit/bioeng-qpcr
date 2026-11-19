"""Molecular docking panel.

Runs rigid-body docking of a ligand against a receptor and reports the ranked
poses with their energy decomposition and the DCK audit rules.  Lower score is
better; every number shown comes from the same scoring engine the CLI and the
web interface use, so a result can be reproduced from any of them.
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

from bioeng_qpcr.docking import (
    SCORE_INTERPRETATION,
    DockingBox,
    DockResult,
    identify_receptor_ligand,
    parse_pdb,
    summarize_docking,
    validate_docking,
)
from bioeng_qpcr.docking.engine import dock
from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.plotting import plot_pose_ranking


class DockingPanel(BasePanel):
    """Panel for rigid-body ligand docking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: DockResult | None = None
        self._build_controls()
        self._install_plot_canvas()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "分子对接"

    def _panel_subtitle(self) -> str:
        return "刚性对接 · 打分分解 · 采样审计"

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        file_card = Card("结构文件")
        self._receptor_selector = FileSelector(
            label="受体文件（蛋白 PDB）",
            file_filter="PDB files (*.pdb *.ent);;All files (*.*)",
        )
        file_card.addWidget(self._receptor_selector)

        self._ligand_selector = FileSelector(
            label="配体文件（小分子 PDB）",
            file_filter="PDB files (*.pdb *.ent);;All files (*.*)",
        )
        file_card.addWidget(self._ligand_selector)

        self._split_check = QCheckBox("从受体文件中提取配体（忽略配体文件）")
        self._split_check.toggled.connect(
            lambda checked: self._ligand_selector.setEnabled(not checked)
        )
        file_card.addWidget(self._split_check)
        self._controls_layout.addWidget(file_card)

        settings = Card("搜索设置")

        self._n_runs_spin = QSpinBox()
        self._n_runs_spin.setRange(1, 50)
        self._n_runs_spin.setValue(4)
        settings.addWidget(self._spin_row("独立运行次数", self._n_runs_spin))

        self._n_steps_spin = QSpinBox()
        self._n_steps_spin.setRange(50, 20000)
        self._n_steps_spin.setSingleStep(100)
        self._n_steps_spin.setValue(1500)
        settings.addWidget(self._spin_row("每次运行步数", self._n_steps_spin))

        self._top_spin = QSpinBox()
        self._top_spin.setRange(1, 50)
        self._top_spin.setValue(5)
        settings.addWidget(self._spin_row("输出姿势数", self._top_spin))

        self._rmsd_spin = QDoubleSpinBox()
        self._rmsd_spin.setRange(0.1, 20.0)
        self._rmsd_spin.setSingleStep(0.5)
        self._rmsd_spin.setValue(2.0)
        self._rmsd_spin.setDecimals(1)
        settings.addWidget(self._spin_row("姿势去重 RMSD (Å)", self._rmsd_spin))

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 10**6)
        self._seed_spin.setValue(1)
        settings.addWidget(self._spin_row("随机种子", self._seed_spin))

        self._box_combo = QComboBox()
        self._box_combo.addItems(["自动（按配体位置）", "手动中心与半径"])
        settings.addWidget(self._spin_row("搜索盒", self._box_combo))

        self._center_edit = QComboBox()
        self._center_edit.setEditable(True)
        self._center_edit.addItem("0,0,0")
        settings.addWidget(self._spin_row("搜索盒中心 x,y,z", self._center_edit))

        self._half_spin = QDoubleSpinBox()
        self._half_spin.setRange(1.0, 100.0)
        self._half_spin.setValue(10.0)
        self._half_spin.setDecimals(1)
        settings.addWidget(self._spin_row("搜索盒半径 (Å)", self._half_spin))

        self._grid_check = QCheckBox("使用确定性网格搜索（可完全复现）")
        settings.addWidget(self._grid_check)

        self._validate_check = QCheckBox("运行 DCK 审计规则")
        self._validate_check.setChecked(True)
        settings.addWidget(self._validate_check)

        self._controls_layout.addWidget(settings)

        action_card = Card("执行")
        self._btn_dock = QPushButton("开始对接")
        self._btn_dock.setObjectName("PrimaryButton")
        self._btn_dock.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_dock.clicked.connect(self._on_dock)
        action_card.addWidget(self._btn_dock)
        self._controls_layout.addWidget(action_card)
        self._controls_layout.addStretch()

    @staticmethod
    def _spin_row(text: str, widget: QWidget) -> QWidget:
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

    def _install_plot_canvas(self) -> None:
        self._plot_canvas = PlotCanvas(self, width=8, height=5, dpi=100)
        self._add_plot_widget(self._plot_canvas, "打分分解")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_figure(self, fig: Any) -> None:
        if self._plot_canvas is None:
            return
        previous = self._plot_canvas.figure
        self._plot_canvas.set_figure(fig)
        if previous is not fig:
            with contextlib.suppress(Exception):
                import matplotlib.pyplot as plt

                plt.close(previous)

    def _build_box(self) -> DockingBox | None:
        if self._box_combo.currentIndex() == 0:
            return None
        import numpy as np

        raw = self._center_edit.currentText().replace("，", ",")
        try:
            components = [float(part) for part in raw.split(",")]
        except ValueError:
            self._show_error("搜索盒中心需要三个逗号分隔的数字，例如 12.5,-3,0。")
            return None
        if len(components) != 3:
            self._show_error("搜索盒中心需要恰好三个数字。")
            return None
        return DockingBox(
            center=np.array(components, dtype=float),
            half_size=np.full(3, float(self._half_spin.value())),
        )

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _on_dock(self) -> None:
        receptor_path = self._receptor_selector.get_path()
        if not receptor_path:
            self._show_error("请先选择受体 PDB 文件。")
            return

        self._emit_status("正在对接...")

        try:
            if self._split_check.isChecked():
                receptor, ligand = identify_receptor_ligand(parse_pdb(receptor_path))
            else:
                ligand_path = self._ligand_selector.get_path()
                if not ligand_path:
                    self._show_error("请选择配体 PDB 文件，或勾选从受体文件提取。")
                    return
                receptor = parse_pdb(receptor_path)
                ligand = parse_pdb(ligand_path)
        except Exception as exc:
            self._show_error(f"读取结构失败: {exc}")
            return

        box = self._build_box()
        if self._box_combo.currentIndex() == 1 and box is None:
            return

        try:
            result = dock(
                receptor,
                ligand,
                box,
                n_runs=self._n_runs_spin.value(),
                seed=self._seed_spin.value(),
                n_steps=self._n_steps_spin.value(),
                use_grid=self._grid_check.isChecked(),
                top_n=self._top_spin.value(),
                rmsd_threshold=self._rmsd_spin.value(),
            )
        except Exception as exc:
            self._show_error(f"对接失败: {exc}")
            return

        self._result = result

        lines: list[str] = []
        lines.append("对接结果")
        lines.append("=" * 46)
        lines.append(
            f"受体 {result.receptor_meta.n_atoms} 原子 / "
            f"{result.receptor_meta.n_residues} 残基"
        )
        lines.append(f"配体 {result.ligand_meta.n_atoms} 原子")
        lines.append(f"打分函数 {result.scoring_name}")
        lines.append("搜索盒中心 " + ", ".join(f"{v:.2f}" for v in result.box.center))
        lines.append(f"搜索盒半径 {float(result.box.half_size[0]):.2f} Å")
        lines.append(
            f"采样 {result.stats.iterations} 步，接受 {result.stats.accepted} 次 "
            f"（{result.stats.acceptance_rate:.1%}），种子 {result.stats.seed}"
        )
        lines.append(result.stats.message)
        lines.append("")
        lines.append("排名   总分      vdW     静电    氢键    疏水   去溶剂   RMSD")
        lines.append("-" * 66)
        for index, pose in enumerate(result.poses):
            deviation = 0.0 if index == 0 else pose.rmsd_to(result.poses[0])
            lines.append(
                f"{index + 1:>3}  {pose.score:8.3f} "
                f"{pose.breakdown.vdw:8.3f} {pose.breakdown.electrostatic:7.3f} "
                f"{pose.breakdown.hbond:7.3f} {pose.breakdown.hydrophobic:7.3f} "
                f"{pose.breakdown.desolvation:7.3f} {deviation:6.2f}"
            )
        lines.append("")
        lines.append(SCORE_INTERPRETATION)

        checks = []
        if self._validate_check.isChecked():
            try:
                checks = validate_docking(result, receptor, ligand)
            except Exception as exc:
                lines.append(f"[错误] 审计规则执行失败: {exc}")
                checks = []
            lines.append("")
            lines.append("审计规则")
            lines.append("-" * 46)
            for check in checks:
                severity = str(getattr(check.severity, "value", check.severity))
                tag = {
                    "error": "[错误]",
                    "warning": "[警告]",
                }.get(severity, "[提示]")
                status = "通过" if check.passed else "未通过"
                lines.append(f"{tag} {check.rule_id} {status} — {check.message}")

        review = summarize_docking(result, checks)
        lines[0:0] = [review["summary"], review["interpretation"], ""]
        lines.extend(["", "下一步复核", *review["next_steps"]])
        self._result_display.set_text("\n".join(lines))

        try:
            fig, _ax = plot_pose_ranking(result.poses)
            self._render_figure(fig)
        except Exception as exc:
            self._show_error(f"绘图失败: {exc}")
            return

        self._emit_status(review["summary"])
