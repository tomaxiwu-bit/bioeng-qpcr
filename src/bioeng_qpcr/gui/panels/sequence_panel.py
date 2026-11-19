"""Sequencing-result comparison panel.

Aligns a Sanger read against a reference, excises primers and low-quality
tails, calls variants, and runs the SEQ-00x audit rules.  Every reported
coordinate is in reference space, because that is the only frame in which a
variant call can be compared with a database entry.
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
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.plotting import plot_read_qualities
from bioeng_qpcr.sequence import (
    BLOSUM62,
    build_consensus,
    compare_to_reference,
    detect_variants,
    dna_matrix,
    needleman_wunsch_affine,
    read_sequence_file,
    smith_waterman_affine,
    trim_low_quality,
    trim_primers,
    validate_sequencing_result,
)

#: Display label to alignment mode, matching the CLI's ``--mode``.
_MODE_LABELS: dict[str, str] = {
    "半全局（读长映射到参考）": "glocal",
    "全局（两端锚定）": "global",
    "局部（只比高分片段）": "local",
}


def _labelled(text: str, widget: QWidget) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    label = QLabel(text)
    label.setObjectName("FieldLabel")
    layout.addWidget(label)
    layout.addWidget(widget)
    return container


class SequencePanel(BasePanel):
    """Panel for sequencing-read alignment, variants and consensus."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_controls()
        self._install_plot_canvas()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "测序比对"

    def _panel_subtitle(self) -> str:
        return "读长比对 · 引物裁剪 · 变异检测 · 共识序列"

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        file_card = Card("序列文件")
        self._query_selector = FileSelector(
            label="待测序列（AB1 / FASTA / FASTQ / 纯文本）",
            file_filter=(
                "Sequence files (*.ab1 *.abi *.fasta *.fa *.fna "
                "*.fastq *.fq *.seq *.txt);;"
                "All files (*.*)"
            ),
        )
        file_card.addWidget(self._query_selector)

        self._reference_selector = FileSelector(
            label="参考序列（FASTA）",
            file_filter=(
                "Sequence files (*.fasta *.fa *.fna *.seq *.txt);;All files (*.*)"
            ),
        )
        file_card.addWidget(self._reference_selector)

        self._read_index_spin = QSpinBox()
        self._read_index_spin.setRange(0, 10000)
        self._read_index_spin.setValue(0)
        file_card.addWidget(_labelled("读长序号（0 起）", self._read_index_spin))
        self._controls_layout.addWidget(file_card)

        settings = Card("比对设置")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(list(_MODE_LABELS))
        settings.addWidget(_labelled("比对模式", self._mode_combo))

        self._matrix_combo = QComboBox()
        self._matrix_combo.addItems(["DNA 匹配/错配", "BLOSUM62（蛋白）"])
        settings.addWidget(_labelled("替换矩阵", self._matrix_combo))

        self._gap_open_spin = QSpinBox()
        self._gap_open_spin.setRange(0, 50)
        self._gap_open_spin.setValue(5)
        settings.addWidget(_labelled("gap 开启罚分", self._gap_open_spin))

        self._gap_extend_spin = QSpinBox()
        self._gap_extend_spin.setRange(0, 50)
        self._gap_extend_spin.setValue(2)
        settings.addWidget(_labelled("gap 延伸罚分", self._gap_extend_spin))

        self._identity_spin = QDoubleSpinBox()
        self._identity_spin.setRange(0.0, 1.0)
        self._identity_spin.setSingleStep(0.01)
        self._identity_spin.setDecimals(2)
        self._identity_spin.setValue(0.90)
        settings.addWidget(_labelled("一致性阈值 SEQ-001", self._identity_spin))

        self._controls_layout.addWidget(settings)

        trim_card = Card("裁剪")

        self._primer5_edit = QLineEdit()
        self._primer5_edit.setPlaceholderText("例如 ACGTACGT")
        trim_card.addWidget(_labelled("5' 引物", self._primer5_edit))

        self._primer3_edit = QLineEdit()
        self._primer3_edit.setPlaceholderText("例如 TGCATGCA")
        trim_card.addWidget(_labelled("3' 引物", self._primer3_edit))

        self._primer_mismatch_spin = QSpinBox()
        self._primer_mismatch_spin.setRange(0, 10)
        self._primer_mismatch_spin.setValue(2)
        trim_card.addWidget(
            _labelled("引物定位允许错配", self._primer_mismatch_spin)
        )

        self._quality_check = QCheckBox("按质量值裁剪低质量末端（需 FASTQ）")
        trim_card.addWidget(self._quality_check)

        self._min_q_spin = QSpinBox()
        self._min_q_spin.setRange(0, 60)
        self._min_q_spin.setValue(20)
        trim_card.addWidget(_labelled("质量阈值 Q", self._min_q_spin))

        self._window_spin = QSpinBox()
        self._window_spin.setRange(2, 100)
        self._window_spin.setValue(10)
        trim_card.addWidget(_labelled("滑窗长度", self._window_spin))

        self._controls_layout.addWidget(trim_card)

        option_card = Card("输出选项")
        self._variants_check = QCheckBox("检测并列出变异")
        self._variants_check.setChecked(True)
        option_card.addWidget(self._variants_check)

        self._coding_check = QCheckBox("按编码序列处理（启用移码检查）")
        option_card.addWidget(self._coding_check)

        self._quality_plot_check = QCheckBox("绘制读长质量图")
        self._quality_plot_check.setChecked(True)
        option_card.addWidget(self._quality_plot_check)
        self._controls_layout.addWidget(option_card)

        action_card = Card("执行")
        self._btn_align = QPushButton("比对并检测变异")
        self._btn_align.setObjectName("PrimaryButton")
        self._btn_align.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_align.clicked.connect(self._on_align)
        action_card.addWidget(self._btn_align)

        self._btn_consensus = QPushButton("由全部读长生成共识序列")
        self._btn_consensus.setObjectName("SecondaryButton")
        self._btn_consensus.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_consensus.clicked.connect(self._on_consensus)
        action_card.addWidget(self._btn_consensus)

        self._controls_layout.addWidget(action_card)
        self._controls_layout.addStretch()

    def _install_plot_canvas(self) -> None:
        self._plot_canvas = PlotCanvas(self, width=8, height=5, dpi=100)
        self._add_plot_widget(self._plot_canvas, "读长质量")

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

    def _substitution_matrix(self):
        return dna_matrix() if self._matrix_combo.currentIndex() == 0 else BLOSUM62

    def _read_paths(self) -> tuple[str, str] | None:
        query = self._query_selector.get_path()
        reference = self._reference_selector.get_path()
        if not query:
            self._show_error("请先选择待测序列文件。")
            return None
        if not reference:
            self._show_error("请先选择参考序列文件。")
            return None
        return query, reference

    @staticmethod
    def _kmer_lines(aligned_query: str, aligned_reference: str, width: int = 60):
        for start in range(0, len(aligned_query), width):
            query_slice = aligned_query[start : start + width]
            reference_slice = aligned_reference[start : start + width]
            matches = "".join(
                "|" if a == b else " "
                for a, b in zip(query_slice, reference_slice, strict=True)
            )
            yield query_slice, matches, reference_slice

    # ------------------------------------------------------------------
    # Action: align
    # ------------------------------------------------------------------
    def _on_align(self) -> None:
        paths = self._read_paths()
        if paths is None:
            return
        query_path, reference_path = paths

        self._emit_status("正在比对...")

        try:
            records = read_sequence_file(query_path)
            references = read_sequence_file(reference_path)
        except Exception as exc:
            self._show_error(f"读取序列失败: {exc}")
            return

        index = self._read_index_spin.value()
        if index >= len(records):
            self._show_error(
                f"读长文件只有 {len(records)} 条记录，序号 {index} 超出范围。"
            )
            return
        if not references:
            self._show_error("参考文件中没有序列。")
            return

        record = records[index]
        query = record.sequence
        quality = list(record.quality) if record.quality else None
        reference = references[0].sequence

        notes: list[str] = []
        primer_result = None
        primer_5 = self._primer5_edit.text().strip() or None
        primer_3 = self._primer3_edit.text().strip() or None
        if primer_5 or primer_3:
            try:
                primer_result = trim_primers(
                    query,
                    primer_5=primer_5,
                    primer_3=primer_3,
                    max_mismatch=self._primer_mismatch_spin.value(),
                )
            except Exception as exc:
                self._show_error(f"引物裁剪失败: {exc}")
                return
            query = primer_result.trimmed
            notes.append(
                f"引物裁剪：5' 去掉 {primer_result.n_trimmed_5} 个碱基，"
                f"3' 去掉 {primer_result.n_trimmed_3} 个"
            )
            if quality is not None:
                quality = quality[
                    primer_result.n_trimmed_5 : len(quality)
                    - primer_result.n_trimmed_3
                ]

        if self._quality_check.isChecked():
            if quality is None:
                self._show_error("质量裁剪需要 FASTQ 或 AB1 质量值。")
                return
            try:
                quality_result = trim_low_quality(
                    query,
                    quality,
                    window=self._window_spin.value(),
                    min_q=self._min_q_spin.value(),
                )
            except Exception as exc:
                self._show_error(f"质量裁剪失败: {exc}")
                return
            query = quality_result.trimmed
            quality = quality[
                quality_result.kept_start : quality_result.kept_end
            ]
            notes.append(
                f"质量裁剪：保留第 {quality_result.kept_start}.."
                f"{quality_result.kept_end} 位（窗口 {self._window_spin.value()}，"
                f"Q{self._min_q_spin.value()}）"
            )

        matrix = self._substitution_matrix()
        mode = _MODE_LABELS[self._mode_combo.currentText()]
        try:
            if mode == "global":
                result = needleman_wunsch_affine(
                    query,
                    reference,
                    matrix,
                    gap_open=self._gap_open_spin.value(),
                    gap_extend=self._gap_extend_spin.value(),
                )
            elif mode == "local":
                result = smith_waterman_affine(
                    query,
                    reference,
                    matrix,
                    gap_open=self._gap_open_spin.value(),
                    gap_extend=self._gap_extend_spin.value(),
                )
            else:
                result = compare_to_reference(
                    query,
                    reference,
                    matrix,
                    gap_open=self._gap_open_spin.value(),
                    gap_extend=self._gap_extend_spin.value(),
                )
        except Exception as exc:
            self._show_error(f"比对失败: {exc}")
            return

        variants = []
        if self._variants_check.isChecked():
            variants = detect_variants(result, reference, query_quality=quality)

        try:
            report = validate_sequencing_result(
                result,
                query,
                reference,
                variants,
                quality=quality,
                coding=self._coding_check.isChecked(),
                primer_trim=primer_result,
                primer_5=primer_5,
                primer_3=primer_3,
                threshold_identity=self._identity_spin.value(),
                min_q=self._min_q_spin.value(),
            )
        except Exception as exc:
            self._show_error(f"审计规则执行失败: {exc}")
            return

        lines: list[str] = []
        lines.append("比对结果")
        lines.append("=" * 46)
        lines.append(f"读长 {record.id}  长度 {len(query)}")
        lines.append(f"参考 {references[0].id if references[0].id else 'reference'}")
        lines.append(f"模式 {mode}  矩阵 {self._matrix_combo.currentText()}")
        lines.append(
            "读长方向 "
            + (
                "反向互补"
                if result.query_orientation == "reverse_complement"
                else "正向"
            )
        )
        for note in notes:
            lines.append(note)
        lines.append("")
        lines.append(
            f"得分 {result.score:.1f}   一致性 {result.identity:.2%}   "
            f"相似性 {result.similarity:.2%}"
        )
        lines.append(
            f"比对长度 {result.length}   gap {result.gaps} "
            f"（开启 {result.gap_opens}）"
        )
        lines.append(
            f"参考坐标 {result.ref_start}..{result.ref_end}   "
            f"读长坐标 {result.query_start}..{result.query_end}"
        )
        lines.append(f"CIGAR {result.cigar}")
        lines.append("")
        lines.append("比对")
        lines.append("-" * 46)
        for query_slice, matches, reference_slice in self._kmer_lines(
            result.aligned_query, result.aligned_reference
        ):
            lines.append(f"q {query_slice}")
            lines.append(f"  {matches}")
            lines.append(f"r {reference_slice}")

        if self._variants_check.isChecked():
            lines.append("")
            lines.append(f"变异（{len(variants)}）")
            lines.append("-" * 46)
            if not variants:
                lines.append("  未检出变异。")
            for variant in variants:
                extra = f"   Q={variant.quality}" if variant.quality else ""
                flag = "  [低复杂度]" if variant.in_low_complexity else ""
                lines.append(
                    f"  {variant.type.value:<10} 位置 {variant.ref_position:<6} "
                    f"{variant.ref_allele or '-'} → {variant.alt_allele or '-'}"
                    f"{extra}{flag}"
                )

        lines.append("")
        lines.append("审计规则")
        lines.append("-" * 46)
        for check in report.results:
            status = "通过" if check.passed else "未通过"
            lines.append(
                f"  {check.rule_id} ({check.severity.value}) {status} — "
                f"{check.message}"
            )
        lines.append("")
        lines.append(f"总体：{'通过' if report.passed else '存在未通过项'}")

        self._result_display.set_text("\n".join(lines))

        if self._quality_plot_check.isChecked() and quality:
            try:
                fig, _ax = plot_read_qualities(
                    quality,
                    threshold=self._min_q_spin.value(),
                    window=self._window_spin.value(),
                )
                self._render_figure(fig)
            except Exception as exc:
                self._show_error(f"无法绘制质量图: {exc}")
                return

        self._emit_status(
            f"比对完成（{len(variants)} 个变异），"
            f"{'通过' if report.passed else '存在未通过项'}。"
        )

    # ------------------------------------------------------------------
    # Action: consensus
    # ------------------------------------------------------------------
    def _on_consensus(self) -> None:
        paths = self._read_paths()
        if paths is None:
            return
        query_path, reference_path = paths

        self._emit_status("正在生成共识序列...")

        try:
            records = read_sequence_file(query_path)
            references = read_sequence_file(reference_path)
        except Exception as exc:
            self._show_error(f"读取序列失败: {exc}")
            return
        if not records or not references:
            self._show_error("读长或参考文件为空。")
            return

        try:
            consensus = build_consensus(
                [record.sequence for record in records], references[0].sequence
            )
        except Exception as exc:
            self._show_error(f"生成共识序列失败: {exc}")
            return

        lines: list[str] = []
        lines.append("共识序列")
        lines.append("=" * 46)
        lines.append(f"读长数 {len(records)}   方法 {consensus.method}")
        lines.append(f"模糊位点数 {consensus.n_ambiguous}")
        lines.append(f"参考长度 {len(references[0].sequence)}")
        lines.append("")
        lines.append("共识序列")
        lines.append("-" * 46)
        sequence = consensus.consensus
        for start in range(0, len(sequence), 60):
            lines.append(f"{start + 1:>6}  {sequence[start : start + 60]}")
        self._result_display.set_text("\n".join(lines))
        self._emit_status(
            f"共识序列生成完成（{consensus.n_ambiguous} 个模糊位点）。"
        )
