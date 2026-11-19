"""Input preview panel: CSV data tables and lab imagery.

Two jobs in one panel, because they answer the same question -- *"what does
the software think this file is?"*  For a CSV the panel renders the import
report (detected encoding/delimiter, inferred column types, and every
structural problem with its row and column).  For an image it renders the
metadata plus a processed thumbnail, with the pseudo-colour and contrast
controls a fluorescence scan or a gel needs.

Neither path modifies the source file.
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
    QVBoxLayout,
    QWidget,
)

from bioeng_qpcr.gui.components.base_panel import BasePanel, Card
from bioeng_qpcr.gui.components.file_selector import FileSelector
from bioeng_qpcr.gui.components.plot_canvas import PlotCanvas
from bioeng_qpcr.io import (
    CHANNEL_PALETTES,
    SCHEMAS,
    inspect_image,
    load_preview,
    preview_csv,
    schema_for,
    write_template_csv,
)

#: File-dialog filter covering every raster format the inspector accepts.
_IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp *.ico *.psd "
    "*.tga *.ppm *.svg);;All files (*.*)"
)


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


class PreviewPanel(BasePanel):
    """Panel for previewing CSV data tables and image files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_controls()
        self._install_plot_canvas()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _panel_title(self) -> str:
        return "数据预览"

    def _panel_subtitle(self) -> str:
        return "CSV 格式校验 · 多格式图片预览"

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        csv_card = Card("CSV 数据表")

        self._csv_selector = FileSelector(
            label="选择 CSV / TSV 文件",
            file_filter=(
                "Table files (*.csv *.tsv *.txt);;All files (*.*)"
            ),
        )
        csv_card.addWidget(self._csv_selector)

        self._format_combo = QComboBox()
        self._format_combo.addItem("自动识别格式", None)
        for key in sorted(SCHEMAS):
            self._format_combo.addItem(f"{key} — {SCHEMAS[key].title}", key)
        csv_card.addWidget(_labelled("导入格式", self._format_combo))

        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 500)
        self._rows_spin.setValue(20)
        csv_card.addWidget(_labelled("显示行数", self._rows_spin))

        self._btn_preview_csv = QPushButton("预览并校验 CSV")
        self._btn_preview_csv.setObjectName("PrimaryButton")
        self._btn_preview_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_preview_csv.clicked.connect(self._on_preview_csv)
        csv_card.addWidget(self._btn_preview_csv)

        self._btn_template = QPushButton("导出所选格式的模板 CSV")
        self._btn_template.setObjectName("SecondaryButton")
        self._btn_template.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_template.clicked.connect(self._on_write_template)
        csv_card.addWidget(self._btn_template)

        hint = QLabel(
            "模板会写到所选数据文件所在目录；未选择数据文件时写入当前目录。"
        )
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        csv_card.addWidget(hint)

        self._controls_layout.addWidget(csv_card)

        image_card = Card("图片")

        self._image_selector = FileSelector(
            label="选择图片文件", file_filter=_IMAGE_FILTER
        )
        image_card.addWidget(self._image_selector)

        self._palette_combo = QComboBox()
        for name in CHANNEL_PALETTES:
            self._palette_combo.addItem(name, name)
        self._palette_combo.setCurrentText("gray")
        image_card.addWidget(_labelled("单通道伪彩表", self._palette_combo))

        self._invert_check = QCheckBox("反相（荧光扫描/凝胶常用）")
        image_card.addWidget(self._invert_check)

        self._contrast_check = QCheckBox("自动对比度拉伸")
        self._contrast_check.setChecked(True)
        image_card.addWidget(self._contrast_check)

        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 5.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setValue(1.0)
        image_card.addWidget(_labelled("Gamma（>1 提亮中间调）", self._gamma_spin))

        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 500)
        self._frame_spin.setValue(0)
        image_card.addWidget(_labelled("帧序号（多页 TIFF / GIF）", self._frame_spin))

        self._btn_inspect = QPushButton("仅读取元数据")
        self._btn_inspect.setObjectName("SecondaryButton")
        self._btn_inspect.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_inspect.clicked.connect(self._on_inspect_image)
        image_card.addWidget(self._btn_inspect)

        self._btn_preview_image = QPushButton("生成图片预览")
        self._btn_preview_image.setObjectName("PrimaryButton")
        self._btn_preview_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_preview_image.clicked.connect(self._on_preview_image)
        image_card.addWidget(self._btn_preview_image)

        self._controls_layout.addWidget(image_card)
        self._controls_layout.addStretch()

    def _install_plot_canvas(self) -> None:
        self._plot_canvas = PlotCanvas(self, width=8, height=5, dpi=100)
        self._add_plot_widget(self._plot_canvas, "图片预览")

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

    # ------------------------------------------------------------------
    # Action: CSV
    # ------------------------------------------------------------------
    def _on_preview_csv(self) -> None:
        path = self._csv_selector.get_path()
        if not path:
            self._show_error("请先选择 CSV 文件。")
            return

        self._emit_status("正在解析并校验 CSV...")

        schema = self._format_combo.currentData()
        try:
            preview = preview_csv(
                path, schema=schema, display_rows=self._rows_spin.value()
            )
        except Exception as exc:
            self._show_error(f"预览失败: {exc}")
            return

        self._result_display.set_text(
            preview.to_text(max_table_rows=self._rows_spin.value())
        )
        errors = len(preview.errors)
        warnings = len(preview.warnings)
        self._emit_status(
            f"CSV 预览完成：{preview.rows_total} 行，"
            f"{errors} 个错误，{warnings} 个警告。"
        )

    def _on_write_template(self) -> None:
        from pathlib import Path

        key = self._format_combo.currentData()
        if not key:
            self._show_error("请先在「导入格式」中选择一个具体格式，再导出模板。")
            return

        source = self._csv_selector.get_path()
        directory = Path(source).parent if source else Path.cwd()
        destination = directory / f"{key}_template.csv"
        try:
            written = write_template_csv(destination, schema_for(key))
        except Exception as exc:
            self._show_error(f"写出模板失败: {exc}")
            return

        self._emit_status(f"模板已写出: {written}")
        self._result_display.set_text(
            f"已写出模板\n{'-' * 46}\n{written}\n\n"
            + schema_for(key).to_markdown()
        )

    # ------------------------------------------------------------------
    # Action: images
    # ------------------------------------------------------------------
    def _on_inspect_image(self) -> None:
        path = self._image_selector.get_path()
        if not path:
            self._show_error("请先选择图片文件。")
            return

        self._emit_status("正在读取图片元数据...")
        try:
            info = inspect_image(path)
        except Exception as exc:
            self._show_error(f"读取失败: {exc}")
            return

        from bioeng_qpcr.io import list_supported_formats

        known = list_supported_formats()
        lines = [info.to_text(), ""]
        lines.append("可用格式")
        lines.append("-" * 46)
        lines.append("  " + "、".join(sorted(known)))
        self._result_display.set_text("\n".join(lines))
        self._emit_status(f"格式 {info.format}，{info.width}×{info.height}。")

    def _on_preview_image(self) -> None:
        path = self._image_selector.get_path()
        if not path:
            self._show_error("请先选择图片文件。")
            return

        self._emit_status("正在生成图片预览...")

        try:
            preview = load_preview(
                path,
                frame=self._frame_spin.value(),
                invert=self._invert_check.isChecked(),
                auto_contrast=self._contrast_check.isChecked(),
                gamma=self._gamma_spin.value(),
                palette=self._palette_combo.currentData() or "gray",
            )
        except Exception as exc:
            self._show_error(f"预览失败: {exc}")
            return

        self._result_display.set_text(preview.to_text())

        try:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(6, 5))
            axis.imshow(preview.thumbnail)
            axis.set_axis_off()
            from pathlib import Path as _Path

            axis.set_title(_Path(path).name)
        except Exception as exc:
            self._show_error(f"绘图失败: {exc}")
            return
        self._render_figure(figure)

        self._emit_status(
            f"图片预览完成：{preview.thumbnail_size[0]}×"
            f"{preview.thumbnail_size[1]} 像素。"
        )
