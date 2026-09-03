"""Base panel class and result display widget for all analysis panels."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bioeng_qpcr.gui.components.background_task import (
    BackgroundTask,
    CancellationToken,
)


class Card(QFrame):
    """A styled card container with a title and body."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(10)

        if title:
            lbl = QLabel(title)
            lbl.setObjectName("CardTitle")
            self._layout.addWidget(lbl)

    def addWidget(self, widget: QWidget) -> None:  # type: override[override]
        self._layout.addWidget(widget)

    def addLayout(self, layout) -> None:
        self._layout.addLayout(layout)

    def addStretch(self) -> None:
        self._layout.addStretch()


class ResultDisplay(QTextEdit):
    """Read-only text display for analysis results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ResultsText")
        self.setReadOnly(True)
        self.setPlaceholderText("分析结果将显示在这里...")
        self.setPlainText(
            "准备开始\n"
            "────────────────────────\n"
            "选择左侧的数据文件与参数，然后运行分析。\n"
            "结果、质量提示与可追溯信息会显示在此处。"
        )

    def append_text(self, text: str) -> None:
        self.append(text)

    def set_text(self, text: str) -> None:
        self.setPlainText(text)

    def clear_text(self) -> None:
        self.clear()


class BasePanel(QWidget):
    """Base class for all analysis panels.

    Provides a standard layout with a left controls area and a right
    results area (text + plot), connected by a splitter.
    """

    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        self._active_task: BackgroundTask | None = None
        self._background_success: Callable[[Any], None] | None = None
        self._background_busy_widgets: tuple[QWidget, ...] = ()
        self._background_cancel_widget: QWidget | None = None
        self._background_error_prefix = "后台任务失败"
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 20)
        outer.setSpacing(18)

        # Header
        self._header = self._build_header()
        outer.addWidget(self._header)

        # Splitter: left = controls, right = results
        # Qt.Orientation replaces Qt.OrientationFlag in PySide6 >= 6.7;
        # fall back for older releases.
        orientation = getattr(Qt, "Orientation", getattr(Qt, "OrientationFlag", None))
        splitter = QSplitter(orientation.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter

        # Left: controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        left_scroll.setStyleSheet("QScrollArea { background: transparent; }")

        self._controls_widget = QWidget()
        self._controls_layout = QVBoxLayout(self._controls_widget)
        self._controls_layout.setContentsMargins(0, 0, 4, 0)
        self._controls_layout.setSpacing(14)
        self._controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        left_scroll.setWidget(self._controls_widget)
        splitter.addWidget(left_scroll)

        # Right: results + plot
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(14)
        self._right_layout = right_layout

        self._result_display = ResultDisplay()
        result_card = Card("分析结果")
        result_card.addWidget(self._result_display)
        self._result_card = result_card
        right_layout.addWidget(result_card, stretch=1)

        # Plot area (lazy: created by subclasses if needed)
        self._plot_canvas = None

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 760])

        outer.addWidget(splitter, stretch=1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        eyebrow = QLabel("BIOENGINEERING ANALYSIS WORKSPACE")
        eyebrow.setObjectName("PanelEyebrow")
        layout.addWidget(eyebrow)

        self._title_label = QLabel(self._panel_title())
        self._title_label.setObjectName("PanelTitle")
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel(self._panel_subtitle())
        self._subtitle_label.setObjectName("SectionHint")
        layout.addWidget(self._subtitle_label)

        return header

    def _add_plot_widget(self, widget: QWidget, title: str = "结果图表") -> Card:
        """Add a consistently styled plot card to the results column."""
        card = Card(title)
        card.addWidget(widget)
        self._right_layout.addWidget(card, stretch=1)
        return card

    def _panel_title(self) -> str:
        return "分析面板"

    def _panel_subtitle(self) -> str:
        return ""

    def _emit_status(self, message: str) -> None:
        self.status_message.emit(message)

    def _show_error(self, message: str) -> None:
        self._result_display.set_text(f"[错误] {message}")
        self._emit_status(f"错误: {message}")

    def _show_info(self, message: str) -> None:
        self._result_display.set_text(message)
        self._emit_status("完成")

    def _append_result(self, text: str) -> None:
        self._result_display.append_text(text)

    # ------------------------------------------------------------------
    # Responsive background operations
    # ------------------------------------------------------------------
    def _start_background_task(
        self,
        operation: Callable[[CancellationToken], Any],
        on_success: Callable[[Any], None],
        *,
        busy_widgets: Iterable[QWidget] = (),
        cancel_widget: QWidget | None = None,
        status: str = "正在执行分析...",
        error_prefix: str = "后台任务失败",
    ) -> bool:
        """Start one scientific operation without blocking the Qt event loop."""
        if self._active_task is not None:
            self._emit_status("已有分析任务正在运行")
            return False

        task = BackgroundTask(operation)
        task.signals.progress.connect(self._on_background_progress)
        task.signals.result.connect(self._on_background_result)
        task.signals.error.connect(self._on_background_error)
        task.signals.cancelled.connect(self._on_background_cancelled)
        task.signals.finished.connect(self._on_background_finished)

        self._active_task = task
        self._background_success = on_success
        self._background_busy_widgets = tuple(busy_widgets)
        self._background_cancel_widget = cancel_widget
        self._background_error_prefix = error_prefix
        for widget in self._background_busy_widgets:
            widget.setEnabled(False)
        if cancel_widget is not None:
            cancel_widget.setEnabled(True)
        self._emit_status(status)
        QThreadPool.globalInstance().start(task)
        return True

    def _cancel_background_task(self) -> None:
        """Request cancellation; an active native solver may finish its step."""
        if self._active_task is None:
            return
        self._active_task.cancel()
        if self._background_cancel_widget is not None:
            self._background_cancel_widget.setEnabled(False)
        self._emit_status("正在取消；当前求解步骤结束后停止...")

    @property
    def _background_task_active(self) -> bool:
        return self._active_task is not None

    @Slot(str)
    def _on_background_progress(self, message: str) -> None:
        self._emit_status(message)

    @Slot(object)
    def _on_background_result(self, result: object) -> None:
        callback = self._background_success
        if callback is None:
            return
        try:
            callback(result)
        except Exception as exc:  # noqa: BLE001 - GUI rendering boundary
            self._show_error(f"结果处理失败: {exc}")

    @Slot(str)
    def _on_background_error(self, message: str) -> None:
        self._show_error(f"{self._background_error_prefix}: {message}")

    @Slot()
    def _on_background_cancelled(self) -> None:
        self._emit_status("分析已取消")

    @Slot()
    def _on_background_finished(self) -> None:
        for widget in self._background_busy_widgets:
            widget.setEnabled(True)
        if self._background_cancel_widget is not None:
            self._background_cancel_widget.setEnabled(False)
        self._active_task = None
        self._background_success = None
        self._background_busy_widgets = ()
        self._background_cancel_widget = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._active_task is not None:
            self._active_task.cancel()
        super().closeEvent(event)
