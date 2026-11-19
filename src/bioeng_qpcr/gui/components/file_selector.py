"""Reusable file selector widget with drag-and-drop support."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FileSelector(QWidget):
    """A file path input with a browse button and file-type filter."""

    file_selected = Signal(str)

    def __init__(
        self,
        label: str = "选择文件",
        file_filter: str = "CSV files (*.csv);;All files (*.*)",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._file_filter = file_filter
        self._path = ""
        self._settings = QSettings("bioeng-qpcr", "bioeng-qpcr")
        self.setAcceptDrops(True)
        self._build_ui(label)

    def _build_ui(self, label: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        lbl = QLabel(label)
        lbl.setObjectName("FieldLabel")

        row = QHBoxLayout()
        row.setSpacing(8)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("点击选择，或将文件拖到这里")
        self._path_edit.setReadOnly(True)
        self._path_edit.setMinimumWidth(0)
        self._path_edit.setAcceptDrops(False)

        btn = QPushButton("选择文件")
        btn.setObjectName("SecondaryButton")
        btn.setFixedWidth(92)
        btn.setToolTip("从本机选择数据文件")
        btn.clicked.connect(self._on_browse)

        row.addWidget(self._path_edit)
        row.addWidget(btn)

        layout.addWidget(lbl)
        layout.addLayout(row)

    def _on_browse(self) -> None:
        default_directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        initial_directory = self._settings.value(
            "last_input_directory", default_directory, type=str
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", initial_directory, self._file_filter
        )
        if path:
            self._set_path(path)

    def _set_path(self, path: str) -> None:
        file_path = Path(path).resolve()
        self._path = str(file_path)
        self._path_edit.setText(file_path.name)
        self._path_edit.setToolTip(self._path)
        self._settings.setValue("last_input_directory", str(file_path.parent))
        self.file_selected.emit(self._path)

    @staticmethod
    def _dropped_file(event) -> str:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) != 1 or not urls[0].isLocalFile():
            return ""
        path = urls[0].toLocalFile()
        return path if Path(path).is_file() else ""

    def _set_drag_active(self, active: bool) -> None:
        self._path_edit.setProperty("dragActive", active)
        self._path_edit.style().unpolish(self._path_edit)
        self._path_edit.style().polish(self._path_edit)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dropped_file(event):
            self._set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._set_drag_active(False)
        event.accept()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._set_drag_active(False)
        path = self._dropped_file(event)
        if not path:
            event.ignore()
            return
        self._set_path(path)
        event.acceptProposedAction()

    def get_path(self) -> str:
        return self._path or self._path_edit.text().strip()

    def clear(self) -> None:
        self._path = ""
        self._path_edit.clear()
        self._path_edit.setToolTip("")
