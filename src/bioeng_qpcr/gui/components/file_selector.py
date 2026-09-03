"""Reusable file selector widget with drag-and-drop support."""

from __future__ import annotations

from PySide6.QtCore import Signal
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
        self._path_edit.setPlaceholderText("尚未选择文件")
        self._path_edit.setReadOnly(True)
        self._path_edit.setMinimumWidth(0)

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
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", self._file_filter
        )
        if path:
            self._path_edit.setText(path)
            self._path_edit.setToolTip(path)
            self.file_selected.emit(path)

    def get_path(self) -> str:
        return self._path_edit.text().strip()

    def clear(self) -> None:
        self._path_edit.clear()
        self._path_edit.setToolTip("")
