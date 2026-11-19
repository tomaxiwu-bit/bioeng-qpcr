"""User-facing file-selection workflow tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtWidgets import QApplication

from bioeng_qpcr.gui.components.file_selector import FileSelector


class _Settings:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def setValue(self, key: str, value: str) -> None:  # noqa: N802 - Qt API
        self.values[key] = value


class _DropEvent:
    def __init__(self, path: Path) -> None:
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(str(path))])
        self.accepted = False

    def mimeData(self) -> QMimeData:  # noqa: N802 - Qt API
        return self._mime

    def acceptProposedAction(self) -> None:  # noqa: N802 - Qt API
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


def test_drop_selects_file_and_remembers_its_directory(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "input.csv"
    path.write_text("sample,cq\nexample,20\n", encoding="utf-8")
    selector = FileSelector()
    settings = _Settings()
    selector._settings = settings
    selected: list[str] = []
    selector.file_selected.connect(selected.append)

    event = _DropEvent(path)
    selector.dropEvent(event)

    assert event.accepted
    assert selector.get_path() == str(path.resolve())
    assert selector._path_edit.text() == path.name
    assert selected == [str(path.resolve())]
    assert settings.values["last_input_directory"] == str(path.parent)
    selector.close()
    app.processEvents()
