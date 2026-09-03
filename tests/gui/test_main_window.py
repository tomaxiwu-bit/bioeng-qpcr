"""Startup tests for the desktop GUI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from bioeng_qpcr.gui.main_window import MainWindow, _run_scientific_smoke_checks


def test_all_gui_panels_load() -> None:
    """A missing plotting dependency must fail CI, not the distributed EXE."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.panel_load_errors == {}
    finally:
        window.close()
        app.processEvents()


def test_scientific_smoke_checks_execute_numerical_paths() -> None:
    _run_scientific_smoke_checks()
