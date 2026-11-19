"""Startup tests for the desktop GUI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from bioeng_qpcr.gui.main_window import (
    _NAV_ITEMS,
    _PANEL_CLASSES,
    MainWindow,
    _run_scientific_smoke_checks,
)


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


def test_every_navigation_entry_has_a_panel() -> None:
    """The sidebar and the panel registry must stay in step.

    A navigation entry without a panel silently does nothing when clicked,
    which is invisible in a screenshot-based review.
    """
    assert {key for key, _, _ in _NAV_ITEMS} == set(_PANEL_CLASSES)


def test_navigation_focuses_on_quantitative_research() -> None:
    for key in ("qpcr", "potency", "enzyme", "ferment", "fba"):
        assert key in _PANEL_CLASSES
        assert key in {entry[0] for entry in _NAV_ITEMS}
    assert not {"docking", "sequence", "preview"} & set(_PANEL_CLASSES)


def test_browser_handoff_selects_desktop_panel_and_navigation() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        launcher = window._panels["workbench"]
        for key in ("enzyme", "ferment", "fba"):
            launcher.open_module_requested.emit(key)
            app.processEvents()
            assert window._stack.currentWidget() is window._panels[key]
            assert window._nav_group.checkedButton().property("panelKey") == key
    finally:
        window.close()
        app.processEvents()
