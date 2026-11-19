"""Desktop and web share the same search review, including skipped audits."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from bioeng_qpcr.gui.panels.docking_panel import DockingPanel


@pytest.mark.parametrize("audit", [True, False])
def test_desktop_displays_review_and_next_steps(monkeypatch, audit):
    app = QApplication.instance() or QApplication([])
    panel = DockingPanel()
    examples = Path(__file__).resolve().parents[2] / "examples"
    errors = []
    monkeypatch.setattr(
        panel._receptor_selector,
        "get_path",
        lambda: str(examples / "docking_receptor.pdb"),
    )
    monkeypatch.setattr(
        panel._ligand_selector, "get_path", lambda: str(examples / "docking_ligand.pdb")
    )
    monkeypatch.setattr(panel, "_show_error", errors.append)
    panel._n_runs_spin.setValue(1)
    panel._n_steps_spin.setValue(50)
    panel._validate_check.setChecked(audit)
    try:
        panel._on_dock()
        assert not errors
        text = panel._result_display.toPlainText()
        assert "计算已完成" in text and "下一步复核" in text
        assert "搜索稳定或规则通过均不等于" in text
        if not audit:
            assert "未执行审计" in text and "启用 DCK 审计规则" in text
    finally:
        panel.close()
        app.processEvents()
