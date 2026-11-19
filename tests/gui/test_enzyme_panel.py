"""An inhibited series must never be pooled into an uninhibited fit."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from bioeng_qpcr.gui.panels import enzyme_panel


@pytest.mark.parametrize("levels", [(1.0, 10.0), (1e-10, 1e-9)])
def test_mixed_levels_without_zero_are_blocked(monkeypatch, levels):
    app = QApplication.instance() or QApplication([])
    panel = enzyme_panel.EnzymePanel()
    errors, fits = [], []
    monkeypatch.setattr(panel, "_get_csv_path", lambda: "unused.csv")
    monkeypatch.setattr(panel, "_show_error", errors.append)
    monkeypatch.setattr(
        enzyme_panel,
        "read_kinetics_csv",
        lambda path: (
            np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            np.repeat(levels, 3),
        ),
    )
    monkeypatch.setattr(
        enzyme_panel, "fit_michaelis_menten", lambda *args: fits.append(args)
    )
    try:
        panel._on_fit_michaelis_menten()
        assert not fits
        assert "不能混合" in errors[0]
    finally:
        panel.close()
        app.processEvents()


def test_global_fit_panel_uses_engine_and_original_rate_plots(monkeypatch):
    from bioeng_qpcr.gui.components.background_task import CancellationToken

    app = QApplication.instance() or QApplication([])
    panel = enzyme_panel.EnzymePanel()
    substrate = np.tile(np.linspace(0.1, 15, 15), 3)
    inhibitor = np.repeat([0.0, 1.0, 5.0], 15)
    velocity = 10 * substrate / (2 * (1 + inhibitor / 3) + substrate)
    errors = []
    monkeypatch.setattr(panel, "_get_csv_path", lambda: "synthetic.csv")
    monkeypatch.setattr(panel, "_show_error", errors.append)
    monkeypatch.setattr(
        enzyme_panel, "read_kinetics_csv", lambda path: (substrate, velocity, inhibitor)
    )
    captured = []

    def run_inline(operation, on_success, **kwargs):
        payload = operation(CancellationToken(lambda message: None))
        captured.append(payload)
        on_success(payload)
        return True

    monkeypatch.setattr(panel, "_start_background_task", run_inline)
    try:
        panel._on_global_inhibition()
        assert not errors
        result, datasets = captured[0]
        assert result.best_type.value == "competitive"
        assert result.best_fit.parameters["ki"] == pytest.approx(3, rel=0.01)
        assert len(result.models) == 5
        assert "AICc" in panel._result_display.toPlainText()
        assert len(panel._plot_canvas.figure.axes) == 2
        assert result.best_fit.predict(np.array([1.0, 2.0]), 1).shape == (2,)
        with pytest.raises(ValueError):
            result.best_fit.predict([-1.0], 0)
    finally:
        panel.close()
        app.processEvents()
