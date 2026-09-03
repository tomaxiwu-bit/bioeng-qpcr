"""Thread-affinity and cancellation tests for scientific GUI tasks."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from bioeng_qpcr.gui.components.background_task import CancellationToken
from bioeng_qpcr.gui.components.base_panel import BasePanel
from bioeng_qpcr.gui.panels import fba_panel, ferment_panel
from bioeng_qpcr.gui.panels.fba_panel import FbaPanel
from bioeng_qpcr.gui.panels.ferment_panel import FermentPanel


class _TaskPanel(BasePanel):
    def _panel_title(self) -> str:
        return "Task test"


@pytest.fixture
def task_panel():
    app = QApplication.instance() or QApplication([])
    panel = _TaskPanel()
    try:
        yield app, panel
    finally:
        assert _wait_until(app, lambda: not panel._background_task_active)
        panel.close()
        app.processEvents()


def _wait_until(app: QApplication, predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def test_operation_runs_off_ui_thread_and_callback_returns_to_ui(task_panel) -> None:
    app, panel = task_panel
    busy = QPushButton()
    cancel = QPushButton()
    cancel.setEnabled(False)
    main_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def operation(token: CancellationToken) -> int:
        token.report("worker progress")
        return threading.get_ident()

    def on_success(worker_thread: int) -> None:
        observed["worker"] = worker_thread
        observed["callback"] = threading.get_ident()

    assert panel._start_background_task(
        operation,
        on_success,
        busy_widgets=(busy,),
        cancel_widget=cancel,
    )
    assert not busy.isEnabled()
    assert cancel.isEnabled()
    assert _wait_until(app, lambda: not panel._background_task_active)

    assert observed["worker"] != main_thread
    assert observed["callback"] == main_thread
    assert busy.isEnabled()
    assert not cancel.isEnabled()


def test_worker_error_is_rendered_and_controls_are_restored(task_panel) -> None:
    app, panel = task_panel
    busy = QPushButton()
    cancel = QPushButton()
    cancel.setEnabled(False)

    def operation(_token: CancellationToken):
        raise ValueError("adversarial failure")

    panel._start_background_task(
        operation,
        lambda _result: None,
        busy_widgets=(busy,),
        cancel_widget=cancel,
        error_prefix="Scientific task failed",
    )
    assert _wait_until(app, lambda: not panel._background_task_active)

    text = panel._result_display.toPlainText()
    assert "Scientific task failed" in text
    assert "ValueError: adversarial failure" in text
    assert busy.isEnabled()
    assert not cancel.isEnabled()


def test_cancellation_discards_late_native_solver_result(task_panel) -> None:
    app, panel = task_panel
    started = threading.Event()
    release_solver = threading.Event()
    rendered: list[object] = []
    statuses: list[str] = []
    panel.status_message.connect(statuses.append)

    def operation(_token: CancellationToken) -> str:
        started.set()
        if not release_solver.wait(timeout=2.0):
            raise TimeoutError("test solver was not released")
        return "late result"

    panel._start_background_task(operation, rendered.append)
    assert _wait_until(app, started.is_set)

    panel._cancel_background_task()
    release_solver.set()
    assert _wait_until(app, lambda: not panel._background_task_active)

    assert rendered == []
    assert any("取消" in status for status in statuses)


def test_second_task_is_rejected_while_first_is_running(task_panel) -> None:
    app, panel = task_panel
    started = threading.Event()
    release = threading.Event()

    def operation(_token: CancellationToken) -> None:
        started.set()
        release.wait(timeout=2.0)

    assert panel._start_background_task(operation, lambda _result: None)
    assert _wait_until(app, started.is_set)
    assert not panel._start_background_task(
        lambda _token: None,
        lambda _result: None,
    )

    panel._cancel_background_task()
    release.set()
    assert _wait_until(app, lambda: not panel._background_task_active)


def test_monod_panel_dispatches_fit_to_worker(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    panel = FermentPanel()
    release = threading.Event()
    rendered: list[tuple[object, int]] = []
    main_thread = threading.get_ident()

    def read_data(_path: str):
        release.wait(timeout=2.0)
        return np.arange(5.0), np.linspace(0.1, 1.0, 5), np.linspace(10, 5, 5)

    passed = SimpleNamespace(passed=True, rule_id="FERM", message="ok")
    fit = SimpleNamespace(mu_max=0.5)
    monkeypatch.setattr(ferment_panel, "read_fermentation_csv", read_data)
    monkeypatch.setattr(
        ferment_panel, "validate_data_sufficiency", lambda *_args: passed
    )
    monkeypatch.setattr(ferment_panel, "fit_monod", lambda *_args: fit)
    monkeypatch.setattr(
        ferment_panel, "validate_mu_max_range", lambda *_args: passed
    )
    panel._complete_monod_fit = lambda payload: rendered.append(
        (payload, threading.get_ident())
    )
    panel._file_selector._path_edit.setText("synthetic.csv")

    try:
        panel._on_fit_monod()
        assert panel._background_task_active
        assert not panel._fit_button.isEnabled()
        assert panel._cancel_button.isEnabled()
        release.set()
        assert _wait_until(app, lambda: not panel._background_task_active)
        assert rendered[0][0][0] is fit
        assert rendered[0][1] == main_thread
    finally:
        release.set()
        assert _wait_until(app, lambda: not panel._background_task_active)
        panel.close()
        app.processEvents()


def test_fba_panel_dispatches_model_load_and_solve_to_worker(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    panel = FbaPanel()
    release = threading.Event()
    rendered: list[tuple[object, int]] = []
    main_thread = threading.get_ident()
    model = object()
    result = object()
    validation = object()

    def load(_source: str):
        release.wait(timeout=2.0)
        return model

    monkeypatch.setattr(fba_panel, "_COBRA_AVAILABLE", True)
    monkeypatch.setattr(fba_panel, "load_model", load)
    monkeypatch.setattr(fba_panel, "run_fba", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        fba_panel, "validate_objective_feasibility", lambda _result: validation
    )
    monkeypatch.setattr(
        fba_panel, "validate_mass_balance", lambda _model, _result: validation
    )
    panel._complete_fba = lambda payload: rendered.append(
        (payload, threading.get_ident())
    )
    panel._source_edit.setText("local-model.xml")

    try:
        panel._on_run_fba()
        assert panel._background_task_active
        assert not panel._run_fba_btn.isEnabled()
        assert panel._cancel_btn.isEnabled()
        release.set()
        assert _wait_until(app, lambda: not panel._background_task_active)
        assert rendered[0][0] == (
            model,
            "local-model.xml",
            result,
            validation,
            validation,
        )
        assert rendered[0][1] == main_thread
    finally:
        release.set()
        assert _wait_until(app, lambda: not panel._background_task_active)
        panel.close()
        app.processEvents()
