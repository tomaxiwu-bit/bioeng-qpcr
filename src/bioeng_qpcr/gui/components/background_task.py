"""Qt thread-pool primitives for responsive scientific GUI operations."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

__all__ = [
    "BackgroundTask",
    "BackgroundTaskSignals",
    "CancellationToken",
    "TaskCancelled",
]


class TaskCancelled(Exception):
    """Internal control-flow exception raised at cancellation checkpoints."""


class BackgroundTaskSignals(QObject):
    """Signals emitted by a :class:`BackgroundTask` from a worker thread."""

    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal()
    finished = Signal()


class CancellationToken:
    """Thread-safe cooperative cancellation and progress-reporting token."""

    def __init__(self, progress_reporter: Callable[[str], None]) -> None:
        self._cancelled = Event()
        self._progress_reporter = progress_reporter

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise TaskCancelled

    def report(self, message: str) -> None:
        """Emit progress unless cancellation has already been requested."""
        self.raise_if_cancelled()
        self._progress_reporter(message)


class BackgroundTask(QRunnable):
    """Execute ``operation(token)`` in Qt's global thread pool."""

    def __init__(self, operation: Callable[[CancellationToken], Any]) -> None:
        super().__init__()
        self.signals = BackgroundTaskSignals()
        self.token = CancellationToken(self.signals.progress.emit)
        self._operation = operation
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self.token.cancel()

    @Slot()
    def run(self) -> None:
        try:
            self.token.raise_if_cancelled()
            result = self._operation(self.token)
            self.token.raise_if_cancelled()
        except TaskCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - forwarded to the GUI boundary
            self.signals.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

