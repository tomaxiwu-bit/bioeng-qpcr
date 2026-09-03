"""Local pytest configuration for the CLI test suite.

This conftest mirrors the patterns established by ``tests/io/conftest.py``
and ``tests/plotting/conftest.py``:

1. Pin a non-interactive matplotlib backend (``Agg``) **before** pyplot is
   imported, so the ``plot`` sub-command runs headless regardless of
   collection order.

2. Override pytest's built-in ``tmp_path`` fixture with a project-local
   variant.  The system temp folder (``%TEMP%``) is not always writable on
   this Windows environment; keeping per-test temp directories inside the
   project tree avoids ``PermissionError`` during fixture setup.

3. Close every matplotlib figure after each test to prevent resource leaks.

The project-root ``conftest.py`` (which adds ``src`` to ``sys.path``) is
intentionally left untouched.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

# Pin the backend before pyplot is imported anywhere in the test process.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

#: Writable base directory for per-test temp folders, kept inside the project
#: tree so it never depends on the (possibly locked) system temp folder.
_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_cli"


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every matplotlib figure after each test."""
    yield
    plt.close("all")


@pytest.fixture
def tmp_path() -> Iterator[Path]:  # type: ignore[override]
    """Per-test temporary directory rooted inside the project tree.

    This is a drop-in replacement for pytest's built-in ``tmp_path`` that
    avoids the system temp directory.
    """
    _BASETEMP.mkdir(parents=True, exist_ok=True)
    directory = _BASETEMP / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_basetemp() -> Iterator[None]:
    """Remove the project-local temp base once the session finishes."""
    yield
    if _BASETEMP.exists():
        shutil.rmtree(_BASETEMP, ignore_errors=True)
