"""Local pytest configuration for the plotting test suite.

pytest's built-in ``tmp_path`` fixture places per-test directories under the
system temp folder (``%TEMP%\\pytest-of-<user>``).  On some Windows
environments that path is not writable, which raises ``PermissionError``
during fixture setup.  This conftest overrides ``tmp_path`` for the plotting
tests only, so that temporary directories are created inside a writable
folder within the project tree instead.

The project-root ``conftest.py`` (which adds ``src`` to ``sys.path``) is
intentionally left untouched.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

# Pin a non-interactive backend before pyplot is imported so the suite runs
# headless regardless of collection order or the platform default backend.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

#: Writable base directory for per-test temp folders, kept inside the project
#: tree so it never depends on the (possibly locked) system temp folder.
_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_plotting"


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every matplotlib figure after each test.

    Plotting tests routinely create figures without closing them; without
    this cleanup matplotlib warns about too many open figures and leaks
    memory.  Closing ``"all"`` after each test keeps the suite clean.
    """
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
