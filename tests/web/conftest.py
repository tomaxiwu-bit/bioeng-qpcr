"""Local pytest configuration for the web test suite.

Mirrors the established conftest pattern: pins the non-interactive
matplotlib backend before pyplot is imported, overrides ``tmp_path`` with a
project-local writable directory, and closes every figure after each test.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

#: Writable base directory for per-test temp folders, kept inside the project
#: tree so it never depends on the (possibly locked) system temp folder.
_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_web"


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every matplotlib figure after each test."""
    yield
    plt.close("all")


@pytest.fixture
def tmp_path() -> Iterator[Path]:  # type: ignore[override]
    """Per-test temporary directory rooted inside the project tree."""
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