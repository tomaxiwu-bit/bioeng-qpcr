"""Shared fixtures for metabolic flux tests.

Pins the Agg matplotlib backend and provides a textbook model fixture.
All tests in this directory are skipped if COBRApy is not installed.
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

_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_fba"

try:
    import cobra  # noqa: F401

    COBRA_AVAILABLE = True
except ImportError:
    COBRA_AVAILABLE = False

skip_if_no_cobra = pytest.mark.skipif(
    not COBRA_AVAILABLE,
    reason="COBRApy not installed; skipping FBA tests.",
)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def tmp_path() -> Iterator[Path]:  # type: ignore[override]
    _BASETEMP.mkdir(parents=True, exist_ok=True)
    directory = _BASETEMP / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_basetemp() -> Iterator[None]:
    yield
    if _BASETEMP.exists():
        shutil.rmtree(_BASETEMP, ignore_errors=True)


@pytest.fixture(scope="session")
def textbook_model():
    """Load the E. coli textbook model once per session."""
    if not COBRA_AVAILABLE:
        pytest.skip("COBRApy not installed")
    from cobra.io import load_model

    return load_model("textbook")
