"""Shared fixtures for widget tests."""

from __future__ import annotations

import shutil
import uuid
import warnings
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

# Suppress the expected UserWarning from plt.show() under the non-interactive
# Agg backend; in real Jupyter usage an interactive backend is active.
warnings.filterwarnings(
    "ignore", message="FigureCanvasAgg is non-interactive", category=UserWarning
)

_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_widgets"


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


EXAMPLE_CSV = str(
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "researcher_example.csv"
)
