"""Shared fixtures for MCP server tests.

Pins the Agg matplotlib backend and provides a project-local tmp_path
fixture (same pattern as tests/cli/conftest.py).
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_mcp"


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

ENZYME_MM_CSV = str(
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "enzyme_mm_kinetics.csv"
)

ENZYME_INHIBITION_CSV = str(
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "enzyme_inhibition_kinetics.csv"
)

FERMENTATION_CSV = str(
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "fermentation_kinetics.csv"
)
