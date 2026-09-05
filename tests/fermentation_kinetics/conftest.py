"""Shared fixtures for fermentation kinetics tests."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bioeng_qpcr.fermentation_kinetics.monod import (
    solve_batch_fermentation,
)

_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_ferm"


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


# ---------------------------------------------------------------------------
# Synthetic fermentation data generators
# ---------------------------------------------------------------------------


def make_fermentation_data(
    mu_max: float = 0.5,
    ks: float = 2.0,
    yxs: float = 0.5,
    x0: float = 0.1,
    s0: float = 10.0,
    *,
    n_points: int = 12,
    t_end: float = 24.0,
    noise_seed: int = 42,
    noise_level: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic batch fermentation data with known parameters.

    Uses the Monod ODE to generate ground truth, then adds Gaussian noise.
    """
    rng = np.random.RandomState(noise_seed)
    time = np.linspace(0, t_end, n_points)
    x_true, s_true = solve_batch_fermentation(time, x0, s0, mu_max, ks, yxs)
    # Add small relative noise.
    x_noise = rng.normal(0, noise_level * max(np.max(x_true), 0.1), n_points)
    s_noise = rng.normal(0, noise_level * max(np.max(s_true), 0.1), n_points)
    biomass = np.maximum(x_true + x_noise, 0)
    substrate = np.maximum(s_true + s_noise, 0)
    return time, biomass, substrate


def make_noisy_fermentation_data(
    mu_max: float = 0.5,
    ks: float = 2.0,
    yxs: float = 0.5,
    x0: float = 0.1,
    s0: float = 10.0,
    *,
    n_points: int = 15,
    t_end: float = 24.0,
    noise_seed: int = 99,
    noise_level: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy fermentation data for identifiability testing."""
    return make_fermentation_data(
        mu_max, ks, yxs, x0, s0,
        n_points=n_points,
        t_end=t_end,
        noise_seed=noise_seed,
        noise_level=noise_level,
    )
