"""Shared fixtures for enzyme kinetics tests."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_BASETEMP = Path(__file__).resolve().parent.parent.parent / "_tmp_enzyme"


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
# Synthetic Michaelis-Menten data generators
# ---------------------------------------------------------------------------


def make_mm_data(
    vmax: float = 100.0,
    km: float = 10.0,
    *,
    n_points: int = 8,
    noise_seed: int = 42,
    noise_level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Michaelis-Menten data with known parameters."""
    rng = np.random.RandomState(noise_seed)
    substrate = np.geomspace(km / 5, km * 5, n_points)
    velocity = vmax * substrate / (km + substrate)
    if noise_level > 0:
        velocity += rng.normal(0, noise_level * np.max(velocity), n_points)
    return substrate, velocity


def make_competitive_inhibition_data(
    vmax: float = 100.0,
    km: float = 10.0,
    ki: float = 5.0,
    *,
    inhibitor_concs: list[float] | None = None,
    n_points: int = 8,
    noise_seed: int = 42,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Generate competitive inhibition data.

    Competitive: Km_app = Km * (1 + [I]/Ki), Vmax unchanged.
    """
    if inhibitor_concs is None:
        inhibitor_concs = [0.0, 2.5, 5.0, 10.0]

    rng = np.random.RandomState(noise_seed)
    datasets = []
    for i_conc in inhibitor_concs:
        km_app = km * (1 + i_conc / ki)
        substrate = np.geomspace(km_app / 5, km_app * 5, n_points)
        velocity = vmax * substrate / (km_app + substrate)
        velocity += rng.normal(0, 0.01 * np.max(velocity), n_points)
        datasets.append((i_conc, substrate, velocity))
    return datasets


def make_noncompetitive_inhibition_data(
    vmax: float = 100.0,
    km: float = 10.0,
    ki: float = 5.0,
    *,
    inhibitor_concs: list[float] | None = None,
    n_points: int = 8,
    noise_seed: int = 42,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Generate non-competitive inhibition data.

    Non-competitive: Vmax_app = Vmax / (1 + [I]/Ki), Km unchanged.
    """
    if inhibitor_concs is None:
        inhibitor_concs = [0.0, 2.5, 5.0, 10.0]

    rng = np.random.RandomState(noise_seed)
    datasets = []
    for i_conc in inhibitor_concs:
        vmax_app = vmax / (1 + i_conc / ki)
        substrate = np.geomspace(km / 5, km * 5, n_points)
        velocity = vmax_app * substrate / (km + substrate)
        velocity += rng.normal(0, 0.01 * np.max(velocity), n_points)
        datasets.append((i_conc, substrate, velocity))
    return datasets


def make_uncompetitive_inhibition_data(
    vmax: float = 100.0,
    km: float = 10.0,
    ki: float = 5.0,
    *,
    inhibitor_concs: list[float] | None = None,
    n_points: int = 8,
    noise_seed: int = 42,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Generate uncompetitive inhibition data.

    Uncompetitive: both Vmax and Km decrease by factor 1/(1+[I]/Ki).
    """
    if inhibitor_concs is None:
        inhibitor_concs = [0.0, 2.5, 5.0, 10.0]

    rng = np.random.RandomState(noise_seed)
    datasets = []
    for i_conc in inhibitor_concs:
        factor = 1 + i_conc / ki
        vmax_app = vmax / factor
        km_app = km / factor
        substrate = np.geomspace(km_app / 5, km_app * 5, n_points)
        velocity = vmax_app * substrate / (km_app + substrate)
        velocity += rng.normal(0, 0.01 * np.max(velocity), n_points)
        datasets.append((i_conc, substrate, velocity))
    return datasets


def make_mixed_inhibition_data(
    vmax: float = 100.0,
    km: float = 10.0,
    ki: float = 5.0,
    ki_prime: float = 10.0,
    *,
    inhibitor_concs: list[float] | None = None,
    n_points: int = 8,
    noise_seed: int = 42,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Generate mixed inhibition data.

    Mixed: Vmax_app = Vmax / (1 + [I]/Ki'), Km_app = Km * (1 + [I]/Ki) / (1 + [I]/Ki').
    """
    if inhibitor_concs is None:
        inhibitor_concs = [0.0, 2.5, 5.0, 10.0]

    rng = np.random.RandomState(noise_seed)
    datasets = []
    for i_conc in inhibitor_concs:
        alpha = 1 + i_conc / ki
        alpha_prime = 1 + i_conc / ki_prime
        vmax_app = vmax / alpha_prime
        km_app = km * alpha / alpha_prime
        substrate = np.geomspace(km_app / 5, km_app * 5, n_points)
        velocity = vmax_app * substrate / (km_app + substrate)
        velocity += rng.normal(0, 0.01 * np.max(velocity), n_points)
        datasets.append((i_conc, substrate, velocity))
    return datasets
