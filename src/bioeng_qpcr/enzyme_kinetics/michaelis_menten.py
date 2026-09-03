"""Michaelis-Menten enzyme kinetics fitting.

Fits the Michaelis-Menten equation ``V = Vmax * [S] / (Km + [S])`` to
substrate-velocity data using :func:`scipy.optimize.curve_fit`.

The module also provides a Lineweaver-Burk (double-reciprocal) linear
regression for inhibition analysis, and a convenience function to read
enzyme-kinetics data from a tidy CSV file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "MichaelisMentenResult",
    "LineweaverBurkResult",
    "fit_michaelis_menten",
    "fit_lineweaver_burk",
    "read_kinetics_csv",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MichaelisMentenResult:
    """Result of a Michaelis-Menten parameter fit.

    Attributes
        vmax:
            Maximum velocity (asymptote).
        km:
            Michaelis constant (substrate concentration at half-maximal velocity).
        r_squared:
            Coefficient of determination of the nonlinear fit, or ``None``
            when the observed velocities have zero variance.
        n_points:
            Number of data points used.
        vmax_ci:
            95% confidence interval for Vmax ``(lower, upper)``.
        km_ci:
            95% confidence interval for Km ``(lower, upper)``.
        residuals:
            Raw residuals (observed - predicted).
    """

    vmax: float
    km: float
    r_squared: float | None
    n_points: int
    vmax_ci: tuple[float, float] | None = None
    km_ci: tuple[float, float] | None = None
    residuals: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LineweaverBurkResult:
    """Result of a Lineweaver-Burk (double-reciprocal) linear regression.

    Attributes
        slope:
            Slope of the LB line (= Km / Vmax).
        intercept:
            Y-intercept of the LB line (= 1 / Vmax).
        r_squared:
            Coefficient of determination.
        km:
            Michaelis constant derived from slope / intercept.
        vmax:
            Maximum velocity derived from 1 / intercept.
        n_points:
            Number of data points.
    """

    slope: float
    intercept: float
    r_squared: float
    km: float
    vmax: float
    n_points: int


# ---------------------------------------------------------------------------
# Michaelis-Menten equation
# ---------------------------------------------------------------------------


def _michaelis_menten(s: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """Michaelis-Menten equation: V = Vmax * [S] / (Km + [S])."""
    return vmax * s / (km + s)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_michaelis_menten(
    substrate: np.ndarray,
    velocity: np.ndarray,
) -> MichaelisMentenResult:
    """Fit Michaelis-Menten parameters to substrate-velocity data.

    Uses :func:`scipy.optimize.curve_fit` with the Levenberg-Marquardt
    method.  Initial estimates are derived from the data: ``Vmax₀`` is
    set to the maximum observed velocity and ``Km₀`` to the substrate
    concentration at the half-maximal velocity.

    Parameters
        substrate:
            Array of substrate concentrations (must be positive).
        velocity:
            Array of initial reaction velocities (must be non-negative).

    Returns
        :class:`MichaelisMentenResult` with fitted Vmax, Km, R² and
        95% confidence intervals.

    Raises
        ValueError:
            If arrays have mismatched shapes, fewer than 4 points, or
            contain non-positive substrate concentrations.
    """
    substrate = np.asarray(substrate, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if substrate.shape != velocity.shape:
        raise ValueError(
            f"substrate and velocity must have the same shape, "
            f"got {substrate.shape} vs {velocity.shape}."
        )
    if substrate.size < 4:
        raise ValueError(
            f"Need at least 4 points for Michaelis-Menten fitting, "
            f"got {substrate.size}."
        )
    if np.any(substrate <= 0):
        raise ValueError("Substrate concentrations must be positive.")
    if np.any(velocity < 0):
        raise ValueError("Velocities must be non-negative.")

    # Initial parameter estimates.
    vmax0 = float(np.max(velocity))
    # Km₀: substrate concentration closest to half-maximal velocity.
    half_vmax = vmax0 / 2.0
    km0 = float(substrate[np.argmin(np.abs(velocity - half_vmax))])
    if km0 <= 0:
        km0 = float(np.median(substrate))

    try:
        popt, pcov = curve_fit(
            _michaelis_menten,
            substrate,
            velocity,
            p0=[vmax0, km0],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Failed to fit Michaelis-Menten equation: {exc}"
        ) from exc

    vmax, km = (float(v) for v in popt)

    # R² of the nonlinear fit.
    predicted = _michaelis_menten(substrate, vmax, km)
    ss_res = float(np.sum((velocity - predicted) ** 2))
    ss_tot = float(np.sum((velocity - np.mean(velocity)) ** 2))
    warnings: list[str] = []
    zero_variance_tolerance = (
        np.finfo(float).eps**2
        * max(float(np.sum(velocity**2)), 1.0)
    )
    if ss_tot > zero_variance_tolerance:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = None
        warnings.append(
            "R-squared is undefined because the observed velocities have zero "
            "variance."
        )

    # 95% confidence intervals from the parameter covariance matrix.
    vmax_ci = None
    km_ci = None
    if pcov is not None and np.all(np.isfinite(pcov)):
        perr = np.sqrt(np.diag(pcov))
        vmax_ci = (float(vmax - 1.96 * perr[0]), float(vmax + 1.96 * perr[0]))
        km_ci = (float(km - 1.96 * perr[1]), float(km + 1.96 * perr[1]))

    residuals = tuple(float(r) for r in (velocity - predicted))

    return MichaelisMentenResult(
        vmax=vmax,
        km=km,
        r_squared=r_squared,
        n_points=int(substrate.size),
        vmax_ci=vmax_ci,
        km_ci=km_ci,
        residuals=residuals,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Lineweaver-Burk (double-reciprocal) linear regression
# ---------------------------------------------------------------------------


def fit_lineweaver_burk(
    substrate: np.ndarray,
    velocity: np.ndarray,
) -> LineweaverBurkResult:
    """Fit a Lineweaver-Burk (double-reciprocal) line.

    Transforms data to ``1/V`` vs ``1/[S]`` and performs linear
    regression.  Useful for inhibition analysis and as a quick
    parameter estimate.

    Parameters
        substrate:
            Array of substrate concentrations (must be positive).
        velocity:
            Array of velocities (must be positive — zero velocities
            cannot be reciprocal-transformed).

    Returns
        :class:`LineweaverBurkResult` with slope, intercept, R², and
        derived Km and Vmax.

    Raises
        ValueError:
            If arrays have mismatched shapes, fewer than 3 points,
            or contain non-positive values.
    """
    substrate = np.asarray(substrate, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if substrate.shape != velocity.shape:
        raise ValueError(
            f"substrate and velocity must have the same shape, "
            f"got {substrate.shape} vs {velocity.shape}."
        )
    if substrate.size < 3:
        raise ValueError(
            f"Need at least 3 points for Lineweaver-Burk fitting, "
            f"got {substrate.size}."
        )
    if np.any(substrate <= 0):
        raise ValueError("Substrate concentrations must be positive.")
    if np.any(velocity <= 0):
        raise ValueError(
            "Velocities must be positive for Lineweaver-Burk transformation."
        )

    inv_s = 1.0 / substrate
    inv_v = 1.0 / velocity

    coeffs = np.polyfit(inv_s, inv_v, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    predicted = slope * inv_s + intercept
    ss_res = float(np.sum((inv_v - predicted) ** 2))
    ss_tot = float(np.sum((inv_v - np.mean(inv_v)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    vmax = 1.0 / intercept if intercept != 0 else float(np.inf)
    km = slope * vmax if intercept != 0 else float(np.inf)

    return LineweaverBurkResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        km=km,
        vmax=vmax,
        n_points=int(substrate.size),
    )


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------


def read_kinetics_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read a tidy enzyme-kinetics CSV file.

    Expected columns:
        - ``substrate_concentration`` (required)
        - ``velocity`` (required)
        - ``inhibitor_concentration`` (optional)

    Parameters
        csv_path:
            Path to the CSV file.

    Returns
        A tuple ``(substrate, velocity, inhibitor)`` where *inhibitor*
        is ``None`` when the column is absent.

    Raises
        ValueError:
            If required columns are missing or the file cannot be parsed.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    required = {"substrate_concentration", "velocity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}"
        )

    substrate = df["substrate_concentration"].to_numpy(dtype=float)
    velocity = df["velocity"].to_numpy(dtype=float)

    if "inhibitor_concentration" in df.columns:
        inhibitor = df["inhibitor_concentration"].to_numpy(dtype=float)
    else:
        inhibitor = None

    return substrate, velocity, inhibitor
