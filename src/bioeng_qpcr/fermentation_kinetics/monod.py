"""Monod fermentation kinetics: ODE solving and parameter fitting.

Provides the Monod equation, batch-fermentation ODE integration, and
nonlinear parameter fitting from time-series biomass/substrate data.

The Monod equation relates specific growth rate to substrate:

    μ = μmax * S / (Ks + S)

The batch fermentation ODE system is:

    dX/dt = μ * X
    dS/dt = -μ * X / Yxs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import t as student_t

__all__ = [
    "MonodFitResult",
    "monod_equation",
    "batch_fermentation_ode",
    "solve_batch_fermentation",
    "fit_monod",
    "read_fermentation_csv",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonodFitResult:
    """Result of Monod parameter fitting.

    Attributes
        mu_max:
            Maximum specific growth rate (h⁻¹).
        ks:
            Half-saturation constant (g/L).
        yxs:
            Apparent yield coefficient (g biomass / g substrate).
        r_squared:
            Mean of the defined response-channel coefficients of determination,
            or ``None`` when neither observed channel has variance.
        n_points:
            Number of time points used.
        mu_max_ci:
            95% confidence interval for μmax.
        ks_ci:
            95% confidence interval for Ks.
        yxs_ci:
            95% confidence interval for Yxs.
        residuals:
            Raw residuals (observed - predicted).
        x0:
            Initial biomass concentration (g/L).
        s0:
            Initial substrate concentration (g/L).
        biomass_scale, substrate_scale:
            Response scales used to make the two residual channels
            dimensionless before fitting.
        residual_variance:
            Estimated variance of the scaled residuals.
        ci_method:
            Method used for parameter confidence intervals.
    """

    mu_max: float
    ks: float
    yxs: float
    r_squared: float | None
    n_points: int
    mu_max_ci: tuple[float, float] | None = None
    ks_ci: tuple[float, float] | None = None
    yxs_ci: tuple[float, float] | None = None
    residuals: tuple[float, ...] = ()
    x0: float = 0.0
    s0: float = 0.0
    biomass_scale: float = 1.0
    substrate_scale: float = 1.0
    residual_variance: float | None = None
    ci_method: str = "scaled-log-wald-t"
    biomass_r_squared: float | None = None
    substrate_r_squared: float | None = None
    warnings: tuple[str, ...] = ()


def _response_scale(values: np.ndarray, declared: float | None) -> float:
    """Resolve a positive scale for one measured response channel."""
    if declared is not None:
        value = float(declared)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Declared response scales must be finite and positive.")
        return value
    dynamic_range = float(np.ptp(values))
    magnitude_floor = float(np.max(np.abs(values))) * 1e-6
    return max(dynamic_range, magnitude_floor, float(np.sqrt(np.finfo(float).eps)))


# ---------------------------------------------------------------------------
# Monod equation
# ---------------------------------------------------------------------------


def monod_equation(
    s: float | np.ndarray,
    mu_max: float,
    ks: float,
) -> float | np.ndarray:
    """Monod equation: μ = μmax * S / (Ks + S)."""
    return mu_max * s / (ks + s)


# ---------------------------------------------------------------------------
# Batch fermentation ODE
# ---------------------------------------------------------------------------


def batch_fermentation_ode(
    t: float,
    y: list[float],
    mu_max: float,
    ks: float,
    yxs: float,
) -> list[float]:
    """Batch fermentation ODE system.

    dX/dt = μ * X
    dS/dt = -μ * X / Yxs
    """
    x, s = y
    x = max(x, 0.0)
    s = max(s, 0.0)
    mu = monod_equation(s, mu_max, ks)
    dxdt = mu * x
    dsdt = -mu * x / yxs if yxs > 0 else 0.0
    return [dxdt, dsdt]


# ---------------------------------------------------------------------------
# ODE solving
# ---------------------------------------------------------------------------


def solve_batch_fermentation(
    time: np.ndarray,
    x0: float,
    s0: float,
    mu_max: float,
    ks: float,
    yxs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the batch fermentation ODE at given time points.

    Parameters
        time:
            Array of time points (h).
        x0:
            Initial biomass concentration (g/L).
        s0:
            Initial substrate concentration (g/L).
        mu_max:
            Maximum specific growth rate (h⁻¹).
        ks:
            Half-saturation constant (g/L).
        yxs:
            Yield coefficient (g/g).

    Returns
        ``(biomass, substrate)`` arrays at the given time points.

    Raises
        RuntimeError:
            If ODE integration fails.
    """
    time = np.asarray(time, dtype=float)
    sol = solve_ivp(
        batch_fermentation_ode,
        t_span=(float(time[0]), float(time[-1])),
        y0=[x0, s0],
        t_eval=time,
        args=(mu_max, ks, yxs),
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.y[0], sol.y[1]


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------


def fit_monod(
    time: np.ndarray,
    biomass: np.ndarray,
    substrate: np.ndarray,
    *,
    biomass_scale: float | None = None,
    substrate_scale: float | None = None,
) -> MonodFitResult:
    """Fit Monod parameters from batch fermentation data.

    Uses :func:`scipy.optimize.least_squares` with the TRF method.
    Initial estimates are derived from the data.

    Parameters
        time:
            Time points (h).
        biomass:
            Biomass concentration at each time point (g/L).
        substrate:
            Substrate concentration at each time point (g/L).
        biomass_scale, substrate_scale:
            Optional known measurement-error scales for the two response
            channels. When omitted, each observed dynamic range is used so
            biomass and substrate contribute comparably despite different
            units or magnitudes.

    Returns
        :class:`MonodFitResult` with fitted μmax, Ks, Yxs and
        confidence intervals.

    Raises
        ValueError:
            If arrays have mismatched shapes, fewer than 5 points,
            or contain negative values.
    """
    time = np.asarray(time, dtype=float)
    biomass = np.asarray(biomass, dtype=float)
    substrate = np.asarray(substrate, dtype=float)

    if not (time.shape == biomass.shape == substrate.shape):
        raise ValueError(
            f"time, biomass, and substrate must have the same shape, "
            f"got {time.shape}, {biomass.shape}, {substrate.shape}."
        )
    if time.size < 5:
        raise ValueError(
            f"Need at least 5 time points for Monod fitting, "
            f"got {time.size}."
        )
    if np.any(time < 0):
        raise ValueError("Time values must be non-negative.")
    if np.any(biomass < 0):
        raise ValueError("Biomass concentrations must be non-negative.")
    if np.any(substrate < 0):
        raise ValueError("Substrate concentrations must be non-negative.")

    x0 = float(biomass[0])
    s0 = float(substrate[0])
    resolved_biomass_scale = _response_scale(biomass, biomass_scale)
    resolved_substrate_scale = _response_scale(substrate, substrate_scale)

    # Initial parameter estimates.
    mu_max0 = 0.5
    positive_s = substrate[substrate > 0]
    ks0 = float(np.median(positive_s)) if positive_s.size > 0 else 1.0
    delta_x = float(biomass[-1] - biomass[0])
    delta_s = float(substrate[0] - substrate[-1])
    yxs0 = delta_x / delta_s if delta_s > 0 else 0.5

    lower = np.array([1e-6, 1e-6, 1e-6], dtype=float)
    upper = np.array([10.0, 1e6, 1e6], dtype=float)
    p0 = np.clip(
        np.array([mu_max0, max(ks0, 1e-6), max(yxs0, 1e-6)], dtype=float),
        lower * (1.0 + 1e-9),
        upper / (1.0 + 1e-9),
    )

    def _residuals(log_p: np.ndarray) -> np.ndarray:
        mu_max, ks, yxs = np.exp(log_p)
        try:
            x_pred, s_pred = solve_batch_fermentation(
                time, x0, s0, mu_max, ks, yxs
            )
            return np.concatenate(
                [
                    (x_pred - biomass) / resolved_biomass_scale,
                    (s_pred - substrate) / resolved_substrate_scale,
                ]
            )
        except (RuntimeError, ValueError):
            return np.full(2 * time.size, 1e6)

    result = least_squares(
        _residuals,
        np.log(p0),
        bounds=(np.log(lower), np.log(upper)),
        method="trf",
        max_nfev=5000,
    )
    if not result.success:
        raise ValueError(f"Monod fitting failed to converge: {result.message}")

    log_parameters = result.x
    mu_max, ks, yxs = (float(v) for v in np.exp(log_parameters))

    # R² of the fit.
    x_pred, s_pred = solve_batch_fermentation(
        time, x0, s0, mu_max, ks, yxs
    )
    biomass_ss_res = float(np.sum((biomass - x_pred) ** 2))
    biomass_ss_tot = float(np.sum((biomass - np.mean(biomass)) ** 2))
    substrate_ss_res = float(np.sum((substrate - s_pred) ** 2))
    substrate_ss_tot = float(np.sum((substrate - np.mean(substrate)) ** 2))
    biomass_zero_variance_tolerance = (
        np.finfo(float).eps**2
        * max(float(np.sum(biomass**2)), 1.0)
    )
    substrate_zero_variance_tolerance = (
        np.finfo(float).eps**2
        * max(float(np.sum(substrate**2)), 1.0)
    )
    warnings: list[str] = []
    biomass_r_squared = (
        1.0 - biomass_ss_res / biomass_ss_tot
        if biomass_ss_tot > biomass_zero_variance_tolerance
        else None
    )
    substrate_r_squared = (
        1.0 - substrate_ss_res / substrate_ss_tot
        if substrate_ss_tot > substrate_zero_variance_tolerance
        else None
    )
    if biomass_r_squared is None:
        warnings.append(
            "Biomass R-squared is undefined because observed biomass has zero "
            "variance."
        )
    if substrate_r_squared is None:
        warnings.append(
            "Substrate R-squared is undefined because observed substrate has zero "
            "variance."
        )
    defined_r_squared = [
        value
        for value in (biomass_r_squared, substrate_r_squared)
        if value is not None
    ]
    r_squared = (
        float(np.mean(defined_r_squared)) if defined_r_squared else None
    )

    # Approximate 95% intervals on log-parameters. Scaling by the residual
    # variance and using a Student-t critical value fixes the previous
    # uncalibrated pinv(J'J) intervals. Exponentiation yields positive,
    # asymmetric intervals appropriate for bounded kinetic parameters.
    mu_max_ci = None
    ks_ci = None
    yxs_ci = None
    residual_variance = None
    if result.jac is not None:
        try:
            jtj = result.jac.T @ result.jac
            degrees_freedom = int(result.fun.size - log_parameters.size)
            if (
                degrees_freedom > 0
                and jtj.size > 0
                and np.all(np.isfinite(jtj))
                and np.linalg.matrix_rank(jtj) == log_parameters.size
            ):
                residual_variance = float(
                    np.sum(result.fun**2) / degrees_freedom
                )
                cov_log = residual_variance * np.linalg.pinv(jtj)
                perr_log = np.sqrt(np.maximum(np.diag(cov_log), 0.0))
                critical = float(student_t.ppf(0.975, degrees_freedom))
                intervals = [
                    (
                        float(np.exp(value - critical * error)),
                        float(np.exp(value + critical * error)),
                    )
                    for value, error in zip(log_parameters, perr_log, strict=True)
                ]
                mu_max_ci, ks_ci, yxs_ci = intervals
        except (np.linalg.LinAlgError, ValueError):
            pass

    residuals_tuple = tuple(
        float(r)
        for r in np.concatenate([biomass - x_pred, substrate - s_pred])
    )

    return MonodFitResult(
        mu_max=mu_max,
        ks=ks,
        yxs=yxs,
        r_squared=r_squared,
        n_points=int(time.size),
        mu_max_ci=mu_max_ci,
        ks_ci=ks_ci,
        yxs_ci=yxs_ci,
        residuals=residuals_tuple,
        x0=x0,
        s0=s0,
        biomass_scale=resolved_biomass_scale,
        substrate_scale=resolved_substrate_scale,
        residual_variance=residual_variance,
        biomass_r_squared=(
            float(biomass_r_squared) if biomass_r_squared is not None else None
        ),
        substrate_r_squared=(
            float(substrate_r_squared) if substrate_r_squared is not None else None
        ),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------


def read_fermentation_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a tidy fermentation kinetics CSV file.

    Expected columns:
        - ``time`` (required)
        - ``biomass`` (required)
        - ``substrate`` (required)

    Parameters
        csv_path:
            Path to the CSV file.

    Returns
        A tuple ``(time, biomass, substrate)``.

    Raises
        ValueError:
            If required columns are missing or the file cannot be parsed.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    required = {"time", "biomass", "substrate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}"
        )

    time = df["time"].to_numpy(dtype=float)
    biomass = df["biomass"].to_numpy(dtype=float)
    substrate = df["substrate"].to_numpy(dtype=float)

    return time, biomass, substrate
