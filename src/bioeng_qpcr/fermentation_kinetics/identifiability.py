"""Parameter identifiability analysis for Monod kinetics.

Assesses whether the fitted Monod parameters (μmax, Ks, Yxs) can be
uniquely determined from the available experimental data using the
Fisher Information Matrix (FIM) approach.

The FIM is computed as ``FIM = J^T * J`` where ``J`` is the sensitivity
matrix (partial derivatives of model outputs with respect to parameters),
estimated via finite differences.  The condition number and parameter
correlation matrix reveal practical identifiability issues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioeng_qpcr.fermentation_kinetics.monod import (
    MonodFitResult,
    solve_batch_fermentation,
)

__all__ = [
    "IdentifiabilityResult",
    "analyze_identifiability",
]


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Outcome of parameter identifiability analysis.

    Attributes
        parameter_names:
            Names of the analysed parameters.
        correlation_matrix:
            Flattened correlation matrix as nested tuples.
        eigenvalues:
            Positive eigenvalues of the FIM, sorted ascending.
        condition_number:
            Ratio of largest to smallest positive eigenvalue.
        is_identifiable:
            Whether parameters are practically identifiable.
        confidence:
            Heuristic confidence (``"high"``, ``"medium"``, ``"low"``).
        message:
            Human-readable explanation.
    """

    parameter_names: tuple[str, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    condition_number: float
    is_identifiable: bool
    confidence: str
    message: str


def _compute_sensitivity_matrix(
    time: np.ndarray,
    x0: float,
    s0: float,
    mu_max: float,
    ks: float,
    yxs: float,
    biomass_scale: float,
    substrate_scale: float,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute sensitivity matrix via central finite differences.

    Returns an ``(n_observations, n_parameters)`` matrix where
    ``n_observations = 2 * n_time_points`` (biomass + substrate).
    """
    log_params = np.log(np.array([mu_max, ks, yxs], dtype=float))
    n_params = len(log_params)
    n_obs = 2 * len(time)

    sens = np.zeros((n_obs, n_params))
    for i in range(n_params):
        h = epsilon

        params_fwd = log_params.copy()
        params_fwd[i] += h
        x_fwd, s_fwd = solve_batch_fermentation(
            time, x0, s0, *np.exp(params_fwd).tolist()
        )
        fwd = np.concatenate(
            [x_fwd / biomass_scale, s_fwd / substrate_scale]
        )

        params_bwd = log_params.copy()
        params_bwd[i] -= h
        x_bwd, s_bwd = solve_batch_fermentation(
            time, x0, s0, *np.exp(params_bwd).tolist()
        )
        bwd = np.concatenate(
            [x_bwd / biomass_scale, s_bwd / substrate_scale]
        )

        sens[:, i] = (fwd - bwd) / (2.0 * h)

    return sens


def analyze_identifiability(
    result: MonodFitResult,
    time: np.ndarray,
    biomass: np.ndarray,
    substrate: np.ndarray,
) -> IdentifiabilityResult:
    """Analyse practical identifiability of fitted Monod parameters.

    Parameters
        result:
            A fitted :class:`~MonodFitResult`.
        time:
            Time points used for fitting (h).
        biomass:
            Observed biomass concentrations (g/L).
        substrate:
            Observed substrate concentrations (g/L).

    Returns
        :class:`IdentifiabilityResult` with correlation matrix,
        eigenvalues, condition number, and a confidence rating.
    """
    time = np.asarray(time, dtype=float)
    biomass = np.asarray(biomass, dtype=float)
    substrate = np.asarray(substrate, dtype=float)
    if not (time.shape == biomass.shape == substrate.shape):
        raise ValueError("time, biomass, and substrate must have the same shape.")

    # Compute sensitivity matrix (Jacobian).
    jacobian = _compute_sensitivity_matrix(
        time,
        result.x0,
        result.s0,
        result.mu_max,
        result.ks,
        result.yxs,
        result.biomass_scale,
        result.substrate_scale,
    )

    # Fisher information for dimensionless outputs and log-parameters. This
    # makes the condition number invariant to concentration and parameter units.
    fim = jacobian.T @ jacobian

    # Covariance and correlation matrices.
    n_params = 3
    try:
        cov = np.linalg.pinv(fim)
        perr = np.sqrt(np.abs(np.diag(cov)))
        corr = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(n_params):
                if perr[i] > 0 and perr[j] > 0:
                    corr[i, j] = cov[i, j] / (perr[i] * perr[j])
                else:
                    corr[i, j] = 0.0
    except np.linalg.LinAlgError:
        corr = np.full((n_params, n_params), np.nan)

    # Eigenvalues of FIM.
    try:
        all_eigenvalues = np.linalg.eigvalsh(fim)
        eigenvalues = np.sort(all_eigenvalues[all_eigenvalues > 1e-30])
    except np.linalg.LinAlgError:
        eigenvalues = np.array([np.nan])

    # Condition number.
    if eigenvalues.size > 0 and eigenvalues[0] > 0:
        condition_number = float(eigenvalues[-1] / eigenvalues[0])
    else:
        condition_number = float("inf")

    # Maximum off-diagonal absolute correlation.
    max_corr = 0.0
    for i in range(n_params):
        for j in range(i + 1, n_params):
            if np.isfinite(corr[i, j]):
                max_corr = max(max_corr, abs(corr[i, j]))

    param_names = ("mu_max", "ks", "yxs")

    if condition_number < 100 and max_corr < 0.9:
        is_identifiable = True
        confidence = "high"
        message = (
            f"Parameters are well identifiable "
            f"(condition number={condition_number:.1f}, "
            f"max correlation={max_corr:.2f})."
        )
    elif condition_number < 1000 and max_corr < 0.95:
        is_identifiable = True
        confidence = "medium"
        message = (
            f"Parameters are practically identifiable but correlated "
            f"(condition number={condition_number:.1f}, "
            f"max correlation={max_corr:.2f})."
        )
    else:
        is_identifiable = False
        confidence = "low"
        message = (
            f"Parameters are poorly identifiable "
            f"(condition number={condition_number:.1f}, "
            f"max correlation={max_corr:.2f}). "
            f"Consider collecting more data near Ks."
        )

    return IdentifiabilityResult(
        parameter_names=param_names,
        correlation_matrix=tuple(tuple(row) for row in corr),
        eigenvalues=tuple(float(e) for e in eigenvalues),
        condition_number=condition_number,
        is_identifiable=is_identifiable,
        confidence=confidence,
        message=message,
    )
