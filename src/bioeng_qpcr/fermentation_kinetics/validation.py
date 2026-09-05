"""Fermentation kinetics validation rules.

Two rules check the physical plausibility of fitted Monod parameters
and the sufficiency of the experimental data:

* **FERM-001** — μmax physical range: fitted μmax must be positive,
  finite, and within a broad biological range.
* **FERM-002** — Data sufficiency: at least 5 time points with both
  biomass growth and substrate consumption detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "FermentationValidationResult",
    "validate_data_sufficiency",
    "validate_mu_max_range",
]


@dataclass(frozen=True)
class FermentationValidationResult:
    """Outcome of a fermentation kinetics validation check.

    Attributes
        rule_id:
            Identifier of the rule (``"FERM-001"`` or ``"FERM-002"``).
        passed:
            Whether the check passed.
        message:
            Human-readable explanation.
        evidence:
            Structured supporting data.
    """

    rule_id: str
    passed: bool
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FERM-001: μmax physical range
# ---------------------------------------------------------------------------

#: Broad biological bounds for μmax (h⁻¹).
_MU_MAX_MIN: float = 1e-3  # 0.001 h⁻¹ (slow-growing organisms)
_MU_MAX_MAX: float = 5.0  # 5.0 h⁻¹ (fast-growing bacteria)


def validate_mu_max_range(mu_max: float) -> FermentationValidationResult:
    """FERM-001: Validate that fitted μmax is physically plausible.

    Parameters
        mu_max:
            Fitted maximum specific growth rate (h⁻¹).

    Returns
        :class:`FermentationValidationResult` with ``passed=True`` when
        μmax is positive, finite, and within ``[1e-3, 5.0]`` h⁻¹.
    """
    evidence: dict[str, Any] = {
        "mu_max": mu_max,
        "range": [_MU_MAX_MIN, _MU_MAX_MAX],
    }

    if not np.isfinite(mu_max):
        return FermentationValidationResult(
            rule_id="FERM-001",
            passed=False,
            message=f"μmax is not finite (got {mu_max}).",
            evidence=evidence,
        )
    if mu_max <= 0:
        return FermentationValidationResult(
            rule_id="FERM-001",
            passed=False,
            message=f"μmax must be positive (got {mu_max}).",
            evidence=evidence,
        )
    if mu_max < _MU_MAX_MIN or mu_max > _MU_MAX_MAX:
        return FermentationValidationResult(
            rule_id="FERM-001",
            passed=False,
            message=(
                f"μmax {mu_max} h⁻¹ is outside the plausible range "
                f"[{_MU_MAX_MIN}, {_MU_MAX_MAX}] h⁻¹."
            ),
            evidence=evidence,
        )
    return FermentationValidationResult(
        rule_id="FERM-001",
        passed=True,
        message=f"μmax {mu_max} h⁻¹ is within the plausible range.",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# FERM-002: Data sufficiency
# ---------------------------------------------------------------------------

#: Minimum number of time points for a reliable Monod fit.
_MIN_POINTS: int = 5


def validate_data_sufficiency(
    time: np.ndarray,
    biomass: np.ndarray,
    substrate: np.ndarray,
) -> FermentationValidationResult:
    """FERM-002: Validate data sufficiency for Monod parameter fitting.

    Checks that the dataset has at least 5 time points and that both
    biomass growth and substrate consumption are observed.

    Parameters
        time:
            Time points (h).
        biomass:
            Biomass concentrations (g/L).
        substrate:
            Substrate concentrations (g/L).

    Returns
        :class:`FermentationValidationResult` with ``passed=True`` when
        the data is sufficient for reliable fitting.
    """
    time = np.asarray(time, dtype=float)
    biomass = np.asarray(biomass, dtype=float)
    substrate = np.asarray(substrate, dtype=float)

    n_points = int(time.size)
    has_growth = bool(np.any(np.diff(biomass) > 0))
    has_consumption = bool(np.any(np.diff(substrate) < 0))

    evidence: dict[str, Any] = {
        "n_points": n_points,
        "min_required": _MIN_POINTS,
        "has_growth": has_growth,
        "has_consumption": has_consumption,
    }

    if n_points < _MIN_POINTS:
        return FermentationValidationResult(
            rule_id="FERM-002",
            passed=False,
            message=(
                f"Need at least {_MIN_POINTS} time points, "
                f"got {n_points}."
            ),
            evidence=evidence,
        )
    if not has_growth:
        return FermentationValidationResult(
            rule_id="FERM-002",
            passed=False,
            message="No biomass growth detected in data.",
            evidence=evidence,
        )
    if not has_consumption:
        return FermentationValidationResult(
            rule_id="FERM-002",
            passed=False,
            message="No substrate consumption detected in data.",
            evidence=evidence,
        )
    return FermentationValidationResult(
        rule_id="FERM-002",
        passed=True,
        message=(
            f"Data sufficient ({n_points} points, growth and "
            f"consumption detected)."
        ),
        evidence=evidence,
    )
