"""Enzyme kinetics validation rules.

Two rules check the physical plausibility of fitted Michaelis-Menten
parameters and the experimental design of the substrate-concentration
series:

* **ENZ-001** — Km physical range: fitted Km must be positive, finite,
  and within a broad biochemical range.
* **ENZ-002** — Substrate concentration span: the measured [S] series
  should bracket Km (at least one point below and one above) for a
  reliable fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "KineticsValidationResult",
    "validate_km_range",
    "validate_substrate_span",
]


@dataclass(frozen=True)
class KineticsValidationResult:
    """Outcome of an enzyme-kinetics validation check.

    Attributes
        rule_id:
            Identifier of the rule (``"ENZ-001"`` or ``"ENZ-002"``).
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
# ENZ-001: Km physical range
# ---------------------------------------------------------------------------

#: Broad biochemical bounds for Km (1 nM to 1 M in molar units).
_KM_MIN: float = 1e-9
_KM_MAX: float = 1e3


def validate_km_range(km: float) -> KineticsValidationResult:
    """ENZ-001: Validate that fitted Km is physically plausible.

    Parameters
        km:
            Fitted Michaelis constant.

    Returns
        :class:`KineticsValidationResult` with ``passed=True`` when Km
        is positive, finite, and within ``[1e-9, 1e3]``.
    """
    evidence: dict[str, Any] = {
        "km": km,
        "range": [_KM_MIN, _KM_MAX],
    }

    if not np.isfinite(km):
        return KineticsValidationResult(
            rule_id="ENZ-001",
            passed=False,
            message=f"Km is not finite (got {km}).",
            evidence=evidence,
        )
    if km <= 0:
        return KineticsValidationResult(
            rule_id="ENZ-001",
            passed=False,
            message=f"Km must be positive (got {km}).",
            evidence=evidence,
        )
    if km < _KM_MIN or km > _KM_MAX:
        return KineticsValidationResult(
            rule_id="ENZ-001",
            passed=False,
            message=(
                f"Km {km} is outside the plausible biochemical range "
                f"[{_KM_MIN}, {_KM_MAX}]."
            ),
            evidence=evidence,
        )
    return KineticsValidationResult(
        rule_id="ENZ-001",
        passed=True,
        message=f"Km {km} is within the plausible range.",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# ENZ-002: Substrate concentration span
# ---------------------------------------------------------------------------


def validate_substrate_span(
    substrate: np.ndarray,
    km: float,
) -> KineticsValidationResult:
    """ENZ-002: Validate that the [S] series brackets Km.

    A reliable Michaelis-Menten fit requires substrate concentrations
    both below and above Km.  This check verifies that at least one
    point falls in ``[0.1*Km, Km]`` and at least one in
    ``[Km, 10*Km]``.

    Parameters
        substrate:
            Array of substrate concentrations used in the assay.
        km:
            Fitted (or expected) Michaelis constant.

    Returns
        :class:`KineticsValidationResult` with ``passed=True`` when the
        [S] series adequately brackets Km.
    """
    substrate = np.asarray(substrate, dtype=float)
    lower = np.any((substrate >= 0.1 * km) & (substrate <= km))
    upper = np.any((substrate >= km) & (substrate <= 10 * km))

    below = np.any(substrate < km)
    above = np.any(substrate > km)

    evidence: dict[str, Any] = {
        "km": km,
        "substrate_min": float(np.min(substrate)) if substrate.size else None,
        "substrate_max": float(np.max(substrate)) if substrate.size else None,
        "has_below_km": bool(below),
        "has_above_km": bool(above),
        "n_points": int(substrate.size),
    }

    if not np.isfinite(km) or km <= 0:
        return KineticsValidationResult(
            rule_id="ENZ-002",
            passed=False,
            message=f"Cannot assess span: Km is invalid ({km}).",
            evidence=evidence,
        )

    if lower and upper:
        return KineticsValidationResult(
            rule_id="ENZ-002",
            passed=True,
            message=(
                f"Substrate series brackets Km ({km}): "
                f"points present both below and above Km."
            ),
            evidence=evidence,
        )
    if below and not upper:
        return KineticsValidationResult(
            rule_id="ENZ-002",
            passed=False,
            message=(
                f"All substrate concentrations are below Km ({km}); "
                f"add higher [S] points for a reliable fit."
            ),
            evidence=evidence,
        )
    if above and not below:
        return KineticsValidationResult(
            rule_id="ENZ-002",
            passed=False,
            message=(
                f"All substrate concentrations are above Km ({km}); "
                f"add lower [S] points for a reliable fit."
            ),
            evidence=evidence,
        )
    return KineticsValidationResult(
        rule_id="ENZ-002",
        passed=False,
        message="Substrate series does not adequately bracket Km.",
        evidence=evidence,
    )
