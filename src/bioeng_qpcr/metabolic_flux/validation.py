"""Metabolic flux validation rules.

Two rules check the quality of an FBA solution:

* **FBA-001** — Objective feasibility: the FBA solution status must be
  ``"optimal"`` and, for growth-maximisation objectives, the objective
  value must be positive.
* **FBA-002** — Mass balance: all internal metabolites must satisfy
  the steady-state assumption (total production = total consumption).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

__all__ = [
    "FluxValidationResult",
    "validate_objective_feasibility",
    "validate_mass_balance",
]

if TYPE_CHECKING:
    from cobra import Model

    from bioeng_qpcr.metabolic_flux.fba import FBAResult


@dataclass(frozen=True)
class FluxValidationResult:
    """Outcome of a metabolic flux validation check.

    Attributes
        rule_id:
            Identifier of the rule (``"FBA-001"`` or ``"FBA-002"``).
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
# FBA-001: Objective feasibility
# ---------------------------------------------------------------------------


def validate_objective_feasibility(
    result: FBAResult,
) -> FluxValidationResult:
    """FBA-001: Validate that the FBA solution is feasible.

    Checks that:
    1. The solver status is ``"optimal"``.
    2. For maximisation objectives, the objective value is positive
       (a zero or negative growth rate suggests an infeasible model).

    Parameters
        result:
            A :class:`~FBAResult` from :func:`run_fba`.

    Returns
        :class:`FluxValidationResult` with ``passed=True`` when the
        solution is optimal and the objective value is positive (for
        maximisation).
    """
    evidence: dict[str, Any] = {
        "status": result.status,
        "objective_value": result.objective_value,
        "objective_direction": (
            result.model_summary.objective_direction
            if result.model_summary
            else "unknown"
        ),
    }

    if result.status != "optimal":
        return FluxValidationResult(
            rule_id="FBA-001",
            passed=False,
            message=(
                f"FBA solution status is '{result.status}', "
                f"expected 'optimal'."
            ),
            evidence=evidence,
        )

    # For maximisation, check positive objective.
    direction = (
        result.model_summary.objective_direction
        if result.model_summary
        else "max"
    )
    if direction == "max" and result.objective_value <= 0:
        return FluxValidationResult(
            rule_id="FBA-001",
            passed=False,
            message=(
                f"Objective value {result.objective_value:.6f} is "
                f"non-positive for a maximisation problem, indicating "
                f"the model cannot grow."
            ),
            evidence=evidence,
        )

    return FluxValidationResult(
        rule_id="FBA-001",
        passed=True,
        message=(
            f"FBA solution is optimal with objective value "
            f"{result.objective_value:.6f}."
        ),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# FBA-002: Mass balance
# ---------------------------------------------------------------------------


def validate_mass_balance(
    model: Model,
    result: FBAResult,
    *,
    tolerance: float = 1e-6,
) -> FluxValidationResult:
    """FBA-002: Validate steady-state mass balance.

    Computes the mass balance for each internal metabolite as the
    sum of (flux × stoichiometric coefficient) over all reactions.
    At steady state this should be zero for every metabolite.

    External metabolites are identified by their participation in
    exchange (``EX_``), demand (``DM_``), or sink (``SK_``) reactions
    rather than by a hard-coded compartment identifier, making the
    check robust to models that use non-standard compartment naming.

    Parameters
        model:
            The :class:`cobra.Model` used for FBA.
        result:
            The :class:`~FBAResult` with flux values.
        tolerance:
            Absolute tolerance for the balance residual.

    Returns
        :class:`FluxValidationResult` with ``passed=True`` when all
        internal metabolites are balanced.
    """
    _BOUNDARY_PREFIXES = ("EX_", "DM_", "SK_")

    def _is_boundary_reaction(rxn_id: str) -> bool:
        return any(rxn_id.startswith(p) for p in _BOUNDARY_PREFIXES)

    # Identify external metabolites as those that participate in
    # exchange / demand / sink reactions.  This is model-agnostic and
    # does not rely on hard-coded compartment identifiers.
    external_met_ids: set[str] = set()
    for rxn in model.reactions:
        if _is_boundary_reaction(rxn.id):
            external_met_ids.update(met.id for met in rxn.metabolites)

    imbalanced: list[str] = []
    max_residual = 0.0
    n_checked = 0

    for met in model.metabolites:
        if met.id in external_met_ids:
            continue
        n_checked += 1

        total = 0.0
        for rxn in met.reactions:
            if _is_boundary_reaction(rxn.id):
                continue
            flux = result.fluxes.get(rxn.id, 0.0)
            coeff = rxn.metabolites[met]
            total += flux * coeff
        residual = abs(total)
        if residual > max_residual:
            max_residual = residual
        if residual > tolerance:
            imbalanced.append(met.id)

    evidence: dict[str, Any] = {
        "n_metabolites_checked": n_checked,
        "n_metabolites_total": len(model.metabolites),
        "n_imbalanced": len(imbalanced),
        "max_residual": max_residual,
        "tolerance": tolerance,
        "imbalanced_ids": imbalanced[:10],
    }

    if imbalanced:
        return FluxValidationResult(
            rule_id="FBA-002",
            passed=False,
            message=(
                f"{len(imbalanced)} of {n_checked} internal metabolite(s) "
                f"violate steady-state mass balance "
                f"(max residual={max_residual:.2e})."
            ),
            evidence=evidence,
        )

    return FluxValidationResult(
        rule_id="FBA-002",
        passed=True,
        message=(
            f"All {n_checked} internal metabolites satisfy "
            f"steady-state mass balance (max residual={max_residual:.2e})."
        ),
        evidence=evidence,
    )
