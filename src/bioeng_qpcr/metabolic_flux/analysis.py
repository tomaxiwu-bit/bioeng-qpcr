"""Metabolic network analysis: blocked reactions and dead-end metabolites.

* **Blocked reactions** are reactions that cannot carry any flux under
  any feasible steady-state condition.  They often indicate missing
  pathways, gaps in the network reconstruction, or dead transport routes.

* **Dead-end metabolites** are metabolites that are either only produced
  or only consumed by the network (no exchange or complete pathway),
  indicating a network gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

__all__ = [
    "BlockedReactionResult",
    "DeadEndResult",
    "find_blocked_reactions",
    "find_dead_end_metabolites",
]

if TYPE_CHECKING:
    from cobra import Model


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockedReactionResult:
    """Result of blocked reaction detection.

    Attributes
        blocked_reactions:
            IDs of reactions that cannot carry any flux.
        n_blocked:
            Number of blocked reactions.
        n_total:
            Total number of reactions tested.
        fraction_blocked:
            Fraction of reactions that are blocked.
    """

    blocked_reactions: tuple[str, ...]
    n_blocked: int
    n_total: int
    fraction_blocked: float


@dataclass(frozen=True)
class DeadEndResult:
    """Result of dead-end metabolite detection.

    Attributes
        dead_end_ids:
            IDs of metabolites that are dead-ends.
        n_dead_ends:
            Number of dead-end metabolites.
        n_total:
            Total number of metabolites in the model.
        details:
            Per-metabolite detail: ``{met_id: {"type": "sink"|"source"}}``.
    """

    dead_end_ids: tuple[str, ...]
    n_dead_ends: int
    n_total: int
    details: dict[str, dict[str, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Blocked reactions
# ---------------------------------------------------------------------------


def find_blocked_reactions(
    model: Model,
    *,
    threshold: float = 1e-6,
) -> BlockedReactionResult:
    """Find reactions that cannot carry any flux.

    Uses COBRApy's ``find_blocked_reactions`` which performs FVA
    (Flux Variability Analysis) to determine whether each reaction
    can carry a non-zero flux under the model's constraints.

    Parameters
        model:
            A :class:`cobra.Model` instance.
        threshold:
            Flux magnitude below which a reaction is considered blocked.
            Must be >= the model's solver tolerance.

    Returns
        :class:`BlockedReactionResult` with the blocked reaction IDs.
    """
    from cobra.flux_analysis import find_blocked_reactions as _cobra_blocked

    # Ensure threshold is not below the model's tolerance.
    model_tolerance = getattr(model, "tolerance", 1e-6)
    effective_threshold = max(threshold, model_tolerance)

    blocked_list = _cobra_blocked(model, zero_cutoff=effective_threshold)
    n_total = len(model.reactions)
    n_blocked = len(blocked_list)
    fraction = n_blocked / n_total if n_total > 0 else 0.0

    return BlockedReactionResult(
        blocked_reactions=tuple(blocked_list),
        n_blocked=n_blocked,
        n_total=n_total,
        fraction_blocked=fraction,
    )


# ---------------------------------------------------------------------------
# Dead-end metabolites
# ---------------------------------------------------------------------------


def find_dead_end_metabolites(
    model: Model,
) -> DeadEndResult:
    """Find dead-end metabolites in a metabolic network.

    A dead-end metabolite is a metabolite that is:
    - **Sink**: only consumed (all reactions produce it with negative
      coefficient, or it has no producing reactions).
    - **Source**: only produced (all reactions consume it with positive
      coefficient, or it has no consuming reactions).

    Exchange and demand reactions are excluded from this analysis
    because they represent external sources/sinks.

    Parameters
        model:
            A :class:`cobra.Model` instance.

    Returns
        :class:`DeadEndResult` with the dead-end metabolite IDs.
    """
    dead_ends: list[str] = []
    details: dict[str, dict[str, str]] = {}

    for met in model.metabolites:
        # Get all reactions involving this metabolite, excluding
        # exchange (EX_) and demand (DM_) reactions.
        internal_reactions = [rxn for rxn in met.reactions if not rxn.boundary]

        if not internal_reactions:
            # Metabolite has no internal reactions at all.
            dead_ends.append(met.id)
            details[met.id] = {"type": "isolated"}
            continue

        # A coefficient sign describes only the forward direction. Reaction
        # bounds decide whether forward and/or reverse flux is feasible.
        has_producer = False
        has_consumer = False
        for rxn in internal_reactions:
            coefficient = float(rxn.metabolites[met])
            forward_possible = rxn.upper_bound > 0.0
            reverse_possible = rxn.lower_bound < 0.0
            if coefficient > 0.0:
                has_producer |= forward_possible
                has_consumer |= reverse_possible
            elif coefficient < 0.0:
                has_consumer |= forward_possible
                has_producer |= reverse_possible

        if not has_producer and not has_consumer:
            dead_ends.append(met.id)
            details[met.id] = {"type": "isolated"}
        elif has_producer and not has_consumer:
            dead_ends.append(met.id)
            details[met.id] = {"type": "source"}
        elif has_consumer and not has_producer:
            dead_ends.append(met.id)
            details[met.id] = {"type": "sink"}
        # If both or neither, it's not a dead-end.

    n_total = len(model.metabolites)

    return DeadEndResult(
        dead_end_ids=tuple(dead_ends),
        n_dead_ends=len(dead_ends),
        n_total=n_total,
        details=details,
    )
