"""Metabolic flux analysis module.

This package provides Flux Balance Analysis (FBA) solving, blocked
reaction detection, dead-end metabolite identification, and metabolic
network validation rules using the COBRApy framework.
"""

from __future__ import annotations

from bioeng_qpcr.metabolic_flux.analysis import (
    BlockedReactionResult,
    DeadEndResult,
    find_blocked_reactions,
    find_dead_end_metabolites,
)
from bioeng_qpcr.metabolic_flux.fba import (
    FBAResult,
    ModelSummary,
    load_model,
    run_fba,
)
from bioeng_qpcr.metabolic_flux.validation import (
    FluxValidationResult,
    validate_mass_balance,
    validate_objective_feasibility,
)

__all__ = [
    # FBA
    "FBAResult",
    "ModelSummary",
    "load_model",
    "run_fba",
    # Analysis
    "BlockedReactionResult",
    "DeadEndResult",
    "find_blocked_reactions",
    "find_dead_end_metabolites",
    # Validation
    "FluxValidationResult",
    "validate_objective_feasibility",
    "validate_mass_balance",
]
