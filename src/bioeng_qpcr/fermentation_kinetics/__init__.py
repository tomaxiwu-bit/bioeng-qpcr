"""Fermentation kinetics analysis module.

This package provides Monod equation parameter fitting, batch
fermentation ODE solving, parameter identifiability analysis, and
fermentation-kinetics validation rules.
"""

from __future__ import annotations

from bioeng_qpcr.fermentation_kinetics.identifiability import (
    IdentifiabilityResult,
    analyze_identifiability,
)
from bioeng_qpcr.fermentation_kinetics.monod import (
    MonodFitResult,
    batch_fermentation_ode,
    fit_monod,
    monod_equation,
    read_fermentation_csv,
    solve_batch_fermentation,
)
from bioeng_qpcr.fermentation_kinetics.validation import (
    FermentationValidationResult,
    validate_data_sufficiency,
    validate_mu_max_range,
)

__all__ = [
    # Validation
    "FermentationValidationResult",
    # Identifiability
    "IdentifiabilityResult",
    # Monod
    "MonodFitResult",
    "analyze_identifiability",
    "batch_fermentation_ode",
    "fit_monod",
    "monod_equation",
    "read_fermentation_csv",
    "solve_batch_fermentation",
    "validate_data_sufficiency",
    "validate_mu_max_range",
]
