"""Enzyme kinetics analysis module.

This package provides Michaelis-Menten parameter fitting, reversible
inhibition type classification, and enzyme-kinetics validation rules.
"""

from __future__ import annotations

from bioeng_qpcr.enzyme_kinetics.inhibition import (
    GlobalInhibitionFit,
    InhibitionDataset,
    InhibitionModelComparison,
    InhibitionResult,
    InhibitionType,
    classify_inhibition,
    compare_inhibition_models,
)
from bioeng_qpcr.enzyme_kinetics.michaelis_menten import (
    LineweaverBurkResult,
    MichaelisMentenResult,
    fit_lineweaver_burk,
    fit_michaelis_menten,
    read_kinetics_csv,
)
from bioeng_qpcr.enzyme_kinetics.validation import (
    KineticsValidationResult,
    validate_km_range,
    validate_substrate_span,
)

__all__ = [
    "GlobalInhibitionFit",
    "InhibitionDataset",
    "InhibitionModelComparison",
    "InhibitionResult",
    # Inhibition
    "InhibitionType",
    # Validation
    "KineticsValidationResult",
    "LineweaverBurkResult",
    # Michaelis-Menten
    "MichaelisMentenResult",
    "classify_inhibition",
    "compare_inhibition_models",
    "fit_lineweaver_burk",
    "fit_michaelis_menten",
    "read_kinetics_csv",
    "validate_km_range",
    "validate_substrate_span",
]
