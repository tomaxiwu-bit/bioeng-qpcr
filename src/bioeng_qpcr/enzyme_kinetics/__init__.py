"""Enzyme kinetics analysis module.

This package provides Michaelis-Menten parameter fitting, reversible
inhibition type classification, and enzyme-kinetics validation rules.
"""

from __future__ import annotations

from bioeng_qpcr.enzyme_kinetics.inhibition import (
    InhibitionDataset,
    InhibitionResult,
    InhibitionType,
    classify_inhibition,
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
    # Michaelis-Menten
    "MichaelisMentenResult",
    "LineweaverBurkResult",
    "fit_michaelis_menten",
    "fit_lineweaver_burk",
    "read_kinetics_csv",
    # Inhibition
    "InhibitionType",
    "InhibitionDataset",
    "InhibitionResult",
    "classify_inhibition",
    # Validation
    "KineticsValidationResult",
    "validate_km_range",
    "validate_substrate_span",
]
