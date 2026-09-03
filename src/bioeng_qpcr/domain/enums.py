"""Domain enums for bioeng-qpcr.

These enumerations define the core vocabulary used across the entire engine.
They are stable, versioned, and form the contract between modules.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class ReplicateType(str, Enum):
    """Type of replicate measurement."""

    TECHNICAL = "technical"
    """Technical replicate — repeated measurement of the same biological sample."""

    BIOLOGICAL = "biological"
    """Biological replicate — distinct biological specimens."""

    SINGLE = "single"
    """No replicates; single measurement per target per sample."""


@unique
class RuleStatus(str, Enum):
    """Four-state validation result.

    The core design principle: a single ``compliant: true`` is scientifically
    indefensible when the input is only Cq CSV.  The engine must distinguish
    "passed" from "cannot be assessed."
    """

    PASS = "PASS"
    """Evidence satisfies the rule."""

    WARN = "WARN"
    """Evidence partially satisfies or is borderline."""

    FAIL = "FAIL"
    """Evidence violates the rule."""

    NOT_ASSESSABLE = "NOT_ASSESSABLE"
    """Required information not provided."""


@unique
class InputLevel(str, Enum):
    """Input data richness level.

    Higher levels carry more metadata, enabling more rules to be assessed.
    """

    LEVEL_A = "level_a"
    """Tidy CSV: well, sample, target, Cq. Minimal but sufficient for calculation."""

    LEVEL_B = "level_b"
    """Vendor export: includes raw fluorescence traces, efficiency, metadata."""

    LEVEL_C = "level_c"
    """RDML or structured format: full experimental context, annotations."""


@unique
class MethodType(str, Enum):
    """Quantification method for relative expression analysis."""

    DELTA_CQ = "delta_cq"
    """ΔCq method — simple fold change relative to a reference sample."""

    DELTA_DELTA_CQ = "delta_delta_cq"
    """ΔΔCq method — assumes ~100% efficiency for target and reference."""

    PFAFFL = "pfaffl"
    """Pfaffl method — efficiency-corrected relative quantification."""

    NORMALIZED_QUANTITY = "normalized_quantity"
    """Efficiency-corrected quantity with geometric reference normalization."""

    EFFICIENCY_FITTING = "efficiency_fitting"
    """Fit efficiency from standard-curve or amplification-curve data."""
