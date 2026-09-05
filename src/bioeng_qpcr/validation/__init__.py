"""Validation framework for bioeng-qpcr.

This package provides the infrastructure for MIQE-compliance rule checking:

* :class:`RuleMetadata` -- immutable description of a validation rule.
* :class:`RuleResult` -- the outcome of evaluating a rule.
* :func:`validation_rule` -- decorator that binds metadata to a rule function.
* :class:`RuleRegistry` -- thread-safe singleton catalogue of all rules.
* :class:`ValidationEngine` -- orchestrates rule execution against an experiment.
* :class:`ValidationReport` -- aggregated, serialisable validation outcome.

The framework deliberately separates *what a rule is* (metadata) from *what
a rule found* (result), and keeps rule definition, decoration, and
registration as independent concerns.
"""

from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.engine import RuleExecutionError, ValidationEngine
from bioeng_qpcr.validation.miqe2 import (
    MIQE2_CATEGORIES,
    MIQE2_FIELDS,
    Miqe2ChecklistItem,
    Miqe2Report,
    miqe2_checklist,
)
from bioeng_qpcr.validation.registry import RuleRegistry
from bioeng_qpcr.validation.report import ValidationReport
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = [
    "MIQE2_CATEGORIES",
    "MIQE2_FIELDS",
    "Miqe2ChecklistItem",
    "Miqe2Report",
    "RuleExecutionError",
    "RuleMetadata",
    "RuleRegistry",
    "RuleResult",
    "ValidationEngine",
    "ValidationReport",
    "miqe2_checklist",
    "validation_rule",
]
