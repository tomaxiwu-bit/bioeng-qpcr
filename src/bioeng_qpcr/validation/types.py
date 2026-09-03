"""Core type definitions for the validation framework.

This module defines the two foundational data structures that every
MIQE-compliance rule in bioeng-qpcr relies upon:

* :class:`RuleMetadata` -- the static, declarative description of a rule
  (identifier, name, MIQE section reference, minimum input level, version,
  citation).  Metadata is attached to a rule once at definition time and
  never changes.

* :class:`RuleResult` -- the outcome of evaluating a single rule against an
  :class:`~bioeng_qpcr.domain.entities.Experiment`.  A result carries the
  rule's metadata, a four-state :class:`~bioeng_qpcr.domain.enums.RuleStatus`,
  a human-readable message, structured evidence, and a UTC timestamp.

Together these types enforce a strict separation between *what a rule is*
(metadata) and *what a rule found* (result), which is essential for
auditability and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bioeng_qpcr.domain.enums import InputLevel, RuleStatus

# ---------------------------------------------------------------------------
# RuleMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleMetadata:
    """Static, declarative description of a single validation rule.

    A :class:`RuleMetadata` instance is attached to every rule function via
    the :func:`~bioeng_qpcr.validation.decorator.validation_rule` decorator.
    Because the dataclass is frozen, metadata serves as an immutable contract
    that can be safely shared across threads and serialised for provenance.

    Parameters
        rule_id:
            Unique, stable rule identifier (e.g. ``"EFF-001"``).  The
            identifier must be unique within a single :class:`RuleRegistry`.
        name:
            Short human-readable name (e.g. ``"Efficiency range check"``).
        description:
            Prose description of what the rule checks and why it matters.
        miqe_section:
            Reference to the relevant section of the MIQE guidelines
            (e.g. ``"MIQE 3.2"``).
        min_input_level:
            The minimum :class:`~bioeng_qpcr.domain.enums.InputLevel` that
            must be present for the rule to be assessable.  Rules requiring
            raw traces or efficiency values should declare ``LEVEL_B`` or
            ``LEVEL_C``.  Defaults to :attr:`InputLevel.LEVEL_A`.
        version:
            Semantic version string for the rule implementation.  Bumping the
            version signals that the rule's behaviour has changed.  Defaults
            to ``"1.0.0"``.
        citation:
            Optional literature citation supporting the rule (e.g. a DOI or
            full reference).  ``None`` when no specific citation applies.

    Attributes
        rule_id, name, description, miqe_section, min_input_level, version, citation:
            See *Parameters* above.
    """

    rule_id: str
    name: str
    description: str
    miqe_section: str
    min_input_level: InputLevel = InputLevel.LEVEL_A
    version: str = "1.0.0"
    citation: str | None = None


# ---------------------------------------------------------------------------
# RuleResult
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    This small factory is used as the default factory for
    :attr:`RuleResult.timestamp` so that each result is stamped at the
    moment of construction.
    """
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RuleResult:
    """Outcome of evaluating a single validation rule.

    A :class:`RuleResult` is produced every time a rule function is invoked
    on an :class:`~bioeng_qpcr.domain.entities.Experiment`.  The result is
    immutable: once recorded, it forms part of the audit trail and must not
    be altered.

    Parameters
        metadata:
            The :class:`RuleMetadata` of the rule that produced this result.
        status:
            The :class:`~bioeng_qpcr.domain.enums.RuleStatus` outcome.  The
            four-state design (``PASS`` / ``WARN`` / ``FAIL`` /
            ``NOT_ASSESSABLE``) is central to the engine's philosophy: a
            missing input must never be conflated with a pass.
        message:
            Human-readable explanation of the outcome, suitable for display
            in a report.
        evidence:
            Structured supporting data (e.g. ``{"efficiency": 0.97,
            "threshold": 0.90}``).  Keys and values are free-form.  Defaults
            to an empty dict.
        timestamp:
            UTC timezone-aware datetime marking when the rule was evaluated.
            Defaults to the moment of construction.

    Attributes
        metadata, status, message, evidence, timestamp:
            See *Parameters* above.
    """

    metadata: RuleMetadata
    status: RuleStatus
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)

    # -- convenience properties ------------------------------------------

    @property
    def rule_id(self) -> str:
        """The rule identifier, delegated from :attr:`metadata`."""
        return self.metadata.rule_id

    @property
    def is_pass(self) -> bool:
        """``True`` when the rule passed."""
        return self.status is RuleStatus.PASS

    @property
    def is_fail(self) -> bool:
        """``True`` when the rule failed."""
        return self.status is RuleStatus.FAIL

    @property
    def is_warn(self) -> bool:
        """``True`` when the rule issued a warning."""
        return self.status is RuleStatus.WARN

    @property
    def is_not_assessable(self) -> bool:
        """``True`` when the rule could not be assessed (missing input)."""
        return self.status is RuleStatus.NOT_ASSESSABLE
