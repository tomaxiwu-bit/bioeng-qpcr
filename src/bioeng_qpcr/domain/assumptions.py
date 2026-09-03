"""Assumption tracking types.

These are placeholder types for future assumption-checking functionality.
In Phase 0 they exist to define the contract; full implementation is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bioeng_qpcr.domain.enums import RuleStatus


@dataclass(frozen=True)
class AssumptionStatus:
    """Status of a methodological assumption check.

    Attributes
        name:
            Human-readable assumption identifier (e.g. ``"equal_efficiency"``).
        status:
            Current assessment status.
        detail:
            Human-readable explanation.
    """

    name: str
    status: RuleStatus = RuleStatus.NOT_ASSESSABLE
    detail: str = ""

    def __str__(self) -> str:
        return f"[{self.status.value}] {self.name}: {self.detail}"


@dataclass
class AssumptionResult:
    """Collection of assumption statuses for an experiment.

    This is a placeholder container.  In Phase 1, the calculations module
    will populate it automatically when a quantification method is selected.
    """

    assumptions: list[AssumptionStatus] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, assumption: AssumptionStatus) -> None:
        self.assumptions.append(assumption)

    @property
    def all_pass(self) -> bool:
        return all(a.status == RuleStatus.PASS for a in self.assumptions)

    @property
    def has_fail(self) -> bool:
        return any(a.status == RuleStatus.FAIL for a in self.assumptions)

    @property
    def has_warn(self) -> bool:
        return any(a.status == RuleStatus.WARN for a in self.assumptions)

    def __iter__(self):
        return iter(self.assumptions)

    def __len__(self) -> int:
        return len(self.assumptions)
