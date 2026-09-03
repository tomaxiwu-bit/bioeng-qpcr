"""The validation report.

:class:`ValidationReport` is the immutable summary of a validation run.
It aggregates a list of :class:`~bioeng_qpcr.validation.types.RuleResult`
objects, exposes convenience properties for filtering by status, and
provides serialisation to JSON and human-readable text.

Design principles
    * ``assessment_passed`` is ``True`` **only** when every executed result
      is ``PASS``.  It describes the implemented checks, not global MIQE
      compliance.
    * ``overall_status`` follows a worst-case escalation:
      ``FAIL`` > ``WARN`` > ``NOT_ASSESSABLE`` > ``PASS``.
"""

from __future__ import annotations

import json
from typing import Any

from bioeng_qpcr.validation.types import RuleResult

__all__ = ["ValidationReport"]

#: Display order for the four statuses in the text report (worst first).
_STATUS_ORDER: list[str] = ["FAIL", "WARN", "NOT_ASSESSABLE", "PASS"]


class ValidationReport:
    """Aggregated result of running a set of validation rules.

    Parameters
        results:
            List of :class:`~bioeng_qpcr.validation.types.RuleResult`
            produced by the :class:`~bioeng_qpcr.validation.engine.ValidationEngine`.
        experiment_name:
            Name of the experiment that was validated.
        experiment_hash:
            Content hash of the experiment (see
            :meth:`~bioeng_qpcr.domain.entities.Experiment.content_hash`).
    """

    def __init__(
        self,
        results: list[RuleResult],
        experiment_name: str,
        experiment_hash: str,
        total_rule_count: int | None = None,
        skipped_rule_ids: list[str] | None = None,
    ) -> None:
        if not results:
            raise ValueError(
                "A validation report requires at least one executed rule; "
                "an empty assessment cannot be interpreted as a pass."
            )
        self._results: list[RuleResult] = list(results)
        self._experiment_name: str = experiment_name
        self._experiment_hash: str = experiment_hash
        self._total_rule_count = total_rule_count or len(results)
        self._skipped_rule_ids = list(skipped_rule_ids or [])

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def results(self) -> list[RuleResult]:
        """All results, in the order they were produced."""
        return list(self._results)

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def experiment_hash(self) -> str:
        return self._experiment_hash

    # ------------------------------------------------------------------
    # Status-based filtering
    # ------------------------------------------------------------------

    @property
    def passed(self) -> list[RuleResult]:
        """Results with ``PASS`` status."""
        return [r for r in self._results if r.is_pass]

    @property
    def warnings(self) -> list[RuleResult]:
        """Results with ``WARN`` status."""
        return [r for r in self._results if r.is_warn]

    @property
    def failures(self) -> list[RuleResult]:
        """Results with ``FAIL`` status."""
        return [r for r in self._results if r.is_fail]

    @property
    def not_assessable(self) -> list[RuleResult]:
        """Results with ``NOT_ASSESSABLE`` status."""
        return [r for r in self._results if r.is_not_assessable]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @property
    def summary(self) -> dict[str, int]:
        """Count of results in each status category."""
        counts: dict[str, int] = {
            "PASS": 0,
            "WARN": 0,
            "FAIL": 0,
            "NOT_ASSESSABLE": 0,
        }
        for r in self._results:
            counts[r.status.value] += 1
        return counts

    @property
    def overall_status(self) -> str:
        """Worst-case status across all results.

        Escalation order: ``FAIL`` > ``WARN`` > ``NOT_ASSESSABLE`` >
        ``PASS``.  Empty result lists are rejected by the constructor.
        """
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        if self.not_assessable:
            return "NOT_ASSESSABLE"
        return "PASS"

    @property
    def compliant(self) -> bool:
        """Backward-compatible alias for :attr:`assessment_passed`.

        This property does **not** certify full MIQE compliance; it only
        reports whether every rule that was executed passed.
        """
        return self.assessment_passed

    @property
    def assessment_passed(self) -> bool:
        """Whether every rule executed by this assessment passed."""
        return self.overall_status == "PASS"

    @property
    def rule_coverage(self) -> dict[str, Any]:
        """Coverage of available rules for the experiment's input level."""
        return {
            "available": self._total_rule_count,
            "executed": len(self._results),
            "skipped_input_level": len(self._skipped_rule_ids),
            "skipped_rule_ids": list(self._skipped_rule_ids),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _result_to_dict(self, r: RuleResult) -> dict[str, Any]:
        return {
            "rule_id": r.rule_id,
            "name": r.metadata.name,
            "description": r.metadata.description,
            "miqe_section": r.metadata.miqe_section,
            "status": r.status.value,
            "message": r.message,
            "evidence": r.evidence,
            "timestamp": r.timestamp.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Full structured dictionary representation."""
        return {
            "experiment_name": self._experiment_name,
            "experiment_hash": self._experiment_hash,
            "summary": self.summary,
            "assessment": "MIQE readiness",
            "overall_status": self.overall_status,
            "assessment_passed": self.assessment_passed,
            "rule_coverage": self.rule_coverage,
            # Retained for 0.1.x consumers; this means all assessed rules
            # passed and must not be interpreted as certification.
            "compliant": self.compliant,
            "results": [self._result_to_dict(r) for r in self._results],
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON serialisation of :meth:`to_dict`."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Human-readable multi-line text report.

        Sections:
            * Header (experiment name, hash)
            * Summary counts
            * Detailed results grouped by status (FAIL, WARN,
              NOT_ASSESSABLE, PASS -- empty groups omitted)
            * Footer (overall status, compliance)
        """
        sep = "=" * 60
        sub = "-" * 60

        lines: list[str] = [
            sep,
            "MIQE READINESS ASSESSMENT",
            sep,
            f"Experiment: {self._experiment_name}",
            f"Hash:       {self._experiment_hash}",
            "",
            "SUMMARY",
            sub,
        ]

        s = self.summary
        lines.append(f"  PASS:           {s['PASS']}")
        lines.append(f"  WARN:           {s['WARN']}")
        lines.append(f"  FAIL:           {s['FAIL']}")
        lines.append(f"  NOT_ASSESSABLE: {s['NOT_ASSESSABLE']}")
        coverage = self.rule_coverage
        lines.append(
            "  RULE COVERAGE:   "
            f"{coverage['executed']}/{coverage['available']} executed"
        )
        if self._skipped_rule_ids:
            lines.append("  SKIPPED:         " + ", ".join(self._skipped_rule_ids))
        lines.append("")

        lines.append("DETAILED RESULTS")
        lines.append(sub)

        groups: list[tuple[str, list[RuleResult]]] = [
            ("FAIL", self.failures),
            ("WARN", self.warnings),
            ("NOT_ASSESSABLE", self.not_assessable),
            ("PASS", self.passed),
        ]

        for status_name, group in groups:
            if not group:
                continue
            lines.append("")
            lines.append(f"[{status_name}]")
            for r in group:
                lines.append(f"  {r.rule_id} - {r.metadata.name}")
                lines.append(f"    Message: {r.message}")
                if r.evidence:
                    lines.append(f"    Evidence: {r.evidence}")
                lines.append("")

        lines.append(sep)
        lines.append(f"Overall status: {self.overall_status}")
        lines.append(f"All assessed rules passed: {self.assessment_passed}")
        lines.append("This assessment is not a certification of full MIQE compliance.")
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return (
            f"ValidationReport(experiment_name={self._experiment_name!r}, "
            f"n_results={len(self._results)}, "
            f"overall_status={self.overall_status!r})"
        )
