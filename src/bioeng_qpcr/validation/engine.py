"""The validation engine.

:class:`ValidationEngine` is the orchestrator that ties together the
:class:`~bioeng_qpcr.validation.registry.RuleRegistry`, the individual
rule functions, and the :class:`~bioeng_qpcr.validation.report.ValidationReport`.

Given an :class:`~bioeng_qpcr.domain.entities.Experiment`, the engine:

1. Determines the effective input level (constructor override or the
   experiment's own ``input_level``).
2. Queries the registry for all rules assessable at that level.
3. Executes each rule function, collecting
   :class:`~bioeng_qpcr.validation.types.RuleResult` objects.
4. Wraps the results into a :class:`~bioeng_qpcr.validation.report.ValidationReport`.

Built-in rules are loaded explicitly when the default engine is created.
Unexpected rule exceptions raise :class:`RuleExecutionError` by default;
software failures must never be presented as missing scientific evidence.
"""

from __future__ import annotations

from collections.abc import Callable

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.registry import RuleRegistry
from bioeng_qpcr.validation.report import ValidationReport
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = ["RuleExecutionError", "ValidationEngine"]


class RuleExecutionError(RuntimeError):
    """Raised when a validation rule fails because of a software error."""

    def __init__(self, rule_id: str, original: Exception) -> None:
        self.rule_id = rule_id
        self.original = original
        super().__init__(
            f"Validation rule '{rule_id}' failed with "
            f"{type(original).__name__}: {original}"
        )


class ValidationEngine:
    """Orchestrate MIQE rule evaluation for an experiment.

    Parameters
        input_level:
            Optional override for the input level.  When ``None`` (the
            default) the engine uses ``experiment.input_level`` at
            run time.  Supplying an explicit level is useful for testing
            or for forcing a higher/lower level than the experiment
            advertises.
    """

    def __init__(
        self,
        input_level: InputLevel | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self._input_level = input_level
        self._registry = RuleRegistry()
        self._strict = strict

        # Import locally to avoid a package-import cycle.  Registration is
        # explicit rather than relying on callers to import a side-effecting
        # module before using the public API.
        from bioeng_qpcr.validation.rules import register_all

        register_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, experiment: Experiment) -> ValidationReport:
        """Run all applicable rules against *experiment*.

        Returns a :class:`ValidationReport` containing one
        :class:`RuleResult` per applicable rule, sorted by ``rule_id``.
        """
        input_level = (
            self._input_level
            if self._input_level is not None
            else experiment.input_level
        )

        functions = self._registry.functions_for_level(input_level)
        if not functions:
            raise RuntimeError(
                "No validation rules are available for this input level; "
                "refusing to produce an empty readiness assessment."
            )

        all_rules = self._registry.all_rules()
        executed_ids = {metadata.rule_id for metadata, _ in functions}
        skipped_rule_ids = [
            metadata.rule_id
            for metadata in all_rules
            if metadata.rule_id not in executed_ids
        ]

        results: list[RuleResult] = []
        for metadata, func in functions:
            try:
                result = func(experiment)
                results.append(result)
            except Exception as exc:
                if self._strict:
                    raise RuleExecutionError(metadata.rule_id, exc) from exc
                results.append(
                    RuleResult(
                        metadata=metadata,
                        status=RuleStatus.NOT_ASSESSABLE,
                        message=(
                            f"Rule '{metadata.rule_id}' raised an exception: {exc}"
                        ),
                        evidence={
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "execution_error": True,
                        },
                    )
                )

        return ValidationReport(
            results=results,
            experiment_name=experiment.name,
            experiment_hash=experiment.content_hash(),
            total_rule_count=len(all_rules),
            skipped_rule_ids=skipped_rule_ids,
        )

    def register_rule(
        self, metadata: RuleMetadata, func: Callable[[Experiment], RuleResult]
    ) -> None:
        """Register a custom rule with the global registry.

        Thin proxy to :meth:`RuleRegistry.register`; provided as a
        convenience so callers need not instantiate a separate
        :class:`RuleRegistry`.
        """
        self._registry.register(metadata, func)
