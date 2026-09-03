"""Thread-safe singleton registry for validation rules.

The :class:`RuleRegistry` is the central catalogue of every MIQE-compliance
rule known to the engine.  Rules are stored at the *class* level, so all
instances of :class:`RuleRegistry` share the same underlying storage -- this
is the intended singleton behaviour.  A :class:`threading.Lock` guards every
mutation and read so the registry is safe to use from concurrent threads.

Rules are keyed by their ``rule_id`` (see :class:`RuleMetadata`) and stored
as ``(metadata, function)`` tuples.  The registry supports filtering by
:class:`~bioeng_qpcr.domain.enums.InputLevel` so that the engine can ask,
"given an experiment at input level *B*, which rules can actually be
assessed?"
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = ["RuleRegistry"]

# Type alias for a rule function: takes an Experiment, returns a RuleResult.
RuleFunc = Callable[[Experiment], RuleResult]

# Total ordering of input levels.  LEVEL_A < LEVEL_B < LEVEL_C because each
# successive level carries strictly more metadata, enabling more rules.
_LEVEL_ORDER: dict[InputLevel, int] = {
    InputLevel.LEVEL_A: 0,
    InputLevel.LEVEL_B: 1,
    InputLevel.LEVEL_C: 2,
}


class RuleRegistry:
    """Singleton, thread-safe catalogue of validation rules.

    All instances share the same class-level ``_rules`` dictionary and
    ``_lock``, so creating a new ``RuleRegistry()`` does **not** give you an
    isolated collection.  This is by design: there is one global set of
    rules per process.

    The registry is intentionally not coupled to the
    :func:`~bioeng_qpcr.validation.decorator.validation_rule` decorator.
    A rule function need only be callable with an :class:`Experiment` and
    return a :class:`RuleResult`; the metadata is supplied separately to
    :meth:`register`.

    Thread safety
        Every public method acquires the class-level :class:`threading.Lock`
        for the duration of its operation, making individual calls atomic.
        Callers that need multi-step atomicity (e.g. register-then-query)
        should perform their own external synchronisation.
    """

    # Class-level shared storage and lock -- the singleton state.
    _rules: ClassVar[dict[str, tuple[RuleMetadata, RuleFunc]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, metadata: RuleMetadata, func: RuleFunc) -> None:
        """Register a rule function under its metadata's ``rule_id``.

        Parameters
            metadata:
                The :class:`RuleMetadata` describing the rule.  The
                ``metadata.rule_id`` is used as the registry key.
            func:
                The callable that implements the rule.  It must accept a
                single :class:`Experiment` argument and return a
                :class:`RuleResult`.

        Raises
            ValueError:
                If a rule with the same ``rule_id`` is already registered.
                Call :meth:`unregister` first to replace an existing rule.
        """
        with self._lock:
            if metadata.rule_id in self._rules:
                raise ValueError(
                    f"Rule '{metadata.rule_id}' is already registered. "
                    f"Call unregister() first if you intend to replace it."
                )
            self._rules[metadata.rule_id] = (metadata, func)

    def unregister(self, rule_id: str) -> None:
        """Remove a rule from the registry.

        This method is idempotent: unregistering a ``rule_id`` that does not
        exist is a no-op and raises no error.

        Parameters
            rule_id:
                The identifier of the rule to remove.
        """
        with self._lock:
            self._rules.pop(rule_id, None)

    def clear(self) -> None:
        """Remove every registered rule.

        Primarily intended for test isolation so that one test's registered
        rules do not leak into another.
        """
        with self._lock:
            self._rules.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, rule_id: str) -> tuple[RuleMetadata, RuleFunc] | None:
        """Retrieve a single rule by its identifier.

        Parameters
            rule_id:
                The rule identifier to look up.

        Returns
            The ``(metadata, function)`` tuple, or ``None`` if no rule with
            the given identifier is registered.
        """
        with self._lock:
            return self._rules.get(rule_id)

    def all_rules(self) -> list[RuleMetadata]:
        """Return the metadata of every registered rule.

        The returned list is sorted by ``rule_id`` for deterministic
        ordering.
        """
        with self._lock:
            return sorted(
                (metadata for metadata, _ in self._rules.values()),
                key=lambda m: m.rule_id,
            )

    def rules_for_level(self, input_level: InputLevel) -> list[RuleMetadata]:
        """Return metadata for all rules assessable at *input_level*.

        A rule is assessable at a given level when the level is greater than
        or equal to the rule's ``min_input_level`` (ordering:
        ``LEVEL_A < LEVEL_B < LEVEL_C``).

        Parameters
            input_level:
                The input level of the experiment being validated.

        Returns
            A list of :class:`RuleMetadata`, sorted by ``rule_id``.
        """
        given = _LEVEL_ORDER[input_level]
        with self._lock:
            return sorted(
                (
                    metadata
                    for metadata, _ in self._rules.values()
                    if given >= _LEVEL_ORDER[metadata.min_input_level]
                ),
                key=lambda m: m.rule_id,
            )

    def functions_for_level(
        self, input_level: InputLevel
    ) -> list[tuple[RuleMetadata, RuleFunc]]:
        """Return ``(metadata, function)`` pairs assessable at *input_level*.

        Identical filtering logic to :meth:`rules_for_level` but also
        returns the callable so the caller can immediately evaluate the
        rules.

        Parameters
            input_level:
                The input level of the experiment being validated.

        Returns
            A list of ``(metadata, function)`` tuples, sorted by ``rule_id``.
        """
        given = _LEVEL_ORDER[input_level]
        with self._lock:
            return sorted(
                (
                    (metadata, func)
                    for metadata, func in self._rules.values()
                    if given >= _LEVEL_ORDER[metadata.min_input_level]
                ),
                key=lambda pair: pair[0].rule_id,
            )

    # ------------------------------------------------------------------
    # Dunder protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of registered rules."""
        with self._lock:
            return len(self._rules)

    def __contains__(self, rule_id: object) -> bool:
        """``True`` if *rule_id* is a registered rule identifier."""
        with self._lock:
            return rule_id in self._rules
