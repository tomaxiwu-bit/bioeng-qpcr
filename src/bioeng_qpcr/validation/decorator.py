"""The ``@validation_rule`` decorator.

This decorator is the standard mechanism for declaring a MIQE-compliance
rule in bioeng-qpcr.  It binds a :class:`~bioeng_qpcr.validation.types.RuleMetadata`
instance to a plain function ``fn(experiment: Experiment) -> RuleResult`` and
returns a callable that:

1. Preserves the original function's identity via :func:`functools.wraps`.
2. Exposes the metadata as the ``.metadata`` attribute on the wrapper.
3. When invoked with an :class:`~bioeng_qpcr.domain.entities.Experiment`,
   delegates to the inner function and guarantees that the returned
   :class:`~bioeng_qpcr.validation.types.RuleResult` carries the *exact*
   metadata object supplied to the decorator.

The decorator deliberately does **not** auto-register the rule with the
:class:`~bioeng_qpcr.validation.registry.RuleRegistry`.  Registration is an
explicit, separate concern so that rule definitions remain side-effect-free
and can be unit-tested in isolation.

Example
-------
.. code-block:: python

    from bioeng_qpcr.validation import validation_rule, RuleMetadata
    from bioeng_qpcr.domain.enums import InputLevel, RuleStatus

    EFF_METADATA = RuleMetadata(
        rule_id="EFF-001",
        name="Efficiency range check",
        description="Amplification efficiency must be within 90-110%.",
        miqe_section="MIQE 3.2",
        min_input_level=InputLevel.LEVEL_B,
    )

    @validation_rule(EFF_METADATA)
    def check_efficiency(experiment):
        if not experiment.has_efficiency:
            return RuleResult(
                metadata=EFF_METADATA,
                status=RuleStatus.NOT_ASSESSABLE,
                message="No efficiency data provided.",
            )
        ...
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = ["validation_rule"]

# A rule function takes an Experiment and returns a RuleResult.
RuleFunc = Callable[[Experiment], RuleResult]
F = TypeVar("F", bound=RuleFunc)


def validation_rule(metadata: RuleMetadata) -> Callable[[F], F]:
    """Bind :class:`RuleMetadata` to a rule function.

    Parameters
        metadata:
            The :class:`RuleMetadata` describing the rule.  This object is
            attached to the wrapped function as ``fn.metadata`` and is also
            guaranteed to appear on every :class:`RuleResult` returned by the
            wrapper.

    Returns
        A decorator that accepts a function with signature
        ``fn(experiment: Experiment) -> RuleResult`` and returns a wrapped
        callable with the same signature plus a ``.metadata`` attribute.

    Notes
        The wrapper inspects the :class:`RuleResult` returned by the inner
        function.  If the result's ``metadata`` is not the *same object* as
        the one supplied to the decorator (checked via ``is``), the wrapper
        substitutes the canonical metadata using
        :func:`dataclasses.replace`.  This guarantees that every result
        produced by a decorated rule is traceable to a single, authoritative
        metadata instance.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(experiment: Experiment) -> RuleResult:
            result = fn(experiment)
            # Guarantee the result carries the canonical metadata object.
            if result.metadata is not metadata:
                result = replace(result, metadata=metadata)
            return result

        # Attach the metadata so callers and the registry can introspect it.
        wrapper.metadata = metadata  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
