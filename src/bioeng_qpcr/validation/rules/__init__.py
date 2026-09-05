"""MIQE validation rules package.

Importing this package triggers registration of all built-in MIQE
validation rules with the global :class:`~bioeng_qpcr.validation.registry.RuleRegistry`.

Each rule lives in its own submodule (``eff_001``, ``ref_001``,
``ntc_001``) and exposes two public attributes:

* ``METADATA`` -- the :class:`~bioeng_qpcr.validation.types.RuleMetadata`.
* The decorated rule function (e.g. ``check_efficiency``).

The :func:`register_all` helper is provided for test isolation: tests can
call ``RuleRegistry().clear()`` followed by ``register_all()`` to ensure a
known, clean set of rules.
"""

from __future__ import annotations

from bioeng_qpcr.validation.registry import RuleRegistry
from bioeng_qpcr.validation.rules import (
    assay_001,
    eff_001,
    input_001,
    ntc_001,
    ref_001,
    ref_002,
    rep_001,
)
from bioeng_qpcr.validation.types import RuleMetadata

__all__ = [
    "BUILTIN_RULES",
    "assay_001",
    "eff_001",
    "input_001",
    "ntc_001",
    "ref_001",
    "ref_002",
    "register_all",
    "rep_001",
]

# ---------------------------------------------------------------------------
# Catalogue of built-in rules
# ---------------------------------------------------------------------------

#: ``(metadata, function)`` pairs for every built-in MIQE rule.
BUILTIN_RULES: list[tuple[RuleMetadata, object]] = [
    (assay_001.METADATA, assay_001.check_assay_characterization),
    (eff_001.METADATA, eff_001.check_efficiency),
    (input_001.METADATA, input_001.check_input_diagnostics),
    (ref_001.METADATA, ref_001.check_reference_gene),
    (ref_002.METADATA, ref_002.check_reference_multiplicity),
    (rep_001.METADATA, rep_001.check_technical_replicates),
    (ntc_001.METADATA, ntc_001.check_ntc),
]


def register_all() -> None:
    """Register every built-in MIQE rule with the global registry.

    The function is **idempotent**: rules that are already registered
    (matched by ``rule_id``) are silently skipped, so it is safe to call
    multiple times -- for instance, after
    :meth:`~bioeng_qpcr.validation.registry.RuleRegistry.clear` in test
    fixtures.
    """
    registry = RuleRegistry()
    for metadata, func in BUILTIN_RULES:
        if metadata.rule_id not in registry:
            registry.register(metadata, func)


# Auto-register on first import.
register_all()
