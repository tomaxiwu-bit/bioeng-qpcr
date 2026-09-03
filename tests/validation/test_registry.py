"""Tests for :class:`bioeng_qpcr.validation.registry.RuleRegistry`.

These tests exercise:

* Basic register / get / unregister lifecycle.
* Duplicate registration rejection.
* ``all_rules``, ``rules_for_level`` and ``functions_for_level`` filtering
  with the ``LEVEL_A < LEVEL_B < LEVEL_C`` ordering.
* ``clear``, ``__len__`` and ``__contains__``.
* Singleton (class-level shared state) behaviour.
* Basic thread safety under concurrent registration.
"""

from __future__ import annotations

import threading

import pytest

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.registry import RuleRegistry
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> RuleMetadata:
    defaults = dict(
        rule_id="EFF-001",
        name="Efficiency range check",
        description="Checks efficiency range.",
        miqe_section="MIQE 3.2",
    )
    defaults.update(overrides)
    return RuleMetadata(**defaults)


def _make_func(rule_id: str = "EFF-001"):
    """Build a trivial rule function that returns a PASS result."""

    def _fn(experiment: Experiment) -> RuleResult:
        return RuleResult(
            metadata=_make_metadata(rule_id=rule_id),
            status=RuleStatus.PASS,
            message="ok",
        )

    return _fn


def _make_experiment(name: str = "test-exp") -> Experiment:
    return Experiment(name=name)


# ---------------------------------------------------------------------------
# Shared fixture: clear the singleton registry around every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure the global registry is empty before and after each test."""
    registry = RuleRegistry()
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# Register / get / unregister
# ---------------------------------------------------------------------------


class TestRegisterAndGet:
    def test_register_then_get(self) -> None:
        registry = RuleRegistry()
        md = _make_metadata()
        func = _make_func()
        registry.register(md, func)

        retrieved = registry.get("EFF-001")
        assert retrieved is not None
        retrieved_md, retrieved_func = retrieved
        assert retrieved_md is md
        assert retrieved_func is func

    def test_get_missing_returns_none(self) -> None:
        registry = RuleRegistry()
        assert registry.get("DOES-NOT-EXIST") is None

    def test_get_returns_tuple_of_two(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        retrieved = registry.get("EFF-001")
        assert retrieved is not None
        assert isinstance(retrieved, tuple)
        assert len(retrieved) == 2


class TestUnregister:
    def test_unregister_existing(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        assert "EFF-001" in registry

        registry.unregister("EFF-001")
        assert "EFF-001" not in registry
        assert registry.get("EFF-001") is None

    def test_unregister_missing_is_noop(self) -> None:
        registry = RuleRegistry()
        # Should not raise.
        registry.unregister("DOES-NOT-EXIST")
        assert len(registry) == 0

    def test_unregister_then_reregister(self) -> None:
        registry = RuleRegistry()
        md = _make_metadata()
        registry.register(md, _make_func())
        registry.unregister("EFF-001")
        # Re-registration should work after unregister.
        registry.register(md, _make_func())
        assert "EFF-001" in registry


class TestDuplicateRegistration:
    def test_duplicate_raises_value_error(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        with pytest.raises(ValueError, match="EFF-001"):
            registry.register(_make_metadata(), _make_func())

    def test_duplicate_after_clear_is_allowed(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        registry.clear()
        # Should not raise after clear.
        registry.register(_make_metadata(), _make_func())
        assert "EFF-001" in registry


# ---------------------------------------------------------------------------
# all_rules
# ---------------------------------------------------------------------------


class TestAllRules:
    def test_empty_registry(self) -> None:
        registry = RuleRegistry()
        assert registry.all_rules() == []

    def test_returns_metadata_only(self) -> None:
        registry = RuleRegistry()
        md_a = _make_metadata(rule_id="AAA-001")
        md_b = _make_metadata(rule_id="BBB-001", name="Other rule")
        registry.register(md_a, _make_func("AAA-001"))
        registry.register(md_b, _make_func("BBB-001"))

        rules = registry.all_rules()
        assert len(rules) == 2
        assert all(isinstance(r, RuleMetadata) for r in rules)

    def test_sorted_by_rule_id(self) -> None:
        registry = RuleRegistry()
        # Register out of order.
        registry.register(_make_metadata(rule_id="CCC-001"), _make_func("CCC-001"))
        registry.register(_make_metadata(rule_id="AAA-001"), _make_func("AAA-001"))
        registry.register(_make_metadata(rule_id="BBB-001"), _make_func("BBB-001"))

        rules = registry.all_rules()
        ids = [r.rule_id for r in rules]
        assert ids == ["AAA-001", "BBB-001", "CCC-001"]


# ---------------------------------------------------------------------------
# Level filtering
# ---------------------------------------------------------------------------


class TestRulesForLevel:
    def test_level_a_returns_only_level_a_rules(self) -> None:
        registry = RuleRegistry()
        registry.register(
            _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A),
            _make_func("R-A"),
        )
        registry.register(
            _make_metadata(rule_id="R-B", min_input_level=InputLevel.LEVEL_B),
            _make_func("R-B"),
        )
        registry.register(
            _make_metadata(rule_id="R-C", min_input_level=InputLevel.LEVEL_C),
            _make_func("R-C"),
        )

        rules = registry.rules_for_level(InputLevel.LEVEL_A)
        ids = [r.rule_id for r in rules]
        assert ids == ["R-A"]

    def test_level_b_returns_a_and_b_rules(self) -> None:
        registry = RuleRegistry()
        registry.register(
            _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A),
            _make_func("R-A"),
        )
        registry.register(
            _make_metadata(rule_id="R-B", min_input_level=InputLevel.LEVEL_B),
            _make_func("R-B"),
        )
        registry.register(
            _make_metadata(rule_id="R-C", min_input_level=InputLevel.LEVEL_C),
            _make_func("R-C"),
        )

        rules = registry.rules_for_level(InputLevel.LEVEL_B)
        ids = [r.rule_id for r in rules]
        assert ids == ["R-A", "R-B"]

    def test_level_c_returns_all_rules(self) -> None:
        registry = RuleRegistry()
        registry.register(
            _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A),
            _make_func("R-A"),
        )
        registry.register(
            _make_metadata(rule_id="R-B", min_input_level=InputLevel.LEVEL_B),
            _make_func("R-B"),
        )
        registry.register(
            _make_metadata(rule_id="R-C", min_input_level=InputLevel.LEVEL_C),
            _make_func("R-C"),
        )

        rules = registry.rules_for_level(InputLevel.LEVEL_C)
        ids = [r.rule_id for r in rules]
        assert ids == ["R-A", "R-B", "R-C"]

    def test_filtered_results_are_sorted(self) -> None:
        registry = RuleRegistry()
        registry.register(
            _make_metadata(rule_id="Z-RULE", min_input_level=InputLevel.LEVEL_A),
            _make_func("Z-RULE"),
        )
        registry.register(
            _make_metadata(rule_id="A-RULE", min_input_level=InputLevel.LEVEL_A),
            _make_func("A-RULE"),
        )

        rules = registry.rules_for_level(InputLevel.LEVEL_B)
        ids = [r.rule_id for r in rules]
        assert ids == ["A-RULE", "Z-RULE"]


class TestFunctionsForLevel:
    def test_returns_tuples_of_metadata_and_func(self) -> None:
        registry = RuleRegistry()
        md_a = _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A)
        func_a = _make_func("R-A")
        registry.register(md_a, func_a)
        registry.register(
            _make_metadata(rule_id="R-C", min_input_level=InputLevel.LEVEL_C),
            _make_func("R-C"),
        )

        pairs = registry.functions_for_level(InputLevel.LEVEL_A)
        assert len(pairs) == 1
        md, func = pairs[0]
        assert md is md_a
        assert func is func_a

    def test_level_b_filters_correctly(self) -> None:
        registry = RuleRegistry()
        registry.register(
            _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A),
            _make_func("R-A"),
        )
        registry.register(
            _make_metadata(rule_id="R-B", min_input_level=InputLevel.LEVEL_B),
            _make_func("R-B"),
        )
        registry.register(
            _make_metadata(rule_id="R-C", min_input_level=InputLevel.LEVEL_C),
            _make_func("R-C"),
        )

        pairs = registry.functions_for_level(InputLevel.LEVEL_B)
        ids = [md.rule_id for md, _ in pairs]
        assert ids == ["R-A", "R-B"]

    def test_functions_are_callable(self) -> None:
        registry = RuleRegistry()
        registry.register(
            _make_metadata(rule_id="R-A", min_input_level=InputLevel.LEVEL_A),
            _make_func("R-A"),
        )

        pairs = registry.functions_for_level(InputLevel.LEVEL_A)
        assert len(pairs) == 1
        _, func = pairs[0]
        result = func(_make_experiment())
        assert isinstance(result, RuleResult)
        assert result.status is RuleStatus.PASS


# ---------------------------------------------------------------------------
# clear, __len__, __contains__
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empties_registry(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        assert len(registry) == 1

        registry.clear()
        assert len(registry) == 0
        assert registry.all_rules() == []

    def test_clear_on_empty_registry(self) -> None:
        registry = RuleRegistry()
        registry.clear()
        assert len(registry) == 0


class TestLenAndContains:
    def test_len_empty(self) -> None:
        registry = RuleRegistry()
        assert len(registry) == 0

    def test_len_after_register(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(rule_id="R1"), _make_func("R1"))
        registry.register(_make_metadata(rule_id="R2"), _make_func("R2"))
        assert len(registry) == 2

    def test_len_after_unregister(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(rule_id="R1"), _make_func("R1"))
        registry.register(_make_metadata(rule_id="R2"), _make_func("R2"))
        registry.unregister("R1")
        assert len(registry) == 1

    def test_contains_registered(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        assert "EFF-001" in registry

    def test_not_contains_missing(self) -> None:
        registry = RuleRegistry()
        assert "EFF-001" not in registry

    def test_not_contains_after_unregister(self) -> None:
        registry = RuleRegistry()
        registry.register(_make_metadata(), _make_func())
        registry.unregister("EFF-001")
        assert "EFF-001" not in registry


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_two_instances_share_state(self) -> None:
        r1 = RuleRegistry()
        r2 = RuleRegistry()
        r1.register(_make_metadata(), _make_func())
        # r2 should see the rule registered via r1.
        assert "EFF-001" in r2
        assert r2.get("EFF-001") is not None

    def test_clear_via_one_instance_affects_all(self) -> None:
        r1 = RuleRegistry()
        r2 = RuleRegistry()
        r1.register(_make_metadata(), _make_func())
        r2.clear()
        assert len(r1) == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_registration_no_loss(self) -> None:
        """Registering from multiple threads must not lose any rule."""
        registry = RuleRegistry()
        n_threads = 10
        rules_per_thread = 20

        def _register(thread_id: int) -> None:
            for i in range(rules_per_thread):
                rid = f"T{thread_id:02d}-R{i:03d}"
                registry.register(
                    _make_metadata(rule_id=rid, name=f"Rule {rid}"),
                    _make_func(rid),
                )

        threads = [
            threading.Thread(target=_register, args=(t,)) for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(registry) == n_threads * rules_per_thread

    def test_concurrent_unregister_is_safe(self) -> None:
        """Unregistering concurrently must not corrupt the registry."""
        registry = RuleRegistry()
        # Pre-register a set of rules.
        rule_ids = [f"R-{i:03d}" for i in range(50)]
        for rid in rule_ids:
            registry.register(_make_metadata(rule_id=rid), _make_func(rid))

        def _unregister(start: int, end: int) -> None:
            for i in range(start, end):
                registry.unregister(f"R-{i:03d}")

        threads = [
            threading.Thread(target=_unregister, args=(0, 25)),
            threading.Thread(target=_unregister, args=(25, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(registry) == 0
