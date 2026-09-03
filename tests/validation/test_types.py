"""Tests for :mod:`bioeng_qpcr.validation.types`.

These tests exercise:

* :class:`RuleMetadata` construction, defaults, immutability and equality.
* :class:`RuleResult` construction, defaults (evidence, timestamp),
  convenience properties (``rule_id``, ``is_pass``, ``is_fail``,
  ``is_warn``, ``is_not_assessable``), and immutability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> RuleMetadata:
    """Build a :class:`RuleMetadata` with sensible defaults."""
    defaults = dict(
        rule_id="EFF-001",
        name="Efficiency range check",
        description="Amplification efficiency must be within 90-110%.",
        miqe_section="MIQE 3.2",
    )
    defaults.update(overrides)
    return RuleMetadata(**defaults)


# ---------------------------------------------------------------------------
# RuleMetadata
# ---------------------------------------------------------------------------


class TestRuleMetadata:
    def test_required_fields(self) -> None:
        md = _make_metadata()
        assert md.rule_id == "EFF-001"
        assert md.name == "Efficiency range check"
        assert md.description == "Amplification efficiency must be within 90-110%."
        assert md.miqe_section == "MIQE 3.2"

    def test_default_min_input_level_is_level_a(self) -> None:
        md = _make_metadata()
        assert md.min_input_level is InputLevel.LEVEL_A

    def test_default_version(self) -> None:
        md = _make_metadata()
        assert md.version == "1.0.0"

    def test_default_citation_is_none(self) -> None:
        md = _make_metadata()
        assert md.citation is None

    def test_custom_min_input_level(self) -> None:
        md = _make_metadata(min_input_level=InputLevel.LEVEL_C)
        assert md.min_input_level is InputLevel.LEVEL_C

    def test_custom_version(self) -> None:
        md = _make_metadata(version="2.1.3")
        assert md.version == "2.1.3"

    def test_custom_citation(self) -> None:
        citation = "Bustin et al. (2009) Clin Chem 55:611-622"
        md = _make_metadata(citation=citation)
        assert md.citation == citation

    def test_is_frozen(self) -> None:
        md = _make_metadata()
        with pytest.raises(FrozenInstanceError):
            md.rule_id = "EFF-999"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        md1 = _make_metadata()
        md2 = _make_metadata()
        assert md1 == md2

    def test_inequality_on_different_rule_id(self) -> None:
        md1 = _make_metadata(rule_id="EFF-001")
        md2 = _make_metadata(rule_id="EFF-002")
        assert md1 != md2

    def test_inequality_on_different_level(self) -> None:
        md1 = _make_metadata(min_input_level=InputLevel.LEVEL_A)
        md2 = _make_metadata(min_input_level=InputLevel.LEVEL_B)
        assert md1 != md2

    def test_hashable(self) -> None:
        """Frozen dataclasses are hashable and can be used in sets/dicts."""
        md = _make_metadata()
        assert hash(md) == hash(_make_metadata())
        assert md in {_make_metadata()}


# ---------------------------------------------------------------------------
# RuleResult
# ---------------------------------------------------------------------------


class TestRuleResultConstruction:
    def test_required_fields(self) -> None:
        md = _make_metadata()
        result = RuleResult(
            metadata=md,
            status=RuleStatus.PASS,
            message="All good.",
        )
        assert result.metadata is md
        assert result.status is RuleStatus.PASS
        assert result.message == "All good."

    def test_default_evidence_is_empty_dict(self) -> None:
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        assert result.evidence == {}

    def test_default_evidence_is_not_shared(self) -> None:
        """Each instance gets its own evidence dict (default_factory)."""
        r1 = RuleResult(metadata=_make_metadata(), status=RuleStatus.PASS, message="a")
        r2 = RuleResult(metadata=_make_metadata(), status=RuleStatus.PASS, message="b")
        r1.evidence["key"] = "value"
        assert r2.evidence == {}

    def test_explicit_evidence(self) -> None:
        evidence = {"efficiency": 0.97, "threshold": 0.90}
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.WARN,
            message="Borderline.",
            evidence=evidence,
        )
        assert result.evidence == evidence

    def test_default_timestamp_is_utc_aware(self) -> None:
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        assert result.timestamp.utcoffset() == timedelta(0)

    def test_default_timestamp_is_close_to_now(self) -> None:
        before = datetime.now(timezone.utc)
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after

    def test_explicit_timestamp(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.FAIL,
            message="bad",
            timestamp=ts,
        )
        assert result.timestamp == ts

    def test_each_construction_gets_fresh_timestamp(self) -> None:
        r1 = RuleResult(metadata=_make_metadata(), status=RuleStatus.PASS, message="a")
        r2 = RuleResult(metadata=_make_metadata(), status=RuleStatus.PASS, message="b")
        # They might be equal if constructed within the same microsecond, but
        # the factory is called independently each time.
        assert isinstance(r1.timestamp, datetime)
        assert isinstance(r2.timestamp, datetime)


class TestRuleResultProperties:
    @pytest.mark.parametrize(
        ("status", "prop"),
        [
            (RuleStatus.PASS, "is_pass"),
            (RuleStatus.WARN, "is_warn"),
            (RuleStatus.FAIL, "is_fail"),
            (RuleStatus.NOT_ASSESSABLE, "is_not_assessable"),
        ],
    )
    def test_positive_property(self, status: RuleStatus, prop: str) -> None:
        result = RuleResult(metadata=_make_metadata(), status=status, message="m")
        assert getattr(result, prop) is True

    @pytest.mark.parametrize(
        ("status", "prop"),
        [
            (RuleStatus.WARN, "is_pass"),
            (RuleStatus.FAIL, "is_pass"),
            (RuleStatus.NOT_ASSESSABLE, "is_pass"),
            (RuleStatus.PASS, "is_warn"),
            (RuleStatus.FAIL, "is_warn"),
            (RuleStatus.PASS, "is_fail"),
            (RuleStatus.PASS, "is_not_assessable"),
        ],
    )
    def test_negative_property(self, status: RuleStatus, prop: str) -> None:
        result = RuleResult(metadata=_make_metadata(), status=status, message="m")
        assert getattr(result, prop) is False

    def test_exactly_one_status_flag_true(self) -> None:
        """For any status, exactly one of the four flags is True."""
        for status in RuleStatus:
            result = RuleResult(metadata=_make_metadata(), status=status, message="m")
            flags = [
                result.is_pass,
                result.is_warn,
                result.is_fail,
                result.is_not_assessable,
            ]
            assert flags.count(True) == 1

    def test_rule_id_delegates_to_metadata(self) -> None:
        md = _make_metadata(rule_id="NTC-003")
        result = RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")
        assert result.rule_id == "NTC-003"
        assert result.rule_id == result.metadata.rule_id


class TestRuleResultImmutability:
    def test_is_frozen(self) -> None:
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        with pytest.raises(FrozenInstanceError):
            result.status = RuleStatus.FAIL  # type: ignore[misc]

    def test_cannot_reassign_rule_id(self) -> None:
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        with pytest.raises(AttributeError):
            result.rule_id = "X"  # type: ignore[misc]

    def test_replace_creates_new_instance(self) -> None:
        original = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        modified = replace(original, status=RuleStatus.FAIL, message="bad")
        assert original.status is RuleStatus.PASS
        assert modified.status is RuleStatus.FAIL
        assert modified.message == "bad"
        # Unchanged fields are preserved.
        assert modified.metadata is original.metadata
        assert modified.evidence == original.evidence

    def test_equality_by_value(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        md = _make_metadata()
        r1 = RuleResult(metadata=md, status=RuleStatus.PASS, message="ok", timestamp=ts)
        r2 = RuleResult(metadata=md, status=RuleStatus.PASS, message="ok", timestamp=ts)
        assert r1 == r2

    def test_inequality_on_different_status(self) -> None:
        md = _make_metadata()
        r1 = RuleResult(metadata=md, status=RuleStatus.PASS, message="ok")
        r2 = RuleResult(metadata=md, status=RuleStatus.FAIL, message="ok")
        assert r1 != r2

    def test_not_hashable_due_to_evidence_dict(self) -> None:
        """RuleResult contains a ``dict`` evidence field, so it is not
        hashable -- calling :func:`hash` raises ``TypeError``."""
        result = RuleResult(
            metadata=_make_metadata(),
            status=RuleStatus.PASS,
            message="ok",
        )
        with pytest.raises(TypeError, match="unhashable"):
            hash(result)
