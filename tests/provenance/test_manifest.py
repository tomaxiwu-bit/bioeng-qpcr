"""Tests for :class:`bioeng_qpcr.provenance.ProvenanceManifest`.

These tests exercise:

* explicit construction and the read-only ``created_at`` property,
* :meth:`ProvenanceManifest.from_experiment` (hash derivation + version capture),
* JSON / dict round-trip serialisation and equality,
* content-hash stability across identical experiments,
* package-version capture, and
* the human-readable :meth:`summary` output.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy
import pandas
import pytest
import scipy

from bioeng_qpcr.domain.entities import Assay, Experiment, RawTrace, Sample, Well
from bioeng_qpcr.domain.enums import InputLevel, ReplicateType
from bioeng_qpcr.provenance import ProvenanceManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wells() -> list[Well]:
    """A small but realistic plate layout used across the tests."""
    return [
        Well(
            position="A01",
            sample_name="Sample1",
            target_name="GAPDH",
            cq=18.23,
            replicate_type=ReplicateType.TECHNICAL,
            replicate_group="s1_gapdh_r1",
        ),
        Well(
            position="A02",
            sample_name="Sample1",
            target_name="GAPDH",
            cq=18.29,
            replicate_type=ReplicateType.TECHNICAL,
            replicate_group="s1_gapdh_r2",
        ),
        Well(
            position="A03",
            sample_name="Sample1",
            target_name="GOI",
            cq=24.10,
            replicate_type=ReplicateType.TECHNICAL,
            replicate_group="s1_goi_r1",
        ),
        Well(
            position="B01",
            sample_name="NTC",
            target_name="GAPDH",
            cq=None,
            replicate_type=ReplicateType.SINGLE,
            cq_status="no_template",
        ),
    ]


def _make_assays() -> dict[str, Assay]:
    return {
        "GAPDH": Assay(
            name="GAPDH",
            efficiency=0.98,
            r2=0.998,
            is_reference=True,
        ),
        "GOI": Assay(
            name="GOI",
            efficiency=1.02,
            r2=0.997,
            is_reference=False,
        ),
    }


def _make_experiment(name: str = "test-experiment") -> Experiment:
    """Build a deterministic Experiment (content hash is stable across calls)."""
    return Experiment(
        name=name,
        wells=_make_wells(),
        samples={"Sample1": Sample(name="Sample1", group="treated")},
        assays=_make_assays(),
        input_level=InputLevel.LEVEL_B,
    )


# ---------------------------------------------------------------------------
# 1. Explicit construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        manifest = ProvenanceManifest()
        assert manifest.input_hash is None
        assert manifest.parameters == {}
        assert manifest.package_versions == {}
        assert isinstance(manifest.created_at, datetime)

    def test_created_at_is_utc_aware(self) -> None:
        manifest = ProvenanceManifest()
        # tz-aware with a zero (UTC) offset.
        assert manifest.created_at.utcoffset() == timedelta(0)

    def test_created_at_is_close_to_now(self) -> None:
        manifest = ProvenanceManifest()
        now = datetime.now(timezone.utc)
        assert (now - manifest.created_at).total_seconds() < 5

    def test_explicit_values(self) -> None:
        params = {"method": "delta_delta_cq", "reference_gene": "GAPDH"}
        versions = {"bioeng_qpcr": "0.1.0", "numpy": "1.26.0"}
        manifest = ProvenanceManifest(
            input_hash="a" * 64,
            parameters=params,
            package_versions=versions,
        )
        assert manifest.input_hash == "a" * 64
        assert manifest.parameters == params
        assert manifest.package_versions == versions

    def test_input_dicts_are_copied(self) -> None:
        """Mutating the caller's dict after construction must not leak in."""
        params = {"k": "v"}
        versions = {"numpy": "1.0"}
        manifest = ProvenanceManifest(
            input_hash="b" * 64,
            parameters=params,
            package_versions=versions,
        )
        params["k"] = "changed"
        versions["numpy"] = "9.9.9"
        assert manifest.parameters == {"k": "v"}
        assert manifest.package_versions == {"numpy": "1.0"}

    def test_created_at_is_read_only(self) -> None:
        manifest = ProvenanceManifest()
        with pytest.raises(AttributeError):
            manifest.created_at = datetime.now(timezone.utc)  # type: ignore[misc]

    def test_repr_contains_key_fields(self) -> None:
        manifest = ProvenanceManifest(input_hash="c" * 64)
        text = repr(manifest)
        assert "ProvenanceManifest" in text
        assert "c" * 64 in text


# ---------------------------------------------------------------------------
# 2. from_experiment
# ---------------------------------------------------------------------------


class TestFromExperiment:
    def test_computes_content_hash(self) -> None:
        experiment = _make_experiment()
        manifest = ProvenanceManifest.from_experiment(experiment)
        assert manifest.input_hash == experiment.content_hash()
        # SHA-256 hex digest -> 64 chars.
        assert len(manifest.input_hash) == 64

    def test_captures_package_versions(self) -> None:
        manifest = ProvenanceManifest.from_experiment(_make_experiment())
        for pkg in ("bioeng_qpcr", "numpy", "scipy", "pandas"):
            assert pkg in manifest.package_versions
            assert manifest.package_versions[pkg]

    def test_bioeng_qpcr_version_matches_module(self) -> None:
        import bioeng_qpcr

        manifest = ProvenanceManifest.from_experiment(_make_experiment())
        assert manifest.package_versions["bioeng_qpcr"] == bioeng_qpcr.__version__

    def test_records_parameters(self) -> None:
        params = {"method": "pfaffl", "reference_gene": "GAPDH"}
        manifest = ProvenanceManifest.from_experiment(
            _make_experiment(), parameters=params
        )
        assert manifest.parameters == params

    def test_parameters_default_to_empty(self) -> None:
        manifest = ProvenanceManifest.from_experiment(_make_experiment())
        assert manifest.parameters == {}


# ---------------------------------------------------------------------------
# 3. Serialisation (to_dict / to_json / from_dict) + equality
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_has_expected_keys(self) -> None:
        manifest = ProvenanceManifest(input_hash="d" * 64, parameters={"x": 1})
        data = manifest.to_dict()
        assert set(data.keys()) == {
            "input_hash",
            "raw_input_hash",
            "parameters",
            "package_versions",
            "ruleset_versions",
            "exclusions",
            "created_at",
        }
        # created_at is a JSON-friendly ISO string.
        assert isinstance(data["created_at"], str)

    def test_to_json_is_valid_json(self) -> None:
        manifest = ProvenanceManifest(
            input_hash="e" * 64,
            parameters={"method": "delta_cq"},
            package_versions={"bioeng_qpcr": "0.1.0"},
        )
        text = manifest.to_json()
        parsed = json.loads(text)
        assert parsed["input_hash"] == "e" * 64
        assert parsed["parameters"] == {"method": "delta_cq"}
        assert parsed["package_versions"] == {"bioeng_qpcr": "0.1.0"}
        assert "created_at" in parsed

    def test_to_json_indent(self) -> None:
        manifest = ProvenanceManifest(input_hash="f" * 64)
        assert "\n" in manifest.to_json(indent=2)
        assert "\n" not in manifest.to_json(indent=None)

    def test_to_json_handles_non_serialisable_parameters(self) -> None:
        class _Custom:
            def __str__(self) -> str:
                return "custom-value"

        manifest = ProvenanceManifest(parameters={"obj": _Custom()})
        parsed = json.loads(manifest.to_json())
        assert parsed["parameters"]["obj"] == "custom-value"

    def test_round_trip_via_json(self) -> None:
        original = ProvenanceManifest.from_experiment(
            _make_experiment(), parameters={"method": "delta_delta_cq"}
        )
        rebuilt = ProvenanceManifest.from_dict(json.loads(original.to_json()))
        assert rebuilt == original

    def test_round_trip_preserves_created_at(self) -> None:
        manifest = ProvenanceManifest(input_hash="1" * 64)
        rebuilt = ProvenanceManifest.from_dict(manifest.to_dict())
        assert rebuilt.created_at == manifest.created_at
        # And it is still timezone-aware UTC.
        assert rebuilt.created_at.utcoffset() == timedelta(0)

    def test_from_dict_handles_missing_fields(self) -> None:
        rebuilt = ProvenanceManifest.from_dict({})
        assert rebuilt.input_hash is None
        assert rebuilt.parameters == {}
        assert rebuilt.package_versions == {}

    def test_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError):
            ProvenanceManifest.from_dict("not-a-dict")  # type: ignore[arg-type]

    def test_from_dict_rejects_bad_timestamp(self) -> None:
        with pytest.raises(ValueError):
            ProvenanceManifest.from_dict({"created_at": "not-a-timestamp"})

    def test_equality_requires_all_fields(self) -> None:
        base = {
            "input_hash": "a" * 64,
            "parameters": {"k": 1},
            "package_versions": {"numpy": "1.0"},
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        a = ProvenanceManifest.from_dict(base)

        # Same data -> equal.
        assert ProvenanceManifest.from_dict(base) == a

        # Different input hash -> not equal.
        diff = dict(base, input_hash="b" * 64)
        assert ProvenanceManifest.from_dict(diff) != a

        # Different parameters -> not equal.
        diff = dict(base, parameters={"k": 2})
        assert ProvenanceManifest.from_dict(diff) != a

        # Different timestamp -> not equal.
        diff = dict(base, created_at="2025-02-01T00:00:00+00:00")
        assert ProvenanceManifest.from_dict(diff) != a

    def test_equality_against_other_types(self) -> None:
        manifest = ProvenanceManifest(input_hash="a" * 64)
        assert manifest != "not-a-manifest"
        assert manifest != 42


# ---------------------------------------------------------------------------
# 4. Hash computation stability
# ---------------------------------------------------------------------------


class TestHashStability:
    def test_same_experiment_same_content_hash(self) -> None:
        exp1 = _make_experiment()
        exp2 = _make_experiment()
        assert exp1.content_hash() == exp2.content_hash()

    def test_same_experiment_same_manifest_hash(self) -> None:
        m1 = ProvenanceManifest.from_experiment(_make_experiment())
        m2 = ProvenanceManifest.from_experiment(_make_experiment())
        assert m1.input_hash == m2.input_hash

    def test_different_wells_different_hash(self) -> None:
        exp = _make_experiment()
        exp_changed = _make_experiment()
        exp_changed.wells.append(
            Well(
                position="C01",
                sample_name="Sample2",
                target_name="GOI",
                cq=20.5,
                replicate_type=ReplicateType.SINGLE,
            )
        )
        assert exp.content_hash() != exp_changed.content_hash()

    def test_different_cq_different_hash(self) -> None:
        exp = _make_experiment()
        exp_changed = _make_experiment()
        # Mutate the Cq of the first well (Experiment.wells is mutable).
        original_well = exp_changed.wells[0]
        exp_changed.wells[0] = Well(
            position=original_well.position,
            sample_name=original_well.sample_name,
            target_name=original_well.target_name,
            cq=99.99,
            replicate_type=original_well.replicate_type,
            replicate_group=original_well.replicate_group,
            cq_status=original_well.cq_status,
        )
        assert exp.content_hash() != exp_changed.content_hash()

    def test_created_at_does_not_affect_content_hash(self) -> None:
        """content_hash is scientific-content-only; timestamps must not matter."""
        exp = _make_experiment()
        first_hash = exp.content_hash()
        exp.created_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        assert exp.content_hash() == first_hash

    def test_replicate_group_affects_hash(self) -> None:
        exp = _make_experiment()
        changed = _make_experiment()
        well = changed.wells[0]
        changed.wells[0] = Well(
            position=well.position,
            sample_name=well.sample_name,
            target_name=well.target_name,
            cq=well.cq,
            replicate_type=well.replicate_type,
            replicate_group="different-group",
            cq_status=well.cq_status,
        )
        assert changed.content_hash() != exp.content_hash()

    def test_sample_group_affects_hash(self) -> None:
        exp = _make_experiment()
        changed = _make_experiment()
        changed.samples["Sample1"] = Sample(
            name="Sample1",
            group="control",
            replicate_type=ReplicateType.SINGLE,
        )
        assert changed.content_hash() != exp.content_hash()

    def test_primer_metadata_affects_hash(self) -> None:
        exp = _make_experiment()
        changed = _make_experiment()
        assay = changed.assays["GOI"]
        changed.assays["GOI"] = Assay(
            name=assay.name,
            efficiency=assay.efficiency,
            r2=assay.r2,
            is_reference=assay.is_reference,
            primer_forward="ACGTACGT",
        )
        assert changed.content_hash() != exp.content_hash()

    def test_raw_trace_affects_hash(self) -> None:
        exp = _make_experiment()
        changed = _make_experiment()
        changed.traces["A01"] = RawTrace(
            position="A01",
            cycles=numpy.array([1.0, 2.0]),
            fluorescence=numpy.array([0.1, 0.2]),
        )
        assert changed.content_hash() != exp.content_hash()

    def test_local_source_path_does_not_affect_hash(self) -> None:
        first = _make_experiment()
        second = _make_experiment()
        first.metadata["source_file"] = "C:/lab-a/input.csv"
        second.metadata["source_file"] = "D:/lab-b/input.csv"
        assert first.content_hash() == second.content_hash()


# ---------------------------------------------------------------------------
# 5. Package version capture
# ---------------------------------------------------------------------------


class TestPackageVersions:
    def test_capture_all_tracked(self) -> None:
        versions = ProvenanceManifest.capture_package_versions()
        assert set(versions) == {"bioeng_qpcr", "numpy", "scipy", "pandas"}

    def test_numpy_version_matches_module(self) -> None:
        versions = ProvenanceManifest.capture_package_versions()
        assert versions["numpy"] == numpy.__version__

    def test_scipy_version_matches_module(self) -> None:
        versions = ProvenanceManifest.capture_package_versions()
        assert versions["scipy"] == scipy.__version__

    def test_pandas_version_matches_module(self) -> None:
        versions = ProvenanceManifest.capture_package_versions()
        assert versions["pandas"] == pandas.__version__

    def test_all_versions_non_empty_strings(self) -> None:
        versions = ProvenanceManifest.capture_package_versions()
        for _name, version in versions.items():
            assert isinstance(version, str) and version


# ---------------------------------------------------------------------------
# 6. Summary text
# ---------------------------------------------------------------------------


class TestSummary:
    def test_contains_header_and_input_hash(self) -> None:
        manifest = ProvenanceManifest.from_experiment(_make_experiment())
        text = manifest.summary()
        assert "Provenance Manifest" in text
        assert manifest.input_hash in text

    def test_lists_package_versions(self) -> None:
        manifest = ProvenanceManifest.from_experiment(_make_experiment())
        text = manifest.summary()
        assert "Package versions:" in text
        for pkg in ("bioeng_qpcr", "numpy", "scipy", "pandas"):
            assert pkg in text

    def test_lists_parameters(self) -> None:
        manifest = ProvenanceManifest.from_experiment(
            _make_experiment(), parameters={"method": "delta_cq"}
        )
        text = manifest.summary()
        assert "Analysis parameters:" in text
        assert "method" in text
        assert "delta_cq" in text

    def test_summary_without_parameters(self) -> None:
        manifest = ProvenanceManifest(input_hash="0" * 64)
        text = manifest.summary()
        assert "0" * 64 in text
        assert "Package versions:" in text
        assert "(none captured)" in text
        assert "(none provided)" in text

    def test_summary_without_input_hash(self) -> None:
        manifest = ProvenanceManifest()
        text = manifest.summary()
        assert "(not set)" in text


# ---------------------------------------------------------------------------
# 7. Package-level export
# ---------------------------------------------------------------------------


def test_package_exports_manifest() -> None:
    from bioeng_qpcr import provenance

    assert provenance.ProvenanceManifest is ProvenanceManifest
    assert "ProvenanceManifest" in provenance.__all__
