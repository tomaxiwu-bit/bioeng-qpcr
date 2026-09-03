"""Tests for :mod:`bioeng_qpcr.io.csv_reader`.

These tests exercise the Level A tidy CSV reader against four committed
fixture files as well as a number of inline CSVs built with ``tmp_path`` for
behaviours that require optional columns (reference genes, efficiency,
sample groups, explicit ``cq_status``, error handling).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bioeng_qpcr.domain import (
    Assay,
    Experiment,
    InputLevel,
    ReplicateType,
    Sample,
    Well,
)
from bioeng_qpcr.io import CSVValidationError, read_tidy_csv

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _fixture(name: str) -> Path:
    """Resolve a fixture path under ``tests/fixtures``."""
    return FIXTURES / name


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    """Write ``content`` to ``tmp_path / name`` and return the path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Basic CSV fixture
# ---------------------------------------------------------------------------


class TestReadBasicCsv:
    """``level_a_basic.csv``: 3 samples x 2 targets, no replicates."""

    def test_returns_experiment_instance(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert isinstance(exp, Experiment)

    def test_input_level_is_level_a(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert exp.input_level is InputLevel.LEVEL_A

    def test_well_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert len(exp.wells) == 6

    def test_sample_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert len(exp.samples) == 3
        assert set(exp.samples) == {"S1", "S2", "S3"}

    def test_assay_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert len(exp.assays) == 2
        assert set(exp.assays) == {"GAPDH", "ACTB"}

    def test_wells_are_well_objects(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert all(isinstance(w, Well) for w in exp.wells)

    def test_samples_are_sample_objects(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert all(isinstance(s, Sample) for s in exp.samples.values())

    def test_assays_are_assay_objects(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert all(isinstance(a, Assay) for a in exp.assays.values())

    def test_cq_values_parsed_as_float(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A01"].cq == 20.5
        assert by_pos["A02"].cq == 18.3
        assert by_pos["A04"].cq == 19.1
        assert by_pos["A06"].cq == 17.9

    def test_all_cq_determined(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert all(w.cq is not None for w in exp.wells)
        assert all(w.cq_status == "determined" for w in exp.wells)

    def test_default_replicate_type_is_single(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert all(w.replicate_type is ReplicateType.SINGLE for w in exp.wells)

    def test_no_ntc_wells(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert exp.ntc_wells == []
        assert exp.has_ntc is False

    def test_no_reference_genes_by_default(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert exp.reference_targets == []
        assert exp.has_reference_genes is False

    def test_experiment_name_defaults_to_filename(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert exp.name == "level_a_basic"

    def test_explicit_experiment_name(self) -> None:
        exp = read_tidy_csv(
            _fixture("level_a_basic.csv"), experiment_name="my_experiment"
        )
        assert exp.name == "my_experiment"

    def test_metadata_contains_source_and_counts(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert "source_file" in exp.metadata
        assert exp.metadata["n_wells"] == 6
        assert exp.metadata["n_samples"] == 3
        assert exp.metadata["n_assays"] == 2

    def test_accepts_string_path(self) -> None:
        exp = read_tidy_csv(str(_fixture("level_a_basic.csv")))
        assert len(exp.wells) == 6

    def test_well_sample_target_mapping(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A01"].sample_name == "S1"
        assert by_pos["A01"].target_name == "GAPDH"
        assert by_pos["A06"].sample_name == "S3"
        assert by_pos["A06"].target_name == "ACTB"


# ---------------------------------------------------------------------------
# NTC fixture
# ---------------------------------------------------------------------------


class TestReadWithNtc:
    """``level_a_with_ntc.csv``: basic data plus two NTC wells."""

    def test_well_count_includes_ntc(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        assert len(exp.wells) == 8

    def test_ntc_wells_detected(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        ntc = exp.ntc_wells
        assert len(ntc) == 2
        assert all(w.sample_name == "NTC" for w in ntc)

    def test_has_ntc_flag(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        assert exp.has_ntc is True

    def test_ntc_with_cq_keeps_value(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        by_pos = {w.position: w for w in exp.wells}
        # NTC well showing amplification (contamination signal).
        assert by_pos["A07"].cq == 35.0
        assert by_pos["A07"].is_ntc is True

    def test_ntc_undetermined_cq_is_none(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A08"].cq is None
        assert by_pos["A08"].is_ntc is True

    def test_ntc_cq_status_is_no_template(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        for w in exp.ntc_wells:
            assert w.cq_status == "no_template"

    def test_ntc_excluded_from_sample_names(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        # ``sample_names`` excludes NTC wells.
        assert set(exp.sample_names) == {"S1", "S2", "S3"}

    def test_ntc_included_in_samples_dict(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        # The raw ``samples`` dict still records NTC as a Sample.
        assert "NTC" in exp.samples
        assert len(exp.samples) == 4

    def test_non_ntc_wells_remain_determined(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_ntc.csv"))
        non_ntc = [w for w in exp.wells if not w.is_ntc]
        assert len(non_ntc) == 6
        assert all(w.cq_status == "determined" for w in non_ntc)


# ---------------------------------------------------------------------------
# Replicates fixture
# ---------------------------------------------------------------------------


class TestReadWithReplicates:
    """``level_a_with_replicates.csv``: 2 samples x 2 targets x 3 technicals."""

    def test_well_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        assert len(exp.wells) == 12

    def test_sample_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        assert set(exp.samples) == {"S1", "S2"}

    def test_assay_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        assert set(exp.assays) == {"GAPDH", "ACTB"}

    def test_replicate_type_parsed_as_technical(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        assert all(w.replicate_type is ReplicateType.TECHNICAL for w in exp.wells)

    def test_replicate_group_captured(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        groups = {w.replicate_group for w in exp.wells}
        assert groups == {"S1_GAPDH", "S1_ACTB", "S2_GAPDH", "S2_ACTB"}

    def test_replicate_group_per_well(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        s1_gapdh = [
            w for w in exp.wells if w.sample_name == "S1" and w.target_name == "GAPDH"
        ]
        assert len(s1_gapdh) == 3
        assert all(w.replicate_group == "S1_GAPDH" for w in s1_gapdh)

    def test_sample_replicate_type_is_technical(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        assert exp.samples["S1"].replicate_type is ReplicateType.TECHNICAL
        assert exp.samples["S2"].replicate_type is ReplicateType.TECHNICAL

    def test_three_replicates_per_combination(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_with_replicates.csv"))
        for sample in ("S1", "S2"):
            for target in ("GAPDH", "ACTB"):
                wells = exp.wells_for(sample, target)
                assert len(wells) == 3


# ---------------------------------------------------------------------------
# Undetermined fixture
# ---------------------------------------------------------------------------


class TestReadUndetermined:
    """``level_a_undetermined.csv``: ``Undetermined`` token + empty Cq fields."""

    def test_well_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        assert len(exp.wells) == 6

    def test_undetermined_string_cq_is_none(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A02"].cq is None  # "Undetermined"

    def test_empty_cq_is_none(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A05"].cq is None  # empty field

    def test_numeric_cq_preserved(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A01"].cq == 20.5
        assert by_pos["A03"].cq == 21.0
        assert by_pos["A04"].cq == 19.1
        assert by_pos["A06"].cq == 17.9

    def test_undetermined_cq_status(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A02"].cq_status == "undetermined"
        assert by_pos["A05"].cq_status == "undetermined"

    def test_determined_cq_status(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A01"].cq_status == "determined"
        assert by_pos["A04"].cq_status == "determined"

    def test_none_cq_count(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_undetermined.csv"))
        none_cq = [w for w in exp.wells if w.cq is None]
        assert len(none_cq) == 2


# ---------------------------------------------------------------------------
# Cq parsing variants (inline)
# ---------------------------------------------------------------------------


class TestCqParsing:
    """Exercise the Cq parser across sentinel tokens and numeric formats."""

    def test_nan_token_is_none(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "nan.csv",
            "position,sample_name,target_name,cq\nA01,S1,GAPDH,NaN\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.wells[0].cq is None

    def test_empty_string_is_none(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "empty.csv",
            "position,sample_name,target_name,cq\nA01,S1,GAPDH,\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.wells[0].cq is None

    def test_integer_cq(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "int.csv",
            "position,sample_name,target_name,cq\nA01,S1,GAPDH,25\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.wells[0].cq == 25.0

    def test_decimal_cq(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "dec.csv",
            "position,sample_name,target_name,cq\nA01,S1,GAPDH,22.34\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.wells[0].cq == pytest.approx(22.34)


# ---------------------------------------------------------------------------
# Replicate-type parsing variants (inline)
# ---------------------------------------------------------------------------


class TestReplicateTypeParsing:
    def test_biological_replicate_type(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bio.csv",
            "position,sample_name,target_name,cq,replicate_type\n"
            "A01,S1,GAPDH,20.5,biological\n"
            "A02,S2,GAPDH,21.0,biological\n",
        )
        exp = read_tidy_csv(csv)
        assert all(w.replicate_type is ReplicateType.BIOLOGICAL for w in exp.wells)

    def test_single_replicate_type(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "single.csv",
            "position,sample_name,target_name,cq,replicate_type\n"
            "A01,S1,GAPDH,20.5,single\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.wells[0].replicate_type is ReplicateType.SINGLE

    def test_case_insensitive_replicate_type(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "ci.csv",
            "position,sample_name,target_name,cq,replicate_type\n"
            "A01,S1,GAPDH,20.5,TECHNICAL\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.wells[0].replicate_type is ReplicateType.TECHNICAL

    def test_invalid_replicate_type_is_rejected(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad_rt.csv",
            "position,sample_name,target_name,cq,replicate_type\n"
            "A01,S1,GAPDH,20.5,bogus\n",
        )
        with pytest.raises(CSVValidationError, match="INVALID_REPLICATE_TYPE"):
            read_tidy_csv(csv)
        exp = read_tidy_csv(csv, strict=False)
        assert exp.wells[0].replicate_type is ReplicateType.SINGLE
        assert exp.metadata["input_diagnostics"][0]["code"] == (
            "INVALID_REPLICATE_TYPE"
        )


# ---------------------------------------------------------------------------
# Reference gene marking
# ---------------------------------------------------------------------------


class TestReferenceGenes:
    def test_reference_via_parameter(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"), reference_targets=["GAPDH"])
        assert exp.assays["GAPDH"].is_reference is True
        assert exp.assays["ACTB"].is_reference is False
        assert exp.reference_targets == ["GAPDH"]
        assert exp.has_reference_genes is True

    def test_reference_via_column(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "ref.csv",
            "position,sample_name,target_name,cq,is_reference\n"
            "A01,S1,GAPDH,20.5,true\n"
            "A02,S1,ACTB,18.3,false\n"
            "A03,S2,GAPDH,21.0,true\n"
            "A04,S2,ACTB,19.1,false\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.assays["GAPDH"].is_reference is True
        assert exp.assays["ACTB"].is_reference is False

    def test_parameter_overrides_column_false(self, tmp_path: Path) -> None:
        # Column says false, parameter says reference -> True.
        csv = _write_csv(
            tmp_path,
            "ref2.csv",
            "position,sample_name,target_name,cq,is_reference\n"
            "A01,S1,GAPDH,20.5,false\n"
            "A02,S1,ACTB,18.3,false\n",
        )
        exp = read_tidy_csv(csv, reference_targets=["GAPDH"])
        assert exp.assays["GAPDH"].is_reference is True
        assert exp.assays["ACTB"].is_reference is False

    def test_multiple_reference_targets(self) -> None:
        exp = read_tidy_csv(
            _fixture("level_a_basic.csv"), reference_targets=["GAPDH", "ACTB"]
        )
        assert exp.assays["GAPDH"].is_reference is True
        assert exp.assays["ACTB"].is_reference is True
        assert sorted(exp.reference_targets) == ["ACTB", "GAPDH"]

    def test_reference_target_not_in_data_is_rejected(self) -> None:
        with pytest.raises(CSVValidationError, match="UNKNOWN_REFERENCE_TARGET"):
            read_tidy_csv(
                _fixture("level_a_basic.csv"),
                reference_targets=["NONEXISTENT"],
            )
        exp = read_tidy_csv(
            _fixture("level_a_basic.csv"),
            reference_targets=["NONEXISTENT"],
            strict=False,
        )
        assert exp.has_reference_genes is False


# ---------------------------------------------------------------------------
# Efficiency capture
# ---------------------------------------------------------------------------


class TestEfficiency:
    def test_efficiency_captured(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "eff.csv",
            "position,sample_name,target_name,cq,efficiency\n"
            "A01,S1,GAPDH,20.5,0.98\n"
            "A02,S1,ACTB,18.3,1.02\n"
            "A03,S2,GAPDH,21.0,0.98\n"
            "A04,S2,ACTB,19.1,1.02\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.assays["GAPDH"].efficiency == pytest.approx(0.98)
        assert exp.assays["ACTB"].efficiency == pytest.approx(1.02)
        assert exp.has_efficiency is True

    def test_efficiency_absent_is_none(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert exp.assays["GAPDH"].efficiency is None
        assert exp.assays["ACTB"].efficiency is None
        assert exp.has_efficiency is False

    def test_efficiency_first_non_null_wins(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "eff2.csv",
            "position,sample_name,target_name,cq,efficiency\n"
            "A01,S1,GAPDH,20.5,0.98\n"
            "A02,S2,GAPDH,21.0,\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.assays["GAPDH"].efficiency == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# Sample group assignment
# ---------------------------------------------------------------------------


class TestSampleGroup:
    def test_sample_group_assigned(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "grp.csv",
            "position,sample_name,target_name,cq,sample_group\n"
            "A01,S1,GAPDH,20.5,treated\n"
            "A02,S1,ACTB,18.3,treated\n"
            "A03,S2,GAPDH,21.0,control\n"
            "A04,S2,ACTB,19.1,control\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.samples["S1"].group == "treated"
        assert exp.samples["S2"].group == "control"

    def test_sample_group_absent_is_none(self) -> None:
        exp = read_tidy_csv(_fixture("level_a_basic.csv"))
        assert all(s.group is None for s in exp.samples.values())


# ---------------------------------------------------------------------------
# Explicit cq_status column
# ---------------------------------------------------------------------------


class TestExplicitCqStatus:
    def test_explicit_cq_status_used(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "status.csv",
            "position,sample_name,target_name,cq,cq_status\n"
            "A01,S1,GAPDH,20.5,determined\n"
            "A02,S1,ACTB,,undetermined\n"
            "A03,NTC,GAPDH,35.0,no_template\n",
        )
        exp = read_tidy_csv(csv)
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A01"].cq_status == "determined"
        assert by_pos["A02"].cq_status == "undetermined"
        assert by_pos["A03"].cq_status == "no_template"

    def test_empty_cq_status_falls_back_to_inference(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "status2.csv",
            "position,sample_name,target_name,cq,cq_status\n"
            "A01,S1,GAPDH,,\n"
            "A02,NTC,GAPDH,35.0,\n",
        )
        exp = read_tidy_csv(csv)
        by_pos = {w.position: w for w in exp.wells}
        assert by_pos["A01"].cq_status == "undetermined"
        assert by_pos["A02"].cq_status == "no_template"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestMissingColumns:
    def test_missing_cq_raises(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad.csv",
            "position,sample_name,target_name\nA01,S1,GAPDH\n",
        )
        with pytest.raises(ValueError, match="cq"):
            read_tidy_csv(csv)

    def test_missing_position_raises(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad2.csv",
            "sample_name,target_name,cq\nS1,GAPDH,20.5\n",
        )
        with pytest.raises(ValueError, match="position"):
            read_tidy_csv(csv)

    def test_missing_multiple_raises_with_all(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad3.csv",
            "sample_name,target_name\nS1,GAPDH\n",
        )
        with pytest.raises(ValueError) as excinfo:
            read_tidy_csv(csv)
        msg = str(excinfo.value)
        assert "position" in msg
        assert "cq" in msg

    def test_column_name_normalised_case_insensitive(self, tmp_path: Path) -> None:
        # Uppercase / spaced headers should still be recognised.
        csv = _write_csv(
            tmp_path,
            "case.csv",
            "Position,Sample_Name,Target_Name,Cq\nA01,S1,GAPDH,20.5\n",
        )
        exp = read_tidy_csv(csv)
        assert len(exp.wells) == 1
        assert exp.wells[0].cq == 20.5


# ---------------------------------------------------------------------------
# Internal helper edge cases (direct unit tests)
# ---------------------------------------------------------------------------

from bioeng_qpcr.io.csv_reader import (  # noqa: E402  -- intentional late import
    _is_missing,
    _parse_bool,
    _parse_cq,
    _parse_float,
    _parse_replicate_type,
)


class TestIsMissingEdgeCases:
    """Direct tests for the ``_is_missing`` parsing helper."""

    def test_none_is_missing(self) -> None:
        # Covers line 59: the ``if value is None`` early return.
        assert _is_missing(None) is True

    def test_nan_float_is_missing(self) -> None:
        # pd.isna(nan) is True -> covers line 62 (``return True`` inside try).
        assert _is_missing(float("nan")) is True

    def test_empty_string_is_missing(self) -> None:
        assert _is_missing("") is True

    def test_whitespace_string_is_missing(self) -> None:
        assert _is_missing("   ") is True

    def test_list_triggers_typeerror_fallthrough(self) -> None:
        # pd.isna on a list returns a numpy array; using it in a boolean
        # context raises ValueError, which is caught (lines 63-64) so
        # execution falls through to the string check. str([1, 2]) is
        # "[1, 2]" (non-empty), so _is_missing returns False rather than
        # crashing -- the important behaviour is that no exception escapes.
        assert _is_missing([1, 2]) is False

    def test_numeric_value_not_missing(self) -> None:
        assert _is_missing(42) is False

    def test_nonempty_string_not_missing(self) -> None:
        assert _is_missing("hello") is False


class TestParseCqEdgeCases:
    """Direct tests for the ``_parse_cq`` parsing helper."""

    def test_none_returns_none(self) -> None:
        # Covers line 75: the ``if raw is None`` early return.
        assert _parse_cq(None) is None

    def test_nan_float_returns_none(self) -> None:
        # pd.isna(nan) is True -> covers line 78 (``return None`` inside try).
        assert _parse_cq(float("nan")) is None

    def test_non_numeric_string_returns_none(self) -> None:
        # "abc" is not an undetermined token and not parseable as float ->
        # covers lines 86-87 (the ValueError/TypeError catch).
        assert _parse_cq("abc") is None

    def test_list_value_does_not_crash(self) -> None:
        # pd.isna on a list raises ValueError (caught, lines 79-80); then
        # str([1, 2]) is "[1, 2]" which is not a token and not a float ->
        # float() raises ValueError -> returns None (lines 86-87).
        assert _parse_cq([1, 2]) is None

    def test_undetermined_token_returns_none(self) -> None:
        assert _parse_cq("undetermined") is None

    def test_na_token_returns_none(self) -> None:
        assert _parse_cq("N/A") is None


class TestParseFloatEdgeCases:
    """Direct tests for the ``_parse_float`` parsing helper."""

    def test_none_returns_none(self) -> None:
        # Covers line 93: ``_is_missing(raw)`` is True.
        assert _parse_float(None) is None

    def test_empty_string_returns_none(self) -> None:
        # Covers line 93: ``_is_missing("")`` is True.
        assert _parse_float("") is None

    def test_non_numeric_returns_none(self) -> None:
        # float("abc") raises ValueError -> covers lines 96-97.
        assert _parse_float("abc") is None

    def test_nan_string_returns_none(self) -> None:
        # float("nan") succeeds, then math.isnan(value) is True -> covers
        # line 99.
        assert _parse_float("nan") is None


class TestParseReplicateTypeEdgeCases:
    """Direct tests for the ``_parse_replicate_type`` parsing helper."""

    def test_none_returns_single(self) -> None:
        # Covers line 106: ``_is_missing(raw)`` is True -> SINGLE.
        assert _parse_replicate_type(None) is ReplicateType.SINGLE

    def test_empty_string_returns_single(self) -> None:
        # Covers line 106: ``_is_missing("")`` is True -> SINGLE.
        assert _parse_replicate_type("") is ReplicateType.SINGLE


class TestParseBoolEdgeCases:
    """Direct tests for the ``_parse_bool`` parsing helper."""

    def test_none_returns_false(self) -> None:
        # Covers line 116: ``_is_missing(raw)`` is True -> False.
        assert _parse_bool(None) is False

    def test_empty_string_returns_false(self) -> None:
        # Covers line 116: ``_is_missing("")`` is True -> False.
        assert _parse_bool("") is False

    def test_yes_returns_true(self) -> None:
        assert _parse_bool("yes") is True

    def test_zero_returns_false(self) -> None:
        assert _parse_bool("0") is False


# ---------------------------------------------------------------------------
# Update-on-second-occurrence logic (integration via CSV files)
# ---------------------------------------------------------------------------


class TestSecondOccurrenceUpdates:
    """Cover the merge/upgrade branches triggered when a sample or assay is
    seen a second time carrying richer metadata than the first row."""

    def test_sample_group_updated_on_second_well(self, tmp_path: Path) -> None:
        # First row for S1 has an empty sample_group (-> group stays None);
        # second row for S1 provides "treated" -> covers line 255.
        csv = _write_csv(
            tmp_path,
            "grp_second.csv",
            "position,sample_name,target_name,cq,sample_group\n"
            "A01,S1,GAPDH,20.5,\n"
            "A02,S1,ACTB,18.3,treated\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.samples["S1"].group == "treated"

    def test_conflicting_replicate_type_is_rejected(self, tmp_path: Path) -> None:
        # First row for S1 is "single" (-> replicate_type SINGLE); second row
        # is "technical" -> covers line 260 (upgrade from SINGLE).
        csv = _write_csv(
            tmp_path,
            "rt_second.csv",
            "position,sample_name,target_name,cq,replicate_type\n"
            "A01,S1,GAPDH,20.5,single\n"
            "A02,S1,ACTB,18.3,technical\n",
        )
        with pytest.raises(CSVValidationError, match="CONFLICTING_REPLICATE_TYPE"):
            read_tidy_csv(csv)
        exp = read_tidy_csv(csv, strict=False)
        assert exp.samples["S1"].replicate_type is ReplicateType.SINGLE

    def test_assay_efficiency_updated_on_second_well(self, tmp_path: Path) -> None:
        # First GAPDH row has an empty efficiency (-> efficiency None);
        # second GAPDH row provides "0.98" -> covers lines 275-277.
        csv = _write_csv(
            tmp_path,
            "eff_second.csv",
            "position,sample_name,target_name,cq,efficiency\n"
            "A01,S1,GAPDH,20.5,\n"
            "A02,S2,GAPDH,21.0,0.98\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.assays["GAPDH"].efficiency == pytest.approx(0.98)

    def test_assay_efficiency_not_updated_when_second_also_missing(
        self, tmp_path: Path
    ) -> None:
        # First GAPDH row has empty efficiency (-> None); second GAPDH row is
        # also empty so _parse_float returns None and the ``if parsed is not
        # None`` branch is False -> efficiency stays None (branch 276->278).
        csv = _write_csv(
            tmp_path,
            "eff_both_missing.csv",
            "position,sample_name,target_name,cq,efficiency\n"
            "A01,S1,GAPDH,20.5,\n"
            "A02,S2,GAPDH,21.0,\n",
        )
        exp = read_tidy_csv(csv)
        assert exp.assays["GAPDH"].efficiency is None

    def test_conflicting_reference_flag_is_rejected(self, tmp_path: Path) -> None:
        # First GAPDH row has is_reference "false" (-> False); second GAPDH
        # row has "true" -> covers line 283 (upgrade to True).
        csv = _write_csv(
            tmp_path,
            "ref_second.csv",
            "position,sample_name,target_name,cq,is_reference\n"
            "A01,S1,GAPDH,20.5,false\n"
            "A02,S2,GAPDH,21.0,true\n",
        )
        with pytest.raises(CSVValidationError, match="CONFLICTING_ASSAY_METADATA"):
            read_tidy_csv(csv)
        exp = read_tidy_csv(csv, strict=False)
        assert exp.assays["GAPDH"].is_reference is False


class TestStructuredScientificDiagnostics:
    def test_malformed_cq_is_not_silently_undetermined(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad_cq.csv",
            "position,sample_name,target_name,cq\nA01,S1,GAPDH,not-a-cq\n",
        )
        with pytest.raises(CSVValidationError) as caught:
            read_tidy_csv(csv)
        issue = caught.value.issues[0]
        assert issue.code == "INVALID_NUMBER"
        assert issue.row == 2
        assert issue.column == "cq"

    def test_invalid_r2_is_structured_and_recoverable(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad_r2.csv",
            "position,sample_name,target_name,cq,r2\nA01,S1,GAPDH,20,1.2\n",
        )
        with pytest.raises(CSVValidationError, match="INVALID_RANGE"):
            read_tidy_csv(csv)
        experiment = read_tidy_csv(csv, strict=False)
        assert experiment.assays["GAPDH"].r2 is None
        assert experiment.metadata["input_diagnostics"][0]["column"] == "r2"

    def test_lloq_above_lod_is_rejected(self, tmp_path: Path) -> None:
        csv = _write_csv(
            tmp_path,
            "bad_limits.csv",
            "position,sample_name,target_name,cq,lod_cq,lloq_cq\n"
            "A01,S1,GAPDH,20,35,38\n",
        )
        with pytest.raises(CSVValidationError, match="lloq_cq"):
            read_tidy_csv(csv)

    def test_numeric_cq_with_undetermined_status_is_rejected(
        self, tmp_path: Path
    ) -> None:
        csv = _write_csv(
            tmp_path,
            "bad_status.csv",
            "position,sample_name,target_name,cq,cq_status\n"
            "A01,S1,GAPDH,20,undetermined\n",
        )
        with pytest.raises(CSVValidationError, match="INCONSISTENT_CQ_STATUS"):
            read_tidy_csv(csv)
