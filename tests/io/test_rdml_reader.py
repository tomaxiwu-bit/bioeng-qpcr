"""Tests for the RDML reader."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pytest

from bioeng_qpcr.domain import InputLevel, ReplicateType
from bioeng_qpcr.io import CSVValidationError, read_rdml

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
EXAMPLE_PATH = EXAMPLES / "rdml_example.xml"

_DEFAULT_TARGETS = (
    '<target id="T1"><type>toi</type><dyeId id="SYBR"/></target>'
    '<target id="T2"><type>toi</type><dyeId id="SYBR"/></target>'
)


def _rdml_doc(
    reacts: str,
    *,
    samples: str | None = None,
    targets: str | None = None,
    runs: str | None = None,
) -> str:
    """Assemble a small RDML document.

    *reacts* fills the single default run; pass *runs* instead to control
    the run structure (e.g. multi-run tests).
    """
    if runs is None:
        runs = (
            "<run id=\"r1\"><pcrFormat><rows>8</rows><columns>12</columns>"
            "<rowLabel>ABC</rowLabel><columnLabel>123</columnLabel></pcrFormat>"
            f"{reacts}</run>"
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rdml version="1.2" xmlns="http://www.rdml.org">
  {samples or '<sample id="S1"><type>unkn</type></sample>'}
  {targets or _DEFAULT_TARGETS}
  <experiment id="exp">{runs}</experiment>
</rdml>
"""


def _react(react_id: str, sample: str, target: str, cq: str | None) -> str:
    cq_xml = f"<cq>{cq}</cq>" if cq is not None else ""
    return (
        f'<react id="{react_id}"><sample id="{sample}"/>'
        f'<data><tar id="{target}"/>{cq_xml}</data></react>'
    )


def _write(tmp_path: Path, content: str, filename: str = "test.rdml") -> Path:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


class TestExampleFile:
    def test_reads_all_wells(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.name == "researcher_example"
        assert experiment.input_level is InputLevel.LEVEL_C
        assert len(experiment.wells) == 30
        assert experiment.metadata["n_wells"] == 30
        assert experiment.metadata["parse_mode"] == "strict"

    def test_plate_positions_from_react_id(self):
        experiment = read_rdml(EXAMPLE_PATH)
        positions = {well.position for well in experiment.wells}

        assert {"A01", "A02", "B01", "C01", "D06", "E06"} <= positions
        c01 = next(w for w in experiment.wells if w.position == "C01")
        assert c01.sample_name == "Treated_1"
        assert c01.target_name == "IL6"

    def test_cq_values(self):
        experiment = read_rdml(EXAMPLE_PATH)
        a01 = next(w for w in experiment.wells if w.position == "A01")

        assert a01.cq == pytest.approx(25.10)
        assert a01.cq_status == "determined"

    def test_ntc_wells(self):
        experiment = read_rdml(EXAMPLE_PATH)
        ntc_wells = [w for w in experiment.wells if w.sample_name == "NTC"]

        assert len(ntc_wells) == 6
        assert all(w.cq_status == "no_template" for w in ntc_wells)
        assert all(w.cq is None for w in ntc_wells)
        assert all(w.is_ntc for w in ntc_wells)

    def test_reference_targets_from_target_type(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.reference_targets == ["GAPDH", "RPLP0"]
        assert experiment.assays["IL6"].is_reference is False

    def test_efficiency_factor_conversion(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.assays["IL6"].efficiency == pytest.approx(0.96)
        assert experiment.assays["GAPDH"].efficiency == pytest.approx(0.98)
        assert experiment.assays["RPLP0"].efficiency == pytest.approx(1.01)

    def test_primer_sequences(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.assays["IL6"].primer_forward == "ATTGCCCTAAGGACCTCAGA"
        assert experiment.assays["IL6"].primer_reverse == "GCTCAGTCTCCTCATCGAATT"

    def test_sample_groups_from_annotations(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.samples["Control_1"].group == "control"
        assert experiment.samples["Control_2"].group == "control"
        assert experiment.samples["Treated_1"].group == "treated"
        assert experiment.samples["Treated_2"].group == "treated"
        assert experiment.samples["NTC"].group is None

    def test_technical_replicates_derived(self):
        experiment = read_rdml(EXAMPLE_PATH)
        wells = [
            w
            for w in experiment.wells
            if w.sample_name == "Control_1" and w.target_name == "IL6"
        ]

        assert len(wells) == 2
        assert all(w.replicate_type is ReplicateType.TECHNICAL for w in wells)
        assert all(w.replicate_group == "Control_1_IL6" for w in wells)

    def test_raw_traces(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.has_traces
        assert set(experiment.traces) == {"A01", "A02"}
        assert experiment.traces["A01"].n_cycles == 8
        assert experiment.traces["A01"].fluorescence[0] == pytest.approx(0.42)

    def test_metadata(self):
        experiment = read_rdml(EXAMPLE_PATH)

        assert experiment.metadata["rdml_version"] == "1.2"
        assert experiment.metadata["container_format"] == "xml"
        assert experiment.metadata["input_diagnostics"] == []
        assert len(experiment.metadata["source_sha256"]) == 64

    def test_matches_csv_analysis_results(self):
        from bioeng_qpcr.calculations import normalized_quantity_all
        from bioeng_qpcr.io import read_tidy_csv

        references = ["GAPDH", "RPLP0"]
        csv_experiment = read_tidy_csv(
            EXAMPLES / "researcher_example.csv", reference_targets=references
        )
        rdml_experiment = read_rdml(EXAMPLE_PATH)

        csv_results = normalized_quantity_all(csv_experiment, "IL6", references)
        rdml_results = normalized_quantity_all(rdml_experiment, "IL6", references)

        csv_map = {r.sample_name: r.normalized_quantity for r in csv_results.results}
        rdml_map = {r.sample_name: r.normalized_quantity for r in rdml_results.results}
        assert set(csv_map) == set(rdml_map)
        for sample, quantity in csv_map.items():
            assert rdml_map[sample] == pytest.approx(quantity)


class TestContainers:
    def test_zip_archive_with_rdml_data_xml(self, tmp_path):
        target = tmp_path / "example.rdml"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("rdml_data.xml", EXAMPLE_PATH.read_text(encoding="utf-8"))

        experiment = read_rdml(target)

        assert experiment.metadata["container_format"] == "zip"
        assert len(experiment.wells) == 30

    def test_gzip_compressed(self, tmp_path):
        target = tmp_path / "example.rdml"
        target.write_bytes(gzip.compress(EXAMPLE_PATH.read_bytes()))

        experiment = read_rdml(target)

        assert experiment.metadata["container_format"] == "gzip"
        assert len(experiment.wells) == 30

    def test_zip_without_xml_member_raises(self, tmp_path):
        target = tmp_path / "broken.rdml"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("readme.txt", "no xml here")

        with pytest.raises(ValueError, match="no XML member"):
            read_rdml(target)


class TestCqParsing:
    def test_na_sentinel(self, tmp_path):
        experiment = read_rdml(
            _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", "-1.0")))
        )

        assert experiment.wells[0].cq is None
        assert experiment.wells[0].cq_status == "undetermined"

    def test_missing_cq_element(self, tmp_path):
        experiment = read_rdml(
            _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", None)))
        )

        assert experiment.wells[0].cq is None
        assert experiment.wells[0].cq_status == "undetermined"

    def test_negative_cq_rejected(self, tmp_path):
        path = _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", "-2.5")))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "INVALID_CQ" for i in excinfo.value.issues)

    def test_nonnumeric_cq_rejected(self, tmp_path):
        path = _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", "abc")))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "INVALID_CQ" for i in excinfo.value.issues)


class TestExcludedData:
    def test_excl_skips_well_without_raising(self, tmp_path):
        excluded = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<excl>manual review</excl></data></react>"
        )
        doc = _rdml_doc(excluded + _react("2", "S1", "T1", "21.0"))

        experiment = read_rdml(_write(tmp_path, doc))

        assert len(experiment.wells) == 1
        assert experiment.wells[0].cq == pytest.approx(21.0)
        assert experiment.metadata["excluded_data_entries"] == 1
        codes = {d["code"] for d in experiment.metadata["input_diagnostics"]}
        assert "EXCLUDED_DATA" in codes


class TestMultiplex:
    def test_position_suffix_for_second_target(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/>'
            '<data><tar id="T1"/><cq>20.0</cq></data>'
            '<data><tar id="T2"/><cq>22.0</cq></data>'
            "</react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react)))

        positions = sorted(w.position for w in experiment.wells)
        assert positions == ["A01", "A01:T2"]
        by_target = {w.target_name: w for w in experiment.wells}
        assert by_target["T1"].cq == pytest.approx(20.0)
        assert by_target["T2"].cq == pytest.approx(22.0)
        assert experiment.metadata["multiplex_data_entries"] == 1
        codes = {d["code"] for d in experiment.metadata["input_diagnostics"]}
        assert "MULTIPLEX_POSITION_SUFFIX" in codes


class TestReferenceResolution:
    def test_unknown_sample_ref(self, tmp_path):
        path = _write(tmp_path, _rdml_doc(_react("1", "GHOST", "T1", "20.0")))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "MISSING_SAMPLE_REF" for i in excinfo.value.issues)

    def test_unknown_target_ref(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/>'
            '<data><tar id="GHOST"/><cq>20.0</cq></data></react>'
        )
        path = _write(tmp_path, _rdml_doc(react))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "MISSING_TARGET_REF" for i in excinfo.value.issues)

    def test_unknown_reference_target_param(self, tmp_path):
        path = _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0")))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path, reference_targets=["GHOST"])

        assert any(i.code == "UNKNOWN_REFERENCE_TARGET" for i in excinfo.value.issues)

    def test_reference_target_param_marks_reference(self, tmp_path):
        path = _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0")))

        experiment = read_rdml(path, reference_targets=["T1"])

        assert experiment.reference_targets == ["T1"]

    def test_ref_type_target_marks_reference(self, tmp_path):
        targets = (
            '<target id="T1"><type>ref</type><dyeId id="SYBR"/></target>'
            '<target id="T2"><type>toi</type><dyeId id="SYBR"/></target>'
        )
        path = _write(
            tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0"), targets=targets)
        )

        experiment = read_rdml(path)

        assert experiment.reference_targets == ["T1"]
        assert experiment.assays["T1"].is_reference is True


class TestEfficiencyValidation:
    @pytest.mark.parametrize("factor", ["0.5", "4.0", "abc"])
    def test_invalid_factor_rejected(self, tmp_path, factor):
        targets = (
            '<target id="T1"><type>toi</type>'
            f"<amplificationEfficiency>{factor}</amplificationEfficiency>"
            '<dyeId id="SYBR"/></target>'
            '<target id="T2"><type>toi</type><dyeId id="SYBR"/></target>'
        )
        path = _write(
            tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0"), targets=targets)
        )

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "INVALID_EFFICIENCY" for i in excinfo.value.issues)

    def test_valid_factor_converted(self, tmp_path):
        targets = (
            '<target id="T1"><type>toi</type>'
            "<amplificationEfficiency>1.95</amplificationEfficiency>"
            '<dyeId id="SYBR"/></target>'
            '<target id="T2"><type>toi</type><dyeId id="SYBR"/></target>'
        )
        path = _write(
            tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0"), targets=targets)
        )

        experiment = read_rdml(path)

        assert experiment.assays["T1"].efficiency == pytest.approx(0.95)


class TestStructureValidation:
    def test_duplicate_react_id(self, tmp_path):
        reacts = _react("1", "S1", "T1", "20.0") + _react("1", "S1", "T1", "21.0")
        path = _write(tmp_path, _rdml_doc(reacts))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "DUPLICATE_REACT" for i in excinfo.value.issues)

    def test_duplicate_react_id_after_int_normalization(self, tmp_path):
        reacts = _react("1", "S1", "T1", "20.0") + _react("01", "S1", "T1", "21.0")
        path = _write(tmp_path, _rdml_doc(reacts))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "DUPLICATE_REACT" for i in excinfo.value.issues)

    def test_no_experiment_raises(self, tmp_path):
        doc = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rdml version="1.2" xmlns="http://www.rdml.org"/>'
        )
        path = _write(tmp_path, doc)

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "EMPTY_FILE" for i in excinfo.value.issues)

    def test_no_run_raises(self, tmp_path):
        doc = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rdml version="1.2" xmlns="http://www.rdml.org">'
            '<experiment id="exp"/></rdml>'
        )
        path = _write(tmp_path, doc)

        with pytest.raises(CSVValidationError):
            read_rdml(path)

    def test_wrong_root_element_raises(self, tmp_path):
        path = _write(
            tmp_path, '<?xml version="1.0"?><html><body>nope</body></html>'
        )

        with pytest.raises(ValueError, match="rdml"):
            read_rdml(path)

    def test_malformed_xml_raises(self, tmp_path):
        path = _write(tmp_path, "<not-xml-at-all")

        with pytest.raises(ValueError, match="valid XML"):
            read_rdml(path)

    def test_namespace_free_document(self, tmp_path):
        doc = _rdml_doc(_react("1", "S1", "T1", "20.0")).replace(
            ' xmlns="http://www.rdml.org"', ""
        )

        experiment = read_rdml(_write(tmp_path, doc))

        assert len(experiment.wells) == 1
        assert experiment.wells[0].cq == pytest.approx(20.0)

    def test_conflicting_group_annotations(self, tmp_path):
        samples = (
            "<sample id=\"S1\"><type>unkn</type>"
            '<annotation><property>group</property><value>a</value></annotation>'
            '<annotation><property>group</property><value>b</value></annotation>'
            "</sample>"
        )
        path = _write(
            tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0"), samples=samples)
        )

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "CONFLICTING_SAMPLE_GROUP" for i in excinfo.value.issues)


class TestPlateGeometry:
    def test_react_id_beyond_plate_falls_back_to_id(self, tmp_path):
        experiment = read_rdml(
            _write(tmp_path, _rdml_doc(_react("97", "S1", "T1", "20.0")))
        )

        assert experiment.wells[0].position == "97"

    def test_missing_pcr_format_falls_back_to_id(self, tmp_path):
        runs = f'<run id="r1">{_react("5", "S1", "T1", "20.0")}</run>'

        experiment = read_rdml(_write(tmp_path, _rdml_doc("", runs=runs)))

        assert experiment.wells[0].position == "5"

    def test_numeric_row_label_format(self, tmp_path):
        runs = (
            '<run id="r1"><pcrFormat><rows>8</rows><columns>12</columns>'
            "<rowLabel>123</rowLabel><columnLabel>123</columnLabel></pcrFormat>"
            f'{_react("1", "S1", "T1", "20.0")}</run>'
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc("", runs=runs)))

        assert experiment.wells[0].position == "R1C01"


class TestMultipleRuns:
    def test_run_id_prefix_keeps_positions_unique(self, tmp_path):
        runs = (
            '<run id="r1"><pcrFormat><rows>8</rows><columns>12</columns>'
            "<rowLabel>ABC</rowLabel><columnLabel>123</columnLabel></pcrFormat>"
            f'{_react("1", "S1", "T1", "20.0")}</run>'
            '<run id="r2"><pcrFormat><rows>8</rows><columns>12</columns>'
            "<rowLabel>ABC</rowLabel><columnLabel>123</columnLabel></pcrFormat>"
            f'{_react("1", "S1", "T1", "21.0")}</run>'
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc("", runs=runs)))

        positions = sorted(w.position for w in experiment.wells)
        assert positions == ["r1:A01", "r2:A01"]
        cq_by_position = {w.position: w.cq for w in experiment.wells}
        assert cq_by_position["r1:A01"] == pytest.approx(20.0)
        assert cq_by_position["r2:A01"] == pytest.approx(21.0)


class TestRawTraces:
    def test_adp_trace_parsed(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<adp><cyc>1</cyc><fluor>0.5</fluor></adp>"
            "<adp><cyc>2</cyc><fluor>0.7</fluor></adp>"
            "<adp><cyc>3</cyc><fluor>1.1</fluor></adp>"
            "</data></react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react)))

        assert experiment.has_traces
        trace = experiment.traces["A01"]
        assert trace.n_cycles == 3
        assert list(trace.cycles) == [1.0, 2.0, 3.0]
        assert list(trace.fluorescence) == [0.5, 0.7, 1.1]

    def test_invalid_trace_point_rejected(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<adp><cyc>1</cyc><fluor>abc</fluor></adp>"
            "</data></react>"
        )
        path = _write(tmp_path, _rdml_doc(react))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "INVALID_TRACE_POINT" for i in excinfo.value.issues)


class TestRdml13:
    """RDML 1.3 additions: per-target sample types, analysis outputs, dye."""

    def test_per_target_ntc_type(self, tmp_path):
        samples = (
            '<sample id="S1"><type>unkn</type>'
            '<type targetId="T1">ntc</type></sample>'
        )
        react = (
            '<react id="1"><sample id="S1"/>'
            '<data><tar id="T1"/><cq>20.0</cq></data>'
            '<data><tar id="T2"/><cq>22.0</cq></data>'
            "</react>"
        )

        experiment = read_rdml(
            _write(tmp_path, _rdml_doc(react, samples=samples))
        )

        by_target = {w.target_name: w for w in experiment.wells}
        assert by_target["T1"].cq_status == "no_template"
        assert by_target["T1"].is_ntc is True
        assert by_target["T2"].cq_status == "determined"
        assert by_target["T2"].is_ntc is False
        assert by_target["T1"].cq == pytest.approx(20.0)

    def test_default_sample_type_applies_to_other_targets(self, tmp_path):
        samples = (
            '<sample id="S1"><type>ntc</type><type targetId="T2">unkn</type></sample>'
        )
        react = (
            '<react id="1"><sample id="S1"/>'
            '<data><tar id="T1"/><cq>20.0</cq></data>'
            '<data><tar id="T2"/><cq>22.0</cq></data>'
            "</react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react, samples=samples)))

        by_target = {w.target_name: w for w in experiment.wells}
        assert by_target["T1"].cq_status == "no_template"
        assert by_target["T2"].cq_status == "determined"

    def test_analysis_outputs_with_correction_factors(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<N0>100.0</N0><ampEff>1.95</ampEff>"
            "<corrF>0.9</corrF><corrP>0.8</corrP><corrCq>21.2</corrCq>"
            "</data></react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react)))

        well = experiment.wells[0]
        assert well.n0 == pytest.approx(112.5)
        assert well.amp_eff == pytest.approx(0.95)
        assert well.corrected_cq == pytest.approx(21.2)

    def test_na_sentinels_in_analysis_outputs(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<N0>-1.0</N0><ampEff>2.0</ampEff><corrCq>-1.0</corrCq>"
            "</data></react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react)))

        well = experiment.wells[0]
        assert well.n0 is None
        assert well.amp_eff == pytest.approx(1.0)
        assert well.corrected_cq is None

    def test_nondefault_amp_eff_converted_to_fractional(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<ampEff>1.90</ampEff></data></react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react)))

        assert experiment.wells[0].amp_eff == pytest.approx(0.90)

    def test_amp_eff_absent_falls_back_to_target_efficiency(self, tmp_path):
        """When a well omits ``ampEff`` its efficiency comes from the declared
        target ``amplificationEfficiency`` instead of being dropped."""
        targets = (
            '<target id="T1"><type>toi</type>'
            "<amplificationEfficiency>1.90</amplificationEfficiency>"
            '<dyeId id="SYBR"/></target>'
            '<target id="T2"><type>toi</type><dyeId id="SYBR"/></target>'
        )
        react = (
            # No <ampEff> in the well's analysis outputs.
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<N0>5.0</N0></data></react>"
        )

        experiment = read_rdml(
            _write(tmp_path, _rdml_doc(react, targets=targets))
        )

        assert experiment.wells[0].amp_eff == pytest.approx(0.90)

    def test_amp_eff_absent_without_target_eff_is_unknown(self, tmp_path):
        """A well with neither per-well ampEff nor a target efficiency leaves
        the well's efficiency unknown (None) rather than discarding data."""
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "</data></react>"
        )

        experiment = read_rdml(_write(tmp_path, _rdml_doc(react)))

        assert experiment.wells[0].amp_eff is None

    def test_dye_chemistry_resolved_from_dye_master(self, tmp_path):
        doc = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rdml version="1.3" xmlns="http://www.rdml.org">'
            '<sample id="S1"><type>unkn</type></sample>'
            '<dye id="SYBR"><dyeChemistry>non-saturating DNA binding dye'
            "</dyeChemistry></dye>"
            '<dye id="FAM"><dyeChemistry>hydrolysis probe</dyeChemistry></dye>'
            '<target id="T1"><type>toi</type><dyeId id="SYBR"/></target>'
            '<target id="T2"><type>toi</type><dyeId id="FAM"/></target>'
            '<target id="T3"><type>toi</type></target>'
            '<experiment id="exp"><run id="r1"><pcrFormat>'
            "<rows>8</rows><columns>12</columns><rowLabel>ABC</rowLabel>"
            "<columnLabel>123</columnLabel></pcrFormat>"
            '<react id="1"><sample id="S1"/>'
            '<data><tar id="T1"/><cq>20.0</cq></data></react>'
            '<react id="2"><sample id="S1"/>'
            '<data><tar id="T2"/><cq>21.0</cq></data></react>'
            '<react id="3"><sample id="S1"/>'
            '<data><tar id="T3"/><cq>22.0</cq></data></react>'
            "</run></experiment></rdml>"
        )

        experiment = read_rdml(_write(tmp_path, doc))

        assert experiment.assays["T1"].dye_chemistry == (
            "non-saturating DNA binding dye"
        )
        assert experiment.assays["T2"].dye_chemistry == "hydrolysis probe"
        assert experiment.assays["T3"].dye_chemistry is None
        assert experiment.metadata["rdml_version"] == "1.3"

    def test_invalid_analysis_number_rejected(self, tmp_path):
        react = (
            '<react id="1"><sample id="S1"/><data><tar id="T1"/><cq>20.0</cq>'
            "<N0>abc</N0></data></react>"
        )
        path = _write(tmp_path, _rdml_doc(react))

        with pytest.raises(CSVValidationError) as excinfo:
            read_rdml(path)

        assert any(i.code == "INVALID_NUMERIC_FIELD" for i in excinfo.value.issues)


class TestLenientMode:
    def test_errors_recorded_not_raised(self, tmp_path):
        doc = _rdml_doc(
            _react("1", "GHOST", "T1", "20.0") + _react("2", "S1", "T1", "21.0")
        )
        path = _write(tmp_path, doc)

        experiment = read_rdml(path, strict=False)

        assert len(experiment.wells) == 1
        assert experiment.metadata["parse_mode"] == "lenient"
        codes = {d["code"] for d in experiment.metadata["input_diagnostics"]}
        assert "MISSING_SAMPLE_REF" in codes

    def test_experiment_name_override(self, tmp_path):
        experiment = read_rdml(
            _write(tmp_path, _rdml_doc(_react("1", "S1", "T1", "20.0"))),
            experiment_name="my-experiment",
        )

        assert experiment.name == "my-experiment"
