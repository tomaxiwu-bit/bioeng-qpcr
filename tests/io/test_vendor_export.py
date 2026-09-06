"""Tests for the commercial instrument export readers."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from bioeng_qpcr.io import read_vendor_export


def _save_workbook(
    tmp_path: Path, filename: str, sheets: dict[str, list[list]]
) -> Path:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        ws = workbook.create_sheet(name)
        for row in rows:
            ws.append(row)
    path = tmp_path / filename
    workbook.save(path)
    return path


class TestQuantStudio:
    def _wb(self, tmp_path: Path) -> Path:
        return _save_workbook(
            tmp_path,
            "qs_results.xlsx",
            {
                "Results": [
                    ["Experiment Name", "myexp"],
                    ["Instrument Type", "QuantStudio 7 Flex"],
                    [],
                    [
                        "Well",
                        "Well Position",
                        "Sample Name",
                        "Target Name",
                        "Task",
                        "CT",
                        "Omit",
                    ],
                    [1, "A1", "S1", "GAPDH", "UNKNOWN", "18.5", "0"],
                    [2, "A2", "S1", "IL6", "UNKNOWN", "21.2", "0"],
                    [3, "A3", "S2", "IL6", "NTC", "Undetermined", "0"],
                    [4, "A4", "S3", "IL6", "UNKNOWN", "20.1", "1"],
                ],
                "Sample Setup": [["A1", "S1"]],
            },
        )

    def test_parses_results_with_metadata_block(self, tmp_path):
        experiment = read_vendor_export(
            self._wb(tmp_path), reference_targets=["GAPDH"]
        )

        assert experiment.metadata["vendor"] == "ThermoQuantStudio"
        assert len(experiment.wells) == 3  # the omitted row is skipped
        by_position = {w.position: w for w in experiment.wells}
        assert by_position["A01"].cq == pytest.approx(18.5)
        assert by_position["A01"].sample_name == "S1"
        assert by_position["A01"].target_name == "GAPDH"
        assert by_position["A02"].cq == pytest.approx(21.2)

    def test_ntc_task_classified_no_template(self, tmp_path):
        experiment = read_vendor_export(self._wb(tmp_path))

        by_pos = {w.position: w for w in experiment.wells}
        ntc = by_pos["A03"]
        assert ntc.cq_status == "no_template"
        assert ntc.is_ntc is True

    def test_undetermined_ct(self, tmp_path):
        experiment = read_vendor_export(self._wb(tmp_path))

        by_pos = {w.position: w for w in experiment.wells}
        assert by_pos["A03"].cq is None

    def test_reference_targets_marked(self, tmp_path):
        experiment = read_vendor_export(
            self._wb(tmp_path), reference_targets=["IL6"]
        )

        assert experiment.assays["IL6"].is_reference is True
        assert experiment.assays["GAPDH"].is_reference is False

    def test_missing_results_sheet_rejected(self, tmp_path):
        path = _save_workbook(tmp_path, "no_results.xlsx", {"Other": [["x"]]})

        with pytest.raises(ValueError, match="Results"):
            read_vendor_export(path)

    def test_multiplex_wells_uniquified(self, tmp_path):
        path = _save_workbook(
            tmp_path,
            "qs_multiplex.xlsx",
            {
                "Results": [
                    [
                        "Well",
                        "Well Position",
                        "Sample Name",
                        "Target Name",
                        "Task",
                        "CT",
                    ],
                    [1, "A1", "S1", "GAPDH", "UNKNOWN", "18.5"],
                    [2, "A1", "S1", "IL6", "UNKNOWN", "21.2"],
                    [3, "A2", "S2", "GAPDH", "UNKNOWN", "19.0"],
                ],
            },
        )

        experiment = read_vendor_export(path)
        positions = {w.position for w in experiment.wells}
        assert len(positions) == len(experiment.wells)  # no duplicate positions
        by_target = {
            w.target_name: w for w in experiment.wells if w.sample_name == "S1"
        }
        assert by_target["GAPDH"].position == "A01:GAPDH"
        assert by_target["IL6"].position == "A01:IL6"
        assert by_target["IL6"].cq == pytest.approx(21.2)


class TestBioRadCfx:
    def _wb(self, tmp_path: Path) -> Path:
        return _save_workbook(
            tmp_path,
            "cfx_results.xlsx",
            {
                "0": [
                    [None, "Well", "Fluor", "Target", "Content", "Sample", "Cq", "SQ"],
                    [None, "A01", "SYBR", "GAPDH", "Unkn", "S1", "18.5", "1000"],
                    [None, "A01", "FAM", "IL6", "Unkn", "S1", "21.1", None],
                    [None, "B01", "SYBR", "GAPDH", "Neg Ctrl", "NTC", None, None],
                ],
                "Run Information": [
                    ["File Name", "test.pcrd"],
                    ["Run Started", "2026-01-01 10:00:00"],
                    ["CFX Maestro Version", "5.2"],
                ],
            },
        )

    def test_parses_result_sheet(self, tmp_path):
        experiment = read_vendor_export(
            self._wb(tmp_path), reference_targets=["GAPDH"]
        )

        assert experiment.metadata["vendor"] == "BioRadCFX"
        assert len(experiment.wells) == 3
        s1_wells = {w.target_name: w for w in experiment.wells if w.sample_name == "S1"}
        gapdh = s1_wells["GAPDH"]
        assert gapdh.cq == pytest.approx(18.5)
        assert gapdh.n0 == pytest.approx(1000)
        # multiplex: A01 measures both GAPDH and IL6 -> unique positions
        il6 = s1_wells["IL6"]
        assert il6.position.startswith("A01:IL6")
        assert il6.cq == pytest.approx(21.1)

    def test_negative_control_marked_ntc(self, tmp_path):
        experiment = read_vendor_export(self._wb(tmp_path))

        ntc = [w for w in experiment.wells if w.is_ntc]
        assert len(ntc) == 1
        assert ntc[0].cq_status == "no_template"

    def test_run_information_captured(self, tmp_path):
        experiment = read_vendor_export(self._wb(tmp_path))

        assert (
            experiment.metadata["run_information"]["CFX Maestro Version"] == "5.2"
        )

    def test_no_result_sheet_rejected(self, tmp_path):
        path = _save_workbook(tmp_path, "no_cfx.xlsx", {"Meta": [["cfx", "q"]]})

        with pytest.raises(ValueError, match="unrecognised Excel vendor"):
            read_vendor_export(path)

    def test_fluor_captured_as_dye_chemistry(self, tmp_path):
        experiment = read_vendor_export(self._wb(tmp_path))

        assert experiment.assays["GAPDH"].dye_chemistry == "SYBR"
        assert experiment.assays["IL6"].dye_chemistry == "FAM"


class TestLightCycler480:
    def _txt(self, tmp_path: Path) -> Path:
        path = tmp_path / "lc480_raw.txt"
        path.write_text(
            "Sample position\tSample name\tProgram number\tSegment number\t"
            "Cycle number\tAcquisition time\tAcquisition temperature\t"
            "Fluorescence data\n"
            "A01\tSample1\t0\t0\t1\tn/a\t25.0\t0.5\n"
            "A01\tSample1\t0\t0\t2\tn/a\t25.0\t0.9\n"
            "A01\tSample1\t0\t0\t3\tn/a\t25.0\t1.8\n"
            "A02\tSample2\t0\t0\t1\tn/a\t25.0\t1.2\n"
            "A02\tSample2\t0\t0\t2\tn/a\t25.0\t1.6\n",
            encoding="utf-8",
        )
        return path

    def test_raw_traces_built(self, tmp_path):
        experiment = read_vendor_export(self._txt(tmp_path))

        assert experiment.metadata["vendor"] == "LightCycler480"
        assert experiment.has_traces
        assert set(experiment.traces) == {"A01", "A02"}
        assert experiment.traces["A01"].n_cycles == 3
        assert experiment.wells[0].sample_name == "Sample1"

    def test_trace_fluorescence_values(self, tmp_path):
        experiment = read_vendor_export(self._txt(tmp_path))

        assert list(experiment.traces["A02"].fluorescence) == [1.2, 1.6]

    def test_multi_channel_export_rejected(self, tmp_path):
        path = tmp_path / "lc480_dualdye.txt"
        path.write_text(
            "Sample position\tCycle number\tFluorescence data\tFluorescence data 2\n"
            "A01\t1\t0.5\t1.8\n"
            "A01\t2\t0.9\t2.1\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="multi-channel"):
            read_vendor_export(path)

    def test_latin1_encoding_tolerated(self, tmp_path):
        path = tmp_path / "lc480_latin1.txt"
        content = (
            "Sample position\tSample name\tCycle number\tFluorescence data\n"
            "A01\tM\xe9ment\t1\t0.5\n"
            "A01\tM\xe9ment\t2\t0.9\n"
        )
        path.write_bytes(content.encode("latin-1"))

        experiment = read_vendor_export(path)
        assert experiment.traces["A01"].n_cycles == 2


class TestDispatcher:
    def test_legacy_xls_rejected(self, tmp_path):
        path = tmp_path / "old.xls"
        path.write_bytes(b"\xd0\xcf\x11\xe0")

        with pytest.raises(ValueError, match="xls"):
            read_vendor_export(path)

    def test_unrecognised_text_rejected(self, tmp_path):
        path = tmp_path / "mystery.txt"
        path.write_text("alpha\tbeta\n1\t2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unrecognised vendor"):
            read_vendor_export(path)