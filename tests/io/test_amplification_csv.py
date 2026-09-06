"""Tests for the long-format amplification-trace CSV reader.

These verify that the reader distinguishes "not a trace file" from "trace-like
but malformed", so callers may safely fall back to a tidy reader only when the
file truly carries no trace columns rather than silently reinterpreting
broken data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bioeng_qpcr.io import NotAmplificationFile, read_amplification_csv


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "traces.csv"
    path.write_text(text, encoding="utf-8")
    return path


class TestExceptionContract:
    def test_missing_trace_columns_raise_not_amplification(self, tmp_path):
        path = _write(tmp_path, "sample,target,cq\nS1,GAPDH,20.1\n")

        with pytest.raises(NotAmplificationFile):
            read_amplification_csv(path)

    def test_empty_file_raise_not_amplification(self, tmp_path):
        path = _write(tmp_path, "position,cycle,fluorescence\n")

        with pytest.raises(NotAmplificationFile):
            read_amplification_csv(path)

    def test_malformed_trace_raise_plain_value_error(self, tmp_path):
        # has trace columns but non-increasing cycles -> malformed, not "not a
        # trace file"
        path = _write(
            tmp_path,
            "position,cycle,fluorescence\n"
            "A01,1,0.5\n"
            "A01,1,0.9\n",
        )

        with pytest.raises(ValueError) as excinfo:
            read_amplification_csv(path)
        assert not isinstance(excinfo.value, NotAmplificationFile)

    def test_non_finite_fluorescence_raise_plain_value_error(self, tmp_path):
        path = _write(
            tmp_path,
            "position,cycle,fluorescence\n"
            "A01,1,0.5\n"
            "A01,2,inf\n",
        )

        with pytest.raises(ValueError) as excinfo:
            read_amplification_csv(path)
        assert not isinstance(excinfo.value, NotAmplificationFile)


class TestReading:
    def test_traces_and_wells_built_from_long_format(self, tmp_path):
        path = _write(
            tmp_path,
            "position,cycle,fluorescence,target_name,sample_name\n"
            "A01,1,0.5,GAPDH,S1\n"
            "A01,2,0.9,GAPDH,S1\n"
            "A02,1,1.2,IL6,S2\n"
            "A02,2,1.6,IL6,S2\n",
        )

        experiment = read_amplification_csv(path)

        assert set(experiment.traces) == {"A01", "A02"}
        assert experiment.traces["A02"].n_cycles == 2
        assert {w.target_name for w in experiment.wells} == {"GAPDH", "IL6"}
        assert experiment.assays["GAPDH"].is_reference is False

    def test_reference_targets_marked(self, tmp_path):
        path = _write(
            tmp_path,
            "position,cycle,fluorescence,target_name\n"
            "A01,1,0.5,GAPDH\n"
            "A01,2,0.9,GAPDH\n",
        )

        experiment = read_amplification_csv(path, reference_targets=["GAPDH"])

        assert experiment.assays["GAPDH"].is_reference is True