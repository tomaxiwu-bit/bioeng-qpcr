"""Tests for the dose-response CSV reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioeng_qpcr.io.dose_response_csv import (
    NotDoseResponseFile,
    read_dose_response_csv,
)
from bioeng_qpcr.potency import CurveDirection, fit_dose_response

#: Repository ``examples/`` directory, located relative to this file so the
#: suite keeps working wherever the checkout lives.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

_LONG = (
    "compound,concentration,response,replicate\n"
    "A,0.1,95,1\n"
    "A,1,80,1\n"
    "A,10,40,1\n"
    "A,100,8,1\n"
    "A,1000,4,1\n"
    "B,0.1,90,1\n"
    "B,1,88,1\n"
    "B,10,70,1\n"
    "B,100,30,1\n"
    "B,1000,6,1\n"
)

_MATRIX = (
    "concentration,CompoundA,CompoundB\n"
    "0.1,95,90\n"
    "1,80,88\n"
    "10,40,70\n"
    "100,8,30\n"
    "1000,4,6\n"
)


def _write(tmp_path, name: str, text: str, encoding: str = "utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


class TestLongLayout:
    def test_groups_by_compound(self, tmp_path):
        datasets = read_dose_response_csv(_write(tmp_path, "l.csv", _LONG))
        assert [dataset.compound for dataset in datasets] == ["A", "B"]
        assert all(len(dataset) == 5 for dataset in datasets)

    def test_values_are_parsed_as_numbers(self, tmp_path):
        dataset = read_dose_response_csv(_write(tmp_path, "l.csv", _LONG))[0]
        assert dataset.concentration.tolist() == [0.1, 1.0, 10.0, 100.0, 1000.0]
        assert dataset.response[0] == pytest.approx(95.0)

    def test_replicate_labels_are_kept(self, tmp_path):
        dataset = read_dose_response_csv(_write(tmp_path, "l.csv", _LONG))[0]
        assert dataset.replicate_group == ("1", "1", "1", "1", "1")

    def test_named_replicates_are_kept(self, tmp_path):
        text = (
            "compound,concentration,response,replicate\n"
            "A,1,90,r1\nA,1,92,r2\nA,1,88,r3\nA,10,50,r1\nA,10,52,r2\nA,10,49,r3\n"
        )
        dataset = read_dose_response_csv(_write(tmp_path, "r.csv", text))[0]
        assert dataset.replicate_group == ("r1", "r2", "r3", "r1", "r2", "r3")

    def test_compound_column_is_optional(self, tmp_path):
        text = "concentration,response\n1,90\n10,50\n100,10\n"
        datasets = read_dose_response_csv(_write(tmp_path, "n.csv", text))
        assert len(datasets) == 1
        assert datasets[0].compound == "n"

    def test_explicit_compound_name_wins(self, tmp_path):
        text = "concentration,response\n1,90\n10,50\n100,10\n"
        datasets = read_dose_response_csv(
            _write(tmp_path, "n.csv", text), compound_name="Vancomycin"
        )
        assert datasets[0].compound == "Vancomycin"

    def test_blank_rows_are_skipped(self, tmp_path):
        text = "compound,concentration,response\nA,1,90\n\nA,10,50\n"
        dataset = read_dose_response_csv(_write(tmp_path, "b.csv", text))[0]
        assert len(dataset) == 2

    def test_rows_with_a_missing_value_are_skipped(self, tmp_path):
        text = "compound,concentration,response\nA,1,90\nA,,\nA,10,50\n"
        dataset = read_dose_response_csv(_write(tmp_path, "s.csv", text))[0]
        assert len(dataset) == 2

    def test_non_numeric_value_reports_the_row(self, tmp_path):
        text = "compound,concentration,response\nA,1,oops\n"
        with pytest.raises(ValueError, match="row 2"):
            read_dose_response_csv(_write(tmp_path, "e.csv", text))


class TestMatrixLayout:
    def test_one_dataset_per_series_column(self, tmp_path):
        datasets = read_dose_response_csv(_write(tmp_path, "m.csv", _MATRIX))
        assert [dataset.compound for dataset in datasets] == [
            "CompoundA",
            "CompoundB",
        ]

    def test_shared_concentration_axis(self, tmp_path):
        datasets = read_dose_response_csv(_write(tmp_path, "m.csv", _MATRIX))
        for dataset in datasets:
            assert dataset.concentration.tolist() == [0.1, 1.0, 10.0, 100.0, 1000.0]

    def test_blank_cells_are_missing_not_zero(self, tmp_path):
        # A blank well is a missing observation; treating it as 0 would drag
        # the fitted curve towards the bottom asymptote.
        text = "concentration,CompoundA\n0.1,95\n1,\n10,40\n100,8\n1000,4\n"
        dataset = read_dose_response_csv(_write(tmp_path, "b.csv", text))[0]
        assert len(dataset) == 4
        assert 0.0 not in dataset.response

    def test_layout_can_be_forced(self, tmp_path):
        datasets = read_dose_response_csv(
            _write(tmp_path, "m.csv", _MATRIX), layout="matrix"
        )
        assert len(datasets) == 2

    def test_compound_named_column_is_not_treated_as_a_series(self, tmp_path):
        text = "concentration,compound,CompoundA\n1,A,90\n10,A,50\n"
        datasets = read_dose_response_csv(
            _write(tmp_path, "c.csv", text), layout="matrix"
        )
        assert [dataset.compound for dataset in datasets] == ["CompoundA"]


class TestNormalisation:
    def test_percent_of_control_rescales_the_top(self, tmp_path):
        dataset = read_dose_response_csv(
            _write(tmp_path, "l.csv", _LONG), normalize="percent_of_control"
        )[0]
        # The lowest concentration is the control and becomes 100.
        assert dataset.response[0] == pytest.approx(100.0)
        assert dataset.response[-1] < 10.0

    def test_none_leaves_the_response_untouched(self, tmp_path):
        dataset = read_dose_response_csv(_write(tmp_path, "l.csv", _LONG))[0]
        assert dataset.response[0] == pytest.approx(95.0)

    def test_zero_control_cannot_be_normalised(self, tmp_path):
        text = "compound,concentration,response\nA,1,0\nA,10,0\nA,100,0\n"
        with pytest.raises(ValueError, match="mean response at the lowest"):
            read_dose_response_csv(
                _write(tmp_path, "z.csv", text), normalize="percent_of_control"
            )

    def test_unknown_normalisation_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="normalize"):
            read_dose_response_csv(
                _write(tmp_path, "l.csv", _LONG), normalize="fraction"
            )


class TestErrors:
    def test_unrelated_file_is_rejected(self, tmp_path):
        with pytest.raises(NotDoseResponseFile, match="not a dose-response table"):
            read_dose_response_csv(_write(tmp_path, "x.csv", "foo,bar\n1,2\n"))

    def test_unknown_layout_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="layout"):
            read_dose_response_csv(_write(tmp_path, "l.csv", _LONG), layout="diagonal")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_dose_response_csv(tmp_path / "nope.csv")

    def test_direction_hint_is_stored(self, tmp_path):
        datasets = read_dose_response_csv(
            _write(tmp_path, "l.csv", _LONG), direction=CurveDirection.DECREASING
        )
        assert all(
            dataset.direction is CurveDirection.DECREASING for dataset in datasets
        )


class TestGeneratedExamples:
    def test_long_example_fits_the_expected_potencies(self):
        datasets = read_dose_response_csv(EXAMPLES / "dose_response_example.csv")
        assert [dataset.compound for dataset in datasets] == [
            "CompoundA",
            "CompoundB",
        ]
        results = {
            result.compound: result for result in map(fit_dose_response, datasets)
        }
        # The generator used IC50 = 3.2 uM and 45 uM.  The shipped example has
        # to demonstrate a clean recovery, so the tolerances are tight.
        assert results["CompoundA"].relative_value == pytest.approx(3.2, rel=0.01)
        assert results["CompoundB"].relative_value == pytest.approx(45.0, rel=0.01)
        for result in results.values():
            assert not result.extrapolated
            assert result.best_fit.r_squared > 0.99

    def test_matrix_example_agrees_with_the_long_example(self):
        long_datasets = {
            dataset.compound: dataset
            for dataset in read_dose_response_csv(
                EXAMPLES / "dose_response_example.csv"
            )
        }
        matrix_datasets = read_dose_response_csv(
            EXAMPLES / "dose_response_matrix_example.csv"
        )
        assert set(long_datasets) == {dataset.compound for dataset in matrix_datasets}
        for dataset in matrix_datasets:
            reference = long_datasets[dataset.compound]
            # The matrix holds one row per level, the long file one row per
            # replicate, so the axes match as *level sets*, not as vectors.
            assert np.array_equal(
                np.unique(dataset.concentration),
                np.unique(reference.concentration),
            )
            long_means = [
                float(np.mean(reference.response[reference.concentration == level]))
                for level in np.unique(reference.concentration)
            ]
            order = np.argsort(dataset.concentration)
            assert dataset.response[order] == pytest.approx(long_means, abs=0.01)
