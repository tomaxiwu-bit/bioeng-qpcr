"""Tests for inhibition type classification."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.enzyme_kinetics.inhibition import (
    InhibitionDataset,
    InhibitionResult,
    InhibitionType,
    classify_inhibition,
)

from .conftest import (
    make_competitive_inhibition_data,
    make_mixed_inhibition_data,
    make_mm_data,
    make_noncompetitive_inhibition_data,
    make_uncompetitive_inhibition_data,
)


class TestClassifyCompetitive:
    def test_competitive_detected(self):
        datasets = make_competitive_inhibition_data()
        result = classify_inhibition(datasets)
        assert isinstance(result, InhibitionResult)
        assert result.inhibition_type is InhibitionType.COMPETITIVE
        assert result.confidence == "high"
        assert "Competitive" in result.message

    def test_km_increases_with_inhibitor(self):
        datasets = make_competitive_inhibition_data()
        result = classify_inhibition(datasets)
        km_apps = [d.mm_result.km for d in result.datasets]
        assert km_apps[-1] > km_apps[0]

    def test_vmax_unchanged(self):
        datasets = make_competitive_inhibition_data()
        result = classify_inhibition(datasets)
        assert result.vmax_cv < 0.05


class TestClassifyNonCompetitive:
    def test_noncompetitive_detected(self):
        datasets = make_noncompetitive_inhibition_data()
        result = classify_inhibition(datasets)
        assert result.inhibition_type is InhibitionType.NON_COMPETITIVE
        assert result.confidence == "high"

    def test_vmax_decreases_with_inhibitor(self):
        datasets = make_noncompetitive_inhibition_data()
        result = classify_inhibition(datasets)
        vmax_apps = [d.mm_result.vmax for d in result.datasets]
        assert vmax_apps[-1] < vmax_apps[0]

    def test_km_unchanged(self):
        datasets = make_noncompetitive_inhibition_data()
        result = classify_inhibition(datasets)
        assert result.km_cv < 0.05


class TestClassifyUncompetitive:
    def test_uncompetitive_detected(self):
        datasets = make_uncompetitive_inhibition_data()
        result = classify_inhibition(datasets)
        assert result.inhibition_type is InhibitionType.UNCOMPETITIVE

    def test_both_decrease(self):
        datasets = make_uncompetitive_inhibition_data()
        result = classify_inhibition(datasets)
        km_apps = [d.mm_result.km for d in result.datasets]
        vmax_apps = [d.mm_result.vmax for d in result.datasets]
        assert km_apps[-1] < km_apps[0]
        assert vmax_apps[-1] < vmax_apps[0]


class TestClassifyMixed:
    def test_mixed_detected(self):
        datasets = make_mixed_inhibition_data()
        result = classify_inhibition(datasets)
        assert result.inhibition_type is InhibitionType.MIXED


class TestClassifyNoInhibition:
    def test_none_detected(self):
        """Two identical datasets with no inhibitor → no inhibition."""
        s, v = make_mm_data(vmax=100.0, km=10.0)
        result = classify_inhibition([(0.0, s, v), (5.0, s, v)])
        assert result.inhibition_type is InhibitionType.NONE


class TestClassifyEdgeCases:
    def test_single_dataset_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            classify_inhibition([(0.0, [1, 2, 3, 4], [1, 2, 3, 4])])

    def test_datasets_sorted_by_inhibitor(self):
        datasets = make_competitive_inhibition_data()
        result = classify_inhibition(list(reversed(datasets)))
        concs = [d.inhibitor_concentration for d in result.datasets]
        assert concs == sorted(concs)

    def test_each_dataset_has_mm_and_lb(self):
        datasets = make_competitive_inhibition_data()
        result = classify_inhibition(datasets)
        for ds in result.datasets:
            assert isinstance(ds, InhibitionDataset)
            assert ds.mm_result is not None
            assert ds.lb_result is not None

    def test_decreasing_km_with_stable_vmax_is_not_competitive(self):
        """The competitive label requires Km to increase with inhibitor."""
        substrate = np.geomspace(1.0, 100.0, 12)
        datasets = []
        for inhibitor, km in ((0.0, 20.0), (5.0, 12.0), (10.0, 7.0)):
            velocity = 100.0 * substrate / (km + substrate)
            datasets.append((inhibitor, substrate, velocity))

        result = classify_inhibition(datasets)

        assert result.inhibition_type is not InhibitionType.COMPETITIVE
        assert result.confidence == "low"

    def test_increasing_vmax_with_stable_km_is_not_noncompetitive(self):
        """The noncompetitive label requires Vmax to decrease."""
        substrate = np.geomspace(1.0, 100.0, 12)
        datasets = []
        for inhibitor, vmax in ((0.0, 80.0), (5.0, 100.0), (10.0, 125.0)):
            velocity = vmax * substrate / (10.0 + substrate)
            datasets.append((inhibitor, substrate, velocity))

        result = classify_inhibition(datasets)

        assert result.inhibition_type is not InhibitionType.NON_COMPETITIVE
        assert result.confidence == "low"
