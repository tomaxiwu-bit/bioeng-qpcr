"""Tests for inhibition classification and global model selection."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.enzyme_kinetics.inhibition import (
    GlobalInhibitionFit,
    InhibitionDataset,
    InhibitionModelComparison,
    InhibitionResult,
    InhibitionType,
    classify_inhibition,
    compare_inhibition_models,
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


# ---------------------------------------------------------------------------
# Global model selection (compare_inhibition_models)
# ---------------------------------------------------------------------------


class TestCompareInhibitionModels:
    """Structural invariants of the AICc model comparison."""

    def test_returns_comparison_object(self):
        comparison = compare_inhibition_models(make_competitive_inhibition_data())

        assert isinstance(comparison, InhibitionModelComparison)
        assert isinstance(comparison.best_fit, GlobalInhibitionFit)
        assert comparison.n_points == 32  # 4 datasets x 8 points
        assert comparison.best_type.value in comparison.message

    def test_all_five_models_present(self):
        comparison = compare_inhibition_models(make_competitive_inhibition_data())

        assert len(comparison.models) == 5
        assert {m.inhibition_type for m in comparison.models} == {
            InhibitionType.COMPETITIVE,
            InhibitionType.NON_COMPETITIVE,
            InhibitionType.UNCOMPETITIVE,
            InhibitionType.MIXED,
            InhibitionType.NONE,
        }

    def test_models_sorted_by_aicc(self):
        comparison = compare_inhibition_models(make_competitive_inhibition_data())

        aiccs = [m.aicc for m in comparison.models]
        assert aiccs == sorted(aiccs)

    def test_best_model_is_first_with_zero_delta(self):
        comparison = compare_inhibition_models(make_competitive_inhibition_data())

        best = comparison.models[0]
        assert best is comparison.best_fit
        assert best.delta_aicc == pytest.approx(0.0)

    def test_akaike_weights_sum_to_one(self):
        comparison = compare_inhibition_models(make_competitive_inhibition_data())

        assert sum(m.akaike_weight for m in comparison.models) == pytest.approx(1.0)

    def test_parameter_std_keys_match_parameters(self):
        comparison = compare_inhibition_models(make_competitive_inhibition_data())

        for model in comparison.models:
            assert set(model.parameter_std) == set(model.parameters)


class TestGlobalModelSelection:
    """The correct mechanism should win the AICc comparison."""

    @pytest.mark.parametrize(
        ("make_data", "expected"),
        [
            (make_competitive_inhibition_data, InhibitionType.COMPETITIVE),
            (make_noncompetitive_inhibition_data, InhibitionType.NON_COMPETITIVE),
            (make_uncompetitive_inhibition_data, InhibitionType.UNCOMPETITIVE),
            (make_mixed_inhibition_data, InhibitionType.MIXED),
        ],
    )
    def test_mechanism_identified(self, make_data, expected):
        comparison = compare_inhibition_models(make_data())

        assert comparison.best_type is expected
        assert comparison.best_fit.converged
        assert comparison.best_fit.akaike_weight > 0.5

    def test_no_inhibition_identified(self):
        """Identical curves at two inhibitor levels → NONE wins by parsimony."""
        s, v = make_mm_data(vmax=100.0, km=10.0, noise_level=0.02)

        comparison = compare_inhibition_models([(0.0, s, v), (5.0, s, v)])

        assert comparison.best_type is InhibitionType.NONE


class TestGlobalParameterRecovery:
    """Global fits should recover the generating parameters."""

    def test_competitive_parameters_recovered(self):
        comparison = compare_inhibition_models(
            make_competitive_inhibition_data(vmax=100.0, km=10.0, ki=5.0)
        )
        params = comparison.best_fit.parameters

        assert params["vmax"] == pytest.approx(100.0, rel=0.05)
        assert params["km"] == pytest.approx(10.0, rel=0.05)
        assert params["ki"] == pytest.approx(5.0, rel=0.15)

    def test_noncompetitive_parameters_recovered(self):
        comparison = compare_inhibition_models(
            make_noncompetitive_inhibition_data(vmax=100.0, km=10.0, ki=5.0)
        )
        params = comparison.best_fit.parameters

        assert params["vmax"] == pytest.approx(100.0, rel=0.05)
        assert params["km"] == pytest.approx(10.0, rel=0.05)
        assert params["ki"] == pytest.approx(5.0, rel=0.15)

    def test_uncompetitive_parameters_recovered(self):
        comparison = compare_inhibition_models(
            make_uncompetitive_inhibition_data(vmax=100.0, km=10.0, ki=5.0)
        )
        params = comparison.best_fit.parameters

        assert params["vmax"] == pytest.approx(100.0, rel=0.05)
        assert params["km"] == pytest.approx(10.0, rel=0.05)
        assert params["ki"] == pytest.approx(5.0, rel=0.15)

    def test_mixed_parameters_recovered(self):
        comparison = compare_inhibition_models(
            make_mixed_inhibition_data(vmax=100.0, km=10.0, ki=5.0, ki_prime=10.0)
        )
        params = comparison.best_fit.parameters

        assert params["vmax"] == pytest.approx(100.0, rel=0.05)
        assert params["km"] == pytest.approx(10.0, rel=0.05)
        assert params["kic"] == pytest.approx(5.0, rel=0.2)
        assert params["kiu"] == pytest.approx(10.0, rel=0.2)


class TestCompareInhibitionModelsEdgeCases:
    def test_single_dataset_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compare_inhibition_models([(0.0, [1, 2, 3], [1, 2, 3])])

    def test_same_inhibitor_concentration_raises(self):
        s, v = make_mm_data()
        with pytest.raises(ValueError, match="distinct"):
            compare_inhibition_models([(5.0, s, v), (5.0, s, v)])

    def test_all_zero_inhibitors_raises(self):
        s, v = make_mm_data()
        with pytest.raises(ValueError, match="zero"):
            compare_inhibition_models([(0.0, s, v), (0.0, s, v)])
