"""Tests for dose-response fitting, model selection and potency reporting."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.potency import (
    CurveDirection,
    DoseResponseDataset,
    DoseResponseModel,
    PotencyKind,
    compare_dose_response_models,
    fit_dose_response,
    fit_dose_response_batch,
    fit_dose_response_curve,
    relative_potency,
)

from .conftest import (
    TRUE_BOTTOM,
    TRUE_HILL,
    TRUE_LOG_IC50,
    TRUE_TOP,
    make_activation_dataset,
    make_asymmetric_dataset,
    make_inhibition_dataset,
)


class TestDatasetValidation:
    def test_mismatched_shapes_rejected(self):
        dataset = DoseResponseDataset(
            compound="bad",
            concentration=np.array([1.0, 2.0, 3.0]),
            response=np.array([1.0, 2.0]),
        )
        with pytest.raises(ValueError, match="same shape"):
            fit_dose_response_curve(dataset)

    def test_non_positive_concentration_rejected(self):
        dataset = DoseResponseDataset(
            compound="zero",
            concentration=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            response=np.array([10.0, 8.0, 6.0, 4.0, 2.0]),
        )
        with pytest.raises(ValueError, match="strictly positive"):
            fit_dose_response_curve(dataset)

    def test_too_few_points_rejected(self):
        dataset = DoseResponseDataset(
            compound="sparse",
            concentration=np.array([1.0, 2.0, 3.0]),
            response=np.array([10.0, 8.0, 6.0]),
        )
        with pytest.raises(ValueError, match="at least 5"):
            fit_dose_response_curve(dataset)

    def test_too_few_distinct_levels_rejected(self):
        dataset = DoseResponseDataset(
            compound="flat",
            concentration=np.array([1.0] * 8),
            response=np.linspace(10.0, 2.0, 8),
        )
        with pytest.raises(ValueError, match="3 distinct concentration levels"):
            fit_dose_response_curve(dataset)

    def test_invalid_confidence_rejected(self):
        dataset = make_inhibition_dataset()
        with pytest.raises(ValueError, match="confidence"):
            fit_dose_response_curve(dataset, confidence=1.5)


class TestParameterRecovery:
    def test_recovers_known_ic50_and_hill_from_clean_data(self):
        dataset = make_inhibition_dataset()
        fit = fit_dose_response_curve(dataset)
        assert fit.half_max_log_concentration == pytest.approx(TRUE_LOG_IC50, abs=0.01)
        assert fit.hill_slope == pytest.approx(abs(TRUE_HILL), abs=0.02)
        assert fit.bottom == pytest.approx(TRUE_BOTTOM, rel=0.02)
        assert fit.top == pytest.approx(TRUE_TOP, rel=0.02)

    def test_perfect_fit_on_noiseless_data(self):
        fit = fit_dose_response_curve(make_inhibition_dataset())
        assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
        assert fit.rmse < 1e-3
        assert fit.converged

    def test_recovers_ec50_of_an_ascending_curve(self):
        dataset = make_activation_dataset(log_ec50=float(np.log10(0.8)))
        result = fit_dose_response(dataset)
        assert result.kind is PotencyKind.EC50
        assert result.relative_value == pytest.approx(0.8, rel=0.02)

    def test_noisy_data_still_locate_the_potency(self):
        dataset = make_inhibition_dataset(noise=2.0)
        fit = fit_dose_response_curve(dataset)
        assert fit.half_max_log_concentration == pytest.approx(TRUE_LOG_IC50, abs=0.15)
        assert fit.r_squared > 0.98

    def test_three_pl_fit_uses_a_zero_lower_asymptote(self):
        dataset = make_inhibition_dataset(bottom=0.0)
        fit = fit_dose_response_curve(dataset, DoseResponseModel.THREE_PL)
        assert fit.bottom == 0.0
        assert "bottom" not in fit.parameters


class TestDirectionHandling:
    def test_direction_inferred_as_decreasing_for_an_inhibitor(self):
        fit = fit_dose_response_curve(make_inhibition_dataset())
        assert fit.direction is CurveDirection.DECREASING
        assert fit.parameters["hill"] < 0

    def test_direction_inferred_as_increasing_for_an_agonist(self):
        fit = fit_dose_response_curve(make_activation_dataset())
        assert fit.direction is CurveDirection.INCREASING
        assert fit.parameters["hill"] > 0

    def test_a_contradictory_direction_is_warned_about(self):
        # Forcing the wrong direction still converges (the symmetric model
        # simply swaps its asymptotes), so it must be warned about rather than
        # silently accepted.
        dataset = make_inhibition_dataset(direction=CurveDirection.INCREASING)
        fit = fit_dose_response_curve(dataset)
        assert fit.direction is CurveDirection.INCREASING
        assert fit.bottom > fit.top  # asymptotes labelled backwards
        assert any("contradicts the trend" in warning for warning in fit.warnings)

    def test_a_consistent_direction_is_not_warned_about(self):
        dataset = make_inhibition_dataset(direction=CurveDirection.DECREASING)
        fit = fit_dose_response_curve(dataset)
        assert not any("contradicts the trend" in warning for warning in fit.warnings)

    def test_kind_follows_direction_when_not_given(self):
        assert fit_dose_response(make_inhibition_dataset()).kind is PotencyKind.IC50
        assert fit_dose_response(make_activation_dataset()).kind is PotencyKind.EC50

    def test_kind_can_be_forced(self):
        result = fit_dose_response(make_inhibition_dataset(), kind=PotencyKind.EC50)
        assert result.kind is PotencyKind.EC50


class TestModelSelection:
    def test_symmetric_data_selects_a_symmetric_model(self):
        comparison = compare_dose_response_models(make_inhibition_dataset())
        assert comparison.best is not None
        assert comparison.best.model is DoseResponseModel.FOUR_PL
        assert not comparison.ambiguous

    def test_asymmetric_data_favours_the_five_pl(self):
        comparison = compare_dose_response_models(make_asymmetric_dataset())
        assert comparison.best is not None
        assert comparison.best.model is DoseResponseModel.FIVE_PL

    def test_ranking_is_sorted_by_aicc(self):
        comparison = compare_dose_response_models(make_asymmetric_dataset())
        aiccs = [fit.aicc for fit in comparison.fits]
        assert aiccs == sorted(aiccs)

    def test_single_candidate_is_allowed(self):
        comparison = compare_dose_response_models(
            make_inhibition_dataset(), [DoseResponseModel.FOUR_PL]
        )
        assert len(comparison.fits) == 1
        assert not comparison.ambiguous

    def test_empty_candidate_list_rejected(self):
        with pytest.raises(ValueError, match="At least one candidate"):
            compare_dose_response_models(make_inhibition_dataset(), [])

    def test_message_names_the_winner(self):
        comparison = compare_dose_response_models(make_inhibition_dataset())
        assert "4pl" in comparison.message

    def test_sparse_sample_falls_back_to_finite_aic(self):
        dataset = make_inhibition_dataset(
            concentrations=np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        )
        comparison = compare_dose_response_models(dataset)
        assert all(not np.isfinite(fit.aicc) for fit in comparison.fits)
        assert "ranked by AIC" in comparison.message
        assert "AICc=inf" not in comparison.message

    def test_mixed_finite_aicc_candidates_use_one_common_aic_scale(self):
        full = make_asymmetric_dataset()
        dataset = DoseResponseDataset(
            compound=full.compound,
            concentration=full.concentration[:7],
            response=full.response[:7],
        )
        comparison = compare_dose_response_models(dataset)
        aiccs = [fit.aicc for fit in comparison.fits]
        assert any(np.isfinite(value) for value in aiccs)
        assert any(not np.isfinite(value) for value in aiccs)
        assert comparison.best is not None
        assert comparison.best.model is DoseResponseModel.FIVE_PL
        assert "ranked by AIC" in comparison.message


class TestRelativeVersusAbsolutePotency:
    def test_they_coincide_when_the_lower_asymptote_is_zero(self):
        dataset = make_inhibition_dataset(bottom=0.0)
        result = fit_dose_response(dataset)
        assert result.absolute_value is not None
        assert result.absolute_value == pytest.approx(result.relative_value, rel=0.02)

    def test_they_differ_when_inhibition_is_incomplete(self):
        # The response only falls to 40 % of the control, so the concentration
        # giving 50 % *of control* is higher than the midpoint of the fitted
        # window.  Reporting the fitted midpoint as "the IC50" would
        # understate it.
        dataset = make_inhibition_dataset(bottom=40.0, top=100.0)
        result = fit_dose_response(dataset)
        assert result.absolute_value is not None
        assert result.absolute_value > result.relative_value
        assert result.control_response == pytest.approx(100.0, rel=0.02)

    def test_control_fraction_is_configurable(self):
        # On a descending curve a *higher* target response is reached at a
        # *lower* concentration, so asking for 80 % of control must give a
        # smaller value than asking for 50 %.
        dataset = make_inhibition_dataset(bottom=40.0, top=100.0)
        half = fit_dose_response(dataset, control_fraction=0.5)
        eighty = fit_dose_response(dataset, control_fraction=0.8)
        assert eighty.absolute_value < half.absolute_value

    def test_unreachable_target_is_reported_rather_than_invented(self):
        # Inhibition stops at 60 % of control, so the 50 %-of-control level is
        # outside the curve and must not be reported.
        dataset = make_inhibition_dataset(bottom=60.0, top=100.0)
        result = fit_dose_response(dataset)
        assert result.absolute_value is None
        assert any("Absolute" in warning for warning in result.warnings)

    def test_explicit_control_response_is_honoured(self):
        dataset = make_inhibition_dataset(bottom=0.0, top=100.0)
        result = fit_dose_response(dataset, control_response=100.0)
        assert result.control_response == pytest.approx(100.0)


class TestIntervals:
    def test_asymptotic_interval_brackets_the_estimate(self):
        fit = fit_dose_response_curve(make_inhibition_dataset(noise=2.0))
        assert fit.half_max_ci is not None
        low, high = fit.half_max_ci
        assert low < fit.half_max_log_concentration < high

    def test_interval_is_asymmetric_in_linear_space(self):
        # Because the fit is done in log space, the concentration interval
        # must not be symmetric about the point estimate.
        fit = fit_dose_response_curve(make_inhibition_dataset(noise=2.0))
        interval = fit.half_max_concentration_ci
        assert interval is not None
        value = fit.half_max_concentration
        assert not (value - interval[0]) == pytest.approx(interval[1] - value)

    def test_noiseless_data_give_a_vanishing_interval(self):
        fit = fit_dose_response_curve(make_inhibition_dataset())
        assert fit.half_max_ci is not None
        width = fit.half_max_ci[1] - fit.half_max_ci[0]
        assert width < 0.05

    def test_bootstrap_interval_is_reproducible_and_brackets(self):
        dataset = make_inhibition_dataset(noise=2.0, replicates=3)
        first = fit_dose_response(
            dataset, ci_method="bootstrap", n_bootstrap=60, seed=3
        )
        second = fit_dose_response(
            dataset, ci_method="bootstrap", n_bootstrap=60, seed=3
        )
        assert first.relative_ci == second.relative_ci
        assert first.relative_ci is not None
        low, high = first.relative_ci
        assert low < first.relative_value < high

    def test_unknown_ci_method_rejected(self):
        with pytest.raises(ValueError, match="ci_method"):
            fit_dose_response(make_inhibition_dataset(), ci_method="magic")

    def test_unbounded_linear_interval_is_omitted_without_overflow(self):
        dataset = DoseResponseDataset(
            compound="poorly identified plateau",
            concentration=np.array(
                [1e-10, 3e-10, 1e-9, 4e-9, 1e-8, 3e-8, 3e-7, 5e-6, 5e-5]
            ),
            response=np.array(
                [-0.20, -0.30, -0.17, -0.23, -0.19, -0.18, -0.70, -1.01, -1.05]
            ),
            direction=CurveDirection.DECREASING,
        )

        result = fit_dose_response(
            dataset,
            models=[DoseResponseModel.FOUR_PL],
            allow_extrapolation=False,
        )

        assert result.relative_ci is None
        assert any("numerically unbounded" in item for item in result.warnings)

    def test_bootstrap_handles_a_sparse_design_with_replicates(self):
        # Five dose levels are sparse for flexible curves, but within-dose
        # replicates let the stratified bootstrap estimate sampling variation
        # without dropping any dose level.
        dataset = make_inhibition_dataset(
            concentrations=np.array([0.01, 0.1, 1.0, 10.0, 100.0]),
            noise=6.0,
            replicates=3,
        )
        bootstrap = fit_dose_response(
            dataset, ci_method="bootstrap", n_bootstrap=80, seed=1
        )
        assert bootstrap.relative_ci is not None
        low, high = bootstrap.relative_ci
        assert low < bootstrap.relative_value < high
        assert high / low > 1.0


class TestExtrapolationFlag:
    def test_potency_outside_the_tested_range_is_flagged(self):
        # Drop the two lowest concentrations, so the true IC50 (3.2) can no
        # longer be measured and the fit has to extrapolate to reach it.
        dataset = make_inhibition_dataset(
            concentrations=np.array([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]),
            log_ic50=float(np.log10(0.05)),
        )
        result = fit_dose_response(dataset)
        assert result.extrapolated
        assert any(
            "outside the tested concentration range" in warning
            for warning in result.warnings
        )

    def test_inside_the_range_is_not_flagged(self):
        result = fit_dose_response(make_inhibition_dataset())
        assert not result.extrapolated

    def test_allow_extrapolation_false_confines_the_search(self):
        dataset = make_inhibition_dataset(
            concentrations=np.array([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]),
            log_ic50=float(np.log10(0.05)),
        )
        fit = fit_dose_response_curve(dataset, allow_extrapolation=False)
        tested_low = float(np.min(dataset.concentration))
        tested_high = float(np.max(dataset.concentration))
        assert tested_low <= fit.half_max_concentration <= tested_high


class TestMessageAndMetadata:
    def test_message_summarises_the_result(self):
        result = fit_dose_response(make_inhibition_dataset())
        assert "IC50" in result.message
        assert "CompoundA" in result.message
        assert "4pl" in result.message

    def test_fold_range_recorded(self):
        result = fit_dose_response(make_inhibition_dataset())
        assert result.fold_range_covered == pytest.approx(
            float(np.log10(300.0 / 0.01)), rel=1e-6
        )


class TestBatch:
    def test_returns_one_result_per_input(self):
        datasets = [
            make_inhibition_dataset(compound="A"),
            make_inhibition_dataset(compound="B", log_ic50=1.0),
        ]
        results = fit_dose_response_batch(datasets)
        assert [result.compound for result in results] == ["A", "B"]

    def test_unfittable_dataset_becomes_a_placeholder(self):
        good = make_inhibition_dataset(compound="good")
        bad = DoseResponseDataset(
            compound="bad",
            concentration=np.array([1.0, 2.0, 3.0]),
            response=np.array([1.0, 2.0, 3.0]),
        )
        results = fit_dose_response_batch([good, bad])
        assert results[0].best_fit.converged
        assert not results[1].best_fit.converged
        assert "could not be determined" in results[1].message


class TestRelativePotency:
    def _pair(self):
        strong = fit_dose_response(
            make_inhibition_dataset(compound="Strong", log_ic50=np.log10(1.0))
        )
        weak = fit_dose_response(
            make_inhibition_dataset(compound="Weak", log_ic50=np.log10(100.0))
        )
        return strong, weak

    def test_ratio_is_a_over_b(self):
        strong, weak = self._pair()
        comparison = relative_potency(strong, weak)
        assert comparison.ratio == pytest.approx(0.01, rel=0.05)
        assert comparison.fold_difference == pytest.approx(100.0, rel=0.05)

    def test_message_states_which_compound_is_more_potent(self):
        strong, weak = self._pair()
        comparison = relative_potency(strong, weak)
        assert "more" in comparison.message
        assert "Strong" in comparison.message

    def test_interval_brackets_the_ratio(self):
        strong, weak = self._pair()
        comparison = relative_potency(strong, weak)
        assert comparison.ci is not None
        assert comparison.ci[0] < comparison.ratio < comparison.ci[1]

    def test_mixing_potency_kinds_is_rejected(self):
        inhibition = fit_dose_response(make_inhibition_dataset())
        activation = fit_dose_response(make_activation_dataset())
        with pytest.raises(ValueError, match="Cannot compare"):
            relative_potency(inhibition, activation)

    def test_unknown_quantity_rejected(self):
        strong, weak = self._pair()
        with pytest.raises(ValueError, match="quantity"):
            relative_potency(strong, weak, quantity="bogus")

    def test_absolute_comparison_requires_absolute_values(self):
        incomplete = fit_dose_response(
            make_inhibition_dataset(compound="Partial", bottom=60.0)
        )
        other = fit_dose_response(make_inhibition_dataset(compound="Full"))
        with pytest.raises(ValueError, match="Absolute potency is unavailable"):
            relative_potency(incomplete, other, quantity="absolute")

    def test_confidence_override_cannot_relabel_existing_interval(self):
        strong, weak = self._pair()
        with pytest.raises(ValueError, match="cannot relabel"):
            relative_potency(strong, weak, confidence=0.99)

    def test_mismatched_input_confidence_is_rejected(self):
        strong = fit_dose_response(
            make_inhibition_dataset(compound="Strong"), confidence=0.90
        )
        weak = fit_dose_response(
            make_inhibition_dataset(compound="Weak"), confidence=0.95
        )
        with pytest.raises(ValueError, match="same confidence"):
            relative_potency(strong, weak)
