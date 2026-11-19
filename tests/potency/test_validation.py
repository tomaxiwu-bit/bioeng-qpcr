"""Tests for the potency validation rules."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bioeng_qpcr.potency import (
    DoseResponseDataset,
    DoseResponseModel,
    fit_dose_response,
    fit_dose_response_curve,
    run_potency_validation,
)
from bioeng_qpcr.potency.validation import (
    validate_assay_window,
    validate_ci_width,
    validate_concentration_levels,
    validate_concentration_span,
    validate_fit_quality,
    validate_hill_slope,
    validate_model_adequacy,
    validate_monotonicity,
    validate_plateau_support,
    validate_replicate_precision,
    validate_within_tested_range,
)

from .conftest import (
    make_asymmetric_dataset,
    make_hook_effect_dataset,
    make_inhibition_dataset,
)

#: Concentrations whose upper end never approaches the lower asymptote, so the
#: fitted bottom is pure extrapolation.
_BEYOND_RANGE = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])


def _by_id(results):
    return {result.rule_id: result for result in results}


class TestWithinTestedRange:
    def test_passes_for_a_resolved_curve(self):
        dataset = make_inhibition_dataset()
        result = fit_dose_response(dataset)
        check = validate_within_tested_range(result, dataset)
        assert check.passed
        assert check.rule_id == "POT-001"

    def test_fails_when_the_potency_is_extrapolated(self):
        dataset = make_inhibition_dataset(
            concentrations=_BEYOND_RANGE, log_ic50=float(np.log10(1000.0))
        )
        result = fit_dose_response(dataset)
        check = validate_within_tested_range(result, dataset)
        assert not check.passed
        assert check.severity == "failure"
        assert "outside the tested range" in check.message


class TestAssayWindow:
    def test_passes_for_a_full_window(self):
        dataset = make_inhibition_dataset()
        result = fit_dose_response(dataset)
        assert validate_assay_window(result, dataset).passed

    def test_fails_when_the_window_is_negligible(self):
        # A 3 % change in signal cannot define a 50 % level.
        dataset = make_inhibition_dataset(bottom=97.0, top=100.0, noise=0.4)
        result = fit_dose_response(dataset)
        check = validate_assay_window(result, dataset)
        assert not check.passed
        assert "assay window" in check.message.lower()


class TestDesignRules:
    def test_enough_levels_pass(self):
        dataset = make_inhibition_dataset()
        assert validate_concentration_levels(dataset).passed

    def test_too_few_levels_fail(self):
        dataset = make_inhibition_dataset(
            concentrations=np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        )
        check = validate_concentration_levels(dataset)
        assert not check.passed
        assert "sparse" in check.message


class TestHillSlopeRule:
    def test_plausible_slope_passes(self):
        dataset = make_inhibition_dataset()
        result = fit_dose_response(dataset)
        assert validate_hill_slope(result).passed

    def test_shallow_slope_fails(self):
        dataset = make_inhibition_dataset(hill=-0.15)
        result = fit_dose_response(dataset)
        check = validate_hill_slope(result)
        assert not check.passed
        assert "outside" in check.message


class TestFitQualityRule:
    def test_good_fit_passes(self):
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        assert validate_fit_quality(result).passed

    def test_threshold_is_configurable(self):
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        assert not validate_fit_quality(result, min_r_squared=0.999999).passed


class TestConcentrationSpanRule:
    def test_wide_span_passes(self):
        dataset = make_inhibition_dataset()
        assert validate_concentration_span(dataset).passed

    def test_narrow_span_fails(self):
        dataset = make_inhibition_dataset(
            concentrations=np.array([1.0, 1.6, 2.5, 4.0, 6.3, 10.0])
        )
        check = validate_concentration_span(dataset)
        assert not check.passed
        assert "decades" in check.message


class TestIntervalWidthRule:
    def test_tight_interval_passes(self):
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        assert validate_ci_width(result).passed

    def test_missing_interval_warns_rather_than_fails(self):
        # Strip the interval from a real fit rather than trying to engineer a
        # dataset whose covariance matrix happens to be singular.
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        without_interval = replace(
            result,
            best_fit=replace(result.best_fit, half_max_ci=None),
        )
        check = validate_ci_width(without_interval)
        assert not check.passed
        assert check.severity == "warning"
        assert "No confidence interval" in check.message


class TestPlateauSupportRule:
    def test_both_plateaus_reached_passes(self):
        dataset = make_inhibition_dataset()
        result = fit_dose_response(dataset)
        assert validate_plateau_support(result, dataset).passed

    def test_unreached_lower_plateau_fails(self):
        # The curve is still near the top at the highest dose, so the lower
        # asymptote is an extrapolation, not an observation.
        dataset = make_inhibition_dataset(
            concentrations=_BEYOND_RANGE, log_ic50=float(np.log10(1000.0))
        )
        result = fit_dose_response(dataset)
        check = validate_plateau_support(result, dataset)
        assert not check.passed
        assert "Plateau not reached" in check.message


class TestModelAdequacyRule:
    def test_adequate_model_passes(self):
        dataset = make_inhibition_dataset(noise=1.5)
        result = fit_dose_response_curve(dataset)
        check = validate_model_adequacy(result, dataset)
        assert check.passed
        assert check.evidence["delta_aicc"] <= 4.0

    def test_asymmetry_forced_into_a_four_pl_is_caught_by_aicc(self):
        # This is the high-power version of the symmetry check: the 5PL nests
        # the 4PL, so an inadequate fit shows up as a large AICc deficit even
        # when the residual sign pattern is too subtle for a runs test.
        dataset = make_asymmetric_dataset()
        result = fit_dose_response_curve(dataset)  # explicitly a 4PL
        check = validate_model_adequacy(result, dataset)
        assert not check.passed
        assert check.evidence["delta_aicc"] > 4.0
        assert "5PL fits substantially better" in check.message

    def test_hook_effect_is_left_to_the_monotonicity_rule(self):
        # Neither sigmoid model fits a hook, so the AICc comparison raises no
        # objection; POT-011 catches it instead.  Keeping the split explicit
        # documents which rule is responsible for which failure mode.
        dataset = make_hook_effect_dataset()
        result = fit_dose_response_curve(dataset)
        assert validate_model_adequacy(result, dataset).passed
        assert not validate_monotonicity(result, dataset).passed

    def test_already_flexible_model_is_not_penalised_for_being_the_best(self):
        # When the fit already uses the 5PL there is nothing more flexible to
        # compare against, and the rule must not fail merely for that reason.
        dataset = make_asymmetric_dataset()
        result = fit_dose_response_curve(dataset, DoseResponseModel.FIVE_PL)
        check = validate_model_adequacy(result, dataset)
        assert check.passed
        assert check.evidence["delta_aicc"] == 0.0

    def test_threshold_is_configurable(self):
        # The asymmetric data give a large positive ΔAICc; raising the
        # threshold above it must silence the rule.
        dataset = make_asymmetric_dataset()
        result = fit_dose_response_curve(dataset)
        assert not validate_model_adequacy(result, dataset).passed
        assert validate_model_adequacy(result, dataset, max_delta_aicc=1e9).passed


class TestMonotonicityRule:
    def test_monotone_series_passes(self):
        dataset = make_inhibition_dataset(noise=1.5)
        result = fit_dose_response_curve(dataset)
        check = validate_monotonicity(result, dataset)
        assert check.passed
        assert check.evidence["reversals"] == []

    def test_hook_effect_fails(self):
        dataset = make_hook_effect_dataset()
        result = fit_dose_response_curve(dataset)
        check = validate_monotonicity(result, dataset)
        assert not check.passed
        assert check.evidence["reversals"]
        assert "hook" in check.message

    def test_noise_below_the_tolerance_is_ignored(self):
        # A 0.5-unit wobble on a 100-unit window is measurement noise, not a
        # reversal, and must not be reported as one.
        dataset = make_inhibition_dataset(noise=2.5, seed=99)
        result = fit_dose_response_curve(dataset)
        check = validate_monotonicity(result, dataset, tolerance_fraction=0.15)
        assert check.passed

    def test_tolerance_is_configurable(self):
        dataset = make_hook_effect_dataset()
        result = fit_dose_response_curve(dataset)
        # A tolerance larger than the window silences every reversal.
        assert validate_monotonicity(
            result, dataset, tolerance_fraction=5.0
        ).passed


class TestReplicatePrecisionRule:
    def test_absent_labels_warn(self):
        dataset = make_inhibition_dataset()
        check = validate_replicate_precision(dataset)
        assert not check.passed
        assert check.severity == "warning"
        assert "replicate labels" in check.message

    def test_tight_replicates_pass(self):
        dataset = make_inhibition_dataset(replicates=3, noise=1.0)
        assert validate_replicate_precision(dataset).passed

    def test_scattered_replicates_fail(self):
        concentrations = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        responses: list[float] = []
        concentrations_out: list[float] = []
        labels: list[str] = []
        base = [95.0, 70.0, 30.0, 8.0, 3.0]
        for level, (concentration, centre) in enumerate(
            zip(concentrations, base, strict=True)
        ):
            for replicate, factor in enumerate((0.5, 1.0, 1.6)):
                concentrations_out.append(float(concentration))
                responses.append(centre * factor)
                labels.append(f"r{level + 1}_{replicate}")
        dataset = DoseResponseDataset(
            compound="Noisy",
            concentration=np.array(concentrations_out),
            response=np.array(responses),
            replicate_group=tuple(labels),
        )
        check = validate_replicate_precision(dataset)
        assert not check.passed
        assert "scatter" in check.message


class TestSuite:
    def test_runs_every_rule_in_order(self):
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        checks = run_potency_validation(result, dataset)
        assert [check.rule_id for check in checks] == [
            f"POT-{index:03d}" for index in range(1, 12)
        ]

    def test_clean_experiment_passes_the_hard_rules(self):
        dataset = make_inhibition_dataset(
            replicates=3, noise=0.8, concentrations=np.array(
                [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
            )
        )
        result = fit_dose_response(dataset)
        failures = [
            check
            for check in run_potency_validation(result, dataset)
            if check.severity == "failure"
        ]
        assert failures == [], [check.message for check in failures]

    def test_thresholds_are_overridable(self):
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        checks = _by_id(
            run_potency_validation(result, dataset, min_r_squared=0.999999)
        )
        assert not checks["POT-005"].passed

    def test_every_rule_reports_structured_evidence(self):
        dataset = make_inhibition_dataset(noise=1.0)
        result = fit_dose_response(dataset)
        for check in run_potency_validation(result, dataset):
            assert isinstance(check.evidence, dict)
            assert check.message
