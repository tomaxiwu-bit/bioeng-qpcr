"""Tests for the dose-response model equations and their inversions."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.potency.models import (
    MODEL_PARAMETERS,
    DoseResponseModel,
    evaluate_model,
    five_pl,
    four_pl,
    gompertz,
    half_max_log_concentration,
    solve_log_concentration_for_response,
    three_pl,
)


class TestFourParameterLogistic:
    def test_inflection_sits_at_declared_log_ec50(self):
        params = {"bottom": 5.0, "top": 95.0, "log_ec50": 1.5, "hill": -1.0}
        value = float(four_pl(np.array([1.5]), **params)[0])
        assert value == pytest.approx(50.0)

    def test_plateaus_reached_far_from_the_inflection(self):
        params = {"bottom": 5.0, "top": 95.0, "log_ec50": 0.0, "hill": -1.0}
        low = float(four_pl(np.array([-8.0]), **params)[0])
        high = float(four_pl(np.array([8.0]), **params)[0])
        assert low == pytest.approx(95.0, abs=1e-4)
        assert high == pytest.approx(5.0, abs=1e-4)

    def test_negative_hill_descends_and_positive_ascends(self):
        descending = four_pl(np.array([-2.0, 2.0]), 0.0, 100.0, 0.0, -1.0)
        ascending = four_pl(np.array([-2.0, 2.0]), 0.0, 100.0, 0.0, 1.0)
        assert descending[0] > descending[1]
        assert ascending[0] < ascending[1]


class TestThreeParameterLogistic:
    def test_lower_asymptote_is_zero(self):
        # Eight decades above the inflection, the logistic is 1e-8 of the top.
        value = float(three_pl(np.array([8.0]), top=80.0, log_ec50=0.0, hill=-1.0)[0])
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_half_of_top_at_the_inflection(self):
        value = float(three_pl(np.array([0.0]), top=80.0, log_ec50=0.0, hill=-1.0)[0])
        assert value == pytest.approx(40.0)


class TestFiveParameterLogistic:
    def test_asymmetry_of_one_reproduces_the_four_pl(self):
        x = np.linspace(-4.0, 4.0, 17)
        four = four_pl(x, 3.0, 90.0, 0.5, -1.3)
        five = five_pl(x, 3.0, 90.0, 0.5, -1.3, 1.0)
        assert five == pytest.approx(four, abs=1e-9)

    def test_inflection_is_not_the_half_maximal_point(self):
        params = {
            "bottom": 0.0,
            "top": 100.0,
            "log_ec50": 0.0,
            "hill": -1.0,
            "asym": 0.4,
        }
        at_inflection = float(five_pl(np.array([0.0]), **params)[0])
        # With asym != 1 the curve value at log_ec50 is not 50.
        assert at_inflection != pytest.approx(50.0, abs=1.0)

    def test_half_max_reaches_exactly_the_midpoint(self):
        params = {
            "bottom": 7.0,
            "top": 93.0,
            "log_ec50": 1.0,
            "hill": -1.2,
            "asym": 0.35,
        }
        half = half_max_log_concentration(DoseResponseModel.FIVE_PL, params)
        value = float(five_pl(np.array([half]), **params)[0])
        assert value == pytest.approx(50.0)


class TestGompertz:
    def test_half_max_reaches_exactly_the_midpoint(self):
        params = {"bottom": 2.0, "top": 98.0, "log_ec50": -0.5, "hill": -0.9}
        half = half_max_log_concentration(DoseResponseModel.GOMPERTZ, params)
        value = float(gompertz(np.array([half]), **params)[0])
        assert value == pytest.approx(50.0)

    def test_inflection_is_not_the_half_maximal_point(self):
        # Gompertz crosses 50 % at its inflection only by coincidence; the
        # correction is the whole reason half_max_log_concentration exists.
        params = {"bottom": 0.0, "top": 100.0, "log_ec50": 0.0, "hill": -1.0}
        at_inflection = float(gompertz(np.array([0.0]), **params)[0])
        assert at_inflection == pytest.approx(100.0 / np.e, abs=0.5)


class TestHalfMaxLogConcentration:
    def test_symmetric_models_return_the_fitted_parameter(self):
        for model, params in (
            (
                DoseResponseModel.THREE_PL,
                {"top": 100.0, "log_ec50": 0.7, "hill": -1.0},
            ),
            (
                DoseResponseModel.FOUR_PL,
                {"bottom": 5.0, "top": 100.0, "log_ec50": 0.7, "hill": -1.0},
            ),
        ):
            assert half_max_log_concentration(model, params) == pytest.approx(0.7)

    def test_zero_hill_is_rejected(self):
        params = {"bottom": 0.0, "top": 100.0, "log_ec50": 0.0, "hill": 0.0}
        with pytest.raises(ValueError, match="Hill slope is zero"):
            half_max_log_concentration(DoseResponseModel.FOUR_PL, params)

    def test_missing_parameter_raises(self):
        with pytest.raises(KeyError):
            half_max_log_concentration(DoseResponseModel.FOUR_PL, {"top": 1.0})


class TestSolveForResponse:
    @pytest.mark.parametrize(
        ("model", "params"),
        [
            (
                DoseResponseModel.THREE_PL,
                {"top": 90.0, "log_ec50": 0.3, "hill": -1.1},
            ),
            (
                DoseResponseModel.FOUR_PL,
                {"bottom": 6.0, "top": 90.0, "log_ec50": 0.3, "hill": -1.1},
            ),
            (
                DoseResponseModel.FIVE_PL,
                {
                    "bottom": 6.0,
                    "top": 90.0,
                    "log_ec50": 0.3,
                    "hill": -1.1,
                    "asym": 0.5,
                },
            ),
            (
                DoseResponseModel.GOMPERTZ,
                {"bottom": 6.0, "top": 90.0, "log_ec50": 0.3, "hill": -1.1},
            ),
        ],
    )
    def test_inversion_round_trips_through_the_model(self, model, params):
        for target in (20.0, 35.0, 48.0, 70.0):
            x = solve_log_concentration_for_response(model, params, target)
            recovered = float(evaluate_model(model, params, np.array([x]))[0])
            assert recovered == pytest.approx(target, abs=1e-6)

    def test_target_outside_the_window_is_rejected(self):
        params = {"bottom": 10.0, "top": 90.0, "log_ec50": 0.0, "hill": -1.0}
        with pytest.raises(ValueError, match="outside the fitted window"):
            solve_log_concentration_for_response(
                DoseResponseModel.FOUR_PL, params, 5.0
            )
        with pytest.raises(ValueError, match="outside the fitted window"):
            solve_log_concentration_for_response(
                DoseResponseModel.FOUR_PL, params, 95.0
            )

    def test_missing_parameter_raises(self):
        with pytest.raises(KeyError):
            solve_log_concentration_for_response(
                DoseResponseModel.FOUR_PL, {"top": 100.0}, 50.0
            )


class TestEvaluateModel:
    def test_parameter_registry_matches_every_model(self):
        expected = {
            DoseResponseModel.THREE_PL: 3,
            DoseResponseModel.FOUR_PL: 4,
            DoseResponseModel.FIVE_PL: 5,
            DoseResponseModel.GOMPERTZ: 4,
        }
        assert {
            model: len(MODEL_PARAMETERS[model]) for model in DoseResponseModel
        } == expected

    def test_missing_parameter_raises_key_error(self):
        with pytest.raises(KeyError):
            evaluate_model(
                DoseResponseModel.FOUR_PL,
                {"bottom": 0.0, "top": 1.0, "log_ec50": 0.0},
                np.array([0.0]),
            )

    def test_extreme_exponents_do_not_overflow(self):
        # A very steep Hill slope far from the inflection would raise
        # OverflowError without the clipping inside the model functions.
        values = four_pl(
            np.array([-50.0, 50.0]), 0.0, 100.0, 0.0, -30.0
        )
        assert np.all(np.isfinite(values))
