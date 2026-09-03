"""Tests for bootstrap resampling (bootstrap.py).

Verifies:

* ``bootstrap_mean`` with a fixed seed for reproducibility.
* CI contains the true mean for large samples.
* ``bootstrap_median``.
* ``bootstrap_statistic`` with a custom function (std).
* Edge cases (n=0, n=1).
* Reproducibility with the same seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.statistics.bootstrap import (
    BootstrapResult,
    bootstrap_mean,
    bootstrap_median,
    bootstrap_statistic,
)

# ---------------------------------------------------------------------------
# bootstrap_mean
# ---------------------------------------------------------------------------


class TestBootstrapMean:
    @pytest.fixture
    def data(self) -> np.ndarray:
        return np.array([25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0])

    def test_returns_bootstrap_result(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=500, seed=42)
        assert isinstance(result, BootstrapResult)

    def test_estimate_is_sample_mean(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=500, seed=42)
        assert result.estimate == pytest.approx(np.mean(data))

    def test_method_is_percentile(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=500, seed=42)
        assert result.method == "percentile"

    def test_ci_contains_estimate(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=1000, seed=42)
        assert result.lower <= result.estimate <= result.upper

    def test_bias_is_small(self, data: np.ndarray) -> None:
        """For the mean, bootstrap bias should be near zero."""
        result = bootstrap_mean(data, n_resamples=5000, seed=42)
        assert abs(result.bias) < 0.5

    def test_std_error_positive(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=500, seed=42)
        assert result.std_error > 0.0

    def test_n_resamples_recorded(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=777, seed=42)
        assert result.n_resamples == 777

    def test_confidence_recorded(self, data: np.ndarray) -> None:
        result = bootstrap_mean(data, n_resamples=500, seed=42, confidence=0.90)
        assert result.confidence == 0.90

    def test_accepts_list(self) -> None:
        result = bootstrap_mean([1.0, 2.0, 3.0, 4.0], n_resamples=100, seed=0)
        assert result.estimate == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# bootstrap_mean: CI contains true mean for large samples
# ---------------------------------------------------------------------------


class TestBootstrapMeanLargeSample:
    def test_ci_contains_sample_mean(self) -> None:
        rng = np.random.default_rng(123)
        data = rng.normal(25.0, 1.0, size=500)
        result = bootstrap_mean(data, n_resamples=5000, seed=42)
        assert result.lower <= result.estimate <= result.upper

    def test_ci_brackets_population_mean(self) -> None:
        """For a large sample the 95 % CI should contain the true mean."""
        rng = np.random.default_rng(999)
        true_mean = 30.0
        data = rng.normal(true_mean, 2.0, size=1000)
        result = bootstrap_mean(data, n_resamples=5000, seed=7)
        assert result.lower <= true_mean <= result.upper

    def test_higher_confidence_wider_ci(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(20.0, 1.0, size=100)
        ci_90 = bootstrap_mean(data, n_resamples=2000, seed=1, confidence=0.90)
        ci_99 = bootstrap_mean(data, n_resamples=2000, seed=1, confidence=0.99)
        width_90 = ci_90.upper - ci_90.lower
        width_99 = ci_99.upper - ci_99.lower
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# bootstrap_median
# ---------------------------------------------------------------------------


class TestBootstrapMedian:
    @pytest.fixture
    def data(self) -> np.ndarray:
        return np.array([25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0])

    def test_returns_bootstrap_result(self, data: np.ndarray) -> None:
        result = bootstrap_median(data, n_resamples=500, seed=42)
        assert isinstance(result, BootstrapResult)

    def test_estimate_is_sample_median(self, data: np.ndarray) -> None:
        result = bootstrap_median(data, n_resamples=500, seed=42)
        assert result.estimate == pytest.approx(np.median(data))

    def test_ci_contains_estimate(self, data: np.ndarray) -> None:
        result = bootstrap_median(data, n_resamples=1000, seed=42)
        assert result.lower <= result.estimate <= result.upper

    def test_method_is_percentile(self, data: np.ndarray) -> None:
        result = bootstrap_median(data, n_resamples=500, seed=42)
        assert result.method == "percentile"


# ---------------------------------------------------------------------------
# bootstrap_statistic: custom function
# ---------------------------------------------------------------------------


class TestBootstrapStatistic:
    def test_bootstrap_std(self) -> None:
        data = [25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0]
        result = bootstrap_statistic(
            data,
            lambda x: float(np.std(x, ddof=1)),
            n_resamples=1000,
            seed=42,
        )
        assert result.estimate == pytest.approx(np.std(data, ddof=1))

    def test_bootstrap_std_ci_contains_estimate(self) -> None:
        data = [25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0]
        result = bootstrap_statistic(
            data,
            lambda x: float(np.std(x, ddof=1)),
            n_resamples=1000,
            seed=42,
        )
        assert result.lower <= result.estimate <= result.upper

    def test_bootstrap_variance(self) -> None:
        data = [10.0, 12.0, 14.0, 11.0, 13.0]
        result = bootstrap_statistic(
            data,
            lambda x: float(np.var(x, ddof=1)),
            n_resamples=500,
            seed=1,
        )
        assert result.estimate == pytest.approx(np.var(data, ddof=1))

    def test_bootstrap_max(self) -> None:
        data = [10.0, 12.0, 14.0, 11.0, 13.0]
        result = bootstrap_statistic(
            data,
            lambda x: float(np.max(x)),
            n_resamples=500,
            seed=1,
        )
        assert result.estimate == pytest.approx(14.0)

    def test_method_is_percentile(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_statistic(data, np.mean, n_resamples=100, seed=0)
        assert result.method == "percentile"

    def test_bootstrap_mean_matches_bootstrap_statistic(self) -> None:
        """bootstrap_mean should agree with bootstrap_statistic(np.mean)."""
        data = [25.0, 26.0, 24.0, 25.0, 27.0]
        r1 = bootstrap_mean(data, n_resamples=1000, seed=99)
        r2 = bootstrap_statistic(data, np.mean, n_resamples=1000, seed=99)
        assert r1.estimate == pytest.approx(r2.estimate)
        assert r1.lower == pytest.approx(r2.lower)
        assert r1.upper == pytest.approx(r2.upper)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBootstrapEdgeCases:
    def test_empty_raises_mean(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_mean([])

    def test_empty_raises_median(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_median([])

    def test_empty_raises_statistic(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_statistic([], np.mean)

    def test_n1_mean(self) -> None:
        result = bootstrap_mean([25.0], n_resamples=100, seed=42)
        assert result.estimate == pytest.approx(25.0)
        assert result.bias == 0.0
        assert result.std_error == 0.0
        assert result.lower == pytest.approx(25.0)
        assert result.upper == pytest.approx(25.0)

    def test_n1_median(self) -> None:
        result = bootstrap_median([25.0], n_resamples=100, seed=42)
        assert result.estimate == pytest.approx(25.0)
        assert result.bias == 0.0
        assert result.std_error == 0.0
        assert result.lower == pytest.approx(25.0)
        assert result.upper == pytest.approx(25.0)

    def test_n1_statistic(self) -> None:
        result = bootstrap_statistic([25.0], np.mean, n_resamples=100, seed=42)
        assert result.estimate == pytest.approx(25.0)
        assert result.bias == 0.0
        assert result.std_error == 0.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestBootstrapReproducibility:
    def test_same_seed_same_result_mean(self) -> None:
        data = [25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0]
        r1 = bootstrap_mean(data, n_resamples=1000, seed=42)
        r2 = bootstrap_mean(data, n_resamples=1000, seed=42)
        assert r1.estimate == r2.estimate
        assert r1.bias == r2.bias
        assert r1.std_error == r2.std_error
        assert r1.lower == r2.lower
        assert r1.upper == r2.upper

    def test_same_seed_same_result_median(self) -> None:
        data = [25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0]
        r1 = bootstrap_median(data, n_resamples=1000, seed=42)
        r2 = bootstrap_median(data, n_resamples=1000, seed=42)
        assert r1.lower == r2.lower
        assert r1.upper == r2.upper

    def test_same_seed_same_result_statistic(self) -> None:
        data = [25.0, 26.0, 24.0, 25.0, 27.0, 26.0, 25.0, 24.0]

        def stat(x):
            return float(np.std(x, ddof=1))

        r1 = bootstrap_statistic(data, stat, n_resamples=1000, seed=42)
        r2 = bootstrap_statistic(data, stat, n_resamples=1000, seed=42)
        assert r1.lower == r2.lower
        assert r1.upper == r2.upper

    def test_different_seed_different_result(self) -> None:
        # Use a larger dataset with distinct values so that the percentile
        # CI bounds have enough resolution to differ between seeds.
        data = np.array(
            [
                25.03,
                26.31,
                24.12,
                25.55,
                27.21,
                26.84,
                25.37,
                24.76,
                25.92,
                26.15,
                24.89,
                25.71,
                27.04,
                26.53,
                24.38,
                25.14,
                26.77,
                24.51,
                25.88,
                26.22,
            ]
        )
        r1 = bootstrap_mean(data, n_resamples=5000, seed=42)
        r2 = bootstrap_mean(data, n_resamples=5000, seed=99)
        # The estimate is the same (it's the sample mean), but the
        # bootstrap distribution should differ due to different seeds.
        assert r1.estimate == r2.estimate
        assert r1.bias != r2.bias or r1.std_error != r2.std_error

    def test_none_seed_runs_without_error(self) -> None:
        data = [25.0, 26.0, 24.0, 25.0, 27.0]
        result = bootstrap_mean(data, n_resamples=100, seed=None)
        assert result.lower <= result.estimate <= result.upper
