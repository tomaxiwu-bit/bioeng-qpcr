"""Tests for replicate-aware statistical models (replicate_models.py).

Verifies:

* ``summarize_replicates`` with known Cq values (mean, std, SEM, n, CI).
* CV% calculation.
* n=1 edge case.
* ``pooled_std`` with known groups.
* ``two_group_ttest`` against ``scipy.stats.ttest_ind(equal_var=False)``.
* ``linear_regression`` against ``numpy.polyfit`` and ``scipy.stats.linregress``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bioeng_qpcr.statistics.intervals import ConfidenceInterval
from bioeng_qpcr.statistics.replicate_models import (
    LinearModelResult,
    ReplicateStats,
    linear_regression,
    pooled_std,
    summarize_replicates,
    two_group_ttest,
)

# ---------------------------------------------------------------------------
# summarize_replicates: known Cq values
# ---------------------------------------------------------------------------


class TestSummarizeReplicates:
    @pytest.fixture
    def cq_values(self) -> np.ndarray:
        return np.array([25.0, 25.2, 24.8, 25.1, 25.0])

    def test_returns_replicate_stats(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        assert isinstance(result, ReplicateStats)

    def test_mean(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        assert result.mean == pytest.approx(np.mean(cq_values))

    def test_std_ddof1(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        assert result.std == pytest.approx(np.std(cq_values, ddof=1))

    def test_sem(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        expected_sem = np.std(cq_values, ddof=1) / np.sqrt(len(cq_values))
        assert result.sem == pytest.approx(expected_sem)

    def test_n(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        assert result.n == 5

    def test_ci_is_confidence_interval(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        assert isinstance(result.ci, ConfidenceInterval)

    def test_ci_method(self, cq_values: np.ndarray) -> None:
        result = summarize_replicates(cq_values)
        assert result.ci.method == "t-distribution"

    def test_accepts_list(self) -> None:
        result = summarize_replicates([20.0, 21.0, 22.0])
        assert result.n == 3
        assert result.mean == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# summarize_replicates: CV% calculation
# ---------------------------------------------------------------------------


class TestCvPercent:
    def test_cv_percent_known(self) -> None:
        cq = [25.0, 25.2, 24.8, 25.1, 25.0]
        result = summarize_replicates(cq)
        mean = np.mean(cq)
        std = np.std(cq, ddof=1)
        expected_cv = std / mean * 100.0
        assert result.cv_percent == pytest.approx(expected_cv)

    def test_cv_percent_low_variability(self) -> None:
        """Typical qPCR technical replicates have CV < 5 %."""
        cq = [25.00, 25.05, 24.95, 25.02, 24.98]
        result = summarize_replicates(cq)
        assert result.cv_percent < 5.0

    def test_cv_percent_high_variability(self) -> None:
        cq = [20.0, 30.0, 25.0]
        result = summarize_replicates(cq)
        assert result.cv_percent > 10.0


# ---------------------------------------------------------------------------
# summarize_replicates: n=1 edge case
# ---------------------------------------------------------------------------


class TestSummarizeReplicatesEdgeCases:
    def test_n1_std_zero(self) -> None:
        result = summarize_replicates([25.0])
        assert result.std == 0.0

    def test_n1_sem_zero(self) -> None:
        result = summarize_replicates([25.0])
        assert result.sem == 0.0

    def test_n1_cv_zero(self) -> None:
        result = summarize_replicates([25.0])
        assert result.cv_percent == 0.0

    def test_n1_ci_collapses(self) -> None:
        result = summarize_replicates([25.0])
        assert result.ci.lower == pytest.approx(25.0)
        assert result.ci.upper == pytest.approx(25.0)
        assert result.ci.estimate == pytest.approx(25.0)

    def test_n1_mean(self) -> None:
        result = summarize_replicates([42.0])
        assert result.mean == pytest.approx(42.0)
        assert result.n == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_replicates([])


# ---------------------------------------------------------------------------
# pooled_std
# ---------------------------------------------------------------------------


class TestPooledStd:
    def test_two_equal_groups(self) -> None:
        g1 = [25.0, 26.0, 24.0]
        g2 = [27.0, 28.0, 26.0]
        result = pooled_std([g1, g2])
        s1 = np.std(g1, ddof=1)
        s2 = np.std(g2, ddof=1)
        n1, n2 = len(g1), len(g2)
        expected = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        assert result == pytest.approx(expected)

    def test_three_groups(self) -> None:
        groups = [
            [25.0, 26.0, 24.0],
            [30.0, 31.0, 29.0],
            [20.0, 21.0, 19.0],
        ]
        result = pooled_std(groups)
        numerator = sum((len(g) - 1) * np.std(g, ddof=1) ** 2 for g in groups)
        denominator = sum(len(g) - 1 for g in groups)
        expected = np.sqrt(numerator / denominator)
        assert result == pytest.approx(expected)

    def test_equal_std_groups(self) -> None:
        """Groups with identical std should yield that std."""
        g1 = [10.0, 12.0, 14.0]  # std = 2
        g2 = [20.0, 22.0, 24.0]  # std = 2
        result = pooled_std([g1, g2])
        assert result == pytest.approx(2.0)

    def test_unequal_group_sizes(self) -> None:
        g1 = [25.0, 26.0, 24.0, 25.0, 26.0]  # n=5
        g2 = [27.0, 28.0]  # n=2
        result = pooled_std([g1, g2])
        s1 = np.std(g1, ddof=1)
        s2 = np.std(g2, ddof=1)
        n1, n2 = len(g1), len(g2)
        expected = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        assert result == pytest.approx(expected)

    def test_empty_groups_list_raises(self) -> None:
        with pytest.raises(ValueError, match="No groups"):
            pooled_std([])

    def test_empty_group_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pooled_std([[1.0, 2.0], []])

    def test_all_single_element_groups_raises(self) -> None:
        """All groups of size 1 have 0 degrees of freedom."""
        with pytest.raises(ValueError, match="degrees of freedom"):
            pooled_std([[1.0], [2.0], [3.0]])

    def test_accepts_numpy_arrays(self) -> None:
        g1 = np.array([25.0, 26.0, 24.0])
        g2 = np.array([27.0, 28.0, 26.0])
        result = pooled_std([g1, g2])
        assert result > 0.0


# ---------------------------------------------------------------------------
# two_group_ttest (Welch's)
# ---------------------------------------------------------------------------


class TestTwoGroupTtest:
    def test_matches_scipy_welch(self) -> None:
        g1 = [25.0, 26.0, 24.0, 25.0, 24.5]
        g2 = [28.0, 29.0, 27.0, 28.0, 27.5]
        t_stat, p_value = two_group_ttest(g1, g2)
        expected = stats.ttest_ind(g1, g2, equal_var=False)
        assert t_stat == pytest.approx(expected.statistic)
        assert p_value == pytest.approx(expected.pvalue)

    def test_significant_difference(self) -> None:
        g1 = [20.0, 21.0, 19.0, 20.0, 20.5]
        g2 = [30.0, 31.0, 29.0, 30.0, 30.5]
        _, p_value = two_group_ttest(g1, g2)
        assert p_value < 0.05

    def test_no_significant_difference(self) -> None:
        g1 = [25.0, 25.1, 24.9, 25.0, 25.05]
        g2 = [25.0, 25.1, 24.9, 25.0, 25.05]
        _, p_value = two_group_ttest(g1, g2)
        assert p_value > 0.05

    def test_returns_tuple_of_floats(self) -> None:
        g1 = [25.0, 26.0, 24.0]
        g2 = [27.0, 28.0, 26.0]
        result = two_group_ttest(g1, g2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_group1_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="group1"):
            two_group_ttest([25.0], [26.0, 27.0])

    def test_group2_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="group2"):
            two_group_ttest([25.0, 26.0], [27.0])

    def test_both_groups_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="group1"):
            two_group_ttest([25.0], [26.0])


# ---------------------------------------------------------------------------
# linear_regression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    @pytest.fixture
    def perfect_data(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        return x, y

    def test_returns_linear_model_result(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        assert isinstance(result, LinearModelResult)

    def test_slope_perfect_fit(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        assert result.slope == pytest.approx(2.0)

    def test_intercept_perfect_fit(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        assert result.intercept == pytest.approx(1.0)

    def test_r_squared_perfect(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        assert result.r_squared == pytest.approx(1.0)

    def test_slope_matches_polyfit(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        coeffs = np.polyfit(x, y, 1)
        assert result.slope == pytest.approx(coeffs[0])

    def test_intercept_matches_polyfit(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        coeffs = np.polyfit(x, y, 1)
        assert result.intercept == pytest.approx(coeffs[1])

    def test_r_squared_matches_linregress(
        self, perfect_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = perfect_data
        result = linear_regression(x, y)
        lr = stats.linregress(x, y)
        assert result.r_squared == pytest.approx(lr.rvalue**2)

    def test_p_value_significant(self) -> None:
        """Strong linear relationship should have small p-value."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        y = 3.0 * x + 2.0
        result = linear_regression(x, y)
        assert result.p_value < 0.05

    def test_noisy_data_r_squared_below_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.1, 4.8, 7.2, 8.7, 11.3])
        result = linear_regression(x, y)
        assert 0.95 < result.r_squared < 1.0

    def test_std_error_positive(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
        result = linear_regression(x, y)
        assert result.std_error > 0.0

    def test_mismatched_lengths_raises(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same shape"):
            linear_regression(x, y)

    def test_too_few_points_raises(self) -> None:
        x = np.array([1.0])
        y = np.array([2.0])
        with pytest.raises(ValueError, match="2"):
            linear_regression(x, y)
