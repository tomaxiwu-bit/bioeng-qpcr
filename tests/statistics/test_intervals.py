"""Tests for confidence interval calculations (intervals.py).

Verifies:

* ``mean_ci`` against a manual t-distribution calculation.
* ``mean_ci`` edge cases (n=1, n=0, all identical values).
* ``proportion_ci`` Wilson score interval against a manual calculation.
* ``fold_change_ci`` conversion from log2 to linear scale.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bioeng_qpcr.statistics.intervals import (
    ConfidenceInterval,
    fold_change_ci,
    mean_ci,
    proportion_ci,
)

# ---------------------------------------------------------------------------
# mean_ci: known data verified against manual t-distribution calculation
# ---------------------------------------------------------------------------


class TestMeanCiKnown:
    """Verify mean_ci against a manual scipy.stats.t calculation."""

    @pytest.fixture
    def data(self) -> np.ndarray:
        return np.array([25.0, 26.0, 24.0, 25.0, 27.0])

    def test_returns_confidence_interval(self, data: np.ndarray) -> None:
        result = mean_ci(data)
        assert isinstance(result, ConfidenceInterval)

    def test_estimate_is_mean(self, data: np.ndarray) -> None:
        result = mean_ci(data)
        assert result.estimate == pytest.approx(np.mean(data))

    def test_lower_matches_manual(self, data: np.ndarray) -> None:
        result = mean_ci(data, confidence=0.95)
        n = data.size
        sem = np.std(data, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        expected_lower = np.mean(data) - t_crit * sem
        assert result.lower == pytest.approx(expected_lower)

    def test_upper_matches_manual(self, data: np.ndarray) -> None:
        result = mean_ci(data, confidence=0.95)
        n = data.size
        sem = np.std(data, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        expected_upper = np.mean(data) + t_crit * sem
        assert result.upper == pytest.approx(expected_upper)

    def test_method_name(self, data: np.ndarray) -> None:
        result = mean_ci(data)
        assert result.method == "t-distribution"

    def test_confidence_level(self, data: np.ndarray) -> None:
        result = mean_ci(data, confidence=0.99)
        assert result.confidence_level == 0.99

    def test_accepts_list(self) -> None:
        result = mean_ci([1.0, 2.0, 3.0, 4.0])
        assert result.estimate == pytest.approx(2.5)

    def test_99_percent_wider_than_95(self, data: np.ndarray) -> None:
        ci_95 = mean_ci(data, confidence=0.95)
        ci_99 = mean_ci(data, confidence=0.99)
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        assert width_99 > width_95


# ---------------------------------------------------------------------------
# mean_ci: edge cases
# ---------------------------------------------------------------------------


class TestMeanCiEdgeCases:
    def test_single_value(self) -> None:
        result = mean_ci([25.0])
        assert result.estimate == pytest.approx(25.0)
        assert result.lower == pytest.approx(25.0)
        assert result.upper == pytest.approx(25.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mean_ci([])

    def test_all_same_values(self) -> None:
        result = mean_ci([25.0, 25.0, 25.0, 25.0])
        assert result.estimate == pytest.approx(25.0)
        assert result.lower == pytest.approx(25.0)
        assert result.upper == pytest.approx(25.0)

    def test_all_same_values_method(self) -> None:
        result = mean_ci([30.0, 30.0, 30.0])
        assert result.method == "t-distribution"


# ---------------------------------------------------------------------------
# proportion_ci: Wilson score interval
# ---------------------------------------------------------------------------


class TestProportionCi:
    def test_returns_confidence_interval(self) -> None:
        result = proportion_ci(7, 12)
        assert isinstance(result, ConfidenceInterval)

    def test_estimate_is_sample_proportion(self) -> None:
        result = proportion_ci(7, 12)
        assert result.estimate == pytest.approx(7.0 / 12.0)

    def test_matches_manual_wilson(self) -> None:
        successes, total = 7, 12
        confidence = 0.95
        result = proportion_ci(successes, total, confidence=confidence)

        p_hat = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)
        n = total
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

        assert result.lower == pytest.approx(center - margin)
        assert result.upper == pytest.approx(center + margin)

    def test_method_name(self) -> None:
        result = proportion_ci(5, 10)
        assert result.method == "wilson"

    def test_bounds_within_zero_one(self) -> None:
        result = proportion_ci(1, 100)
        assert 0.0 <= result.lower <= result.upper <= 1.0

    def test_all_successes(self) -> None:
        """10/10 successes -- interval should be wide and include 1."""
        result = proportion_ci(10, 10)
        assert result.estimate == pytest.approx(1.0)
        assert result.upper == pytest.approx(1.0)
        assert result.lower < 1.0

    def test_zero_successes(self) -> None:
        """0/10 successes -- interval should include 0."""
        result = proportion_ci(0, 10)
        assert result.estimate == pytest.approx(0.0)
        assert result.lower == pytest.approx(0.0)
        assert result.upper > 0.0

    def test_total_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="total must be positive"):
            proportion_ci(0, 0)

    def test_negative_total_raises(self) -> None:
        with pytest.raises(ValueError, match="total must be positive"):
            proportion_ci(0, -5)

    def test_successes_exceed_total_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed total"):
            proportion_ci(15, 10)


# ---------------------------------------------------------------------------
# fold_change_ci: log2 to linear conversion
# ---------------------------------------------------------------------------


class TestFoldChangeCi:
    def test_returns_confidence_interval(self) -> None:
        result = fold_change_ci(2.0, 0.5)
        assert isinstance(result, ConfidenceInterval)

    def test_estimate_is_linear_fold_change(self) -> None:
        result = fold_change_ci(2.0, 0.5)
        assert result.estimate == pytest.approx(2.0**2.0)  # = 4.0

    def test_estimate_for_negative_log2fc(self) -> None:
        """log2_fc = -1 means fold change = 0.5 (down-regulation)."""
        result = fold_change_ci(-1.0, 0.3)
        assert result.estimate == pytest.approx(0.5)

    def test_estimate_for_zero_log2fc(self) -> None:
        """log2_fc = 0 means fold change = 1 (no change)."""
        result = fold_change_ci(0.0, 0.2)
        assert result.estimate == pytest.approx(1.0)

    def test_lower_upper_match_manual(self) -> None:
        log2_fc = 2.0
        se = 0.5
        confidence = 0.95
        result = fold_change_ci(log2_fc, se, confidence=confidence)

        z = stats.norm.ppf((1 + confidence) / 2)
        expected_lower = 2.0 ** (log2_fc - z * se)
        expected_upper = 2.0 ** (log2_fc + z * se)

        assert result.lower == pytest.approx(expected_lower)
        assert result.upper == pytest.approx(expected_upper)

    def test_lower_less_than_estimate_less_than_upper(self) -> None:
        result = fold_change_ci(1.5, 0.4)
        assert result.lower < result.estimate < result.upper

    def test_method_name(self) -> None:
        result = fold_change_ci(1.0, 0.2)
        assert result.method == "log2_fold_change"

    def test_zero_se_collapses_to_point(self) -> None:
        result = fold_change_ci(2.0, 0.0)
        assert result.lower == pytest.approx(result.estimate)
        assert result.upper == pytest.approx(result.estimate)

    def test_symmetric_on_log2_scale(self) -> None:
        """On the log2 scale the CI is symmetric; on linear it is not."""
        log2_fc = 1.0
        se = 0.3
        result = fold_change_ci(log2_fc, se)
        # On log2 scale: lower_dist = log2_fc - z*se, upper_dist = log2_fc + z*se
        # So estimate - lower on log2 scale == upper - estimate on log2 scale
        log2_est = np.log2(result.estimate)
        log2_lower = np.log2(result.lower)
        log2_upper = np.log2(result.upper)
        assert (log2_est - log2_lower) == pytest.approx(log2_upper - log2_est)
