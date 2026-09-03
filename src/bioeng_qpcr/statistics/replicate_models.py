"""Replicate-aware statistical models for qPCR data.

This module provides statistical utilities that operate on replicate Cq
(quantification cycle) values:

* :func:`summarize_replicates` -- descriptive statistics (mean, std, SEM,
  CV %, CI) for a set of technical replicates.
* :func:`pooled_std` -- pooled standard deviation across multiple groups.
* :func:`two_group_ttest` -- Welch's t-test (unequal variances) for
  comparing two groups of replicates.
* :func:`linear_regression` -- ordinary least-squares linear regression,
  used for standard-curve fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from bioeng_qpcr.statistics.intervals import (
    ConfidenceInterval,
    mean_ci,
)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplicateStats:
    """Descriptive statistics for a set of replicate measurements.

    Attributes
        mean:
            Arithmetic mean of the replicates.
        std:
            Sample standard deviation (ddof = 1).
        sem:
            Standard error of the mean (std / sqrt(n)).
        n:
            Number of replicates.
        cv_percent:
            Coefficient of variation in percent (std / mean * 100).
        ci:
            Confidence interval for the mean (see :func:`mean_ci`).
    """

    mean: float
    std: float
    sem: float
    n: int
    cv_percent: float
    ci: ConfidenceInterval


@dataclass
class LinearModelResult:
    """Result of an ordinary least-squares linear regression.

    Attributes
        slope:
            Slope of the regression line.
        intercept:
            Y-intercept of the regression line.
        r_squared:
            Coefficient of determination (R^2).
        p_value:
            Two-tailed p-value for the t-test on the slope.
        std_error:
            Standard error of the slope.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float


# ---------------------------------------------------------------------------
# Replicate summary
# ---------------------------------------------------------------------------


def summarize_replicates(
    cq_values: np.ndarray | list[float],
    confidence: float = 0.95,
) -> ReplicateStats:
    """Compute descriptive statistics for a set of replicate Cq values.

    Parameters
        cq_values:
            Array (or list) of Cq values from technical replicates.
        confidence:
            Confidence level for the mean CI.  Default 0.95.

    Returns
        :class:`ReplicateStats` with mean, std, SEM, n, CV %, and CI.

    Raises
        ValueError:
            If *cq_values* is empty.
    """
    data = np.asarray(cq_values, dtype=float)
    n = int(data.size)

    if n == 0:
        raise ValueError("Cannot summarize empty replicate data.")

    mean = float(np.mean(data))
    ci = mean_ci(data, confidence)

    # Edge case: single replicate -- no variability.
    if n == 1:
        return ReplicateStats(
            mean=mean,
            std=0.0,
            sem=0.0,
            n=1,
            cv_percent=0.0,
            ci=ci,
        )

    std = float(np.std(data, ddof=1))
    sem = std / np.sqrt(n)
    cv_percent = (std / mean * 100.0) if mean != 0.0 else 0.0

    return ReplicateStats(
        mean=mean,
        std=std,
        sem=sem,
        n=n,
        cv_percent=cv_percent,
        ci=ci,
    )


# ---------------------------------------------------------------------------
# Pooled standard deviation
# ---------------------------------------------------------------------------


def pooled_std(groups: list[np.ndarray]) -> float:
    """Pooled standard deviation across multiple groups.

    Computes the weighted average of group variances, weighted by
    ``n_i - 1`` (degrees of freedom)::

        s_p = sqrt( sum( (n_i - 1) * s_i^2 ) / sum( n_i - 1 ) )

    Parameters
        groups:
            List of arrays, each containing the values for one group.

    Returns
        The pooled standard deviation as a float.

    Raises
        ValueError:
            If *groups* is empty, any individual group is empty, or all
            groups have fewer than 2 elements (no degrees of freedom).
    """
    if not groups:
        raise ValueError("No groups provided.")

    numerator = 0.0
    denominator = 0.0

    for i, group in enumerate(groups):
        g = np.asarray(group, dtype=float)
        ni = int(g.size)

        if ni == 0:
            raise ValueError(f"Group {i} is empty.")

        if ni == 1:
            # Contributes 0 degrees of freedom; skip.
            continue

        si = float(np.std(g, ddof=1))
        numerator += (ni - 1) * si**2
        denominator += ni - 1

    if denominator == 0:
        raise ValueError(
            "Cannot compute pooled std: all groups have size 1 (no degrees of freedom)."
        )

    return float(np.sqrt(numerator / denominator))


# ---------------------------------------------------------------------------
# Two-group t-test (Welch's)
# ---------------------------------------------------------------------------


def two_group_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
) -> tuple[float, float]:
    """Welch's t-test (unequal variances) between two groups.

    Parameters
        group1:
            Array of values for the first group.
        group2:
            Array of values for the second group.

    Returns
        A tuple ``(t_statistic, p_value)`` where *p_value* is two-tailed.

    Raises
        ValueError:
            If either group has fewer than 2 elements.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    if g1.size < 2:
        raise ValueError(f"group1 must have at least 2 elements, got {g1.size}.")
    if g2.size < 2:
        raise ValueError(f"group2 must have at least 2 elements, got {g2.size}.")

    result = stats.ttest_ind(g1, g2, equal_var=False)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Linear regression (OLS)
# ---------------------------------------------------------------------------


def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> LinearModelResult:
    """Ordinary least-squares linear regression.

    Fits ``y = slope * x + intercept`` and returns the slope, intercept,
    R^2, two-tailed p-value (t-test on the slope), and the standard error
    of the slope.

    Parameters
        x:
            Independent variable values.
        y:
            Dependent variable values.

    Returns
        :class:`LinearModelResult` with slope, intercept, R^2, p-value,
        and std_error of the slope.

    Raises
        ValueError:
            If *x* and *y* have different lengths or fewer than 2 points
            are provided.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} vs {y.shape}."
        )
    if x.size < 2:
        raise ValueError(f"Need at least 2 points for linear regression, got {x.size}.")

    result = stats.linregress(x, y)

    return LinearModelResult(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
        p_value=float(result.pvalue),
        std_error=float(result.stderr),
    )
