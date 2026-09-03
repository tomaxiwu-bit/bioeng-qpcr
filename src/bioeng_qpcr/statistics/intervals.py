"""Confidence interval calculations for qPCR data.

This module provides confidence interval (CI) utilities for three common
qPCR scenarios:

* **Mean CI** (:func:`mean_ci`) -- CI for the mean of replicate Cq values
  using the Student's t-distribution.
* **Proportion CI** (:func:`proportion_ci`) -- Wilson score interval for a
  binomial proportion (e.g. positive-call rate).
* **Fold-change CI** (:func:`fold_change_ci`) -- CI for a fold-change
  estimate that is computed on the log2 scale and converted back to the
  linear scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceInterval:
    """A confidence interval for a point estimate.

    Attributes
        estimate:
            The point estimate (e.g. mean, proportion, fold change).
        lower:
            Lower bound of the confidence interval.
        upper:
            Upper bound of the confidence interval.
        confidence_level:
            The confidence level (e.g. 0.95 for 95 % CI).
        method:
            Name of the method used to compute the interval
            (e.g. ``"t-distribution"``, ``"wilson"``, ``"log2_fold_change"``).
    """

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str


# ---------------------------------------------------------------------------
# Mean confidence interval (t-distribution)
# ---------------------------------------------------------------------------


def mean_ci(
    data: np.ndarray | list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Confidence interval for the mean using the t-distribution.

    Computes the two-sided confidence interval for the sample mean as::

        CI = mean +/- t_crit * (std / sqrt(n))

    where ``t_crit`` is the critical value of the t-distribution with
    ``n - 1`` degrees of freedom.

    Parameters
        data:
            Array (or list) of numeric values.
        confidence:
            Confidence level in the range (0, 1).  Default 0.95.

    Returns
        :class:`ConfidenceInterval` with ``method="t-distribution"``.

    Raises
        ValueError:
            If *data* is empty.
    """
    data = np.asarray(data, dtype=float)
    n = int(data.size)

    if n == 0:
        raise ValueError("Cannot compute confidence interval for empty data.")

    mean = float(np.mean(data))

    # Edge case: single observation -- no variability, CI collapses to point.
    if n == 1:
        return ConfidenceInterval(
            estimate=mean,
            lower=mean,
            upper=mean,
            confidence_level=confidence,
            method="t-distribution",
        )

    std = float(np.std(data, ddof=1))

    # Edge case: zero variance (all values identical).
    if std == 0.0:
        return ConfidenceInterval(
            estimate=mean,
            lower=mean,
            upper=mean,
            confidence_level=confidence,
            method="t-distribution",
        )

    sem = std / np.sqrt(n)
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    margin = t_crit * sem

    return ConfidenceInterval(
        estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
        method="t-distribution",
    )


# ---------------------------------------------------------------------------
# Proportion confidence interval (Wilson score)
# ---------------------------------------------------------------------------


def proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score confidence interval for a binomial proportion.

    The Wilson score interval has better coverage properties than the
    naive (Wald) interval, especially for small samples or proportions
    near 0 or 1.

    Formula::

        z          = Phi^-1((1 + confidence) / 2)
        p_hat      = successes / total
        denom      = 1 + z^2 / n
        center     = (p_hat + z^2 / (2 n)) / denom
        margin     = (z / denom) * sqrt(p_hat (1 - p_hat) / n + z^2 / (4 n^2))

    Parameters
        successes:
            Number of successes (non-negative integer).
        total:
            Total number of trials (positive integer).
        confidence:
            Confidence level in the range (0, 1).  Default 0.95.

    Returns
        :class:`ConfidenceInterval` with ``method="wilson"``.  The
        ``estimate`` field holds the sample proportion ``successes / total``.

    Raises
        ValueError:
            If *total* <= 0 or *successes* > *total*.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}.")
    if successes > total:
        raise ValueError(f"successes ({successes}) cannot exceed total ({total}).")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}.")

    n = float(total)
    p_hat = successes / n
    z = float(stats.norm.ppf((1.0 + confidence) / 2.0))

    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n + z**2 / (4.0 * n**2))

    lower = float(center - margin)
    upper = float(center + margin)

    # Clamp to [0, 1] -- Wilson interval can theoretically exceed these
    # bounds for extreme proportions, but a proportion is always in [0, 1].
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return ConfidenceInterval(
        estimate=float(p_hat),
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        method="wilson",
    )


# ---------------------------------------------------------------------------
# Fold-change confidence interval (log2 scale -> linear)
# ---------------------------------------------------------------------------


def fold_change_ci(
    log2_fold_change: float,
    se: float,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Confidence interval for a fold-change estimate.

    The fold change and its standard error are assumed to be on the log2
    scale.  The CI is constructed on the log2 scale and then converted back
    to the linear scale by exponentiating with base 2::

        lower_fold = 2^(log2_fc - z_crit * se)
        upper_fold = 2^(log2_fc + z_crit * se)
        estimate   = 2^log2_fc

    Because the standard error is provided externally (it may originate
    from a delta-delta-Cq calculation with replicates from multiple genes),
    the degrees of freedom are not known.  The normal-distribution critical
    value is therefore used as a large-sample approximation.

    Parameters
        log2_fold_change:
            Fold change on the log2 scale.
        se:
            Standard error of *log2_fold_change*.
        confidence:
            Confidence level in the range (0, 1).  Default 0.95.

    Returns
        :class:`ConfidenceInterval` with ``method="log2_fold_change"``.
        All bounds are on the **linear** scale.
    """
    z_crit = float(stats.norm.ppf((1.0 + confidence) / 2.0))

    lower_log2 = log2_fold_change - z_crit * se
    upper_log2 = log2_fold_change + z_crit * se

    lower_fold = float(2.0**lower_log2)
    upper_fold = float(2.0**upper_log2)
    estimate = float(2.0**log2_fold_change)

    return ConfidenceInterval(
        estimate=estimate,
        lower=lower_fold,
        upper=upper_fold,
        confidence_level=confidence,
        method="log2_fold_change",
    )
