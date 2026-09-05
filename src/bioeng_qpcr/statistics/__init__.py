"""Statistics module: statistical analysis utilities for qPCR data.

This package provides three sub-modules:

* **Intervals** (:mod:`~bioeng_qpcr.statistics.intervals`) -- confidence
  intervals for means, proportions, and fold changes.
* **Replicate models** (:mod:`~bioeng_qpcr.statistics.replicate_models`) --
  descriptive statistics, pooled std, t-tests, and linear regression for
  replicate Cq data.
* **Bootstrap** (:mod:`~bioeng_qpcr.statistics.bootstrap`) --
  non-parametric bootstrap resampling for arbitrary statistics.
"""

from __future__ import annotations

from bioeng_qpcr.statistics.bootstrap import (
    BootstrapResult,
    bootstrap_mean,
    bootstrap_median,
    bootstrap_statistic,
)
from bioeng_qpcr.statistics.intervals import (
    ConfidenceInterval,
    fold_change_ci,
    mean_ci,
    proportion_ci,
)
from bioeng_qpcr.statistics.replicate_models import (
    LinearModelResult,
    ReplicateStats,
    linear_regression,
    pooled_std,
    summarize_replicates,
    two_group_ttest,
)

__all__ = [
    # Bootstrap
    "BootstrapResult",
    # Intervals
    "ConfidenceInterval",
    "LinearModelResult",
    # Replicate models
    "ReplicateStats",
    "bootstrap_mean",
    "bootstrap_median",
    "bootstrap_statistic",
    "fold_change_ci",
    "linear_regression",
    "mean_ci",
    "pooled_std",
    "proportion_ci",
    "summarize_replicates",
    "two_group_ttest",
]
