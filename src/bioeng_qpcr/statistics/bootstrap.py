"""Bootstrap resampling for qPCR data.

This module provides non-parametric bootstrap confidence intervals for
arbitrary statistics.  The bootstrap is particularly useful in qPCR
analysis when the sample size is small and parametric assumptions
(normality) may not hold.

Three convenience functions are provided:

* :func:`bootstrap_mean` -- bootstrap the mean.
* :func:`bootstrap_median` -- bootstrap the median.
* :func:`bootstrap_statistic` -- general bootstrap for any statistic
  function.

All functions use the **percentile method** for confidence intervals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of a bootstrap resampling procedure.

    Attributes
        estimate:
            The statistic computed on the original data.
        bias:
            Bootstrap bias estimate: mean(bootstrap_stats) - estimate.
        std_error:
            Bootstrap standard error: std(bootstrap_stats, ddof=1).
        lower:
            Lower bound of the percentile confidence interval.
        upper:
            Upper bound of the percentile confidence interval.
        confidence:
            Confidence level (e.g. 0.95).
        n_resamples:
            Number of bootstrap resamples performed.
        method:
            Name of the CI method (always ``"percentile"``).
    """

    estimate: float
    bias: float
    std_error: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    method: str


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _percentile_ci(
    bootstrap_stats: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Percentile-method confidence interval.

    Parameters
        bootstrap_stats:
            1-D array of bootstrap replicate statistics.
        confidence:
            Confidence level in (0, 1).

    Returns
        ``(lower, upper)`` bounds.
    """
    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_stats, 100.0 * alpha / 2.0))
    upper = float(np.percentile(bootstrap_stats, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


# ---------------------------------------------------------------------------
# General bootstrap
# ---------------------------------------------------------------------------


def bootstrap_statistic(
    data: np.ndarray | list[float],
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap any statistic using the percentile method.

    Resamples *data* with replacement *n_resamples* times, computes
    *statistic* on each resample, and derives a percentile confidence
    interval.

    Parameters
        data:
            Array (or list) of numeric values.
        statistic:
            A callable that takes a 1-D ``np.ndarray`` and returns a
            ``float`` (e.g. ``np.mean``, ``np.median``, or a custom
            function).
        n_resamples:
            Number of bootstrap resamples.  Default 10 000.
        confidence:
            Confidence level in (0, 1).  Default 0.95.
        seed:
            Random seed for reproducibility.  If ``None`` the RNG is
            seeded from the operating system.

    Returns
        :class:`BootstrapResult` with ``method="percentile"``.

    Raises
        ValueError:
            If *data* is empty.
    """
    data = np.asarray(data, dtype=float)
    n = int(data.size)

    if n == 0:
        raise ValueError("Cannot bootstrap empty data.")

    estimate = float(statistic(data))

    # Edge case: single observation -- every resample is identical.
    if n == 1:
        return BootstrapResult(
            estimate=estimate,
            bias=0.0,
            std_error=0.0,
            lower=estimate,
            upper=estimate,
            confidence=confidence,
            n_resamples=n_resamples,
            method="percentile",
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))

    # Apply the statistic to each bootstrap resample.
    bootstrap_stats = np.array([float(statistic(data[idx])) for idx in indices])

    bias = float(np.mean(bootstrap_stats) - estimate)
    std_error = float(np.std(bootstrap_stats, ddof=1))
    lower, upper = _percentile_ci(bootstrap_stats, confidence)

    return BootstrapResult(
        estimate=estimate,
        bias=bias,
        std_error=std_error,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_resamples=n_resamples,
        method="percentile",
    )


# ---------------------------------------------------------------------------
# Convenience: bootstrap mean
# ---------------------------------------------------------------------------


def bootstrap_mean(
    data: np.ndarray | list[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap the mean using the percentile method.

    Parameters
        data:
            Array (or list) of numeric values.
        n_resamples:
            Number of bootstrap resamples.  Default 10 000.
        confidence:
            Confidence level in (0, 1).  Default 0.95.
        seed:
            Random seed for reproducibility.

    Returns
        :class:`BootstrapResult` with ``method="percentile"``.

    Raises
        ValueError:
            If *data* is empty.
    """
    return bootstrap_statistic(
        data,
        np.mean,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Convenience: bootstrap median
# ---------------------------------------------------------------------------


def bootstrap_median(
    data: np.ndarray | list[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap the median using the percentile method.

    Parameters
        data:
            Array (or list) of numeric values.
        n_resamples:
            Number of bootstrap resamples.  Default 10 000.
        confidence:
            Confidence level in (0, 1).  Default 0.95.
        seed:
            Random seed for reproducibility.

    Returns
        :class:`BootstrapResult` with ``method="percentile"``.

    Raises
        ValueError:
            If *data* is empty.
    """
    return bootstrap_statistic(
        data,
        np.median,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
