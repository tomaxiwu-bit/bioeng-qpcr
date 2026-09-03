"""Efficiency-corrected normalized quantities for qPCR expression analysis.

Quantities are calculated in log space as ``Q = (1 + efficiency) ** -Cq``.
For multiple reference genes, their quantities are combined with a geometric
mean before target normalization. Each biological sample remains an
independent observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bioeng_qpcr.calculations.batch import AnalysisExclusion, BatchAnalysisResult
from bioeng_qpcr.calculations.delta_cq import _mean_cq
from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.statistics.replicate_models import two_group_ttest


@dataclass(frozen=True)
class NormalizedQuantityResult:
    """Efficiency-corrected normalized expression for one biological sample."""

    sample_name: str
    target_name: str
    reference_targets: tuple[str, ...]
    mean_cq_target: float
    mean_cq_references: dict[str, float]
    target_efficiency: float
    reference_efficiencies: dict[str, float]
    normalized_quantity: float
    log2_normalized_quantity: float


@dataclass(frozen=True)
class GroupComparisonResult:
    """Group fold difference based on per-sample normalized quantities."""

    target_name: str
    treated_group: str
    control_group: str
    reference_targets: tuple[str, ...]
    treated_values: dict[str, float]
    control_values: dict[str, float]
    treated_mean: float
    control_mean: float
    fold_difference: float
    log2_fold_difference: float
    welch_t_statistic: float | None
    p_value: float | None


def _reference_targets(
    experiment: Experiment, reference_targets: list[str] | tuple[str, ...] | None
) -> tuple[str, ...]:
    references = tuple(dict.fromkeys(reference_targets or experiment.reference_targets))
    if not references:
        raise ValueError(
            "At least one designated reference target is required for "
            "normalized quantities."
        )
    missing = [name for name in references if name not in experiment.assays]
    if missing:
        raise ValueError(f"Reference assay(s) not found: {', '.join(missing)}")
    return references


def _efficiency(experiment: Experiment, target_name: str) -> float:
    assay = experiment.assays.get(target_name)
    if assay is None:
        raise ValueError(f"Assay '{target_name}' is not present in the experiment.")
    if assay.efficiency is None:
        raise ValueError(
            f"Measured amplification efficiency is required for assay '{target_name}'."
        )
    if assay.efficiency <= 0:
        raise ValueError(
            f"Assay '{target_name}' has non-positive efficiency {assay.efficiency}."
        )
    return assay.efficiency


def normalized_quantity(
    experiment: Experiment,
    sample_name: str,
    target_name: str,
    reference_targets: list[str] | tuple[str, ...] | None = None,
) -> NormalizedQuantityResult:
    """Calculate an efficiency-corrected normalized quantity for one sample."""
    references = _reference_targets(experiment, reference_targets)
    if target_name in references:
        raise ValueError("Target of interest must not also be a reference target.")

    target_efficiency = _efficiency(experiment, target_name)
    reference_efficiencies = {
        reference: _efficiency(experiment, reference) for reference in references
    }
    target_cq = _mean_cq(experiment, sample_name, target_name)
    reference_cqs = {
        reference: _mean_cq(experiment, sample_name, reference)
        for reference in references
    }

    log_target_quantity = -target_cq * math.log1p(target_efficiency)
    log_reference_quantities = [
        -reference_cqs[reference] * math.log1p(reference_efficiencies[reference])
        for reference in references
    ]
    log_normalized = log_target_quantity - float(np.mean(log_reference_quantities))
    normalized = math.exp(log_normalized)

    return NormalizedQuantityResult(
        sample_name=sample_name,
        target_name=target_name,
        reference_targets=references,
        mean_cq_target=target_cq,
        mean_cq_references=reference_cqs,
        target_efficiency=target_efficiency,
        reference_efficiencies=reference_efficiencies,
        normalized_quantity=normalized,
        log2_normalized_quantity=log_normalized / math.log(2.0),
    )


def normalized_quantity_all(
    experiment: Experiment,
    target_name: str,
    reference_targets: list[str] | tuple[str, ...] | None = None,
) -> BatchAnalysisResult[NormalizedQuantityResult]:
    """Calculate per-sample quantities and explicitly retain all exclusions."""
    references = _reference_targets(experiment, reference_targets)
    _efficiency(experiment, target_name)
    for reference in references:
        _efficiency(experiment, reference)

    results: list[NormalizedQuantityResult] = []
    exclusions: list[AnalysisExclusion] = []
    for sample_name in experiment.sample_names:
        try:
            result = normalized_quantity(
                experiment, sample_name, target_name, references
            )
        except ValueError as exc:
            exclusions.append(
                AnalysisExclusion(
                    sample_name=sample_name,
                    target_name=target_name,
                    code="INCOMPLETE_NORMALIZATION_INPUT",
                    reason=str(exc),
                )
            )
            continue
        results.append(result)

    return BatchAnalysisResult(
        results=results,
        exclusions=exclusions,
        parameters={
            "method": "normalized_quantity",
            "target_name": target_name,
            "reference_targets": list(references),
            "quantity_formula": "(1 + efficiency) ** -Cq",
            "reference_aggregation": "geometric_mean",
        },
    )


def compare_groups(
    experiment: Experiment,
    target_name: str,
    treated_group: str,
    control_group: str,
    reference_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[GroupComparisonResult, BatchAnalysisResult[NormalizedQuantityResult]]:
    """Compare group means of per-sample normalized quantities.

    Welch's t-test is performed on log2 normalized quantities when both groups
    contain at least two valid biological samples. The returned batch object
    preserves exclusions and should be reported alongside the comparison.
    """
    batch = normalized_quantity_all(experiment, target_name, reference_targets)
    by_sample = {result.sample_name: result for result in batch.results}
    treated = {
        name: by_sample[name]
        for name, sample in experiment.samples.items()
        if sample.group == treated_group and name in by_sample
    }
    control = {
        name: by_sample[name]
        for name, sample in experiment.samples.items()
        if sample.group == control_group and name in by_sample
    }
    if not treated:
        raise ValueError(f"No valid samples found in treated group '{treated_group}'.")
    if not control:
        raise ValueError(f"No valid samples found in control group '{control_group}'.")

    treated_values = {
        name: result.normalized_quantity for name, result in treated.items()
    }
    control_values = {
        name: result.normalized_quantity for name, result in control.items()
    }
    treated_log2 = np.array(
        [item.log2_normalized_quantity for item in treated.values()]
    )
    control_log2 = np.array(
        [item.log2_normalized_quantity for item in control.values()]
    )
    # Expression quantities are positive and multiplicative. Group summaries
    # therefore use geometric means, while inference is performed in log2
    # space on independent biological samples.
    treated_mean = float(2.0 ** np.mean(treated_log2))
    control_mean = float(2.0 ** np.mean(control_log2))
    if control_mean <= 0:
        raise ValueError("Control-group mean normalized quantity must be positive.")
    fold_difference = treated_mean / control_mean

    t_statistic: float | None = None
    p_value: float | None = None
    if len(treated) >= 2 and len(control) >= 2:
        t_statistic, p_value = two_group_ttest(treated_log2, control_log2)
    else:
        batch.warnings.append(
            "Welch's t-test was not run because each group requires at least "
            "two valid biological samples."
        )

    references = _reference_targets(experiment, reference_targets)
    return (
        GroupComparisonResult(
            target_name=target_name,
            treated_group=treated_group,
            control_group=control_group,
            reference_targets=references,
            treated_values=treated_values,
            control_values=control_values,
            treated_mean=treated_mean,
            control_mean=control_mean,
            fold_difference=fold_difference,
            log2_fold_difference=math.log2(fold_difference),
            welch_t_statistic=t_statistic,
            p_value=p_value,
        ),
        batch,
    )
