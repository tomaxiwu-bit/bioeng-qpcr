"""ΔΔCq (Livak) relative quantification method.

The ΔΔCq method extends ΔCq by comparing a sample's ΔCq to that of a
control sample.  It assumes ~100 % amplification efficiency for both
target and reference genes.

Formulas
--------
    ΔCq(sample)   = Cq(target) − Cq(reference)
    ΔΔCq          = ΔCq(sample) − ΔCq(control)
    Fold change   = 2^(−ΔΔCq)

A fold change > 1 indicates up-regulation; < 1 indicates down-regulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioeng_qpcr.calculations.batch import AnalysisExclusion, BatchAnalysisResult
from bioeng_qpcr.calculations.delta_cq import (
    DeltaCqResult,
    _resolve_reference,
    delta_cq,
)
from bioeng_qpcr.domain.entities import Experiment

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeltaDeltaCqResult:
    """Result of a ΔΔCq calculation for a single sample / target pair.

    Attributes
        sample_name:
            Name of the biological sample.
        target_name:
            Name of the target (GOI) assay.
        delta_cq:
            ΔCq of the sample (Cq_target − Cq_reference).
        delta_delta_cq:
            ΔΔCq = ΔCq(sample) − ΔCq(control).
        fold_change:
            Fold change = 2^(−ΔΔCq).
        control_sample:
            Name of the control sample used as the reference point.
        reference_target:
            Name of the reference gene used for normalisation.
    """

    sample_name: str
    target_name: str
    delta_cq: float
    delta_delta_cq: float
    fold_change: float
    control_sample: str
    reference_target: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delta_delta_cq(
    experiment: Experiment,
    sample_name: str,
    target_name: str,
    control_sample: str,
    reference_target: str | None = None,
) -> DeltaDeltaCqResult:
    """Calculate ΔΔCq and fold change for a single sample.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        sample_name:
            Name of the biological sample.
        target_name:
            Name of the target gene of interest.
        control_sample:
            Name of the control (calibrator) sample.
        reference_target:
            Name of the reference gene.  If ``None`` the first reference gene
            on the experiment is used.

    Returns
        :class:`DeltaDeltaCqResult` with ΔCq, ΔΔCq, and fold change.

    Raises
        ValueError:
            If the sample, control, target, or reference is not found, or if
            Cq values are ``None`` for any required measurement.
    """
    ref = _resolve_reference(experiment, reference_target)

    # ΔCq for the test sample (raises ValueError if data missing).
    sample_result: DeltaCqResult = delta_cq(experiment, sample_name, target_name, ref)

    # ΔCq for the control sample (raises ValueError if control missing data).
    try:
        control_result: DeltaCqResult = delta_cq(
            experiment, control_sample, target_name, ref
        )
    except ValueError as exc:
        raise ValueError(
            f"Control sample '{control_sample}' is missing data for "
            f"target '{target_name}' or reference '{ref}': {exc}"
        ) from exc

    dd_cq = sample_result.delta_cq - control_result.delta_cq
    fold_change = 2.0 ** (-dd_cq)

    return DeltaDeltaCqResult(
        sample_name=sample_name,
        target_name=target_name,
        delta_cq=sample_result.delta_cq,
        delta_delta_cq=dd_cq,
        fold_change=fold_change,
        control_sample=control_sample,
        reference_target=ref,
    )


def delta_delta_cq_all(
    experiment: Experiment,
    target_name: str,
    control_sample: str,
    reference_target: str | None = None,
    *,
    fail_on_exclusion: bool = True,
) -> list[DeltaDeltaCqResult]:
    """Calculate ΔΔCq for all samples, failing on omissions by default.

    Iterates over non-NTC samples that measured *target_name* and computes
    ΔΔCq relative to *control_sample*.  The control sample itself is
    included (its ΔΔCq will be 0 and fold change 1).  Samples with missing
    or undetermined Cq data are silently skipped.

    Parameters
        experiment:
            The qPCR experiment.
        target_name:
            Name of the target gene of interest.
        control_sample:
            Name of the control (calibrator) sample.
        reference_target:
            Name of the reference gene (default: first reference gene).

    Returns
        List of :class:`DeltaDeltaCqResult`, one per sample with valid data.
    """
    batch = delta_delta_cq_batch(
        experiment, target_name, control_sample, reference_target
    )
    if fail_on_exclusion:
        batch.raise_for_exclusions()
    return batch.results


def delta_delta_cq_batch(
    experiment: Experiment,
    target_name: str,
    control_sample: str,
    reference_target: str | None = None,
) -> BatchAnalysisResult[DeltaDeltaCqResult]:
    """Calculate ΔΔCq and retain an audit record of all omissions."""
    ref = _resolve_reference(experiment, reference_target)
    results: list[DeltaDeltaCqResult] = []
    exclusions: list[AnalysisExclusion] = []
    for sample_name in experiment.sample_names:
        if target_name not in experiment.targets_for(sample_name):
            exclusions.append(
                AnalysisExclusion(
                    sample_name=sample_name,
                    target_name=target_name,
                    code="MISSING_TARGET",
                    reason=f"target '{target_name}' was not measured",
                )
            )
            continue
        try:
            result = delta_delta_cq(
                experiment, sample_name, target_name, control_sample, ref
            )
        except ValueError as exc:
            exclusions.append(
                AnalysisExclusion(
                    sample_name=sample_name,
                    target_name=target_name,
                    code="INVALID_OR_MISSING_CQ",
                    reason=str(exc),
                )
            )
            continue
        results.append(result)
    return BatchAnalysisResult(
        results=results,
        exclusions=exclusions,
        warnings=[
            "ΔΔCq assumes equal, near-perfect amplification efficiencies; "
            "prefer normalized_quantity for publication-oriented analysis."
        ],
        parameters={
            "method": "delta_delta_cq",
            "target_name": target_name,
            "reference_target": ref,
            "control_sample": control_sample,
        },
    )
