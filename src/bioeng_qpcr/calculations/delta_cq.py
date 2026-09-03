"""ΔCq (delta-Cq) relative quantification method.

The ΔCq method compares the Cq of a target gene to that of a reference
(housekeeping) gene within the same sample.  It is the simplest relative
quantification approach and forms the basis for both the ΔΔCq (Livak) and
Pfaffl methods.

Formula
-------
    ΔCq = Cq(target) − Cq(reference)

The Cq values are averaged across technical replicates before subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioeng_qpcr.calculations.batch import AnalysisExclusion, BatchAnalysisResult
from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import ReplicateType

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeltaCqResult:
    """Result of a ΔCq calculation for a single sample / target pair.

    Attributes
        sample_name:
            Name of the biological sample.
        target_name:
            Name of the target (GOI) assay.
        delta_cq:
            ΔCq = mean_cq_target − mean_cq_reference.
        mean_cq_target:
            Mean Cq of the target across technical replicates.
        mean_cq_reference:
            Mean Cq of the reference gene across technical replicates.
        reference_target:
            Name of the reference gene used for normalisation.
    """

    sample_name: str
    target_name: str
    delta_cq: float
    mean_cq_target: float
    mean_cq_reference: float
    reference_target: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_reference(experiment: Experiment, reference_target: str | None) -> str:
    """Resolve the reference target name.

    If *reference_target* is ``None`` the first reference gene registered on
    the experiment is used.  Raises :class:`ValueError` when no reference gene
    is available.
    """
    if reference_target is not None:
        return reference_target
    refs = experiment.reference_targets
    if not refs:
        raise ValueError(
            "No reference gene found in experiment. "
            "Specify reference_target explicitly or mark an assay as "
            "reference (is_reference=True)."
        )
    return refs[0]


def _mean_cq(experiment: Experiment, sample_name: str, target_name: str) -> float:
    """Compute a replicate-safe mean Cq for one biological sample.

    Technical replicates are averaged within ``replicate_group`` and then
    across groups. Biological replicates must use distinct sample names; more
    than one biological well under the same sample/target is rejected to avoid
    pseudoreplication. Multiple wells marked ``single`` are also rejected.
    """
    wells = experiment.wells_for(sample_name, target_name)
    if not wells:
        raise ValueError(
            f"No wells found for sample '{sample_name}' and target '{target_name}'."
        )
    determined = [well for well in wells if well.cq is not None]
    if not determined:
        raise ValueError(
            f"All Cq values are None for sample '{sample_name}' and target "
            f"'{target_name}'."
        )
    replicate_types = {well.replicate_type for well in determined}
    if len(replicate_types) > 1:
        values = ", ".join(sorted(item.value for item in replicate_types))
        raise ValueError(
            f"Mixed replicate types ({values}) for sample '{sample_name}' and "
            f"target '{target_name}'."
        )
    replicate_type = next(iter(replicate_types))
    if replicate_type is ReplicateType.BIOLOGICAL and len(determined) > 1:
        raise ValueError(
            "Biological replicates must have distinct sample names; refusing "
            f"to average {len(determined)} wells under sample '{sample_name}'."
        )
    if replicate_type is ReplicateType.SINGLE and len(determined) > 1:
        raise ValueError(
            f"Multiple wells for sample '{sample_name}' and target "
            f"'{target_name}' are marked single; label technical replicates."
        )
    if replicate_type is not ReplicateType.TECHNICAL:
        return float(determined[0].cq)

    grouped: dict[str, list[float]] = {}
    for well in determined:
        group = well.replicate_group or "__ungrouped__"
        grouped.setdefault(group, []).append(float(well.cq))
    group_means = [float(np.mean(values)) for values in grouped.values()]
    return float(np.mean(group_means))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delta_cq(
    experiment: Experiment,
    sample_name: str,
    target_name: str,
    reference_target: str | None = None,
) -> DeltaCqResult:
    """Calculate ΔCq for a single sample.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        sample_name:
            Name of the biological sample.
        target_name:
            Name of the target gene of interest.
        reference_target:
            Name of the reference gene.  If ``None`` the first reference gene
            on the experiment is used.

    Returns
        :class:`DeltaCqResult` with the ΔCq value and intermediate means.

    Raises
        ValueError:
            If the target or reference is not found, or if Cq values are
            ``None`` (undetermined) for all replicates.
    """
    ref = _resolve_reference(experiment, reference_target)
    mean_cq_target = _mean_cq(experiment, sample_name, target_name)
    mean_cq_reference = _mean_cq(experiment, sample_name, ref)
    delta = mean_cq_target - mean_cq_reference
    return DeltaCqResult(
        sample_name=sample_name,
        target_name=target_name,
        delta_cq=delta,
        mean_cq_target=mean_cq_target,
        mean_cq_reference=mean_cq_reference,
        reference_target=ref,
    )


def delta_cq_all(
    experiment: Experiment,
    target_name: str,
    reference_target: str | None = None,
    *,
    fail_on_exclusion: bool = True,
) -> list[DeltaCqResult]:
    """Calculate ΔCq for all samples, failing on omissions by default.

    Iterates over non-NTC samples that measured *target_name* and computes
    ΔCq for each.  Samples with missing or undetermined Cq data are silently
    skipped.

    Parameters
        experiment:
            The qPCR experiment.
        target_name:
            Name of the target gene of interest.
        reference_target:
            Name of the reference gene (default: first reference gene).

    Returns
        List of :class:`DeltaCqResult`, one per sample with valid data.
    """
    batch = delta_cq_batch(experiment, target_name, reference_target)
    if fail_on_exclusion:
        batch.raise_for_exclusions()
    return batch.results


def delta_cq_batch(
    experiment: Experiment,
    target_name: str,
    reference_target: str | None = None,
) -> BatchAnalysisResult[DeltaCqResult]:
    """Calculate ΔCq while explicitly recording every excluded sample."""
    ref = _resolve_reference(experiment, reference_target)
    results: list[DeltaCqResult] = []
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
            result = delta_cq(experiment, sample_name, target_name, ref)
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
        parameters={
            "method": "delta_cq",
            "target_name": target_name,
            "reference_target": ref,
        },
    )
