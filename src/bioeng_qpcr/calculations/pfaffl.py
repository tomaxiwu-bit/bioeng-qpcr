"""Pfaffl efficiency-corrected relative quantification method.

The Pfaffl method extends the ΔΔCq approach by incorporating individual
amplification efficiencies for the target and reference genes, yielding a
more accurate ratio when efficiencies deviate from 100 %.

Formula
-------
    ratio = (E_target ^ ΔCq_target) / (E_reference ^ ΔCq_reference)

where:

    ΔCq_target     = Cq(target, control) − Cq(target, sample)
    ΔCq_reference  = Cq(reference, control) − Cq(reference, sample)
    E              = efficiency + 1   (so 100 % efficiency → E = 2)

When both efficiencies equal 1.0 (100 %), the Pfaffl ratio reduces to the
Livak (ΔΔCq) fold change.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioeng_qpcr.calculations.batch import AnalysisExclusion, BatchAnalysisResult
from bioeng_qpcr.calculations.delta_cq import _mean_cq, _resolve_reference
from bioeng_qpcr.domain.entities import Experiment

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PfafflResult:
    """Result of a Pfaffl calculation for a single sample / target pair.

    Attributes
        sample_name:
            Name of the biological sample.
        target_name:
            Name of the target (GOI) assay.
        ratio:
            Efficiency-corrected expression ratio.
        target_efficiency:
            Amplification efficiency of the target assay (1.0 = 100 %).
        reference_efficiency:
            Amplification efficiency of the reference assay.
        delta_cq_target:
            ΔCq for the target = Cq(control) − Cq(sample).
        delta_cq_reference:
            ΔCq for the reference = Cq(control) − Cq(sample).
        control_sample:
            Name of the control (calibrator) sample.
        reference_target:
            Name of the reference gene used for normalisation.
    """

    sample_name: str
    target_name: str
    ratio: float
    target_efficiency: float
    reference_efficiency: float
    delta_cq_target: float
    delta_cq_reference: float
    control_sample: str
    reference_target: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_efficiency(experiment: Experiment, target_name: str) -> float:
    """Return the amplification efficiency for *target_name*.

    Missing efficiency is an error. Assuming 100% without evidence would turn
    an efficiency-corrected method into an undisclosed ΔΔCq approximation.
    """
    assay = experiment.assays.get(target_name)
    if assay is not None and assay.efficiency is not None:
        return assay.efficiency
    raise ValueError(
        f"No measured amplification efficiency is available for assay '{target_name}'."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pfaffl(
    experiment: Experiment,
    sample_name: str,
    target_name: str,
    control_sample: str,
    reference_target: str | None = None,
) -> PfafflResult:
    """Calculate the Pfaffl efficiency-corrected ratio for a single sample.

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
        :class:`PfafflResult` with the ratio and intermediate values.

    Raises
        ValueError:
            If the sample, control, target, or reference is not found, or if
            Cq values are ``None`` for any required measurement.
    """
    ref = _resolve_reference(experiment, reference_target)

    # Mean Cq for the test sample.
    cq_target_sample = _mean_cq(experiment, sample_name, target_name)
    cq_ref_sample = _mean_cq(experiment, sample_name, ref)

    # Mean Cq for the control sample (raises ValueError if missing).
    try:
        cq_target_control = _mean_cq(experiment, control_sample, target_name)
        cq_ref_control = _mean_cq(experiment, control_sample, ref)
    except ValueError as exc:
        raise ValueError(
            f"Control sample '{control_sample}' is missing data for "
            f"target '{target_name}' or reference '{ref}': {exc}"
        ) from exc

    # ΔCq values (control − sample, per Pfaffl convention).
    delta_cq_target = cq_target_control - cq_target_sample
    delta_cq_reference = cq_ref_control - cq_ref_sample

    # Efficiencies (E = efficiency + 1).
    target_eff = _get_efficiency(experiment, target_name)
    ref_eff = _get_efficiency(experiment, ref)
    e_target = target_eff + 1.0
    e_reference = ref_eff + 1.0

    ratio = (e_target**delta_cq_target) / (e_reference**delta_cq_reference)

    return PfafflResult(
        sample_name=sample_name,
        target_name=target_name,
        ratio=ratio,
        target_efficiency=target_eff,
        reference_efficiency=ref_eff,
        delta_cq_target=delta_cq_target,
        delta_cq_reference=delta_cq_reference,
        control_sample=control_sample,
        reference_target=ref,
    )


def pfaffl_all(
    experiment: Experiment,
    target_name: str,
    control_sample: str,
    reference_target: str | None = None,
    *,
    fail_on_exclusion: bool = True,
) -> list[PfafflResult]:
    """Calculate Pfaffl ratios for all samples, failing on omissions by default.

    Iterates over non-NTC samples that measured *target_name* and computes
    the efficiency-corrected ratio relative to *control_sample*.  The control
    sample itself is included (its ratio will be 1.0).  Samples with missing
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
        List of :class:`PfafflResult`, one per sample with valid data.
    """
    batch = pfaffl_batch(experiment, target_name, control_sample, reference_target)
    if fail_on_exclusion:
        batch.raise_for_exclusions()
    return batch.results


def pfaffl_batch(
    experiment: Experiment,
    target_name: str,
    control_sample: str,
    reference_target: str | None = None,
) -> BatchAnalysisResult[PfafflResult]:
    """Calculate Pfaffl ratios and explicitly record all exclusions."""
    ref = _resolve_reference(experiment, reference_target)
    # Fail before iterating so a global assay-metadata problem cannot be
    # misreported as one exclusion per sample.
    _get_efficiency(experiment, target_name)
    _get_efficiency(experiment, ref)
    results: list[PfafflResult] = []
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
            result = pfaffl(experiment, sample_name, target_name, control_sample, ref)
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
            "method": "pfaffl",
            "target_name": target_name,
            "reference_target": ref,
            "control_sample": control_sample,
        },
    )
