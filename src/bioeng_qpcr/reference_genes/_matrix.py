"""Shared reference-gene matrix construction.

Reference-gene algorithms operate on log relative quantities, not directly
on Cq when candidate assays have different amplification efficiencies.  This
module also centralises the biological/technical replicate rules used by the
main qPCR calculation path.
"""

from __future__ import annotations

import math

import numpy as np

from bioeng_qpcr.calculations.delta_cq import _mean_cq
from bioeng_qpcr.domain.entities import Experiment


def candidate_efficiencies(
    experiment: Experiment, candidates: list[str]
) -> dict[str, float]:
    """Return measured positive amplification efficiencies for candidates."""
    efficiencies: dict[str, float] = {}
    for target in candidates:
        assay = experiment.assays.get(target)
        if assay is None:
            raise ValueError(f"Candidate assay '{target}' is not present.")
        if assay.efficiency is None:
            raise ValueError(
                "Measured amplification efficiency is required for reference "
                f"candidate '{target}'."
            )
        if assay.efficiency <= 0:
            raise ValueError(
                f"Reference candidate '{target}' has non-positive efficiency "
                f"{assay.efficiency}."
            )
        efficiencies[target] = float(assay.efficiency)
    return efficiencies


def build_reference_matrix(
    experiment: Experiment,
    candidates: list[str],
    *,
    efficiency_corrected: bool = True,
) -> tuple[list[str], np.ndarray, list[str], dict[str, float]]:
    """Build a complete samples-by-genes matrix for stability analysis.

    Determined wells are reduced with :func:`_mean_cq`, which averages true
    technical replicates but rejects mixed, duplicate-single, and collapsed
    biological replicate declarations. Samples without a determined Cq for
    any candidate are excluded complete-case and returned to the caller.

    With ``efficiency_corrected=True`` each value is
    ``-Cq * log2(1 + efficiency)``, the log2 relative-quantity scale used by
    geNorm/NormFinder. At 100% efficiency this is simply ``-Cq`` and therefore
    preserves the conventional algorithms' variance and ranking.
    """
    efficiencies = (
        candidate_efficiencies(experiment, candidates)
        if efficiency_corrected
        else {}
    )
    sample_names: list[str] = []
    excluded_samples: list[str] = []
    rows: list[list[float]] = []

    for sample_name in experiment.sample_names:
        row: list[float] = []
        for target in candidates:
            wells = experiment.wells_for(sample_name, target)
            if not wells or not any(well.cq is not None for well in wells):
                excluded_samples.append(sample_name)
                break
            mean_cq = _mean_cq(experiment, sample_name, target)
            if efficiency_corrected:
                value = -mean_cq * math.log2(1.0 + efficiencies[target])
            else:
                value = mean_cq
            row.append(float(value))
        else:
            sample_names.append(sample_name)
            rows.append(row)

    matrix = (
        np.asarray(rows, dtype=float)
        if rows
        else np.empty((0, len(candidates)), dtype=float)
    )
    return sample_names, matrix, excluded_samples, efficiencies
