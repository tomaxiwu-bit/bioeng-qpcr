"""Reference-gene stability estimators.

``normfinder`` implements the model-based estimator used by the maintained
Bioconductor NormqPCR implementation of NormFinder. Candidate Cq values are
first converted to efficiency-corrected log2 relative quantities. The model
then separates sample, group and gene effects, estimates group-specific
residual variances, and shrinks gene-by-group effects with an empirical-Bayes
variance estimate.

``variance_stability_score`` preserves the project's earlier descriptive
within/between-group score for callers that intentionally need that simpler
measure.  The two methods are deliberately separate because their numerical
values are not interchangeable.

References
----------
Andersen CL, Jensen JL, Orntoft TF. Cancer Res. 2004;64(15):5245-5250.
Perkins JR et al. NormqPCR: Functions for normalisation of RT-qPCR data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.reference_genes._matrix import build_reference_matrix


@dataclass
class NormFinderResult:
    """Result of the legacy descriptive variance stability score."""

    ranking: list[str]
    stability_values: dict[str, float]
    inter_group_var: dict[str, float]
    intra_group_var: dict[str, float]


@dataclass
class ModelBasedNormFinderResult:
    """Result of the model-based NormFinder estimator.

    ``stability_values`` are the NormqPCR ``rho`` values for grouped data:
    the mean, across groups, of the absolute shrunken group effect plus its
    standard uncertainty.  Lower values indicate greater stability.
    ``group_effects`` and ``group_effect_variances`` expose the corresponding
    empirical-Bayes quantities for auditability.
    """

    ranking: list[str]
    stability_values: dict[str, float]
    group_effects: dict[str, dict[str, float]]
    group_effect_variances: dict[str, dict[str, float]]
    group_specific_variances: dict[str, dict[str, float]]
    common_group_variance: float
    excluded_samples: list[str]
    n_samples: int
    n_groups: int
    efficiencies: dict[str, float] = field(default_factory=dict)
    quantity_scale: str = "log2_relative_quantity_efficiency_corrected"


def _resolve_candidates(
    experiment: Experiment, candidate_targets: list[str] | None
) -> list[str]:
    """Resolve and deterministically order candidate reference genes."""
    if candidate_targets is not None:
        raw = list(candidate_targets)
    else:
        raw = list(experiment.reference_targets)
        if not raw:
            raise ValueError(
                "No reference-gene candidates are designated. Mark assays "
                "with is_reference=True or pass candidate_targets explicitly."
            )
    return sorted(dict.fromkeys(raw))


def _resolve_groups(
    experiment: Experiment,
    sample_names: list[str],
    sample_groups: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve explicit groups, then Sample.group, then ``ungrouped``."""
    groups: dict[str, str] = {}
    for name in sample_names:
        if sample_groups is not None and name in sample_groups:
            groups[name] = str(sample_groups[name])
            continue
        sample = experiment.samples.get(name)
        groups[name] = (
            str(sample.group)
            if sample is not None and sample.group is not None
            else "ungrouped"
        )
    return groups


def _replace_negative_variances(
    matrix: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Replace negative corrected estimates using pairwise gene differences."""
    corrected = np.asarray(variances, dtype=float).copy()
    for gene_index in np.flatnonzero(corrected < 0):
        pairwise = [
            np.var(matrix[:, gene_index] - matrix[:, other], ddof=1) / 4.0
            for other in range(matrix.shape[1])
            if other != gene_index
        ]
        corrected[gene_index] = float(np.mean(pairwise))
    return np.maximum(corrected, 0.0)


def _single_group_model(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return single-group stability SDs and corrected residual variances."""
    n_samples, n_genes = matrix.shape
    sample_means = np.mean(matrix, axis=1)
    gene_means = np.mean(matrix, axis=0)
    overall_mean = float(np.mean(matrix))
    residuals = matrix - sample_means[:, None] - gene_means + overall_mean
    raw = np.sum(residuals**2, axis=0) / (n_samples - 1)
    corrected = (raw - np.sum(raw) / (n_genes * (n_genes - 1))) / (
        1.0 - 2.0 / n_genes
    )
    corrected = _replace_negative_variances(matrix, corrected)
    return np.sqrt(corrected), corrected


def _grouped_model(
    matrix: np.ndarray, group_labels: list[str], unique_groups: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the grouped empirical-Bayes model used by NormqPCR."""
    n_genes = matrix.shape[1]
    n_groups = len(unique_groups)
    labels = np.asarray(group_labels)
    sample_means = np.mean(matrix, axis=1)

    group_means = np.vstack(
        [np.mean(matrix[labels == group], axis=0) for group in unique_groups]
    )
    group_sizes = np.asarray(
        [np.count_nonzero(labels == group) for group in unique_groups], dtype=float
    )
    group_overall_means = np.mean(group_means, axis=1)

    group_variances = np.empty((n_groups, n_genes), dtype=float)
    correction_denominator = 1.0 - 2.0 / n_genes
    for group_index, group in enumerate(unique_groups):
        mask = labels == group
        residuals = (
            matrix[mask]
            - group_means[group_index]
            - sample_means[mask, None]
            + group_overall_means[group_index]
        )
        raw = np.sum(residuals**2, axis=0) / (group_sizes[group_index] - 1.0)
        corrected = (
            raw - np.sum(raw) / (n_genes * (n_genes - 1))
        ) / correction_denominator
        # This matches NormqPCR's pmax(..., 0) boundary estimate.
        group_variances[group_index] = np.maximum(corrected, 0.0)

    gene_group_mean = np.mean(group_means, axis=0)
    group_mean = np.mean(group_means, axis=1)
    overall_group_mean = float(np.mean(group_mean))
    group_differences = (
        (group_means - group_mean[:, None]).T
        - gene_group_mean[:, None]
        + overall_group_mean
    )

    mean_variances = group_variances / group_sizes[:, None]
    common_variance = max(
        float(
            np.sum(group_differences**2)
            / ((n_groups - 1) * (n_genes - 1))
            - np.mean(mean_variances)
        ),
        0.0,
    )

    sampling_variances = mean_variances.T
    denominators = common_variance + sampling_variances
    shrinkage = np.divide(
        common_variance,
        denominators,
        out=np.zeros_like(sampling_variances),
        where=denominators > 0,
    )
    shrunken_effects = group_differences * shrinkage
    effect_variances = sampling_variances + np.divide(
        common_variance * sampling_variances,
        denominators,
        out=np.zeros_like(sampling_variances),
        where=denominators > 0,
    )
    stability = np.mean(
        np.abs(shrunken_effects) + np.sqrt(effect_variances), axis=1
    )
    return (
        stability,
        shrunken_effects,
        effect_variances,
        group_variances,
        common_variance,
    )


def variance_stability_score(
    experiment: Experiment,
    candidate_targets: list[str] | None = None,
    sample_groups: dict[str, str] | None = None,
) -> NormFinderResult:
    """Calculate the legacy descriptive within/between-group variance score."""
    candidates = _resolve_candidates(experiment, candidate_targets)
    if len(candidates) < 2:
        raise ValueError(
            "Variance stability scoring requires at least 2 candidate genes, got "
            f"{len(candidates)}."
        )

    sample_names, matrix, _, _ = build_reference_matrix(
        experiment, candidates, efficiency_corrected=False
    )
    if len(sample_names) < 3:
        raise ValueError(
            "Variance stability scoring requires at least 3 samples with complete "
            f"Cq data for all candidate genes, got {len(sample_names)}."
        )

    groups = _resolve_groups(experiment, sample_names, sample_groups)
    group_labels = [groups[name] for name in sample_names]
    unique_groups = sorted(set(group_labels))
    labels = np.asarray(group_labels)

    inter_group_var: dict[str, float] = {}
    intra_group_var: dict[str, float] = {}
    stability_values: dict[str, float] = {}
    for gene_index, gene in enumerate(candidates):
        values = matrix[:, gene_index]
        group_values = [values[labels == group] for group in unique_groups]
        intra = float(np.mean([np.var(values, ddof=0) for values in group_values]))
        inter = float(np.var([np.mean(values) for values in group_values], ddof=0))
        intra_group_var[gene] = intra
        inter_group_var[gene] = inter
        stability_values[gene] = intra + inter

    ranking = sorted(candidates, key=lambda gene: (stability_values[gene], gene))
    return NormFinderResult(
        ranking=ranking,
        stability_values=stability_values,
        inter_group_var=inter_group_var,
        intra_group_var=intra_group_var,
    )


def normfinder(
    experiment: Experiment,
    candidate_targets: list[str] | None = None,
    sample_groups: dict[str, str] | None = None,
) -> ModelBasedNormFinderResult:
    """Estimate model-based NormFinder reference-gene stability.

    At least three candidate genes are required because the corrected residual
    variance contains the factor ``1 - 2/k``. Every candidate requires a
    measured positive amplification efficiency. Grouped analyses require at
    least two complete biological samples in every group. Samples missing a
    candidate Cq are excluded complete-case and listed in ``excluded_samples``.
    """
    candidates = _resolve_candidates(experiment, candidate_targets)
    if len(candidates) < 3:
        raise ValueError(
            "Model-based NormFinder requires at least 3 candidate genes, got "
            f"{len(candidates)}."
        )

    (
        sample_names,
        matrix,
        excluded_samples,
        efficiencies,
    ) = build_reference_matrix(experiment, candidates)
    if len(sample_names) < 3:
        raise ValueError(
            "Model-based NormFinder requires at least 3 samples with complete Cq "
            f"data for all candidate genes, got {len(sample_names)}."
        )

    groups = _resolve_groups(experiment, sample_names, sample_groups)
    group_labels = [groups[name] for name in sample_names]
    unique_groups = sorted(set(group_labels))

    if len(unique_groups) == 1:
        stability, residual_variances = _single_group_model(matrix)
        effects = np.zeros((len(candidates), 1), dtype=float)
        effect_variances = (residual_variances / len(sample_names))[:, None]
        group_variances = residual_variances[None, :]
        common_variance = 0.0
    else:
        group_sizes = {
            group: group_labels.count(group) for group in unique_groups
        }
        undersized = [group for group, size in group_sizes.items() if size < 2]
        if undersized:
            details = ", ".join(
                f"{group!r} ({group_sizes[group]})" for group in undersized
            )
            raise ValueError(
                "Model-based NormFinder requires at least 2 complete samples per "
                f"group; undersized group(s): {details}."
            )
        (
            stability,
            effects,
            effect_variances,
            group_variances,
            common_variance,
        ) = _grouped_model(matrix, group_labels, unique_groups)

    stability_values = {
        gene: float(stability[index]) for index, gene in enumerate(candidates)
    }
    ranking = sorted(candidates, key=lambda gene: (stability_values[gene], gene))

    return ModelBasedNormFinderResult(
        ranking=ranking,
        stability_values=stability_values,
        group_effects={
            gene: {
                group: float(effects[gene_index, group_index])
                for group_index, group in enumerate(unique_groups)
            }
            for gene_index, gene in enumerate(candidates)
        },
        group_effect_variances={
            gene: {
                group: float(effect_variances[gene_index, group_index])
                for group_index, group in enumerate(unique_groups)
            }
            for gene_index, gene in enumerate(candidates)
        },
        group_specific_variances={
            gene: {
                group: float(group_variances[group_index, gene_index])
                for group_index, group in enumerate(unique_groups)
            }
            for gene_index, gene in enumerate(candidates)
        },
        common_group_variance=float(common_variance),
        excluded_samples=excluded_samples,
        n_samples=len(sample_names),
        n_groups=len(unique_groups),
        efficiencies=efficiencies,
    )
