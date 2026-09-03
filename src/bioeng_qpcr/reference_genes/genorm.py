"""geNorm reference-gene stability analysis.

Implements the geNorm algorithm (Vandesompele et al., 2002) for identifying
the most stably expressed reference (housekeeping) genes from qPCR data.

Reference
---------
Vandesompele J, De Preter K, Pattyn F, et al.  Accurate normalisation of
real-time quantitative RT-PCR data by geometric averaging of multiple
internal control genes.  Genome Biol. 2002;3(7):research0034.

Algorithm overview
------------------
For each candidate gene *j* geNorm computes an expression-stability measure
*M_j*:

1.  For every pair of genes (j, k) the log2-transformed expression ratio is
    obtained for each sample. Candidate Cq values are converted with their
    measured amplification efficiencies::

        log2(Q_j) = -Cq_j * log2(1 + efficiency_j)

    At 100% efficiency this reduces to ``-Cq_j``. The sign of a pairwise
    difference is irrelevant for its standard deviation.

2.  *V_{j/k}* -- the standard deviation of that log2 ratio across samples --
    quantifies how much the two genes disagree.

3.  *M_j* = mean of *V_{j/k}* over all k != j.  Lower *M* = more stable.

4.  The gene with the highest *M* (least stable) is removed stepwise and *M*
    is recomputed for the remaining genes until only two genes remain.  Those
    two constitute the most stable pair.  The *M* value reported for each gene
    is the one computed in the elimination step at which it was dropped (or
    the final-step *M* for the two survivors).

The pairwise variation *V_{n/n+1}* indicates whether adding an (n+1)-th gene
to the *n* most stable genes materially changes the normalisation factor.
A threshold of 0.15 is conventionally used to decide the optimal number of
reference genes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.reference_genes._matrix import build_reference_matrix

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GeNormResult:
    """Result of a geNorm stability analysis.

    Attributes
        ranking:
            Candidate genes ordered from most stable to least stable.
        m_values:
            *M* stability value for each gene (lower = more stable).  The
            value corresponds to the elimination step at which the gene was
            dropped, or the final step for the two surviving genes.
        pairwise_variation:
            *V_{n/n+1}* pairwise variation keyed by ``"V{n}/{n+1}"`` (e.g.
            ``"V2/3"``).  Empty when fewer than three candidate genes are
            evaluated.
        recommended_count:
            The first *n* for which *V_{n/n+1}* < 0.15.  ``None`` means the
            threshold was not reached by the evaluated candidate set.  With
            exactly two candidates, the value is necessarily 2 because no
            pairwise-variation comparison is possible.
    """

    ranking: list[str]
    m_values: dict[str, float]
    pairwise_variation: dict[str, float]
    recommended_count: int | None
    efficiencies: dict[str, float] = field(default_factory=dict)
    excluded_samples: list[str] = field(default_factory=list)
    quantity_scale: str = "log2_relative_quantity_efficiency_corrected"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Conventional geNorm cut-off for the pairwise variation.
_PAIRWISE_THRESHOLD = 0.15


def _resolve_candidates(
    experiment: Experiment, candidate_targets: list[str] | None
) -> list[str]:
    """Resolve the list of candidate reference genes.

    If *candidate_targets* is ``None`` the registered reference genes are
    used.  Targets of interest are never silently promoted to reference-gene
    candidates.  Returns a sorted list of unique candidate names.
    """
    if candidate_targets is not None:
        raw = list(candidate_targets)
    else:
        raw = list(experiment.reference_targets)
        if not raw:
            raise ValueError(
                "No reference-gene candidates are designated. Mark assays "
                "with is_reference=True or pass candidate_targets explicitly."
            )
    seen: set[str] = set()
    unique: list[str] = []
    for name in raw:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return sorted(unique)


def _compute_m_values(matrix: np.ndarray, gene_indices: list[int]) -> dict[int, float]:
    """Compute the geNorm *M* value for each gene index in *gene_indices*.

    *matrix* has shape ``(n_samples, n_genes)``.  Only the columns listed in
    *gene_indices* participate in the pairwise comparison.  Standard
    deviations use the population estimator (``ddof=0``).
    """
    m: dict[int, float] = {}
    for j in gene_indices:
        v_values: list[float] = []
        for k in gene_indices:
            if k == j:
                continue
            # Difference of log2 relative quantities is the log2 ratio.
            log_ratio = matrix[:, j] - matrix[:, k]
            v_values.append(float(np.std(log_ratio, ddof=0)))
        m[j] = float(np.mean(v_values)) if v_values else 0.0
    return m


def _compute_pairwise_variation(
    matrix: np.ndarray, ranking: list[str], gene_index: dict[str, int]
) -> dict[str, float]:
    """Compute *V_{n/n+1}* for n = 2 .. (n_genes - 1).

    *ranking* lists gene names from most to least stable.  *V_{n/n+1}* is the
    standard deviation across samples of the change in the (log2) normalisation
    factor when the (n+1)-th gene is added to the top-n set.  The log2
    normalisation factor is the arithmetic mean of the log2 quantities (which
    equals the log2 of the geometric mean of relative quantities).
    """
    pairwise: dict[str, float] = {}
    n_genes = len(ranking)
    for n in range(2, n_genes):
        top_n_idx = [gene_index[ranking[i]] for i in range(n)]
        top_n1_idx = [gene_index[ranking[i]] for i in range(n + 1)]
        nf_n = matrix[:, top_n_idx].mean(axis=1)
        nf_n1 = matrix[:, top_n1_idx].mean(axis=1)
        diff = nf_n - nf_n1
        pairwise[f"V{n}/{n + 1}"] = float(np.std(diff, ddof=0))
    return pairwise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def genorm(
    experiment: Experiment, candidate_targets: list[str] | None = None
) -> GeNormResult:
    """Run the geNorm reference-gene stability analysis.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        candidate_targets:
            Candidate reference genes to evaluate.  If ``None`` the
            experiment's registered reference genes are used (or every target
            when none are registered).

    Returns
        :class:`GeNormResult` with the stability ranking and measures.

    Raises
        ValueError:
            If fewer than 2 candidate genes are available, or fewer than 3
            samples with complete Cq data remain, or a candidate lacks a
            measured positive amplification efficiency.
    """
    candidates = _resolve_candidates(experiment, candidate_targets)
    if len(candidates) < 2:
        raise ValueError(
            f"geNorm requires at least 2 candidate genes, got {len(candidates)}."
        )

    sample_names, matrix, excluded_samples, efficiencies = build_reference_matrix(
        experiment, candidates
    )
    if len(sample_names) < 3:
        raise ValueError(
            f"geNorm requires at least 3 samples with complete Cq data for "
            f"all candidate genes, got {len(sample_names)}."
        )

    n_genes = len(candidates)
    gene_index = {name: i for i, name in enumerate(candidates)}
    remaining = list(range(n_genes))

    m_values: dict[str, float] = {}
    elimination_order: list[int] = []

    # Stepwise elimination until 2 genes remain.
    while len(remaining) > 2:
        m = _compute_m_values(matrix, remaining)
        # Least stable = highest M.  Ties are broken by gene name so that the
        # alphabetically-later gene is removed first (deterministic output).
        worst = max(remaining, key=lambda g: (m[g], candidates[g]))
        m_values[candidates[worst]] = m[worst]
        elimination_order.append(worst)
        remaining.remove(worst)

    # Final 2 genes: record their M at the last step.
    m = _compute_m_values(matrix, remaining)
    for g in remaining:
        m_values[candidates[g]] = m[g]

    # Build ranking: most stable first.
    # The final 2 are most stable (ordered by ascending M, ties broken by
    # name); then the elimination order reversed (last removed = next most
    # stable).
    final_two = sorted(
        remaining, key=lambda g: (m_values[candidates[g]], candidates[g])
    )
    ranking_idx = final_two + list(reversed(elimination_order))
    ranking = [candidates[g] for g in ranking_idx]

    pairwise_variation = _compute_pairwise_variation(matrix, ranking, gene_index)

    # Recommended count: first n where V_{n/n+1} < 0.15.  If the threshold
    # is never reached, do not return an evidence-free default of two.
    recommended_count: int | None = 2 if n_genes == 2 else None
    for n in range(2, n_genes):
        key = f"V{n}/{n + 1}"
        if key in pairwise_variation and pairwise_variation[key] < _PAIRWISE_THRESHOLD:
            recommended_count = n
            break

    return GeNormResult(
        ranking=ranking,
        m_values=m_values,
        pairwise_variation=pairwise_variation,
        recommended_count=recommended_count,
        efficiencies=efficiencies,
        excluded_samples=excluded_samples,
    )
