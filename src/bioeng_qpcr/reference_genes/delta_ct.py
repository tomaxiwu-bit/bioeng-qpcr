"""delta-Ct reference-gene stability analysis.

Implements the delta-Ct method (Silver et al., 2006) for ranking candidate
reference (housekeeping) genes from qPCR data.

Reference
---------
Silver N, Best S, Jiang J, Thein SL.  Selection of housekeeping genes for gene
expression studies in human reticulocytes using real-time PCR.  BMC Mol Biol.
2006;7:33.

Algorithm overview
------------------
For every pair of candidate genes (j, k), the delta-Ct between the two genes
is computed for each sample and its standard deviation across samples is
recorded.  A pair whose delta-Ct is nearly constant expresses the two genes at
a constant ratio across conditions, which is the hallmark of stable reference
genes.  The stability measure assigned to a gene is the geometric mean of the
pairwise standard deviations involving that gene; lower is more stable.

As elsewhere in this module, candidate Cq values are converted to the
efficiency-corrected log2 relative-quantity scale
(``-Cq * log2(1 + efficiency)``).  At 100% amplification efficiency this
reduces to ``-Cq``, so the pairwise-standard-deviation ranking is identical to
the conventional raw-Ct form of the method while remaining consistent with the
scale used by geNorm and NormFinder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.reference_genes._matrix import build_reference_matrix

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeltaCtResult:
    """Result of a delta-Ct reference-gene stability analysis.

    Attributes
        ranking:
            Candidate genes ordered from most stable to least stable.
        stability_values:
            delta-Ct stability measure for each gene (lower = more stable):
            the geometric mean, across all pairing partners, of the standard
            deviation of the log2 relative-quantity difference.
        pairwise_sd:
            Standard deviation of the log2 relative-quantity difference for
            each evaluated pair, keyed by ``(gene1, gene2)`` with gene names
            sorted alphabetically.
        excluded_samples:
            Samples removed because they lacked a determined Cq for at least
            one candidate.
        efficiencies:
            Measured amplification efficiencies used to build the
            quantity matrix.
        quantity_scale:
            Scale of the quantity matrix used for the computation.
    """

    ranking: list[str]
    stability_values: dict[str, float]
    pairwise_sd: dict[tuple[str, str], float]
    excluded_samples: list[str] = field(default_factory=list)
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


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean of non-negative values.

    A single zero factor drives the geometric mean to zero, which is the
    behaviour Silver-style stability wants: a gene that is perfectly co-constant
    with any partner is maximally stable, so its overall measure converges to 0.
    """
    values = [v for v in values if v >= 0]
    if not values or any(v == 0.0 for v in values):
        return 0.0
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delta_ct(
    experiment: Experiment, candidate_targets: list[str] | None = None
) -> DeltaCtResult:
    """Run the delta-Ct reference-gene stability analysis.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        candidate_targets:
            Candidate reference genes to evaluate.  If ``None`` the
            experiment's registered reference genes are used.  At least two
            candidates must be designated explicitly or on the assays.

    Returns
        :class:`DeltaCtResult` with the stability ranking and pairwise
        measures.

    Raises
        ValueError:
            If fewer than 2 candidate genes are available, fewer than 3
            samples with complete Cq data remain, or a candidate lacks a
            measured positive amplification efficiency.
    """
    candidates = _resolve_candidates(experiment, candidate_targets)
    if len(candidates) < 2:
        raise ValueError(
            f"delta-Ct requires at least 2 candidate genes, got {len(candidates)}."
        )

    sample_names, matrix, excluded_samples, efficiencies = build_reference_matrix(
        experiment, candidates
    )
    if len(sample_names) < 3:
        raise ValueError(
            "delta-Ct requires at least 3 samples with complete Cq data for "
            f"all candidate genes, got {len(sample_names)}."
        )

    n_genes = len(candidates)
    # Standard deviation of the log2 relative-quantity difference for each
    # unordered pair, keyed by alphabetically sorted gene names.
    pairwise_sd: dict[tuple[str, str], float] = {}
    partner_sd: dict[str, list[float]] = {gene: [] for gene in candidates}
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            gi, gj = candidates[i], candidates[j]
            sd = float(np.std(matrix[:, i] - matrix[:, j], ddof=0))
            pairwise_sd[(gi, gj)] = sd
            partner_sd[gi].append(sd)
            partner_sd[gj].append(sd)

    stability_values = {
        gene: _geometric_mean(partner_sd[gene]) for gene in candidates
    }
    # Lower stability value = more stable.  Ties broken by gene name for
    # deterministic output.
    ranking = sorted(candidates, key=lambda g: (stability_values[g], g))

    return DeltaCtResult(
        ranking=ranking,
        stability_values=stability_values,
        pairwise_sd=dict(sorted(pairwise_sd.items())),
        excluded_samples=excluded_samples,
        efficiencies=efficiencies,
    )