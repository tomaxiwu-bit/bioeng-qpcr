"""Consensus ranking of reference genes.

Combines geNorm with model-based NormFinder into a single consensus ranking
using rank aggregation. For each candidate gene,
the rank position (1 = best) from each method is averaged; genes are ordered by
this
consensus score (lower = better).  Ties are broken by the geNorm rank and
then by gene name for deterministic output.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.reference_genes.genorm import genorm
from bioeng_qpcr.reference_genes.normfinder import normfinder

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsensusResult:
    """Result of a consensus reference-gene ranking.

    Attributes
        ranking:
            Candidate genes ordered by consensus score (most stable first).
        consensus_scores:
            Average rank across geNorm and NormFinder for each gene
            (lower = better; 1.0 = first in both methods).
        genorm_ranking:
            Ranking produced by geNorm (most stable first).
        normfinder_ranking:
            Ranking produced by NormFinder (most stable first).
        recommended_genes:
            Top *N* genes from the consensus ranking, where *N* is the
            geNorm recommended reference-gene count.  Empty when the
            pairwise-variation threshold was not reached.
    """

    ranking: list[str]
    consensus_scores: dict[str, float]
    genorm_ranking: list[str]
    normfinder_ranking: list[str]
    recommended_genes: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def consensus_ranking(
    experiment: Experiment,
    candidate_targets: list[str] | None = None,
    sample_groups: dict[str, str] | None = None,
) -> ConsensusResult:
    """Compute a consensus reference-gene ranking.

    Runs both geNorm and NormFinder on the same experiment and aggregates
    their per-gene rank positions into a consensus score.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        candidate_targets:
            Candidate reference genes to evaluate.  If ``None`` the
            experiment's registered reference genes are used.  At least two
            candidates must be designated explicitly or on the assays.
        sample_groups:
            Optional mapping of sample name -> group name forwarded to
            NormFinder.

    Returns
        :class:`ConsensusResult` with the aggregated ranking.

    Raises
        ValueError:
            Propagated from geNorm or NormFinder when the inputs do not meet
            their minimum requirements (fewer than 3 candidate genes, fewer
            than 3 complete samples, or fewer than 2 samples in a group).
    """
    ge_result = genorm(experiment, candidate_targets)
    nf_result = normfinder(experiment, candidate_targets, sample_groups)

    genorm_rank = {gene: i + 1 for i, gene in enumerate(ge_result.ranking)}
    normfinder_rank = {gene: i + 1 for i, gene in enumerate(nf_result.ranking)}

    all_genes = sorted(set(ge_result.ranking) | set(nf_result.ranking))

    consensus_scores: dict[str, float] = {}
    for gene in all_genes:
        # Both rankings are guaranteed to contain every candidate gene, so
        # the lookups below never miss.
        consensus_scores[gene] = (genorm_rank[gene] + normfinder_rank[gene]) / 2.0

    ranking = sorted(
        all_genes,
        key=lambda g: (consensus_scores[g], genorm_rank[g], g),
    )

    recommended_count = ge_result.recommended_count
    recommended_genes = (
        ranking[:recommended_count] if recommended_count is not None else []
    )

    return ConsensusResult(
        ranking=ranking,
        consensus_scores=consensus_scores,
        genorm_ranking=list(ge_result.ranking),
        normfinder_ranking=list(nf_result.ranking),
        recommended_genes=recommended_genes,
    )
