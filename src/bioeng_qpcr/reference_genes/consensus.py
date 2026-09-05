"""Consensus ranking of reference genes.

Combines multiple reference-gene stability methods into a single consensus
ranking using rank aggregation.  For each candidate gene, the rank position
(1 = best) from each selected method is averaged; genes are ordered by this
consensus score (lower = better).  Ties are broken by the rank of the first
selected method and then by gene name for deterministic output.

By default the consensus combines geNorm and model-based NormFinder, matching
the historical behaviour.  ``delta_ct`` and ``bestkeeper`` may be added (or
used on their own) via the ``methods`` argument.

Note that geNorm, NormFinder and delta-Ct require measured positive
amplification efficiencies for every candidate, whereas BestKeeper operates
on raw Cq and does not.  A request that mixes BestKeeper with an
efficiency-requiring method therefore still needs efficiencies for all
candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.reference_genes.bestkeeper import bestkeeper
from bioeng_qpcr.reference_genes.delta_ct import delta_ct
from bioeng_qpcr.reference_genes.genorm import genorm
from bioeng_qpcr.reference_genes.normfinder import normfinder

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_METHODS = {"genorm", "normfinder", "delta_ct", "bestkeeper"}
_DEFAULT_METHODS = ("genorm", "normfinder")


@dataclass
class ConsensusResult:
    """Result of a consensus reference-gene ranking.

    Attributes
        ranking:
            Candidate genes ordered by consensus score (most stable first).
        consensus_scores:
            Average rank across the selected methods for each gene
            (lower = better; 1.0 = first in every method).
        methods:
            The stability methods that were aggregated.
        genorm_ranking:
            Ranking produced by geNorm (most stable first), or empty if
            geNorm was not among the selected methods.
        normfinder_ranking:
            Ranking produced by NormFinder, or empty if not selected.
        delta_ct_ranking:
            Ranking produced by delta-Ct, or empty if not selected.
        bestkeeper_ranking:
            Ranking produced by BestKeeper, or empty if not selected.
        recommended_genes:
            Top *N* genes from the consensus ranking, where *N* is the
            geNorm recommended reference-gene count.  Empty when geNorm was
            not one of the selected methods or when the pairwise-variation
            threshold was not reached.
    """

    ranking: list[str]
    consensus_scores: dict[str, float]
    genorm_ranking: list[str] = field(default_factory=list)
    normfinder_ranking: list[str] = field(default_factory=list)
    delta_ct_ranking: list[str] = field(default_factory=list)
    bestkeeper_ranking: list[str] = field(default_factory=list)
    recommended_genes: list[str] = field(default_factory=list)
    methods: tuple[str, ...] = _DEFAULT_METHODS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def consensus_ranking(
    experiment: Experiment,
    candidate_targets: list[str] | None = None,
    sample_groups: dict[str, str] | None = None,
    methods: tuple[str, ...] = _DEFAULT_METHODS,
) -> ConsensusResult:
    """Compute a consensus reference-gene ranking.

    Runs the selected stability methods on the same experiment and aggregates
    their per-gene rank positions into a consensus score.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        candidate_targets:
            Candidate reference genes to evaluate.  If ``None`` the
            experiment's registered reference genes are used.
        sample_groups:
            Optional mapping of sample name -> group name forwarded to
            NormFinder.
        methods:
            Stability methods to aggregate, chosen from
            ``"genorm"``, ``"normfinder"``, ``"delta_ct"`` and
            ``"bestkeeper"``.  Defaults to ``("genorm", "normfinder")``.

    Returns
        :class:`ConsensusResult` with the aggregated ranking.

    Raises
        ValueError:
            If no methods are selected, if an unknown method is requested, or
            if one of the underlying methods rejects the inputs (fewer than 3
            candidate genes, fewer than 3 complete samples, or -- for the
            efficiency-requiring methods -- a missing efficiency).
    """
    unknown = set(methods) - _METHODS
    if unknown:
        raise ValueError(
            "Unknown reference-gene method(s): " + ", ".join(sorted(unknown))
        )
    if not methods:
        raise ValueError("At least one aggregation method must be selected.")

    per_method_ranking: dict[str, list[str]] = {}
    ge_result = None
    for method in methods:
        if method == "genorm":
            result = genorm(experiment, candidate_targets)
        elif method == "normfinder":
            result = normfinder(experiment, candidate_targets, sample_groups)
        elif method == "delta_ct":
            result = delta_ct(experiment, candidate_targets)
        else:  # bestkeeper
            result = bestkeeper(experiment, candidate_targets)
        per_method_ranking[method] = list(result.ranking)
        if method == "genorm":
            ge_result = result

    recommended_count = ge_result.recommended_count if ge_result is not None else None

    rank_position = {
        method: {gene: i + 1 for i, gene in enumerate(per_method_ranking[method])}
        for method in methods
    }

    all_genes = sorted(
        set().union(*(set(per_method_ranking[m]) for m in methods))
    )

    consensus_scores: dict[str, float] = {}
    for gene in all_genes:
        scores = [rank_position[m][gene] for m in methods]
        consensus_scores[gene] = sum(scores) / len(scores)

    primary = methods[0]
    ranking = sorted(
        all_genes,
        key=lambda g: (consensus_scores[g], rank_position[primary][g], g),
    )

    recommended_genes = (
        ranking[:recommended_count] if recommended_count is not None else []
    )

    def _optional_ranking(method: str) -> list[str]:
        return list(per_method_ranking[method]) if method in methods else []

    return ConsensusResult(
        ranking=ranking,
        consensus_scores=consensus_scores,
        genorm_ranking=_optional_ranking("genorm"),
        normfinder_ranking=_optional_ranking("normfinder"),
        delta_ct_ranking=_optional_ranking("delta_ct"),
        bestkeeper_ranking=_optional_ranking("bestkeeper"),
        recommended_genes=recommended_genes,
        methods=tuple(methods),
    )