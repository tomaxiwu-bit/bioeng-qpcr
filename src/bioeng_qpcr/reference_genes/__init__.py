"""Reference-gene stability analysis module.

This package provides algorithms for evaluating and selecting stably
expressed reference (housekeeping) genes from qPCR data:

* **geNorm** (:mod:`~bioeng_qpcr.reference_genes.genorm`) -- pairwise
  stability measure *M* with stepwise elimination (Vandesompele et al., 2002).
* **NormFinder** (:mod:`~bioeng_qpcr.reference_genes.normfinder`) -- a
  model-based empirical-Bayes group stability estimator.
* **Variance stability score** -- the earlier descriptive within/between-group
  score, retained under an explicit name for compatibility.
* **delta-Ct** (:mod:`~bioeng_qpcr.reference_genes.delta_ct`) -- stability
  from the geometric mean of pairwise delta-Ct standard deviations
  (Silver et al., 2006).
* **BestKeeper** (:mod:`~bioeng_qpcr.reference_genes.bestkeeper`) -- raw-Cq
  descriptive statistics, Cq consistency flag, and correlation with the
  BestKeeper index (Pfaffl et al., 2004).
* **Consensus** (:mod:`~bioeng_qpcr.reference_genes.consensus`) -- rank
  aggregation of the selected stability methods into a single recommendation.

All operate on the :class:`~bioeng_qpcr.domain.entities.Experiment`
domain model and use numpy for numerical computation.
"""

from __future__ import annotations

from bioeng_qpcr.reference_genes.bestkeeper import (
    BestKeeperResult,
    bestkeeper,
)
from bioeng_qpcr.reference_genes.consensus import (
    ConsensusResult,
    consensus_ranking,
)
from bioeng_qpcr.reference_genes.delta_ct import DeltaCtResult, delta_ct
from bioeng_qpcr.reference_genes.genorm import GeNormResult, genorm
from bioeng_qpcr.reference_genes.normfinder import (
    ModelBasedNormFinderResult,
    NormFinderResult,
    normfinder,
    variance_stability_score,
)

__all__ = [
    "BestKeeperResult",
    "ConsensusResult",
    "DeltaCtResult",
    "GeNormResult",
    "ModelBasedNormFinderResult",
    "NormFinderResult",
    "bestkeeper",
    "consensus_ranking",
    "delta_ct",
    "genorm",
    "normfinder",
    "variance_stability_score",
]
