"""Reference-gene stability analysis module.

This package provides algorithms for evaluating and selecting stably
expressed reference (housekeeping) genes from qPCR data:

* **geNorm** (:mod:`~bioeng_qpcr.reference_genes.genorm`) -- pairwise
  stability measure *M* with stepwise elimination (Vandesompele et al., 2002).
* **NormFinder** (:mod:`~bioeng_qpcr.reference_genes.normfinder`) -- a
  model-based empirical-Bayes group stability estimator.
* **Variance stability score** -- the earlier descriptive within/between-group
  score, retained under an explicit name for compatibility.
* **Consensus** (:mod:`~bioeng_qpcr.reference_genes.consensus`) -- rank
  aggregation of geNorm and NormFinder into a single recommendation.

All three operate on the :class:`~bioeng_qpcr.domain.entities.Experiment`
domain model and use numpy for numerical computation.
"""

from __future__ import annotations

from bioeng_qpcr.reference_genes.consensus import (
    ConsensusResult,
    consensus_ranking,
)
from bioeng_qpcr.reference_genes.genorm import GeNormResult, genorm
from bioeng_qpcr.reference_genes.normfinder import (
    ModelBasedNormFinderResult,
    NormFinderResult,
    normfinder,
    variance_stability_score,
)

__all__ = [
    "GeNormResult",
    "genorm",
    "NormFinderResult",
    "ModelBasedNormFinderResult",
    "normfinder",
    "variance_stability_score",
    "ConsensusResult",
    "consensus_ranking",
]
