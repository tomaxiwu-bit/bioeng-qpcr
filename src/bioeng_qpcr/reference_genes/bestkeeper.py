"""BestKeeper reference-gene stability analysis.

Implements the BestKeeper approach (Pfaffl et al., 2004) for assessing
candidate reference (housekeeping) genes from qPCR data.

Reference
---------
Pfaffl MW, Tichopad A, Prgomet C, Neuvians TP.  Determination of stable
housekeeping genes, differentially regulated target genes and sample
integrity: BestKeeper — Excel-based tool using pair-wise correlations.
Biotechnol Lett. 2004;26(6):509-515.

Algorithm overview
------------------
BestKeeper works on raw crossing points (Cq) and does not apply amplification-
efficiency correction.  For every candidate gene the descriptive statistics of
its Cq across samples are computed (arithmetic mean, standard deviation and
coefficient of variation).  A BestKeeper index is formed per sample as the
geometric mean of the Cq values across all candidate genes, and each gene is
then correlated (Pearson) with that index.

A gene whose Cq is very constant across samples (low standard deviation) and
highly correlated with the index is considered stable.  Genes whose Cq
standard deviation exceeds a conventional threshold are flagged as
inconsistent and candidates for exclusion from normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.reference_genes._matrix import (
    build_reference_matrix,
    resolve_candidate_genes,
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BestKeeperResult:
    """Result of a BestKeeper reference-gene stability analysis.

    Attributes
        ranking:
            Candidate genes ordered from most stable (lowest Cq standard
            deviation) to least stable.
        ct_mean:
            Arithmetic mean Cq of each gene across samples.
        ct_sd:
            Sample standard deviation of Cq of each gene across samples.
        ct_cv:
            Coefficient of variation of Cq (percent) of each gene.
        bestkeeper_index:
            BestKeeper index for each sample: the geometric mean of Cq across
            the candidate genes.
        pearson_r:
            Pearson correlation of each gene's Cq with the BestKeeper index.
            ``None`` when the gene's Cq has zero variance across samples.
        consistent:
            ``True`` when a gene's Cq standard deviation does not exceed
            *sd_threshold*.
        excluded_samples:
            Samples removed because they lacked a determined Cq for at least
            one candidate.
        sd_threshold:
            Conventional Cq standard-deviation cutoff for consistency.
        quantity_scale:
            Scale of the quantity matrix used for the computation.
    """

    ranking: list[str]
    ct_mean: dict[str, float]
    ct_sd: dict[str, float]
    ct_cv: dict[str, float]
    bestkeeper_index: dict[str, float]
    pearson_r: dict[str, float | None]
    consistent: dict[str, bool]
    excluded_samples: list[str] = field(default_factory=list)
    sd_threshold: float = 1.0
    quantity_scale: str = "cq"


# Conventional BestKeeper cutoff for Cq consistency (in Cq units).
_DEFAULT_SD_THRESHOLD = 1.0


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation coefficient, ``None`` if either input has no
    variance or there are fewer than two finite points."""
    if x.size < 2 or y.size < 2:
        return None
    x_var = np.var(x)
    y_var = np.var(y)
    tolerance = 1e-12
    if x_var <= tolerance or y_var <= tolerance:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bestkeeper(
    experiment: Experiment,
    candidate_targets: list[str] | None = None,
    *,
    sd_threshold: float = _DEFAULT_SD_THRESHOLD,
) -> BestKeeperResult:
    """Run the BestKeeper reference-gene stability analysis.

    Parameters
        experiment:
            The qPCR experiment containing wells and assays.
        candidate_targets:
            Candidate reference genes to evaluate.  If ``None`` the
            experiment's registered reference genes are used.  At least two
            candidates must be designated explicitly or on the assays.
        sd_threshold:
            Conventional Cq standard-deviation cutoff used to flag
            inconsistent genes.  Defaults to 1.0 (Cq).

    Returns
        :class:`BestKeeperResult` with the descriptive statistics, the
        BestKeeper index, correlations, and a consistency flag per gene.

    Raises
        ValueError:
            If fewer than 2 candidate genes are available or fewer than 3
            samples with complete Cq data remain, or *sd_threshold* is not
            positive.

    Note
    ----
    BestKeeper is defined on raw crossing points, so amplification
    efficiencies are **not** required or applied.  This contrasts with
    geNorm, NormFinder and delta-Ct, all of which operate on the
    efficiency-corrected log2 relative-quantity scale.
    """
    if sd_threshold <= 0:
        raise ValueError("sd_threshold must be positive.")

    candidates = resolve_candidate_genes(experiment, candidate_targets)
    if len(candidates) < 2:
        raise ValueError(
            f"BestKeeper requires at least 2 candidate genes, got {len(candidates)}."
        )

    # Raw Cq matrix (no efficiency correction, consistent with the method).
    sample_names, matrix, excluded_samples, _ = build_reference_matrix(
        experiment, candidates, efficiency_corrected=False
    )
    if len(sample_names) < 3:
        raise ValueError(
            "BestKeeper requires at least 3 samples with complete Cq data "
            f"for all candidate genes, got {len(sample_names)}."
        )

    ct_mean: dict[str, float] = {}
    ct_sd: dict[str, float] = {}
    ct_cv: dict[str, float] = {}
    for col, gene in enumerate(candidates):
        values = matrix[:, col]
        mean = float(np.mean(values))
        # BestKeeper reports the sample standard deviation (ddof=1).
        sd = float(np.std(values, ddof=1))
        ct_mean[gene] = mean
        ct_sd[gene] = sd
        ct_cv[gene] = sd / mean * 100.0 if mean != 0 else 0.0

    # BestKeeper index per sample: geometric mean of Cq across candidates.
    # A geometric mean is only defined for strictly positive values, and any
    # non-positive crossing point is non-physical, so fail rather than silently
    # produce a nonsensical index.
    if np.any(matrix <= 0.0):
        raise ValueError(
            "BestKeeper indexes require strictly positive Cq values, but the "
            "crossing-point matrix contains non-positive entries."
        )
    index_per_sample = np.exp(np.mean(np.log(matrix), axis=1))
    bestkeeper_index = dict(zip(sample_names, index_per_sample, strict=True))

    pearson_r: dict[str, float | None] = {}
    for col, gene in enumerate(candidates):
        pearson_r[gene] = _pearson(matrix[:, col], index_per_sample)

    consistent = {
        gene: ct_sd[gene] <= sd_threshold for gene in candidates
    }

    # Lower Cq standard deviation = more stable.  Ties broken by gene name.
    ranking = sorted(candidates, key=lambda g: (ct_sd[g], g))

    return BestKeeperResult(
        ranking=ranking,
        ct_mean=ct_mean,
        ct_sd=ct_sd,
        ct_cv=ct_cv,
        bestkeeper_index=bestkeeper_index,
        pearson_r=pearson_r,
        consistent=consistent,
        excluded_samples=excluded_samples,
        sd_threshold=sd_threshold,
    )