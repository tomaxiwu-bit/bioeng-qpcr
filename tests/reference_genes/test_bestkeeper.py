"""Tests for the BestKeeper reference-gene stability analysis.

Verifies the raw-Cq (efficiency-free) descriptive statistics, the per-sample
BestKeeper index, the Pearson correlation with that index, and the
consistency flag produced by a conventional Cq standard-deviation cutoff.
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.reference_genes.bestkeeper import BestKeeperResult, bestkeeper


def _make_experiment(
    gene_cqs: dict[str, list[float | None]],
    *,
    efficiencies: dict[str, float] | None = None,
) -> Experiment:
    """Build a synthetic experiment from a ``{gene: [cq_per_sample]}`` map."""
    n_samples = len(next(iter(gene_cqs.values())))
    sample_names = [f"S{i + 1}" for i in range(n_samples)]
    wells: list[Well] = []
    pos = 0
    for gene, cqs in gene_cqs.items():
        for sname, cq in zip(sample_names, cqs, strict=True):
            pos += 1
            wells.append(
                Well(
                    position=f"A{pos:02d}",
                    sample_name=sname,
                    target_name=gene,
                    cq=cq,
                    cq_status="undetermined" if cq is None else "determined",
                )
            )
    effs = efficiencies or {}
    assays = {
        gene: Assay(name=gene, efficiency=effs.get(gene, 1.0), is_reference=True)
        for gene in gene_cqs
    }
    samples = {sname: Sample(name=sname) for sname in sample_names}
    return Experiment(
        name="bestkeeper_test", wells=wells, samples=samples, assays=assays
    )


# ---------------------------------------------------------------------------
# Descriptive statistics on raw Cq
# ---------------------------------------------------------------------------


class TestDescriptiveStatistics:
    def test_returns_result_type(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
            }
        )
        assert isinstance(bestkeeper(exp), BestKeeperResult)

    def test_sample_standard_deviation(self) -> None:
        # B = [20,22,24] has sample SD 2.0 (ddof=1).
        result = bestkeeper(
            _make_experiment(
                {"A": [20.0, 20.0, 20.0], "B": [20.0, 22.0, 24.0]}
            )
        )
        assert result.ct_sd["B"] == pytest.approx(2.0)

    def test_coefficient_of_variation_is_percent(self) -> None:
        exp = _make_experiment(
            {"A": [20.0, 20.0, 20.0], "B": [20.0, 22.0, 24.0]}
        )
        result = bestkeeper(exp)
        # mean(B)=22, sd=2 -> CV = 2/22*100.
        assert result.ct_cv["B"] == pytest.approx(2.0 / 22.0 * 100.0)
        assert result.ct_cv["A"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BestKeeper index and correlation
# ---------------------------------------------------------------------------


class TestIndexAndCorrelation:
    def test_index_is_geometric_mean_across_genes(self) -> None:
        exp = _make_experiment(
            {
                "A": [20.0, 21.0, 22.0],
                "B": [30.0, 31.0, 32.0],
            }
        )
        result = bestkeeper(exp)
        # index[S1] = sqrt(20*30), etc. (a geometric mean, not an average).
        assert result.bestkeeper_index["S1"] == pytest.approx((20 * 30) ** 0.5)
        assert result.bestkeeper_index["S2"] == pytest.approx((21 * 31) ** 0.5)
        assert result.bestkeeper_index["S3"] == pytest.approx((22 * 32) ** 0.5)

    def test_perfectly_correlated_gene_has_r_one(self) -> None:
        exp = _make_experiment(
            {
                "A": [20.0, 21.0, 22.0],
                "B": [30.0, 31.0, 32.0],
            }
        )
        result = bestkeeper(exp)
        assert result.pearson_r["A"] == pytest.approx(1.0)
        assert result.pearson_r["B"] == pytest.approx(1.0)

    def test_constant_gene_has_none_correlation(self) -> None:
        exp = _make_experiment(
            {
                "A": [20.0, 20.0, 20.0],
                "B": [20.0, 21.0, 22.0],
            }
        )
        result = bestkeeper(exp)
        # A has zero variance -> undefined correlation. B tracks the geometric-
        # mean index closely (though the index also contains the constant A, so
        # the correlation is not exactly unity).
        assert result.pearson_r["A"] is None
        assert result.pearson_r["B"] > 0.99


# ---------------------------------------------------------------------------
# Ranking and consistency
# ---------------------------------------------------------------------------


class TestRankingAndConsistency:
    def test_constant_gene_ranked_first(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
            }
        )
        result = bestkeeper(exp)
        assert result.ranking[0] == "ACTB"

    def test_consistent_flag_uses_default_threshold(self) -> None:
        exp = _make_experiment(
            {
                "STABLE": [20.0, 20.1, 19.9, 20.0],
                "UNSTABLE": [15.0, 20.0, 25.0, 30.0],
            }
        )
        result = bestkeeper(exp)
        assert result.consistent["STABLE"] is True
        assert result.consistent["UNSTABLE"] is False

    def test_custom_threshold(self) -> None:
        exp = _make_experiment(
            {
                "STABLE": [20.0, 20.1, 19.9, 20.0],
                "UNSTABLE": [15.0, 20.0, 25.0, 30.0],
            }
        )
        result = bestkeeper(exp, sd_threshold=0.2)
        assert result.consistent["STABLE"] is True

    def test_does_not_require_efficiency(self) -> None:
        # BestKeeper must work even when no efficiencies are declared.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
            },
            efficiencies={"ACTB": None, "GAPDH": None},
        )
        result = bestkeeper(exp)
        assert isinstance(result, BestKeeperResult)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestBestKeeperErrors:
    def test_fewer_than_two_genes_raises(self) -> None:
        exp = _make_experiment({"ACTB": [20.0, 21.0, 22.0]})
        with pytest.raises(ValueError):
            bestkeeper(exp)

    def test_fewer_than_three_samples_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0],
                "GAPDH": [20.0, 21.0],
            }
        )
        with pytest.raises(ValueError):
            bestkeeper(exp)

    def test_non_positive_sd_threshold_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0, 22.0],
                "GAPDH": [20.0, 21.0, 22.0],
            }
        )
        with pytest.raises(ValueError):
            bestkeeper(exp, sd_threshold=0.0)