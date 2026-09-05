"""Tests for the consensus reference-gene ranking.

Uses synthetic experiments with known Cq values to verify:

* The consensus combines geNorm and NormFinder rankings.
* A gene ranked #1 by both methods is #1 in the consensus.
* Rank aggregation works when the two methods disagree.
* Recommended genes are the top-N from the consensus ranking.
* Errors from the underlying methods propagate.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.reference_genes.consensus import (
    ConsensusResult,
    consensus_ranking,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(
    gene_cqs: dict[str, list[float | None]],
    sample_names: list[str] | None = None,
    groups: dict[str, str] | None = None,
    is_reference: bool = True,
) -> Experiment:
    """Build a synthetic experiment from a ``{gene: [cq_per_sample]}`` map."""
    n_samples = len(next(iter(gene_cqs.values())))
    if sample_names is None:
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
    assays = {
        gene: Assay(name=gene, efficiency=1.0, is_reference=is_reference)
        for gene in gene_cqs
    }
    samples = {
        sname: Sample(name=sname, group=(groups.get(sname) if groups else None))
        for sname in sample_names
    }
    return Experiment(
        name="consensus_test", wells=wells, samples=samples, assays=assays
    )


# ---------------------------------------------------------------------------
# Consensus combines both methods
# ---------------------------------------------------------------------------


class TestConsensusCombination:
    """ACTB is constant; GAPDH/TUBULIN anti-correlate.  Both methods rank
    ACTB first."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        return _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )

    def test_returns_result_type(self, experiment: Experiment) -> None:
        result = consensus_ranking(experiment)
        assert isinstance(result, ConsensusResult)

    def test_ranking_contains_all_genes(self, experiment: Experiment) -> None:
        result = consensus_ranking(experiment)
        assert sorted(result.ranking) == ["ACTB", "GAPDH", "TUBULIN"]

    def test_genorm_and_normfinder_rankings_populated(
        self, experiment: Experiment
    ) -> None:
        result = consensus_ranking(experiment)
        assert sorted(result.genorm_ranking) == ["ACTB", "GAPDH", "TUBULIN"]
        assert sorted(result.normfinder_ranking) == ["ACTB", "GAPDH", "TUBULIN"]

    def test_consensus_score_is_average_of_ranks(self, experiment: Experiment) -> None:
        result = consensus_ranking(experiment)
        for gene in ("ACTB", "GAPDH", "TUBULIN"):
            ge_rank = result.genorm_ranking.index(gene) + 1
            nf_rank = result.normfinder_ranking.index(gene) + 1
            expected = (ge_rank + nf_rank) / 2.0
            assert result.consensus_scores[gene] == pytest.approx(expected)

    def test_ranking_sorted_by_consensus_score(self, experiment: Experiment) -> None:
        result = consensus_ranking(experiment)
        scores = [result.consensus_scores[g] for g in result.ranking]
        assert scores == sorted(scores)

    def test_top_gene_ranked_first_by_both(self, experiment: Experiment) -> None:
        """A gene ranked #1 by both methods is #1 in the consensus."""
        result = consensus_ranking(experiment)
        assert result.genorm_ranking[0] == "ACTB"
        assert result.normfinder_ranking[0] == "ACTB"
        assert result.ranking[0] == "ACTB"
        assert result.consensus_scores["ACTB"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rank aggregation when methods disagree
# ---------------------------------------------------------------------------


class TestConsensusDisagreement:
    """geNorm prefers a co-varying pair; NormFinder prefers the constant
    gene.  The consensus averages their ranks."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        groups = {
            "S1": "control",
            "S2": "control",
            "S3": "control",
            "S4": "treated",
            "S5": "treated",
            "S6": "treated",
        }
        # GAPDH and TUBULIN co-vary perfectly across a group shift, whereas
        # ACTB is stable between groups. This exposes the methods' different
        # assumptions; RPLP0 is deliberately noisy.
        return _make_experiment(
            {
                "ACTB": [20.0, 20.1, 19.9, 20.0, 20.1, 19.9],
                "GAPDH": [19.0, 19.1, 18.9, 24.0, 24.1, 23.9],
                "TUBULIN": [19.0, 19.1, 18.9, 24.0, 24.1, 23.9],
                "RPLP0": [18.0, 22.0, 20.0, 18.0, 22.0, 20.0],
            },
            groups=groups,
        )

    def test_methods_disagree_on_first(self, experiment: Experiment) -> None:
        result = consensus_ranking(experiment)
        assert result.genorm_ranking[0] == "GAPDH"
        assert result.normfinder_ranking[0] == "ACTB"

    def test_consensus_ranking(self, experiment: Experiment) -> None:
        # geNorm: GAPDH(1), TUBULIN(2), ACTB(3), RPLP0(4)
        # NormFinder: ACTB(1), GAPDH(2), TUBULIN(3), RPLP0(4)
        result = consensus_ranking(experiment)
        assert result.ranking == ["GAPDH", "ACTB", "TUBULIN", "RPLP0"]
        assert result.consensus_scores["GAPDH"] == pytest.approx(1.5)
        assert result.consensus_scores["ACTB"] == pytest.approx(2.0)
        assert result.consensus_scores["TUBULIN"] == pytest.approx(2.5)
        assert result.consensus_scores["RPLP0"] == pytest.approx(4.0)

    def test_consensus_first_is_lowest_score(self, experiment: Experiment) -> None:
        result = consensus_ranking(experiment)
        first_score = result.consensus_scores[result.ranking[0]]
        assert first_score == min(result.consensus_scores.values())


# ---------------------------------------------------------------------------
# Recommended genes
# ---------------------------------------------------------------------------


class TestConsensusRecommendedGenes:
    def test_no_recommendation_when_threshold_not_reached(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )
        result = consensus_ranking(exp)
        assert result.recommended_genes == []

    def test_recommended_genes_match_genorm_count(self) -> None:
        # Dataset where geNorm recommends 3 reference genes.
        exp = _make_experiment(
            {
                "A": [18.206, 19.187, 19.603, 20.218],
                "B": [20.644, 21.697, 22.261, 23.727],
                "C": [20.137, 21.553, 22.959, 23.132],
                "D": [18.770, 19.883, 21.674, 22.487],
            }
        )
        result = consensus_ranking(exp)
        assert len(result.recommended_genes) == 3
        assert result.recommended_genes == result.ranking[:3]

    def test_recommended_genes_subset_of_ranking(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )
        result = consensus_ranking(exp)
        for gene in result.recommended_genes:
            assert gene in result.ranking


# ---------------------------------------------------------------------------
# Candidate / group forwarding
# ---------------------------------------------------------------------------


class TestConsensusForwarding:
    def test_explicit_candidates_forwarded(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
                "GOI": [15.0, 25.0, 18.0, 28.0],
            }
        )
        result = consensus_ranking(exp, candidate_targets=["ACTB", "GAPDH", "TUBULIN"])
        assert sorted(result.ranking) == ["ACTB", "GAPDH", "TUBULIN"]

    def test_sample_groups_forwarded_to_normfinder(self) -> None:
        groups = {"C1": "control", "C2": "control", "T1": "treated", "T2": "treated"}
        exp = _make_experiment(
            {
                "STABLE": [20.0, 20.0, 20.0, 20.0],
                "BETWEEN": [20.0, 20.0, 25.0, 25.0],
                "WITHIN": [18.0, 22.0, 18.0, 22.0],
            },
            sample_names=["C1", "C2", "T1", "T2"],
        )
        result = consensus_ranking(exp, sample_groups=groups)
        # NormFinder should rank STABLE first; geNorm too (constant gene).
        assert result.normfinder_ranking[0] == "STABLE"
        assert result.ranking[0] == "STABLE"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestConsensusErrors:
    def test_fewer_than_two_genes_raises(self) -> None:
        exp = _make_experiment({"ACTB": [20.0, 21.0, 22.0]})
        with pytest.raises(ValueError):
            consensus_ranking(exp)

    def test_fewer_than_three_samples_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0],
                "GAPDH": [20.0, 21.0],
            }
        )
        with pytest.raises(ValueError):
            consensus_ranking(exp)

    def test_missing_data_propagates_error(self) -> None:
        # S3 missing TUBULIN -> fewer than 3 complete samples.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [15.0, 20.0, None],
            }
        )
        with pytest.raises(ValueError):
            consensus_ranking(exp)


# ---------------------------------------------------------------------------
# Selectable methods
# ---------------------------------------------------------------------------


class TestConsensusSelectableMethods:
    """The consensus can aggregate an arbitrary subset of the stability
    algorithms via the ``methods`` argument."""

    def _exp(self) -> Experiment:
        return _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )

    def test_default_methods_unchanged(self) -> None:
        result = consensus_ranking(self._exp())
        assert result.methods == ("genorm", "normfinder")
        assert result.delta_ct_ranking == []
        assert result.bestkeeper_ranking == []

    def test_delta_ct_and_bestkeeper_rankings_populated(self) -> None:
        result = consensus_ranking(
            self._exp(),
            methods=("genorm", "delta_ct", "bestkeeper"),
        )
        assert sorted(result.delta_ct_ranking) == ["ACTB", "GAPDH", "TUBULIN"]
        assert sorted(result.bestkeeper_ranking) == ["ACTB", "GAPDH", "TUBULIN"]
        assert result.normfinder_ranking == []

    def test_scores_average_all_selected_methods(self) -> None:
        result = consensus_ranking(
            self._exp(),
            methods=("delta_ct", "bestkeeper"),
        )
        for gene in ("ACTB", "GAPDH", "TUBULIN"):
            dc_rank = result.delta_ct_ranking.index(gene) + 1
            bk_rank = result.bestkeeper_ranking.index(gene) + 1
            assert result.consensus_scores[gene] == pytest.approx(
                (dc_rank + bk_rank) / 2.0
            )

    def test_bestkeeper_only_needs_no_efficiency(self) -> None:
        # A dataset where efficiency is missing for every candidate can still
        # be ranked by BestKeeper alone.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0],
            },
        )
        # Replace assays with efficiency-less copies so bestkeeper-only works.
        exp.assays = {
            name: replace(assay, efficiency=None)
            for name, assay in exp.assays.items()
        }
        result = consensus_ranking(exp, methods=("bestkeeper",))
        assert sorted(result.ranking) == ["ACTB", "GAPDH"]

    def test_mixing_bestkeeper_with_efficiency_method_requires_efficiency(
        self,
    ) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0],
            },
        )
        exp.assays = {
            name: replace(assay, efficiency=None)
            for name, assay in exp.assays.items()
        }
        with pytest.raises(ValueError):
            consensus_ranking(exp, methods=("bestkeeper", "delta_ct"))

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            consensus_ranking(self._exp(), methods=("genorm", "fake"))

    def test_empty_methods_raises(self) -> None:
        with pytest.raises(ValueError):
            consensus_ranking(self._exp(), methods=())

    def test_no_recommendation_without_ge_norm(self) -> None:
        result = consensus_ranking(self._exp(), methods=("delta_ct",))
        assert result.recommended_genes == []
