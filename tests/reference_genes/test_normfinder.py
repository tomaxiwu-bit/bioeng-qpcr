"""Tests for the NormFinder reference-gene stability algorithm.

Uses synthetic experiments with known Cq values to verify:

* A constant gene gets a low (zero) stability value.
* A variable gene gets a high stability value.
* Group structure correctly separates inter- from intra-group variance.
* Group labels can come from ``Sample.group`` or an explicit mapping.
* Error cases (fewer than 2 genes, fewer than 3 samples).
"""

from __future__ import annotations

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.reference_genes.normfinder import NormFinderResult
from bioeng_qpcr.reference_genes.normfinder import (
    variance_stability_score as normfinder,
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
        name="normfinder_test", wells=wells, samples=samples, assays=assays
    )


# ---------------------------------------------------------------------------
# Constant vs variable genes (no groups)
# ---------------------------------------------------------------------------


class TestNormFinderStability:
    """Without group information stability reduces to overall variance."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        # ACTB constant, GAPDH variable across 3 samples.
        return _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [18.0, 20.0, 22.0],
            }
        )

    def test_returns_result_type(self, experiment: Experiment) -> None:
        result = normfinder(experiment)
        assert isinstance(result, NormFinderResult)

    def test_constant_gene_has_zero_stability(self, experiment: Experiment) -> None:
        result = normfinder(experiment)
        assert result.stability_values["ACTB"] == pytest.approx(0.0)

    def test_variable_gene_has_high_stability(self, experiment: Experiment) -> None:
        # var([18, 20, 22], ddof=0) = 8/3 ~ 2.6667
        result = normfinder(experiment)
        assert result.stability_values["GAPDH"] == pytest.approx(8.0 / 3.0)

    def test_constant_gene_ranked_first(self, experiment: Experiment) -> None:
        result = normfinder(experiment)
        assert result.ranking[0] == "ACTB"
        assert result.ranking[-1] == "GAPDH"

    def test_no_group_inter_is_zero(self, experiment: Experiment) -> None:
        """With a single (ungrouped) bucket the inter-group variance is 0."""
        result = normfinder(experiment)
        assert result.inter_group_var["ACTB"] == pytest.approx(0.0)
        assert result.inter_group_var["GAPDH"] == pytest.approx(0.0)

    def test_no_group_intra_equals_overall_variance(
        self, experiment: Experiment
    ) -> None:
        result = normfinder(experiment)
        assert result.intra_group_var["GAPDH"] == pytest.approx(8.0 / 3.0)

    def test_stability_equals_inter_plus_intra(self, experiment: Experiment) -> None:
        result = normfinder(experiment)
        for gene in ("ACTB", "GAPDH"):
            assert result.stability_values[gene] == pytest.approx(
                result.inter_group_var[gene] + result.intra_group_var[gene]
            )


# ---------------------------------------------------------------------------
# Group structure: inter- vs intra-group variance
# ---------------------------------------------------------------------------


class TestNormFinderGroupStructure:
    """Two groups expose the inter/intra variance decomposition."""

    GROUPS = {"C1": "control", "C2": "control", "T1": "treated", "T2": "treated"}
    SAMPLES = ["C1", "C2", "T1", "T2"]

    @pytest.fixture
    def experiment(self) -> Experiment:
        # STABLE: constant everywhere.
        # BETWEEN: differs between groups, constant within -> high inter.
        # WITHIN: constant group means, noisy within -> high intra.
        return _make_experiment(
            {
                "STABLE": [20.0, 20.0, 20.0, 20.0],
                "BETWEEN": [20.0, 20.0, 25.0, 25.0],
                "WITHIN": [18.0, 22.0, 18.0, 22.0],
            },
            sample_names=self.SAMPLES,
            groups=self.GROUPS,
        )

    def test_ranking(self, experiment: Experiment) -> None:
        result = normfinder(experiment)
        assert result.ranking == ["STABLE", "WITHIN", "BETWEEN"]

    def test_stable_gene_has_zero_stability(self, experiment: Experiment) -> None:
        result = normfinder(experiment)
        assert result.stability_values["STABLE"] == pytest.approx(0.0)
        assert result.inter_group_var["STABLE"] == pytest.approx(0.0)
        assert result.intra_group_var["STABLE"] == pytest.approx(0.0)

    def test_between_gene_has_high_inter_zero_intra(
        self, experiment: Experiment
    ) -> None:
        """BETWEEN is constant within each group -> intra 0, inter 6.25."""
        result = normfinder(experiment)
        assert result.inter_group_var["BETWEEN"] == pytest.approx(6.25)
        assert result.intra_group_var["BETWEEN"] == pytest.approx(0.0)
        assert result.stability_values["BETWEEN"] == pytest.approx(6.25)

    def test_within_gene_has_zero_inter_high_intra(
        self, experiment: Experiment
    ) -> None:
        """WITHIN has equal group means -> inter 0, intra 4.0."""
        result = normfinder(experiment)
        assert result.inter_group_var["WITHIN"] == pytest.approx(0.0)
        assert result.intra_group_var["WITHIN"] == pytest.approx(4.0)
        assert result.stability_values["WITHIN"] == pytest.approx(4.0)

    def test_between_less_stable_than_within(self, experiment: Experiment) -> None:
        """BETWEEN (6.25) is less stable than WITHIN (4.0)."""
        result = normfinder(experiment)
        assert result.stability_values["BETWEEN"] > result.stability_values["WITHIN"]

    def test_sample_groups_param_matches_sample_group(
        self, experiment: Experiment
    ) -> None:
        """Passing groups via the ``sample_groups`` parameter gives the same
        result as embedding them in ``Sample.group``."""
        # Experiment without group info on the samples.
        exp_no_group = _make_experiment(
            {
                "STABLE": [20.0, 20.0, 20.0, 20.0],
                "BETWEEN": [20.0, 20.0, 25.0, 25.0],
                "WITHIN": [18.0, 22.0, 18.0, 22.0],
            },
            sample_names=self.SAMPLES,
        )
        result = normfinder(exp_no_group, sample_groups=self.GROUPS)
        assert result.ranking == ["STABLE", "WITHIN", "BETWEEN"]
        assert result.stability_values["BETWEEN"] == pytest.approx(6.25)
        assert result.stability_values["WITHIN"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Candidate resolution
# ---------------------------------------------------------------------------


class TestNormFinderCandidateResolution:
    def test_uses_reference_targets_by_default(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
                "GOI": [15.0, 20.0, 25.0],
            }
        )
        result = normfinder(exp)
        assert sorted(result.ranking) == ["ACTB", "GAPDH", "GOI"]

    def test_explicit_candidates_override(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
                "GOI": [15.0, 20.0, 25.0],
            }
        )
        result = normfinder(exp, candidate_targets=["ACTB", "GAPDH"])
        assert sorted(result.ranking) == ["ACTB", "GAPDH"]

    def test_requires_designated_candidates_when_no_reference(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
            },
            is_reference=False,
        )
        with pytest.raises(ValueError, match="No reference-gene candidates"):
            normfinder(exp)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestNormFinderErrors:
    def test_fewer_than_two_genes_raises(self) -> None:
        exp = _make_experiment({"ACTB": [20.0, 21.0, 22.0]})
        with pytest.raises(ValueError, match="2 candidate genes"):
            normfinder(exp)

    def test_fewer_than_three_samples_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0],
                "GAPDH": [20.0, 21.0],
            }
        )
        with pytest.raises(ValueError, match="3 samples"):
            normfinder(exp)

    def test_skips_samples_with_missing_cq(self) -> None:
        # S3 missing GAPDH -> only 2 complete samples -> ValueError.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [20.0, 21.0, None],
            }
        )
        with pytest.raises(ValueError, match="3 samples"):
            normfinder(exp)
