"""Tests for the geNorm reference-gene stability algorithm.

Uses synthetic experiments with known Cq values to verify:

* A gene with constant Cq across samples gets a low (zero) M value.
* A gene that varies wildly gets a high M and is removed first.
* Stepwise elimination produces the correct stability ranking.
* Pairwise variation V_{n/n+1} and the recommended gene count.
* Error cases (fewer than 2 genes, fewer than 3 samples).
"""

from __future__ import annotations

import math

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.domain.enums import ReplicateType
from bioeng_qpcr.reference_genes.genorm import GeNormResult, genorm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(
    gene_cqs: dict[str, list[float | None]],
    sample_names: list[str] | None = None,
    groups: dict[str, str] | None = None,
    is_reference: bool = True,
) -> Experiment:
    """Build a synthetic experiment from a ``{gene: [cq_per_sample]}`` map.

    ``None`` entries produce undetermined wells.  Sample names default to
    ``S1, S2, ...``.  One well per sample/target is created (technical
    replicates can be added by passing repeated sample/gene keys via a
    separate helper when needed).
    """
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
    return Experiment(name="genorm_test", wells=wells, samples=samples, assays=assays)


# ---------------------------------------------------------------------------
# Constant vs variable genes
# ---------------------------------------------------------------------------


class TestGeNormStability:
    """Constant genes should be stable; variable genes unstable."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        # ACTB and GAPDH are constant; TUBULIN varies across the 3 samples.
        return _make_experiment(
            {
                "ACTB": [18.0, 18.0, 18.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [15.0, 20.0, 25.0],
            }
        )

    def test_returns_result_type(self, experiment: Experiment) -> None:
        result = genorm(experiment)
        assert isinstance(result, GeNormResult)

    def test_ranking_contains_all_genes(self, experiment: Experiment) -> None:
        result = genorm(experiment)
        assert sorted(result.ranking) == ["ACTB", "GAPDH", "TUBULIN"]

    def test_constant_gene_has_zero_m(self, experiment: Experiment) -> None:
        """A constant gene surviving to the final pair has M == 0."""
        result = genorm(experiment)
        assert result.m_values["ACTB"] == pytest.approx(0.0)
        assert result.m_values["GAPDH"] == pytest.approx(0.0)

    def test_variable_gene_has_high_m(self, experiment: Experiment) -> None:
        """TUBULIN varies; its M equals sqrt(50/3) ~ 4.0825."""
        result = genorm(experiment)
        expected = math.sqrt(50.0 / 3.0)
        assert result.m_values["TUBULIN"] == pytest.approx(expected)

    def test_variable_gene_is_least_stable(self, experiment: Experiment) -> None:
        """The variable gene is ranked last (removed first)."""
        result = genorm(experiment)
        assert result.ranking[-1] == "TUBULIN"

    def test_constant_genes_ranked_before_variable(
        self, experiment: Experiment
    ) -> None:
        result = genorm(experiment)
        assert result.ranking.index("ACTB") < result.ranking.index("TUBULIN")
        assert result.ranking.index("GAPDH") < result.ranking.index("TUBULIN")

    def test_m_values_are_non_negative(self, experiment: Experiment) -> None:
        result = genorm(experiment)
        assert all(v >= 0.0 for v in result.m_values.values())


# ---------------------------------------------------------------------------
# Stepwise elimination
# ---------------------------------------------------------------------------


class TestGeNormStepwiseElimination:
    """Four genes with a clear stability hierarchy."""

    @pytest.fixture
    def experiment(self) -> Experiment:
        # ACTB constant, GAPDH nearly constant, TUBULIN moderate, HPRT wild.
        return _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 21.0, 19.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
                "HPRT": [10.0, 30.0, 15.0, 25.0],
            }
        )

    def test_ranking_most_to_least_stable(self, experiment: Experiment) -> None:
        result = genorm(experiment)
        # HPRT (wildest) last; ACTB/GAPDH (most stable) in the top two.
        assert result.ranking[-1] == "HPRT"
        assert set(result.ranking[:2]) == {"ACTB", "GAPDH"}

    def test_wild_gene_removed_first(self, experiment: Experiment) -> None:
        """HPRT has the highest M and is eliminated in the first step."""
        result = genorm(experiment)
        assert result.m_values["HPRT"] == pytest.approx(8.365392, rel=1e-5)
        assert result.m_values["HPRT"] == max(result.m_values.values())

    def test_stepwise_ranking_order(self, experiment: Experiment) -> None:
        result = genorm(experiment)
        # Full deterministic order: ACTB, GAPDH (final pair) then TUBULIN
        # (removed second) then HPRT (removed first).
        assert result.ranking == ["ACTB", "GAPDH", "TUBULIN", "HPRT"]

    def test_elimination_m_values(self, experiment: Experiment) -> None:
        result = genorm(experiment)
        # TUBULin removed at the second step with this M.
        assert result.m_values["TUBULIN"] == pytest.approx(1.790569, rel=1e-5)
        # The final pair share the same M.
        assert result.m_values["ACTB"] == pytest.approx(0.707107, rel=1e-5)
        assert result.m_values["GAPDH"] == pytest.approx(0.707107, rel=1e-5)


# ---------------------------------------------------------------------------
# Pairwise variation
# ---------------------------------------------------------------------------


class TestGeNormPairwiseVariation:
    """V_{n/n+1} keys and values, plus the recommended gene count."""

    def test_keys_for_four_genes(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 21.0, 19.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
                "HPRT": [10.0, 30.0, 15.0, 25.0],
            }
        )
        result = genorm(exp)
        assert set(result.pairwise_variation.keys()) == {"V2/3", "V3/4"}

    def test_keys_for_three_genes(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [18.0, 18.0, 18.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [15.0, 20.0, 25.0],
            }
        )
        result = genorm(exp)
        assert set(result.pairwise_variation.keys()) == {"V2/3"}

    def test_pairwise_values_stepwise(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 21.0, 19.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
                "HPRT": [10.0, 30.0, 15.0, 25.0],
            }
        )
        result = genorm(exp)
        assert result.pairwise_variation["V2/3"] == pytest.approx(0.589256, rel=1e-5)
        assert result.pairwise_variation["V3/4"] == pytest.approx(2.0825, rel=1e-4)

    def test_pairwise_value_three_genes(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [18.0, 18.0, 18.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [15.0, 20.0, 25.0],
            }
        )
        result = genorm(exp)
        assert result.pairwise_variation["V2/3"] == pytest.approx(1.360827, rel=1e-5)

    def test_empty_pairwise_for_two_genes(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [20.0, 21.0, 22.0],
            }
        )
        result = genorm(exp)
        assert result.pairwise_variation == {}
        assert result.recommended_count == 2

    def test_no_recommendation_when_threshold_not_reached(self) -> None:
        # All V >= 0.15 -> the evidence does not support a count.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 21.0, 19.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
                "HPRT": [10.0, 30.0, 15.0, 25.0],
            }
        )
        result = genorm(exp)
        assert result.recommended_count is None

    def test_recommended_count_two_when_v_below_threshold(self) -> None:
        # Three genes co-vary perfectly (V2/3 == 0 < 0.15) -> 2 suffice.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0, 22.0, 23.0],
                "GAPDH": [20.0, 21.0, 22.0, 23.0],
                "TUBULIN": [20.0, 21.0, 22.0, 23.0],
                "HPRT": [15.0, 20.0, 25.0, 30.0],
            }
        )
        result = genorm(exp)
        assert result.pairwise_variation["V2/3"] == pytest.approx(0.0)
        assert result.recommended_count == 2

    def test_recommended_count_three(self) -> None:
        # A constructed dataset where V2/3 >= 0.15 but V3/4 < 0.15, so the
        # optimal number of reference genes is 3.
        exp = _make_experiment(
            {
                "A": [18.206, 19.187, 19.603, 20.218],
                "B": [20.644, 21.697, 22.261, 23.727],
                "C": [20.137, 21.553, 22.959, 23.132],
                "D": [18.770, 19.883, 21.674, 22.487],
            }
        )
        result = genorm(exp)
        assert result.pairwise_variation["V2/3"] >= 0.15
        assert result.pairwise_variation["V3/4"] < 0.15
        assert result.recommended_count == 3


# ---------------------------------------------------------------------------
# Candidate resolution
# ---------------------------------------------------------------------------


class TestGeNormCandidateResolution:
    """Default candidate selection and explicit overrides."""

    def test_uses_reference_targets_by_default(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
                "GOI": [15.0, 20.0, 25.0],
            }
        )
        # All three are marked is_reference=True by the helper.
        result = genorm(exp)
        assert sorted(result.ranking) == ["ACTB", "GAPDH", "GOI"]

    def test_explicit_candidates_override(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
                "GOI": [15.0, 20.0, 25.0],
            }
        )
        result = genorm(exp, candidate_targets=["ACTB", "GAPDH"])
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
            genorm(exp)

    def test_skips_samples_with_missing_cq(self) -> None:
        # S3 is missing TUBULIN; only S1 and S2 are complete -> ValueError.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [15.0, 20.0, None],
            }
        )
        with pytest.raises(ValueError, match="3 samples"):
            genorm(exp)

    def test_technical_replicates_averaged(self) -> None:
        # Two technical replicates for ACTB on S1 should be averaged.
        wells = [
            Well(
                position="A01",
                sample_name="S1",
                target_name="ACTB",
                cq=20.0,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group="S1_ACTB",
            ),
            Well(
                position="A02",
                sample_name="S1",
                target_name="ACTB",
                cq=20.4,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group="S1_ACTB",
            ),
            Well(position="A03", sample_name="S1", target_name="GAPDH", cq=20.0),
            Well(position="A04", sample_name="S2", target_name="ACTB", cq=20.0),
            Well(position="A05", sample_name="S2", target_name="GAPDH", cq=21.0),
            Well(position="A06", sample_name="S3", target_name="ACTB", cq=20.0),
            Well(position="A07", sample_name="S3", target_name="GAPDH", cq=22.0),
        ]
        assays = {
            "ACTB": Assay(name="ACTB", efficiency=1.0, is_reference=True),
            "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
        }
        samples = {s: Sample(name=s) for s in ["S1", "S2", "S3"]}
        exp = Experiment(name="reps", wells=wells, samples=samples, assays=assays)
        result = genorm(exp)
        # No error and both genes ranked.
        assert sorted(result.ranking) == ["ACTB", "GAPDH"]

    def test_duplicate_single_wells_are_not_silently_averaged(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
            }
        )
        exp.wells.append(
            Well(position="H12", sample_name="S1", target_name="ACTB", cq=20.2)
        )
        with pytest.raises(ValueError, match="marked single"):
            genorm(exp)

    def test_efficiency_correction_changes_misleading_raw_cq_pair(self) -> None:
        exp = _make_experiment(
            {
                "A": [20.0, 21.0, 22.0, 23.0],
                "B": [20.0, 21.0, 22.0, 23.0],
                "C": [18.0, 19.1, 19.9, 21.0],
            }
        )
        exp.assays["B"] = Assay(
            name="B", efficiency=0.5, is_reference=True
        )
        result = genorm(exp)
        assert result.ranking[-1] == "B"
        assert result.efficiencies["B"] == pytest.approx(0.5)
        assert result.quantity_scale == "log2_relative_quantity_efficiency_corrected"

    def test_missing_candidate_efficiency_is_rejected(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [21.0, 21.0, 21.0],
            }
        )
        exp.assays["ACTB"] = Assay(name="ACTB", is_reference=True)
        with pytest.raises(ValueError, match="Measured amplification efficiency"):
            genorm(exp)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestGeNormErrors:
    """geNorm must reject insufficient inputs."""

    def test_fewer_than_two_genes_raises(self) -> None:
        exp = _make_experiment({"ACTB": [20.0, 21.0, 22.0]})
        with pytest.raises(ValueError, match="2 candidate genes"):
            genorm(exp)

    def test_fewer_than_three_samples_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0],
                "GAPDH": [20.0, 21.0],
            }
        )
        with pytest.raises(ValueError, match="3 samples"):
            genorm(exp)

    def test_explicit_single_candidate_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0, 22.0],
                "GAPDH": [20.0, 21.0, 22.0],
            }
        )
        with pytest.raises(ValueError, match="2 candidate genes"):
            genorm(exp, candidate_targets=["ACTB"])
