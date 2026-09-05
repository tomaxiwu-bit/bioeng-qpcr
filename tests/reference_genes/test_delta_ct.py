"""Tests for the delta-Ct reference-gene stability analysis.

Uses synthetic experiments to verify the linkage with the established
efficiency-corrected quantity scale and the geometric-mean of pairwise
standard deviations that distinguishes the method from geNorm's arithmetic
mean.
"""

from __future__ import annotations

import math

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.reference_genes.delta_ct import DeltaCtResult, delta_ct


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
        name="delta_ct_test", wells=wells, samples=samples, assays=assays
    )


# ---------------------------------------------------------------------------
# Stability ranking
# ---------------------------------------------------------------------------


class TestDeltaCtRanking:
    def test_returns_result_type(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )
        assert isinstance(delta_ct(exp), DeltaCtResult)

    def test_constant_gene_ranked_first(self) -> None:
        # ACTB is perfectly constant; both others vary. ACTB's pairwise SDs
        # are the smallest, so it ranks first.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )
        result = delta_ct(exp)
        assert result.ranking[0] == "ACTB"

    def test_constant_genes_have_zero_stability(self) -> None:
        # Two genes both constant across samples form a delta that is constant,
        # so each gene's pairwise SD (and hence stability) is exactly zero.
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [30.0, 30.0, 30.0, 30.0],
            }
        )
        result = delta_ct(exp)
        assert result.stability_values["ACTB"] == pytest.approx(0.0)
        assert result.stability_values["GAPDH"] == pytest.approx(0.0)

    def test_ranking_matches_pairwise_geomean(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0, 20.0],
                "GAPDH": [19.0, 21.0, 22.0, 18.0],
                "TUBULIN": [22.0, 18.0, 19.0, 21.0],
            }
        )
        result = delta_ct(exp)
        # GAPDH's stability = geomean of SD(GAPDH,ACTB) and SD(GAPDH,TUBULIN).
        sd_actb = result.pairwise_sd[("ACTB", "GAPDH")]
        sd_tub = result.pairwise_sd[("GAPDH", "TUBULIN")]
        expected = math.exp((math.log(sd_actb) + math.log(sd_tub)) / 2)
        assert result.stability_values["GAPDH"] == pytest.approx(expected)

    def test_pairwise_sd_uses_population_estimator(self) -> None:
        exp = _make_experiment(
            {
                "G1": [20.0, 20.0, 20.0],
                "G2": [22.0, 23.0, 24.0],
            }
        )
        result = delta_ct(exp)
        # SD of the delta between G2 (constant) and a linear ramp:
        # deltas = [2, 3, 4], population SD = sqrt(2/3).
        expected = (2 / 3) ** 0.5
        assert result.pairwise_sd[("G1", "G2")] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Efficiency correction at 100% is identical to raw-Ct ordering
# ---------------------------------------------------------------------------


class TestEfficiencyCorrection:
    def test_non_unit_efficiency_scales_the_pairwise_sd(self) -> None:
        # B varies, A is constant, so stability(A) collapses to
        # |log2(1 + eff_B)| * population_sd(B). This must reflect the declared
        # efficiency, not the raw-Ct SD.
        exp = _make_experiment(
            {
                "A": [20.0, 20.0, 20.0],
                "B": [17.0, 22.0, 24.0],
            },
            efficiencies={"A": 0.9, "B": 1.1},
        )
        result = delta_ct(exp)
        assert result.efficiencies == {"A": 0.9, "B": 1.1}
        sd_b = float((sum((v - 21) ** 2 for v in (17, 22, 24)) / 3) ** 0.5)
        expected = math.log2(2.1) * sd_b
        assert result.stability_values["A"] == pytest.approx(expected)

    def test_100_percent_efficiency_matches_raw_ct_scale(self) -> None:
        exp = _make_experiment(
            {
                "A": [20.0, 20.0, 20.0],
                "B": [17.0, 22.0, 24.0],
            }
        )
        result = delta_ct(exp)
        expected = float((sum((v - 21) ** 2 for v in (17, 22, 24)) / 3) ** 0.5)
        assert result.stability_values["A"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Sample completeness and errors
# ---------------------------------------------------------------------------


class TestDeltaCtErrors:
    def test_fewer_than_two_genes_raises(self) -> None:
        exp = _make_experiment({"ACTB": [20.0, 21.0, 22.0]})
        with pytest.raises(ValueError):
            delta_ct(exp)

    def test_fewer_than_three_samples_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 21.0],
                "GAPDH": [20.0, 21.0],
            }
        )
        with pytest.raises(ValueError):
            delta_ct(exp)

    def test_missing_data_excludes_incomplete_samples(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [15.0, 20.0, None],
            }
        )
        # Only S3 lacks TUBULIN, leaving S1/S2 -> fewer than 3 complete.
        with pytest.raises(ValueError):
            delta_ct(exp)

    def test_missing_efficiency_raises(self) -> None:
        exp = _make_experiment(
            {
                "ACTB": [20.0, 20.0, 20.0],
                "GAPDH": [20.0, 20.0, 20.0],
                "TUBULIN": [20.0, 20.0, 20.0],
            },
            efficiencies={"ACTB": 1.0, "GAPDH": 0.95, "TUBULIN": None},
        )
        # TUBULIN has no measured efficiency.
        with pytest.raises(ValueError):
            delta_ct(exp)