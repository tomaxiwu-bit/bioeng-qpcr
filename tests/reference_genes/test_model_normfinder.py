"""Regression and adversarial tests for model-based NormFinder."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.reference_genes.normfinder import (
    ModelBasedNormFinderResult,
    normfinder,
)


def _make_experiment(
    gene_cqs: dict[str, list[float | None]],
    groups: list[str] | None = None,
) -> Experiment:
    n_samples = len(next(iter(gene_cqs.values())))
    sample_names = [f"S{index + 1}" for index in range(n_samples)]
    wells: list[Well] = []
    for gene_index, (gene, values) in enumerate(gene_cqs.items()):
        for sample_index, (sample_name, cq) in enumerate(
            zip(sample_names, values, strict=True)
        ):
            wells.append(
                Well(
                    position=f"P{gene_index:02d}{sample_index:03d}",
                    sample_name=sample_name,
                    target_name=gene,
                    cq=cq,
                )
            )
    return Experiment(
        name="model_normfinder_test",
        wells=wells,
        samples={
            name: Sample(name=name, group=groups[index] if groups else None)
            for index, name in enumerate(sample_names)
        },
        assays={
            gene: Assay(name=gene, efficiency=1.0, is_reference=True)
            for gene in gene_cqs
        },
    )


def _load_colon_experiment() -> Experiment:
    """Convert the official linear-expression fixture to Cq-like log2 data."""
    path = Path(__file__).parent / "data" / "colon_normqpcr.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    genes = [name for name in rows[0] if name not in {"sample", "group"}]
    wells: list[Well] = []
    for sample_index, row in enumerate(rows):
        for gene_index, gene in enumerate(genes):
            # NormqPCR uses log2(expression). Negating and adding a constant
            # maps it into the Cq convention without changing model stability.
            cq = 40.0 - math.log2(float(row[gene]))
            wells.append(
                Well(
                    position=f"P{sample_index:03d}{gene_index:02d}",
                    sample_name=row["sample"],
                    target_name=gene,
                    cq=cq,
                )
            )
    return Experiment(
        name="NormqPCR Colon benchmark",
        wells=wells,
        samples={
            row["sample"]: Sample(name=row["sample"], group=row["group"])
            for row in rows
        },
        assays={
            gene: Assay(name=gene, efficiency=1.0, is_reference=True)
            for gene in genes
        },
    )


@pytest.fixture(scope="module")
def colon_result() -> ModelBasedNormFinderResult:
    return normfinder(_load_colon_experiment())


class TestNormqPCRColonBenchmark:
    """Expected values are Table 3 values reproduced by NormqPCR 1.58.0."""

    EXPECTED = {
        "UBC": 0.1821707,
        "GAPD": 0.2146061,
        "TPT1": 0.2202956,
        "UBB": 0.2471573,
        "TUBA6": 0.2700641,
        "RPS13": 0.2813039,
        "NACA": 0.2862397,
        "CFL1": 0.2870467,
        "SUI1": 0.3139404,
        "ACTB": 0.3235918,
        "CLTC": 0.3692880,
        "RPS23": 0.3784909,
        "FLJ20030": 0.3935173,
    }

    def test_result_metadata(self, colon_result: ModelBasedNormFinderResult) -> None:
        assert isinstance(colon_result, ModelBasedNormFinderResult)
        assert colon_result.n_samples == 40
        assert colon_result.n_groups == 4
        assert colon_result.excluded_samples == []

    def test_ranking_matches_reference(
        self, colon_result: ModelBasedNormFinderResult
    ) -> None:
        assert colon_result.ranking == list(self.EXPECTED)

    def test_values_match_reference(
        self, colon_result: ModelBasedNormFinderResult
    ) -> None:
        for gene, expected in self.EXPECTED.items():
            assert colon_result.stability_values[gene] == pytest.approx(
                expected, abs=5e-8
            )

    def test_common_variance_matches_reference_port(
        self, colon_result: ModelBasedNormFinderResult
    ) -> None:
        assert colon_result.common_group_variance == pytest.approx(
            0.04603131533796071
        )


class TestModelBehavior:
    def test_group_shift_is_penalized(self) -> None:
        experiment = _make_experiment(
            {
                "STABLE": [20.0, 20.1, 19.9, 20.0, 20.1, 19.9],
                "SHIFTED": [20.0, 20.1, 19.9, 25.0, 25.1, 24.9],
                "NOISY": [18.0, 22.0, 20.0, 18.0, 22.0, 20.0],
            },
            groups=["control"] * 3 + ["treated"] * 3,
        )
        result = normfinder(experiment)
        assert result.ranking[0] == "STABLE"
        assert result.stability_values["SHIFTED"] > result.stability_values["STABLE"]

    def test_single_group_returns_nonnegative_finite_scores(self) -> None:
        experiment = _make_experiment(
            {
                "A": [20.0, 20.1, 19.9, 20.0],
                "B": [21.0, 20.8, 21.2, 21.0],
                "C": [19.0, 21.0, 18.0, 22.0],
            }
        )
        result = normfinder(experiment)
        assert result.n_groups == 1
        assert all(
            math.isfinite(value) and value >= 0
            for value in result.stability_values.values()
        )

    def test_missing_complete_case_is_reported(self) -> None:
        experiment = _make_experiment(
            {
                "A": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                "B": [21.0, 21.0, None, 21.0, 21.0, 21.0],
                "C": [22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
            },
            groups=["control"] * 3 + ["treated"] * 3,
        )
        result = normfinder(experiment)
        assert result.excluded_samples == ["S3"]
        assert result.n_samples == 5

    def test_group_with_one_complete_sample_is_rejected(self) -> None:
        experiment = _make_experiment(
            {
                "A": [20.0, 20.0, 20.0, 20.0],
                "B": [21.0, 21.0, 21.0, 21.0],
                "C": [22.0, 22.0, 22.0, 22.0],
            },
            groups=["control", "treated", "treated", "treated"],
        )
        with pytest.raises(ValueError, match="at least 2 complete samples per group"):
            normfinder(experiment)

    def test_two_candidates_are_rejected(self) -> None:
        experiment = _make_experiment(
            {"A": [20.0, 20.0, 20.0], "B": [21.0, 21.0, 21.0]}
        )
        with pytest.raises(ValueError, match="at least 3 candidate genes"):
            normfinder(experiment)

    def test_efficiency_correction_penalizes_misleading_raw_cq_pair(self) -> None:
        experiment = _make_experiment(
            {
                "A": [20.0, 21.0, 22.0, 23.0],
                "B": [20.0, 21.0, 22.0, 23.0],
                "C": [18.0, 19.1, 19.9, 21.0],
            },
            groups=["control", "control", "treated", "treated"],
        )
        experiment.assays["B"] = Assay(
            name="B", efficiency=0.5, is_reference=True
        )
        result = normfinder(experiment)
        assert result.ranking == ["A", "C", "B"]
        assert result.efficiencies["B"] == pytest.approx(0.5)
        assert result.quantity_scale == "log2_relative_quantity_efficiency_corrected"

    def test_missing_candidate_efficiency_is_rejected(self) -> None:
        experiment = _make_experiment(
            {
                "A": [20.0, 20.0, 20.0],
                "B": [21.0, 21.0, 21.0],
                "C": [22.0, 22.0, 22.0],
            }
        )
        experiment.assays["A"] = Assay(name="A", is_reference=True)
        with pytest.raises(ValueError, match="Measured amplification efficiency"):
            normfinder(experiment)
