"""Known-answer tests for efficiency-corrected normalized quantities."""

from __future__ import annotations

import math

import pytest

from bioeng_qpcr.calculations import (
    compare_groups,
    normalized_quantity,
    normalized_quantity_all,
)
from bioeng_qpcr.domain import Assay, Experiment, Sample, Well


def _experiment(*, incomplete: bool = False) -> Experiment:
    assays = {
        "GOI": Assay(name="GOI", efficiency=1.0),
        "GAPDH": Assay(name="GAPDH", efficiency=1.0, is_reference=True),
        "RPLP0": Assay(name="RPLP0", efficiency=1.0, is_reference=True),
    }
    samples = {
        "C1": Sample(name="C1", group="control"),
        "C2": Sample(name="C2", group="control"),
        "T1": Sample(name="T1", group="treated"),
        "T2": Sample(name="T2", group="treated"),
    }
    wells: list[Well] = []
    position = 1
    for sample_name in samples:
        target_cq = 24.0 if sample_name.startswith("T") else 25.0
        if sample_name.endswith("2"):
            target_cq += 0.2
        values = {"GOI": target_cq, "GAPDH": 20.0, "RPLP0": 21.0}
        if incomplete and sample_name == "T2":
            values.pop("RPLP0")
        for target_name, cq in values.items():
            wells.append(
                Well(
                    position=f"A{position:02d}",
                    sample_name=sample_name,
                    target_name=target_name,
                    cq=cq,
                )
            )
            position += 1
    return Experiment(
        name="known-answer",
        wells=wells,
        samples=samples,
        assays=assays,
    )


def test_multiple_references_use_geometric_quantity_mean() -> None:
    result = normalized_quantity(_experiment(), "C1", "GOI")
    assert result.log2_normalized_quantity == pytest.approx(-4.5)
    assert result.normalized_quantity == pytest.approx(2.0**-4.5)
    assert result.reference_targets == ("GAPDH", "RPLP0")


def test_group_comparison_uses_independent_sample_values() -> None:
    result, audit = compare_groups(
        _experiment(),
        "GOI",
        treated_group="treated",
        control_group="control",
    )
    assert result.fold_difference == pytest.approx(2.0)
    assert result.log2_fold_difference == pytest.approx(1.0)
    assert set(result.treated_values) == {"T1", "T2"}
    assert set(result.control_values) == {"C1", "C2"}
    assert not math.isnan(result.welch_t_statistic)
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert audit.is_complete


def test_batch_preserves_incomplete_sample_exclusion() -> None:
    batch = normalized_quantity_all(_experiment(incomplete=True), "GOI")
    assert {result.sample_name for result in batch.results} == {"C1", "C2", "T1"}
    assert len(batch.exclusions) == 1
    assert batch.exclusions[0].sample_name == "T2"
    assert batch.exclusions[0].code == "INCOMPLETE_NORMALIZATION_INPUT"
    with pytest.raises(ValueError, match="T2"):
        batch.raise_for_exclusions()


def test_efficiency_is_required() -> None:
    experiment = _experiment()
    experiment.assays["GOI"] = Assay(name="GOI", efficiency=None)
    with pytest.raises(ValueError, match="Measured amplification efficiency"):
        normalized_quantity(experiment, "C1", "GOI")
