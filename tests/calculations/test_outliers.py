"""Tests for Chauvenet-criterion technical-replicate outlier detection.

Synthetic replicate groups with known Cq values verify:

* The Chauvenet rejection mathematics (z-score, tail probability,
  threshold) against hand-calculated values.
* Iterative removal behaviour and the evidence recorded per flag.
* The review tier for groups whose spread exceeds ``sd_trigger`` without
  a statistical flag.
* Group formation, ordering, serialisation, and parameter validation.
"""

from __future__ import annotations

import math

import pytest

from bioeng_qpcr.calculations import (
    ChauvenetFlag,
    ReplicateGroupReport,
    ReplicateOutlierResult,
    detect_replicate_outliers,
)
from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.domain.enums import ReplicateType


def _experiment_with_groups(
    groups: dict[str, list[float | None]],
) -> Experiment:
    """Build an experiment from ``{group_name: [cq values]}`` maps.

    Group names use the ``sample/target`` convention; every well is a
    technical replicate within its own group.
    """
    wells: list[Well] = []
    row = 0
    for label, cqs in groups.items():
        sample_name, target_name = label.split("/", 1)
        replicate_group = f"{sample_name}_{target_name}"
        for offset, cq in enumerate(cqs):
            row += 1
            position = f"{chr(ord('A') + row - 1)}{offset + 1:02d}"
            wells.append(
                Well(
                    position=position,
                    sample_name=sample_name,
                    target_name=target_name,
                    cq=None if cq is None else float(cq),
                    replicate_type=ReplicateType.TECHNICAL,
                    replicate_group=replicate_group,
                    cq_status=(
                        "undetermined" if cq is None
                        else "determined" if cq >= 0
                        else "determined"
                    ),
                )
            )
    return Experiment(
        name="outlier-test",
        wells=wells,
        samples={name: Sample(name=name) for name in
                {w.sample_name for w in wells}},
        assays={name: Assay(name=name, efficiency=1.0) for name in
                {w.target_name for w in wells}},
    )


class TestChauvenetMath:
    def test_six_well_group_flags_one_outlier(self):
        # Five tight values plus one gross deviation: the classical case
        # Chauvenet was designed for.
        experiment = _experiment_with_groups(
            {"S1/T": [20.01, 20.02, 20.00, 20.03, 20.02, 21.30]}
        )

        result = detect_replicate_outliers(experiment)

        assert result.n_groups == 1
        group = result.groups[0]
        assert group.status == "chauvenet_flagged"
        assert len(group.chauvenet_flags) == 1
        flag = group.chauvenet_flags[0]
        assert flag.cq == pytest.approx(21.30)
        assert flag.removal_order == 1

        # Hand-checked statistics over all six values.
        values = [20.01, 20.02, 20.00, 20.03, 20.02, 21.30]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance)
        z = abs(21.30 - mean) / std
        tail = math.erfc(z / math.sqrt(2.0))
        assert flag.group_mean_cq == pytest.approx(mean)
        assert flag.group_std_cq == pytest.approx(std)
        assert flag.z_score == pytest.approx(z)
        assert flag.tail_probability == pytest.approx(tail)
        assert flag.threshold_probability == pytest.approx(1.0 / 12.0)
        assert flag.tail_probability < flag.threshold_probability

    def test_iterative_rejection_of_two_outliers(self):
        # Two gross outliers in a group of eight: the criterion should
        # reject the worst first, recompute, and reject the second.
        experiment = _experiment_with_groups(
            {
                "S1/T": [
                    20.01, 20.02, 20.00, 20.03, 20.02, 20.01, 23.40, 16.80,
                ]
            }
        )

        result = detect_replicate_outliers(experiment)

        group = result.groups[0]
        assert group.status == "chauvenet_flagged"
        assert len(group.chauvenet_flags) == 2
        orders = [flag.removal_order for flag in group.chauvenet_flags]
        assert orders == [1, 2]
        retained = [20.01, 20.02, 20.00, 20.03, 20.02, 20.01]
        assert group.retained_mean_cq == pytest.approx(
            sum(retained) / len(retained)
        )
        assert result.total_flagged == 2
        assert len(result.flagged_positions) == 2

    def test_identical_values_never_flag(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.00, 20.00, 20.00, 20.00]}
        )

        result = detect_replicate_outliers(experiment)

        group = result.groups[0]
        assert group.status == "pass"
        assert group.chauvenet_flags == ()
        assert group.std_cq == pytest.approx(0.0)

    def test_three_wells_cannot_flag_statistically(self):
        # With the sample standard deviation the maximum attainable
        # z-score for n=3 is 2/sqrt(3) ≈ 1.155, below the Chauvenet
        # threshold ≈ 1.383: no statistical flag is possible, however
        # extreme the third value looks.
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.00, 22.00]}
        )

        result = detect_replicate_outliers(experiment)

        group = result.groups[0]
        assert group.chauvenet_flags == ()
        assert group.status == "review"
        assert group.sd_trigger_exceeded is True
        assert group.review_candidate_position is not None
        flagged = next(
            w for w in experiment.wells
            if w.position == group.review_candidate_position
        )
        assert flagged.cq == pytest.approx(22.00)


class TestReviewTier:
    def test_spread_above_trigger_recommends_review(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.01, 20.02, 20.80]}
        )

        result = detect_replicate_outliers(experiment)

        group = result.groups[0]
        assert group.status == "review"
        assert group.chauvenet_flags == ()
        assert group.review_candidate_position is not None
        assert group.review_candidate_deviation == pytest.approx(0.795, abs=0.01)

    def test_two_well_group_review_has_no_candidate(self):
        # Two values are symmetric around their mean; neither well is
        # more deviant, so no candidate is named.
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.90]}
        )

        result = detect_replicate_outliers(experiment)

        group = result.groups[0]
        assert group.status == "review"
        assert group.review_candidate_position is None
        assert group.review_candidate_deviation is None

    def test_narrow_spread_passes(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.01, 20.02, 20.01, 20.00]}
        )

        result = detect_replicate_outliers(experiment)

        assert result.groups[0].status == "pass"
        assert result.n_pass == 1
        assert result.summary().startswith("1 group(s) assessed")

    def test_custom_sd_trigger(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.10, 20.20]}
        )

        default = detect_replicate_outliers(experiment)
        strict = detect_replicate_outliers(experiment, sd_trigger=0.05)

        assert default.groups[0].status == "pass"
        assert strict.groups[0].status == "review"

    def test_zero_sd_trigger(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.00, 20.01]}
        )

        result = detect_replicate_outliers(experiment, sd_trigger=0.0)

        assert result.groups[0].status == "review"


class TestGrouping:
    def test_undetermined_wells_excluded(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00, 20.01, 20.02, None]}
        )

        result = detect_replicate_outliers(experiment)

        assert result.n_groups == 1
        assert result.groups[0].n_wells == 3

    def test_single_well_group_not_reported(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.00], "S2/T": [20.00, 20.01]}
        )

        result = detect_replicate_outliers(experiment)

        labels = [group.label for group in result.groups]
        assert labels == ["S2/T/S2_T"]

    def test_groups_sorted_by_sample_target_group(self):
        experiment = _experiment_with_groups(
            {
                "S2/T1": [20.00, 20.01],
                "S1/T2": [20.00, 20.01],
                "S1/T1": [20.00, 20.01],
            }
        )

        result = detect_replicate_outliers(experiment)

        labels = [group.label for group in result.groups]
        assert labels == [
            "S1/T1/S1_T1",
            "S1/T2/S1_T2",
            "S2/T1/S2_T1",
        ]

    def test_separate_replicate_groups_kept_apart(self):
        # Two replicate groups under one sample/target pair must not be
        # merged (matching REP-001's grouping key).
        wells = [
            Well(
                position="A01",
                sample_name="S1",
                target_name="T",
                cq=20.00,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group="g1",
            ),
            Well(
                position="A02",
                sample_name="S1",
                target_name="T",
                cq=20.01,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group="g1",
            ),
            Well(
                position="B01",
                sample_name="S1",
                target_name="T",
                cq=25.00,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group="g2",
            ),
            Well(
                position="B02",
                sample_name="S1",
                target_name="T",
                cq=25.01,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group="g2",
            ),
        ]
        experiment = Experiment(name="split", wells=wells)

        result = detect_replicate_outliers(experiment)

        labels = [group.label for group in result.groups]
        assert labels == ["S1/T/g1", "S1/T/g2"]
        assert all(group.status == "pass" for group in result.groups)

    def test_non_technical_wells_ignored(self):
        wells = [
            Well(
                position="A01",
                sample_name="S1",
                target_name="T",
                cq=20.00,
                replicate_type=ReplicateType.SINGLE,
            ),
            Well(
                position="A02",
                sample_name="S1",
                target_name="T",
                cq=30.00,
                replicate_type=ReplicateType.SINGLE,
            ),
        ]
        experiment = Experiment(name="singles", wells=wells)

        result = detect_replicate_outliers(experiment)

        assert result.n_groups == 0
        assert result.groups == ()
        assert "No technical replicate group" in result.summary()

    def test_experiment_not_modified(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.01, 20.02, 20.00, 20.03, 20.02, 21.30]}
        )
        before = [well.position for well in experiment.wells]

        detect_replicate_outliers(experiment)

        assert [well.position for well in experiment.wells] == before
        assert len(experiment.wells) == 6


class TestParameterValidation:
    def test_min_group_size_below_three_rejected(self):
        experiment = _experiment_with_groups({"S1/T": [20.0, 20.1]})

        with pytest.raises(ValueError, match="min_group_size"):
            detect_replicate_outliers(experiment, min_group_size=2)

    def test_negative_sd_trigger_rejected(self):
        experiment = _experiment_with_groups({"S1/T": [20.0, 20.1]})

        with pytest.raises(ValueError, match="sd_trigger"):
            detect_replicate_outliers(experiment, sd_trigger=-0.1)


class TestSerialization:
    def test_to_dict_round_trip_fields(self):
        experiment = _experiment_with_groups(
            {"S1/T": [20.01, 20.02, 20.00, 20.03, 20.02, 21.30]}
        )

        result = detect_replicate_outliers(experiment)
        data = result.to_dict()

        assert data["summary"]["n_groups"] == 1
        assert data["summary"]["total_flagged_wells"] == 1
        assert len(data["summary"]["flagged_positions"]) == 1
        assert data["parameters"]["sd_trigger"] == pytest.approx(0.3)
        assert data["parameters"]["min_group_size"] == 3
        group = data["groups"][0]
        assert group["status"] == "chauvenet_flagged"
        assert group["sample_name"] == "S1"
        assert len(group["chauvenet_flags"]) == 1
        flag = group["chauvenet_flags"][0]
        assert flag["position"] == data["summary"]["flagged_positions"][0]
        assert flag["removal_order"] == 1

    def test_result_helpers(self):
        experiment = _experiment_with_groups(
            {
                "S1/T": [20.01, 20.02, 20.00, 20.03, 20.02, 21.30],
                "S2/T": [20.00, 20.90],
                "S3/T": [20.00, 20.01],
            }
        )

        result = detect_replicate_outliers(experiment)

        assert result.n_groups == 3
        assert result.total_flagged == 1
        assert len(result.chauvenet_groups) == 1
        assert len(result.review_groups) == 1
        assert result.review_groups[0].sample_name == "S2"
        assert result.n_pass == 1
        assert isinstance(result, ReplicateOutlierResult)
        assert isinstance(result.groups[0], ReplicateGroupReport)
        assert isinstance(result.groups[0].chauvenet_flags[0], ChauvenetFlag)
