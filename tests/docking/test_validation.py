"""Tests for the DCK-00x docking validation rules."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bioeng_qpcr.docking.engine import DockResult, StructureMeta
from bioeng_qpcr.docking.scoring import ScoreBreakdown
from bioeng_qpcr.docking.search import Pose, SearchStats
from bioeng_qpcr.docking.structure import DockingBox, RigidTransform, parse_pdb_string
from bioeng_qpcr.docking.validation import (
    ValidationResult,
    check_box_covers_ligand,
    check_favourable_top_pose,
    check_interface_contact,
    check_no_clash,
    check_rigid_geometry_preserved,
    check_sampling_adequacy,
    check_top_pose_in_box,
    summarize_docking,
    validate_docking,
)

from .conftest import make_complex_pdb_string, make_ligand, make_receptor


def _meta(n_atoms: int = 10) -> StructureMeta:
    return StructureMeta(
        n_atoms=n_atoms,
        n_residues=1,
        residue_names=("LIG",),
        chain_ids=("A",),
    )


def _stats() -> SearchStats:
    return SearchStats(
        iterations=100,
        accepted=50,
        acceptance_rate=0.5,
        seed=1,
        converged=True,
        message="test",
    )


def _pose(
    coords: np.ndarray, score: float, breakdown: ScoreBreakdown | None = None
) -> Pose:
    if breakdown is None:
        breakdown = ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, score)
    return Pose(
        transform=RigidTransform(rotation=np.eye(3), translation=np.zeros(3)),
        score=score,
        breakdown=breakdown,
        ligand_coords=np.asarray(coords, dtype=float),
    )


def _result(poses, box) -> DockResult:
    return DockResult(
        poses=tuple(poses),
        receptor_meta=_meta(22),
        ligand_meta=_meta(12),
        box=box,
        stats=_stats(),
        weights=(1.0, 0.5, 1.0, 0.8, 0.5),
        scoring_name="ScoringFunction",
    )


class TestClashRule:
    def test_detects_severe_overlap(self):
        receptor = make_receptor()
        ligand = make_ligand()
        # Place the ligand exactly on top of the receptor atoms -> clash.
        pose = _pose(receptor.coordinates, score=-5.0)
        result = _result(
            [pose], DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0))
        )
        rule = check_no_clash(result, receptor, ligand)
        assert rule.rule_id == "DCK-003"
        assert rule.passed is False
        assert rule.severity == "error"

    def test_passes_when_separated(self):
        receptor = make_receptor()
        ligand = make_ligand()
        displaced = ligand.coordinates + np.array([50.0, 0.0, 0.0])
        pose = _pose(displaced, score=-5.0)
        result = _result(
            [pose],
            DockingBox(center=np.array([50.0, 0.0, 0.0]), half_size=np.full(3, 20.0)),
        )
        rule = check_no_clash(result, receptor, ligand)
        assert rule.passed is True


class TestBoxCoverageRule:
    def test_fails_when_box_excludes_ligand(self):
        rec, lig = _real_split()
        far_box = DockingBox(
            center=np.array([200.0, 0.0, 0.0]), half_size=np.full(3, 5.0)
        )
        pose = _pose(lig.coordinates, score=-1.0)
        result = _result([pose], far_box)
        rule = check_box_covers_ligand(result, lig)
        assert rule.passed is False
        assert rule.rule_id == "DCK-001"

    def test_passes_when_box_covers(self):
        rec, lig = _real_split()
        center = np.mean(lig.coordinates, axis=0)
        box = DockingBox(center=center, half_size=np.full(3, 10.0))
        pose = _pose(lig.coordinates, score=-1.0)
        result = _result([pose], box)
        rule = check_box_covers_ligand(result, lig)
        assert rule.passed is True


class TestTopPoseInBox:
    def test_fails_when_pose_escapes(self):
        rec, lig = _real_split()
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 3.0))
        escaped = lig.coordinates + np.array([100.0, 0.0, 0.0])
        pose = _pose(escaped, score=-1.0)
        result = _result([pose], box)
        rule = check_top_pose_in_box(result)
        assert rule.passed is False


class TestSamplingAdequacy:
    def test_fails_with_single_pose(self):
        rec, lig = _real_split()
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0))
        pose = _pose(lig.coordinates, score=-1.0)
        result = _result([pose], box)
        rule = check_sampling_adequacy(result)
        assert rule.passed is False

    def test_passes_with_clear_gap(self):
        rec, lig = _real_split()
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0))
        poses = [
            _pose(lig.coordinates, score=-5.0),
            _pose(lig.coordinates + 1.0, score=-3.0),
        ]
        result = _result(poses, box)
        rule = check_sampling_adequacy(result)
        assert rule.passed is True

    @pytest.mark.parametrize("gap", [0.01, 20.0])
    @pytest.mark.parametrize("settled", [False, True])
    def test_score_gap_never_determines_convergence(self, gap, settled):
        _, lig = _real_split()
        result = _result(
            [_pose(lig.coordinates, -30), _pose(lig.coordinates + 1, -30 + gap)],
            DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0)),
        )
        result = replace(result, stats=replace(result.stats, converged=settled))
        check = check_sampling_adequacy(result)
        assert check.passed is settled
        assert check.evidence["gap"] == pytest.approx(gap)
        assert check.evidence["search_diagnostic_passed"] is settled
        review = summarize_docking(result, [check])
        assert review["status"] == ("exploratory" if settled else "review_required")
        assert "不等于" in review["interpretation"]
        if not settled:
            assert "增加每次运行步数" in review["next_steps"][0]

    def test_grid_enumeration_does_not_claim_global_convergence(self):
        _, lig = _real_split()
        result = _result(
            [_pose(lig.coordinates, -5), _pose(lig.coordinates + 1, -3)],
            DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0)),
        )
        result = replace(result, stats=replace(result.stats, seed=None))
        check = check_sampling_adequacy(result)
        assert check.passed
        assert "有限网格" in check.message and "不代表" in check.message
        assert summarize_docking(result, [check])["search_label"] == "网格枚举完成"


@pytest.mark.parametrize("state", ["error", "warning", "skipped", "empty"])
def test_review_never_labels_failed_or_unaudited_results_as_passed(state):
    _, ligand = _real_split()
    result = _result(
        [_pose(ligand.coordinates, -5)],
        DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0)),
    )
    checks = (
        []
        if state == "skipped"
        else [
            ValidationResult(
                "DCK-003", False, "error" if state == "error" else "warning", "test"
            )
        ]
    )
    if state == "empty":
        result = replace(result, poses=())
    review = summarize_docking(result, checks)
    assert (
        review["status"]
        == {
            "error": "blocked",
            "warning": "review_required",
            "skipped": "not_audited",
            "empty": "blocked",
        }[state]
    )
    assert review["next_steps"]


class TestFavourablePose:
    def test_fails_when_purely_repulsive(self):
        rec, lig = _real_split()
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 20.0))
        bd = ScoreBreakdown(
            vdw=2.0,
            electrostatic=1.0,
            hbond=0.0,
            hydrophobic=0.0,
            desolvation=0.5,
            total=3.5,
        )
        pose = _pose(lig.coordinates, score=3.5, breakdown=bd)
        result = _result([pose], box)
        rule = check_favourable_top_pose(result)
        assert rule.passed is False


class TestPoseBustersStyleGeometry:
    def test_detects_ligand_distortion_and_missing_contact(self):
        receptor = make_receptor()
        ligand = make_ligand()
        distorted = ligand.coordinates.copy()
        distorted[0] += np.array([0.5, 0.0, 0.0])
        far = distorted + np.array([100.0, 0.0, 0.0])
        result = _result(
            [_pose(far, score=-1.0)],
            DockingBox(center=np.array([100.0, 0.0, 0.0]), half_size=np.full(3, 20.0)),
        )

        assert check_rigid_geometry_preserved(result, ligand).passed is False
        assert check_interface_contact(result, receptor, ligand).passed is False


class TestFullValidation:
    def test_runs_all_rules_and_counts(self):
        rec, lig = _real_split()
        box = DockingBox(
            center=np.mean(lig.coordinates, axis=0), half_size=np.full(3, 15.0)
        )
        poses = [
            _pose(lig.coordinates, score=-5.0),
            _pose(lig.coordinates + 2.0, score=-3.0),
        ]
        result = _result(poses, box)
        rules = validate_docking(result, rec, lig)
        assert len(rules) == 10
        for r in rules:
            assert isinstance(r, ValidationResult)
            assert r.rule_id.startswith("DCK-")
            assert isinstance(r.passed, bool)


def _real_split():
    structure = parse_pdb_string(make_complex_pdb_string())
    from bioeng_qpcr.docking.structure import identify_receptor_ligand

    return identify_receptor_ligand(structure)
