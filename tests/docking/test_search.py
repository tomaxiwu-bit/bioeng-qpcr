"""Tests for rigid-body search, simulated annealing and pose clustering."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.docking.scoring import ScoreBreakdown, build_atom_table
from bioeng_qpcr.docking.search import (
    Pose,
    cluster_poses,
    grid_search,
    monte_carlo_dock,
    random_quaternion,
)
from bioeng_qpcr.docking.structure import RigidTransform, kabsch_rmsd

from .conftest import WellScoring, make_box, make_ligand, make_well_target


class TestRandomQuaternion:
    def test_unit_norm(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            q = random_quaternion(rng)
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)

    def test_orthonormal_rotation(self):
        rng = np.random.default_rng(7)
        q = random_quaternion(rng)
        tf = RigidTransform(rotation=_quat_to_matrix(q), translation=np.zeros(3))
        assert tf.rotation @ tf.rotation.T == pytest.approx(np.eye(3), abs=1e-9)


class TestMonteCarloWell:
    @pytest.mark.parametrize(
        "pattern, expected",
        [
            ([0, -1, -2, -3, -4, -4, -4], True),
            ([0, 0, 0, 0, 0, -1, -2], False),
            ([0, 0, 0, 0, 0, -0.8e-9, -1.6e-9], False),
        ],
    )
    def test_diagnostic_uses_cumulative_improvement_in_final_third(
        self, pattern, expected
    ):
        class ScheduledScoring:
            def __init__(self):
                self.values = iter(pattern)

            def score_table(self, receptor, ligand):
                return ScoreBreakdown(0, 0, 0, 0, 0, next(self.values))

        _, stats = monte_carlo_dock(
            build_atom_table(_dummy_receptor()),
            build_atom_table(make_ligand()),
            make_box(half=100),
            ScheduledScoring(),
            n_steps=6,
            seed=1,
        )
        assert stats.converged is expected

    def test_recentered_pose_stays_in_box_and_transform_matches(self):
        ligand = make_ligand()
        shifted = build_atom_table(ligand)
        shifted.coords[:] += np.array([20.0, -10.0, 5.0])
        box = make_box(center=[0.0, 0.0, 0.0], half=2.0)
        poses, _ = monte_carlo_dock(
            build_atom_table(_dummy_receptor()),
            shifted,
            box,
            WellScoring(np.zeros_like(shifted.coords)),
            n_steps=0,
            seed=4,
        )
        pose = poses[0]
        assert box.contains(np.mean(pose.ligand_coords, axis=0))
        assert pose.transform.apply(shifted.coords) == pytest.approx(pose.ligand_coords)

    def test_finds_known_target_pose(self):
        ligand = make_ligand()
        target = make_well_target(ligand)
        rec_table = build_atom_table(_dummy_receptor())
        lig_table = build_atom_table(ligand)
        box = make_box(center=[3.0, -2.0, 1.0], half=8.0)

        scoring = WellScoring(target)
        poses, stats = monte_carlo_dock(
            rec_table, lig_table, box, scoring, n_steps=6000, seed=1
        )
        best = min(poses, key=lambda p: p.score)
        best_coords = best.ligand_coords
        rms = kabsch_rmsd(best_coords, target)
        assert rms < 2.0
        assert best.score < 0.5  # well score is squared RMSD

    def test_reproducible_with_same_seed(self):
        ligand = make_ligand()
        target = make_well_target(ligand)
        rec_table = build_atom_table(_dummy_receptor())
        lig_table = build_atom_table(ligand)
        box = make_box(center=[3.0, -2.0, 1.0], half=8.0)
        scoring = WellScoring(target)

        poses_a, _ = monte_carlo_dock(
            rec_table, lig_table, box, scoring, n_steps=2000, seed=42
        )
        poses_b, _ = monte_carlo_dock(
            rec_table, lig_table, box, scoring, n_steps=2000, seed=42
        )
        best_a = min(poses_a, key=lambda p: p.score)
        best_b = min(poses_b, key=lambda p: p.score)
        assert best_a.score == pytest.approx(best_b.score, abs=1e-9)
        assert best_a.transform.translation == pytest.approx(
            best_b.transform.translation, abs=1e-9
        )
        assert best_a.transform.rotation == pytest.approx(
            best_b.transform.rotation, abs=1e-9
        )

    def test_acceptance_rate_recorded(self):
        ligand = make_ligand()
        target = make_well_target(ligand)
        rec_table = build_atom_table(_dummy_receptor())
        lig_table = build_atom_table(ligand)
        box = make_box(center=[3.0, -2.0, 1.0], half=8.0)
        _, stats = monte_carlo_dock(
            rec_table, lig_table, box, WellScoring(target), n_steps=500, seed=1
        )
        assert 0.0 <= stats.acceptance_rate <= 1.0
        assert stats.iterations == 500


class TestGridSearch:
    def test_recentered_pose_transform_matches_original_frame(self):
        ligand = build_atom_table(make_ligand())
        ligand.coords[:] += np.array([20.0, -10.0, 5.0])
        box = make_box(center=[0.0, 0.0, 0.0], half=2.0)
        poses, _ = grid_search(
            build_atom_table(_dummy_receptor()),
            ligand,
            box,
            WellScoring(np.zeros_like(ligand.coords)),
            translation_steps=1,
            rotation_samples=1,
        )
        for pose in poses:
            assert box.contains(np.mean(pose.ligand_coords, axis=0))
            assert pose.transform.apply(ligand.coords) == pytest.approx(
                pose.ligand_coords
            )

    def test_deterministic_and_finds_target(self):
        ligand = make_ligand()
        target = make_well_target(ligand)
        rec_table = build_atom_table(_dummy_receptor())
        lig_table = build_atom_table(ligand)
        box = make_box(center=[3.0, -2.0, 1.0], half=8.0)
        poses_a, _ = grid_search(
            rec_table,
            lig_table,
            box,
            WellScoring(target),
            translation_steps=3,
            rotation_samples=6,
        )
        poses_b, _ = grid_search(
            rec_table,
            lig_table,
            box,
            WellScoring(target),
            translation_steps=3,
            rotation_samples=6,
        )
        best_a = min(poses_a, key=lambda p: p.score)
        best_b = min(poses_b, key=lambda p: p.score)
        assert best_a.score == pytest.approx(best_b.score, abs=1e-12)
        assert best_a.score < 4.0  # coarse grid should still get reasonably close


class TestClusterPoses:
    def test_removes_near_duplicate(self):
        base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        poses = [
            _pose_with_coords(base, score=0.0),
            # nearly identical orientation (tiny rotation) -> duplicate
            _pose_with_coords(_rot_z(0.02, base), score=0.1),
            # clearly distinct orientation (90 deg) -> kept
            _pose_with_coords(_rot_z(np.pi / 2, base), score=0.2),
        ]
        # poses must be sorted best-first by score for clustering semantics.
        poses.sort(key=lambda p: p.score)
        selected = cluster_poses(poses, rmsd_threshold=2.0)
        assert len(selected) == 2

    def test_empty_input(self):
        assert cluster_poses([], rmsd_threshold=2.0) == []

    def test_keeps_all_distinct(self):
        base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        # Five distinct rotations; each differs structurally from the first.
        poses = [
            _pose_with_coords(_rot_z(angle, base), score=float(i))
            for i, angle in enumerate([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ]
        selected = cluster_poses(poses, rmsd_threshold=1.0)
        assert len(selected) == 5


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )


def _dummy_receptor():
    """A trivial 1-atom receptor so search tests focus on the ligand/well."""
    from bioeng_qpcr.docking.structure import Atom, Structure

    atom = Atom(
        serial=1,
        name="O",
        altloc=" ",
        res_name="DUM",
        chain_id="A",
        res_seq=1,
        ins_code=" ",
        xyz=np.array([100.0, 100.0, 100.0]),
        occupancy=1.0,
        temp_factor=0.0,
        element="O",
        charge="",
        is_hetatm=False,
    )
    return Structure(atoms=(atom,), residues=(), source="<dummy>")


def _pose_with_coords(coords: np.ndarray, score: float) -> Pose:
    from bioeng_qpcr.docking.scoring import ScoreBreakdown

    tf = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
    return Pose(
        transform=tf,
        score=score,
        breakdown=ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, score),
        ligand_coords=np.asarray(coords, dtype=float),
    )


def _rot_z(angle: float, coords: np.ndarray) -> np.ndarray:
    """Rotate *coords* about the z-axis by *angle* (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return np.asarray(coords, dtype=float) @ rot.T
