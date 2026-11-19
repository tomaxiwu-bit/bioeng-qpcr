"""Conformational search for rigid-body molecular docking.

This module performs the *search* half of docking: it explores rigid-body
placements of a ligand inside a :class:`DockingBox` and returns ranked
candidates.  Two strategies are provided:

* **Monte Carlo with simulated annealing** (:func:`monte_carlo_dock`) — random
  translational + rotational (quaternion) moves are accepted/rejected with the
  Metropolis criterion at a linearly-annealed temperature.  This is the default
  production strategy.
* **Grid enumeration** (:func:`grid_search`) — a coarse, fully reproducible
  scan over a lattice of translations and a set of discrete rotations.  Useful
  for validation, audits, and small boxes.

Both strategies minimise the total score (see ``scoring.py``: *lower is
better*).  Results are collected as :class:`Pose` objects carrying the
:class:`RigidTransform`, the :class:`ScoreBreakdown`, and the resulting ligand
coordinates.

Because docking produces a *family* of similar low-energy poses, final results
are de-duplicated by heavy-atom RMSD via :func:`cluster_poses`, which greedily
keeps only poses that differ by more than ``rmsd_threshold`` from any already
selected pose.  This guarantees the top-N poses returned by the engine are
mutually distinct.

All randomness flows from an explicit ``numpy.random.Generator`` seeded by the
caller, so a fixed seed yields a byte-identical trajectory (auditability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioeng_qpcr.docking.scoring import (
    AtomTable,
    ScoreBreakdown,
    ScoringFunction,
    transformed_table,
)
from bioeng_qpcr.docking.structure import (
    DockingBox,
    RigidTransform,
    centroid,
    transform_from_quaternion,
)


@dataclass(frozen=True)
class Pose:
    """A single docking pose (one rigid ligand placement).

    Attributes
        transform:
            Rigid transform placing the ligand's original frame into the box.
        score:
            Weighted total score of the pose (lower is better).
        breakdown:
            Per-term :class:`ScoreBreakdown` for auditability.
        ligand_coords:
            Ligand heavy-atom coordinates after applying *transform*.
    """

    transform: RigidTransform
    score: float
    breakdown: ScoreBreakdown
    ligand_coords: np.ndarray

    def rmsd_to(self, other: Pose) -> float:
        """Return the receptor-frame RMSD between this pose and *other*.

        This is the docking RMSD used by CAPRI: the two ligand placements are
        compared *directly in the receptor coordinate frame*, with no
        superposition.

        Superimposing first (a Kabsch fit) would be a serious error here: a
        rigid ligand and its rotated/translated copy are congruent point sets,
        so their optimal-fit RMSD is zero no matter how different the binding
        modes are.  Pose de-duplication built on that would silently collapse
        every sampled orientation into a single "unique" pose.

        Parameters
            other:
                The pose to compare against.

        Returns
            Root-mean-square deviation over all ligand atoms (angstrom).

        Raises
            ValueError:
                If the two poses do not describe the same ligand atom set.
        """
        if self.ligand_coords.shape != other.ligand_coords.shape:
            raise ValueError(
                "Poses must describe the same ligand atom set, got shapes "
                f"{self.ligand_coords.shape} and {other.ligand_coords.shape}."
            )
        difference = self.ligand_coords - other.ligand_coords
        return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


@dataclass(frozen=True)
class SearchStats:
    """Bookkeeping statistics for a search run.

    Attributes
        iterations:
            Number of Monte Carlo steps taken (or grid points evaluated).
        accepted:
            Number of accepted *moves* (Monte Carlo only).  The initial pose
            is not a move and is therefore not counted, so
            ``accepted <= iterations`` always holds.
        acceptance_rate:
            ``accepted / iterations``, in ``[0, 1]``.
        seed:
            Seed that drove this run (``None`` for deterministic grid search).
        converged:
            ``True`` when the best score stopped improving during the final
            third of the search (Monte Carlo), or when the enumeration
            completed (grid).
        message:
            Human-readable note about the run.
    """

    iterations: int
    accepted: int
    acceptance_rate: float
    seed: int | None
    converged: bool
    message: str


# ---------------------------------------------------------------------------
# Move generators
# ---------------------------------------------------------------------------


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random unit quaternion ``(w, x, y, z)``.

    Uses the standard uniform-rotation sampling on the 4-sphere so that the
    induced rotations are isotropic (no pole bias).

    Parameters
        rng:
            Seeded NumPy random generator.

    Returns
        A unit quaternion, shape ``(4,)``.
    """
    u1, u2, u3 = rng.random(3)
    a = np.sqrt(1.0 - u1)
    b = np.sqrt(u1)
    q = np.array(
        [
            a * np.sin(2.0 * np.pi * u2),
            a * np.cos(2.0 * np.pi * u2),
            b * np.sin(2.0 * np.pi * u3),
            b * np.cos(2.0 * np.pi * u3),
        ],
        dtype=float,
    )
    return q


def _random_translation(box: DockingBox, rng: np.random.Generator) -> np.ndarray:
    """Sample a translation that keeps the ligand centroid inside *box*."""
    return box.random_point(rng)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3x3 rotation matrix."""
    q = random_quaternion(rng)
    return transform_from_quaternion(q, np.zeros(3)).rotation


def _random_transform(box: DockingBox, rng: np.random.Generator) -> RigidTransform:
    """Sample a fully random rigid transform with centroid inside *box*."""
    q = random_quaternion(rng)
    t = box.random_point(rng)
    return transform_from_quaternion(q, t)


# ---------------------------------------------------------------------------
# Monte Carlo with simulated annealing
# ---------------------------------------------------------------------------


def monte_carlo_dock(
    receptor: AtomTable,
    ligand: AtomTable,
    box: DockingBox,
    scoring: ScoringFunction,
    *,
    n_steps: int = 2000,
    seed: int | None = 1,
    start_temp: float = 5.0,
    end_temp: float = 0.05,
    trans_step: float = 2.0,
    rot_step: float = 0.4,
    recenter: bool = True,
) -> tuple[list[Pose], SearchStats]:
    """Dock a rigid ligand into *box* via Monte Carlo simulated annealing.

    The ligand centroid is initialised at the box centre (optionally moved to
    its own centroid first) and randomised; the temperature is annealed
    linearly from *start_temp* to *end_temp* across *n_steps*.  Each step
    proposes a small translation (length up to *trans_step*) and rotation
    (magnitude up to *rot_step*) and is accepted with the Metropolis criterion.

    Parameters
        receptor:
            Fixed receptor :class:`AtomTable`.
        ligand:
            Ligand :class:`AtomTable` (its coordinates are the untransformed
            reference frame).
        box:
            Search region; translations are drawn so the ligand centroid stays
            inside it.
        scoring:
            Scoring function to minimise.
        n_steps:
            Number of Monte Carlo iterations.
        seed:
            RNG seed for reproducibility; ``None`` uses fresh entropy.
        start_temp, end_temp:
            Annealing temperature bounds (score units).
        trans_step:
            Max translation per step (angstrom).
        rot_step:
            Max rotation angle per step (radian).
        recenter:
            If ``True``, pre-center the ligand on its centroid so the rotation
            pivot is the ligand centre of mass.

    Returns
        A ``(poses, stats)`` tuple where *poses* is the full list of accepted
        poses (including the initial) in visitation order, and *stats* carries
        run bookkeeping.
    """
    rng = np.random.default_rng(seed)
    n = ligand.coords.shape[0]

    # Optionally recenter ligand so rotations pivot about its centroid.
    if recenter and n > 0:
        lig_center = centroid(ligand.coords)
        moved_coords = ligand.coords - lig_center
        lig_table = AtomTable(
            coords=moved_coords,
            elements=ligand.elements,
            charges=ligand.charges,
            is_acceptor=ligand.is_acceptor,
            is_donor_h=ligand.is_donor_h,
            donor_vec=ligand.donor_vec,
            is_hydrophobic=ligand.is_hydrophobic,
            is_polar=ligand.is_polar,
        )
    else:
        lig_table = ligand
        lig_center = np.zeros(3)

    current_tf = transform_from_quaternion(
        random_quaternion(rng), box.random_point(rng)
    )
    current_table = transformed_table(lig_table, current_tf)
    current_coords = current_table.coords
    current_tf_eff = RigidTransform(
        rotation=current_tf.rotation,
        translation=current_tf.translation - lig_center @ current_tf.rotation.T,
    )
    current_score = scoring.score_table(receptor, current_table)

    poses: list[Pose] = [
        Pose(
            transform=current_tf_eff,
            score=current_score.total,
            breakdown=current_score,
            ligand_coords=current_coords,
        )
    ]
    accepted = 0
    best_score = current_score.total
    tail_start = n_steps - max(1, n_steps // 3)
    best_at_tail_start = best_score

    for step in range(n_steps):
        if step == tail_start:
            best_at_tail_start = best_score
        frac = (step + 1) / max(n_steps, 1)
        temp = start_temp * (1.0 - frac) + end_temp * frac
        # Propose a move around the current transform: a small random rotation
        # (random axis, magnitude <= rot_step) composed with a small translation.
        axis = rng.standard_normal(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / axis_norm
        angle = rng.uniform(0.0, rot_step)
        w = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
        move_q = np.array([w, s * axis[0], s * axis[1], s * axis[2]], dtype=float)
        # Compose: R_new = R_move @ R_current.
        new_rot = _compose_quaternions(move_q, current_tf.rotation)
        delta_t = rng.uniform(-trans_step, trans_step, size=3)
        new_t = current_tf.translation + delta_t

        new_tf = RigidTransform(rotation=new_rot, translation=new_t)
        new_table = transformed_table(lig_table, new_tf)
        new_coords = new_table.coords
        # Enforce box containment of the ligand centroid.
        new_centroid = centroid(new_coords)
        if not box.contains(new_centroid):
            # Reflect the offending translation back into the box (soft reject).
            new_t = new_t - (new_centroid - box.center)
            new_tf = RigidTransform(rotation=new_rot, translation=new_t)
            new_table = transformed_table(lig_table, new_tf)
            new_coords = new_table.coords
            new_centroid = centroid(new_coords)
            if not box.contains(new_centroid):
                continue

        new_tf_eff = RigidTransform(
            rotation=new_tf.rotation,
            translation=new_tf.translation - lig_center @ new_tf.rotation.T,
        )
        new_score = scoring.score_table(receptor, new_table)

        delta = new_score.total - current_score.total
        if delta <= 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-9))):
            current_tf = new_tf
            current_score = new_score
            current_coords = new_coords
            accepted += 1
            best_score = min(best_score, new_score.total)
            poses.append(
                Pose(
                    transform=new_tf_eff,
                    score=new_score.total,
                    breakdown=new_score,
                    ligand_coords=new_coords,
                )
            )

    # The anneal has settled when the best score stopped improving over its
    # final third; the tolerance is scaled so it also behaves on score scales
    # far from unity.
    tail_tolerance = 1e-9 + 1e-6 * abs(best_score)
    converged = n_steps > 0 and best_at_tail_start - best_score <= tail_tolerance
    message = (
        f"Monte Carlo annealing completed {n_steps} steps; best score {best_score:.4f}."
    )
    if not converged:
        message += (
            " The best score was still improving at the end of the schedule; "
            "increase n_steps before trusting the ranking."
        )
    stats = SearchStats(
        iterations=n_steps,
        accepted=accepted,
        acceptance_rate=accepted / max(n_steps, 1),
        seed=seed,
        converged=bool(converged),
        message=message,
    )
    return poses, stats


def _compose_quaternions(q: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Compose a quaternion *q* (applied first) with an existing rotation.

    Returns the combined rotation matrix ``R_q @ rotation_matrix``.
    """
    tf = transform_from_quaternion(q, np.zeros(3))
    return tf.rotation @ rotation_matrix


# ---------------------------------------------------------------------------
# Grid enumeration
# ---------------------------------------------------------------------------


def grid_search(
    receptor: AtomTable,
    ligand: AtomTable,
    box: DockingBox,
    scoring: ScoringFunction,
    *,
    translation_steps: int = 3,
    rotation_samples: int = 6,
    recenter: bool = True,
) -> tuple[list[Pose], SearchStats]:
    """Exhaustively enumerate a coarse grid of rigid transformations.

    Translations sample a ``translation_steps^3`` lattice spanning the box;
    rotations sample ``rotation_samples`` orientations about each Cartesian
    axis.  This is fully deterministic (no RNG), which makes it ideal for
    audits and for testing convergence of the stochastic strategy.

    Parameters
        receptor:
            Fixed receptor :class:`AtomTable`.
        ligand:
            Ligand :class:`AtomTable`.
        box:
            Search region.
        scoring:
            Scoring function to minimise.
        translation_steps:
            Number of lattice points per axis (>=1).
        rotation_samples:
            Number of discrete rotation angles tested per axis.
        recenter:
            Pre-center the ligand on its centroid before sampling.

    Returns
        A ``(poses, stats)`` tuple with every enumerated pose and its stats.
    """
    if recenter and ligand.coords.shape[0] > 0:
        lig_center = centroid(ligand.coords)
        moved_coords = ligand.coords - lig_center
        lig_table = AtomTable(
            coords=moved_coords,
            elements=ligand.elements,
            charges=ligand.charges,
            is_acceptor=ligand.is_acceptor,
            is_donor_h=ligand.is_donor_h,
            donor_vec=ligand.donor_vec,
            is_hydrophobic=ligand.is_hydrophobic,
            is_polar=ligand.is_polar,
        )
    else:
        lig_table = ligand
        lig_center = np.zeros(3)

    # Translation lattice.
    t_vals = np.linspace(
        box.center[0] - box.half_size[0],
        box.center[0] + box.half_size[0],
        max(translation_steps, 1),
    )
    translations = [
        np.array([tx, ty, tz], dtype=float)
        for tx in t_vals
        for ty in t_vals
        for tz in t_vals
    ]

    # Rotation samples about each axis.
    angles = np.linspace(0.0, 2.0 * np.pi, max(rotation_samples, 1), endpoint=False)
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    rotations: list[np.ndarray] = [np.eye(3)]
    for axis in axes:
        for ang in angles:
            c, s = np.cos(ang), np.sin(ang)
            if axis[0] == 1.0:
                rm = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
            elif axis[1] == 1.0:
                rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
            else:
                rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
            rotations.append(rm)

    poses: list[Pose] = []
    iterations = 0
    for translation in translations:
        for rotation in rotations:
            iterations += 1
            search_tf = RigidTransform(rotation=rotation, translation=translation)
            table = transformed_table(lig_table, search_tf)
            coords = table.coords
            tf = RigidTransform(
                rotation=rotation,
                translation=translation - lig_center @ rotation.T,
            )
            score = scoring.score_table(receptor, table)
            poses.append(
                Pose(
                    transform=tf,
                    score=score.total,
                    breakdown=score,
                    ligand_coords=coords,
                )
            )

    stats = SearchStats(
        iterations=iterations,
        # A grid enumeration has no accept/reject step, so nothing is
        # "accepted"; every placement is simply evaluated.
        accepted=0,
        acceptance_rate=0.0,
        seed=None,
        converged=True,
        message=f"Grid search enumerated {iterations} rigid placements.",
    )
    return poses, stats


# ---------------------------------------------------------------------------
# Pose de-duplication by RMSD
# ---------------------------------------------------------------------------


def cluster_poses(
    poses: list[Pose],
    rmsd_threshold: float = 2.0,
) -> list[Pose]:
    """Greedily select mutually-distinct poses from a candidate list.

    The input is assumed to be sorted best-first.  The first (best) pose is
    always selected; each subsequent pose is kept only if its Kabsch RMSD to
    every already-selected pose exceeds *rmsd_threshold*.  This removes
    near-duplicate solutions so the returned top-N are structurally diverse.

    Parameters
        poses:
            Candidate poses, ideally sorted by score ascending.
        rmsd_threshold:
            Minimum pairwise RMSD (angstrom) required to keep a new pose.

    Returns
        A list of distinct poses, preserving the input ranking order.
    """
    if not poses:
        return []
    selected: list[Pose] = [poses[0]]
    for pose in poses[1:]:
        if all(pose.rmsd_to(chosen) > rmsd_threshold for chosen in selected):
            selected.append(pose)
    return selected


__all__ = [
    "Pose",
    "SearchStats",
    "monte_carlo_dock",
    "grid_search",
    "cluster_poses",
    "random_quaternion",
]
