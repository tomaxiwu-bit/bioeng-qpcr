"""High-level molecular docking engine.

This module ties together the parsing, scoring and search layers into a single
auditable entry point.  :func:`dock` takes a receptor :class:`Structure`, a
ligand :class:`Structure`, and a :class:`DockingBox`, runs the configured
search strategy (Monte Carlo simulated annealing by default, optionally a
deterministic grid scan), de-duplicates the resulting poses by heavy-atom RMSD,
and returns a :class:`DockResult` containing:

* the ranking of distinct poses (each with its transform, total score and
  per-term :class:`ScoreBreakdown`),
* run statistics (iterations, acceptance rate, convergence note, seed), and
* lightweight receptor/ligand metadata for the audit trail.

**Score convention (consistent everywhere):** lower (more negative) total
score means a better predicted binding pose.  The engine sorts poses ascending
by total score.

The seed is propagated deterministically: run ``k`` of ``n_runs`` is driven by
``seed + k`` so the whole multi-run campaign is reproducible from a single
integer, while still exploring distinct basins per run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioeng_qpcr.docking.scoring import (
    ScoringFunction,
    build_atom_table,
)
from bioeng_qpcr.docking.search import (
    Pose,
    SearchStats,
    cluster_poses,
    grid_search,
    monte_carlo_dock,
)
from bioeng_qpcr.docking.structure import (
    DockingBox,
    Structure,
    make_box_from_structures,
)

SCORE_INTERPRETATION = (
    "实验性刚性对接的简化经验评分；仅用于相同输入与设置下的姿势排序，"
    "不代表实测亲和力、结合自由能或已证实的结合能力。"
)


@dataclass(frozen=True)
class StructureMeta:
    """Compact metadata about one docking partner.

    Attributes
        n_atoms:
            Number of atoms.
        n_residues:
            Number of residues.
        residue_names:
            Sorted unique residue names.
        chain_ids:
            Sorted unique chain identifiers.
    """

    n_atoms: int
    n_residues: int
    residue_names: tuple[str, ...]
    chain_ids: tuple[str, ...]


@dataclass(frozen=True)
class DockResult:
    """Outcome of a docking calculation.

    Attributes
        poses:
            Top distinct poses, sorted by score ascending (lower is better).
        receptor_meta, ligand_meta:
            Metadata for the two partners.
        box:
            Search box actually used.
        stats:
            Aggregate run statistics across all runs.
        weights:
            Scoring weights actually applied (for auditability).
        scoring_name:
            Human-readable name of the scoring function class used.
    """

    poses: tuple[Pose, ...]
    receptor_meta: StructureMeta
    ligand_meta: StructureMeta
    box: DockingBox
    stats: SearchStats
    weights: tuple[float, float, float, float, float]
    scoring_name: str


def _meta(structure: Structure) -> StructureMeta:
    """Build a :class:`StructureMeta` summary of *structure*."""
    res_names = tuple(sorted({r.res_name for r in structure.residues}))
    chain_ids = tuple(sorted({a.chain_id for a in structure.atoms if a.chain_id}))
    return StructureMeta(
        n_atoms=structure.n_atoms,
        n_residues=len(structure.residues),
        residue_names=res_names,
        chain_ids=chain_ids,
    )


def dock(
    receptor: Structure,
    ligand: Structure,
    box: DockingBox | None = None,
    scoring: ScoringFunction | None = None,
    *,
    n_runs: int = 4,
    seed: int = 1,
    n_steps: int = 1500,
    use_grid: bool = False,
    grid_translation_steps: int = 3,
    grid_rotation_samples: int = 6,
    top_n: int = 10,
    rmsd_threshold: float = 2.0,
    padding: float = 8.0,
    ligand_center: np.ndarray | None = None,
) -> DockResult:
    """Dock a rigid ligand into a receptor and return ranked poses.

    Parameters
        receptor:
            Receptor :class:`Structure` (protein / nucleic acid).
        ligand:
            Ligand :class:`Structure` (small molecule).
        box:
            Search :class:`DockingBox`.  When ``None`` a box is auto-built
            around the receptor with *padding* margin via
            :func:`make_box_from_structures`.
        scoring:
            Scoring function; a default :class:`ScoringFunction` is used when
            ``None``.
        n_runs:
            Number of independent Monte Carlo campaigns (ignored for grid).
        seed:
            Base RNG seed; run ``k`` uses ``seed + k``.
        n_steps:
            Monte Carlo steps per run.
        use_grid:
            If ``True`` use the deterministic :func:`grid_search` instead of
            Monte Carlo.
        grid_translation_steps, grid_rotation_samples:
            Grid resolution when *use_grid* is ``True``.
        top_n:
            Maximum number of distinct poses returned.
        rmsd_threshold:
            Minimum pairwise RMSD for pose de-duplication.
        padding:
            Box margin (angstrom) when auto-building the box.
        ligand_center:
            Explicit reference centre for the ligand used by auto-boxing.

    Returns
        A :class:`DockResult` with ranked, de-duplicated poses and statistics.

    Raises
        ValueError:
            If either structure has zero atoms or a search parameter is invalid.
    """
    if receptor.n_atoms == 0:
        raise ValueError("Receptor structure contains no atoms.")
    if ligand.n_atoms == 0:
        raise ValueError("Ligand structure contains no atoms.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if rmsd_threshold < 0:
        raise ValueError("rmsd_threshold must be non-negative.")
    if use_grid:
        if grid_translation_steps < 1 or grid_rotation_samples < 1:
            raise ValueError("Grid search dimensions must be at least 1.")
    elif n_runs < 1 or n_steps < 1:
        raise ValueError("Monte Carlo n_runs and n_steps must be at least 1.")

    if scoring is None:
        scoring = ScoringFunction()
    if box is None:
        box = make_box_from_structures(
            receptor, ligand, padding=padding, ligand_center=ligand_center
        )

    rec_table = build_atom_table(receptor)
    lig_table = build_atom_table(ligand)

    all_poses: list[Pose] = []
    total_iterations = 0
    total_accepted = 0
    run_messages: list[str] = []
    run_convergence: list[bool] = []

    if use_grid:
        poses, stats = grid_search(
            rec_table,
            lig_table,
            box,
            scoring,
            translation_steps=grid_translation_steps,
            rotation_samples=grid_rotation_samples,
        )
        all_poses.extend(poses)
        total_iterations += stats.iterations
        total_accepted += stats.accepted
        run_messages.append(stats.message)
        run_convergence.append(stats.converged)
    else:
        for run in range(n_runs):
            run_seed = seed + run
            poses, stats = monte_carlo_dock(
                rec_table,
                lig_table,
                box,
                scoring,
                n_steps=n_steps,
                seed=run_seed,
            )
            all_poses.extend(poses)
            total_iterations += stats.iterations
            total_accepted += stats.accepted
            run_messages.append(f"run {run} (seed {run_seed}): {stats.message}")
            run_convergence.append(stats.converged)

    # Rank best-first (lower total score is better) then de-duplicate.
    all_poses.sort(key=lambda p: p.score)
    distinct = cluster_poses(all_poses, rmsd_threshold=rmsd_threshold)
    top_poses = distinct[:top_n]

    agg_stats = SearchStats(
        iterations=total_iterations,
        accepted=total_accepted,
        acceptance_rate=total_accepted / max(total_iterations, 1),
        seed=None if use_grid else seed,
        converged=bool(top_poses) and bool(run_convergence) and all(run_convergence),
        message=" | ".join(run_messages),
    )

    weights = (
        scoring.weights.vdw,
        scoring.weights.electrostatic,
        scoring.weights.hbond,
        scoring.weights.hydrophobic,
        scoring.weights.desolvation,
    )

    return DockResult(
        poses=tuple(top_poses),
        receptor_meta=_meta(receptor),
        ligand_meta=_meta(ligand),
        box=box,
        stats=agg_stats,
        weights=weights,
        scoring_name=type(scoring).__name__,
    )


def dock_multiple(
    receptor: Structure,
    ligands: list[Structure],
    box: DockingBox | None = None,
    scoring: ScoringFunction | None = None,
    **kwargs,
) -> list[DockResult]:
    """Dock several ligands against one receptor.

    Parameters
        receptor:
            Shared receptor structure.
        ligands:
            List of ligand structures.
        box, scoring:
            Optional shared search box / scoring (otherwise auto-built per
            ligand from *receptor*).
        kwargs:
            Forwarded to :func:`dock` (``n_runs``, ``seed``, ``n_steps``...).

    Returns
        A list of :class:`DockResult`, one per ligand in input order.
    """
    results: list[DockResult] = []
    for ligand in ligands:
        # Auto-box per ligand so each search region fits its own extent.
        shared_box = box
        if shared_box is None:
            shared_box = make_box_from_structures(
                receptor, ligand, padding=kwargs.get("padding", 8.0)
            )
        result = dock(
            receptor,
            ligand,
            box=shared_box,
            scoring=scoring,
            **kwargs,
        )
        results.append(result)
    return results


__all__ = [
    "StructureMeta",
    "DockResult",
    "dock",
    "dock_multiple",
]
