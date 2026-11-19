"""Molecular docking for the bioeng-qpcr analysis engine.

Rigid-body ligand docking against a receptor with an auditable, dependency-free
(empirical) scoring pipeline.  The public API mirrors the rest of the package:
frozen dataclasses, NumPy-style docstrings, and explicit ``__all__`` exports.

Lower scores rank poses within the same empirical model and settings.
They are not measured affinities or experimentally validated binding energies.

Typical workflow::

    from bioeng_qpcr.docking import (
        parse_pdb,
        identify_receptor_ligand,
        dock,
        validate_docking,
    )

    structure = parse_pdb("complex.pdb")
    receptor, ligand = identify_receptor_ligand(structure)
    result = dock(receptor, ligand, n_runs=4, seed=1)
    report = validate_docking(result, receptor, ligand)
"""

from __future__ import annotations

from bioeng_qpcr.docking.engine import (
    SCORE_INTERPRETATION,
    DockResult,
    StructureMeta,
    dock,
    dock_multiple,
)
from bioeng_qpcr.docking.scoring import (
    AtomTable,
    ScoreBreakdown,
    ScoringFunction,
    ScoringWeights,
    build_atom_table,
    score_pose,
    transformed_table,
)
from bioeng_qpcr.docking.search import (
    Pose,
    SearchStats,
    cluster_poses,
    grid_search,
    monte_carlo_dock,
    random_quaternion,
)
from bioeng_qpcr.docking.structure import (
    Atom,
    DockingBox,
    Residue,
    RigidTransform,
    Structure,
    atom_to_pdb_line,
    centroid,
    identify_receptor_ligand,
    inertia_tensor,
    infer_element,
    kabsch_rmsd,
    make_box_from_structures,
    parse_pdb,
    parse_pdb_string,
    principal_axes,
    rmsd,
    transform_from_quaternion,
    write_pdb,
    write_pdb_string,
)
from bioeng_qpcr.docking.validation import (
    ValidationResult,
    check_binding_pocket,
    check_box_covers_ligand,
    check_favourable_top_pose,
    check_interface_contact,
    check_ligand_not_too_large,
    check_no_clash,
    check_partners_identified,
    check_rigid_geometry_preserved,
    check_sampling_adequacy,
    check_top_pose_in_box,
    summarize_docking,
    validate_docking,
)

__all__ = [
    "SCORE_INTERPRETATION",
    # structure
    "Atom",
    "Residue",
    "Structure",
    "RigidTransform",
    "DockingBox",
    "infer_element",
    "parse_pdb",
    "parse_pdb_string",
    "write_pdb",
    "write_pdb_string",
    "atom_to_pdb_line",
    "identify_receptor_ligand",
    "centroid",
    "inertia_tensor",
    "principal_axes",
    "rmsd",
    "kabsch_rmsd",
    "transform_from_quaternion",
    "make_box_from_structures",
    # scoring
    "ScoringWeights",
    "ScoreBreakdown",
    "AtomTable",
    "ScoringFunction",
    "build_atom_table",
    "transformed_table",
    "score_pose",
    # search
    "Pose",
    "SearchStats",
    "monte_carlo_dock",
    "grid_search",
    "cluster_poses",
    "random_quaternion",
    # engine
    "StructureMeta",
    "DockResult",
    "dock",
    "dock_multiple",
    # validation
    "ValidationResult",
    "check_box_covers_ligand",
    "check_top_pose_in_box",
    "check_no_clash",
    "check_sampling_adequacy",
    "check_ligand_not_too_large",
    "check_binding_pocket",
    "check_partners_identified",
    "check_favourable_top_pose",
    "check_rigid_geometry_preserved",
    "check_interface_contact",
    "validate_docking",
    "summarize_docking",
]
