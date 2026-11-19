"""Synthetic fixtures for docking tests.

All fixtures are generated procedurally in-memory (no network, no large data
files).  They provide a small protein-like receptor with a polar pocket, a
small ring-like ligand with heteroatoms, and a number of helpers used across
the docking test suite.
"""

from __future__ import annotations

import numpy as np

from bioeng_qpcr.docking.scoring import AtomTable, ScoreBreakdown
from bioeng_qpcr.docking.search import Pose
from bioeng_qpcr.docking.structure import (
    Atom,
    DockingBox,
    RigidTransform,
    Structure,
    centroid,
    kabsch_rmsd,
)

# (name, resname, chain, resseq, x, y, z, element, is_hetatm)
_RECEPTOR_SPEC = [
    ("N", "ALA", "A", 1, 0.0, 0.0, 8.0, "N", False),
    ("CA", "ALA", "A", 1, 0.7, 0.7, 7.0, "C", False),
    ("C", "ALA", "A", 1, 1.5, 0.0, 6.0, "C", False),
    ("O", "ALA", "A", 1, 2.5, 0.3, 5.5, "O", False),
    ("N", "SER", "A", 2, 1.5, -1.0, 5.0, "N", False),
    ("CA", "SER", "A", 2, 2.0, -1.5, 4.0, "C", False),
    ("C", "SER", "A", 2, 1.5, -2.5, 3.0, "C", False),
    ("O", "SER", "A", 2, 2.0, -3.5, 2.5, "O", False),
    ("OG", "SER", "A", 2, 0.5, -1.5, 4.5, "O", False),
    ("N", "ASP", "A", 3, 0.5, -3.0, 2.5, "N", False),
    ("CA", "ASP", "A", 3, 0.0, -3.5, 1.5, "C", False),
    ("C", "ASP", "A", 3, -0.5, -4.5, 0.5, "C", False),
    ("O", "ASP", "A", 3, -1.5, -4.8, -0.5, "O", False),
    ("OD1", "ASP", "A", 3, -1.5, -3.0, 1.0, "O", False),
    ("OD2", "ASP", "A", 3, 0.8, -4.8, -0.2, "O", False),
    ("N", "HIS", "A", 4, -1.0, -5.0, 0.0, "N", False),
    ("CA", "HIS", "A", 4, -1.5, -5.5, -1.0, "C", False),
    ("C", "HIS", "A", 4, -2.0, -6.5, -2.0, "C", False),
    ("O", "HIS", "A", 4, -3.0, -6.8, -3.0, "O", False),
    ("NE2", "HIS", "A", 4, -2.5, -5.0, -2.5, "N", False),
    ("ND1", "HIS", "A", 4, -0.5, -6.0, -1.5, "N", False),
]

# (name, x, y, z, element)
_LIGAND_SPEC = [
    ("C1", -1.0, 0.0, 0.0, "C"),
    ("C2", 0.0, -1.0, 0.0, "C"),
    ("C3", 1.0, 0.0, 0.0, "C"),
    ("C4", 0.0, 1.0, 0.0, "C"),
    ("N5", 0.0, 0.0, 1.0, "N"),
    ("O6", 0.0, 0.0, -1.0, "O"),
    ("H1", -2.0, 0.0, 0.0, "H"),
    ("H2", 0.0, -2.0, 0.0, "H"),
    ("H3", 2.0, 0.0, 0.0, "H"),
    ("H4", 0.0, 2.0, 0.0, "H"),
    ("HN", 0.0, 0.0, 2.0, "H"),
    ("HO", 0.0, 0.0, -2.0, "O"),
]


def _make_structure(
    spec,
    res_name: str = "LIG",
    chain: str = "A",
    res_seq: int = 5,
    hetatm: bool = False,
):
    atoms = []
    for i, entry in enumerate(spec, start=1):
        if len(entry) == 9:
            name, rname, ch, rseq, x, y, z, element, is_het = entry
        else:
            name, x, y, z, element = entry
            rname, ch, rseq, is_het = res_name, chain, res_seq, hetatm
        atoms.append(
            Atom(
                serial=i,
                name=name,
                altloc=" ",
                res_name=rname,
                chain_id=ch,
                res_seq=rseq,
                ins_code=" ",
                xyz=np.array([x, y, z], dtype=float),
                occupancy=1.0,
                temp_factor=0.0,
                element=element,
                charge="",
                is_hetatm=is_het,
            )
        )
    return Structure(atoms=tuple(atoms), source="<fixture>")


def make_receptor() -> Structure:
    """Return a small protein-like receptor with a polar pocket."""
    return _make_structure(_RECEPTOR_SPEC)


def make_ligand() -> Structure:
    """Return a small ring-like ligand with heteroatoms."""
    return _make_structure(_LIGAND_SPEC, hetatm=True)


def make_complex_pdb_string() -> str:
    """Return a combined PDB string of receptor + ligand as one structure."""
    from bioeng_qpcr.docking.structure import write_pdb_string

    rec = make_receptor()
    lig = make_ligand()
    combined = Structure(
        atoms=rec.atoms + lig.atoms,
        residues=rec.residues + lig.residues,
        source="<fixture>",
    )
    return write_pdb_string(combined)


# ---------------------------------------------------------------------------
# Analytic "energy well" scoring for deterministic search tests
# ---------------------------------------------------------------------------


class WellScoring:
    """Toy scoring that rewards matching a target point cloud (min RMSD 0).

    The total score is the squared Kabsch RMSD between the ligand coordinates
    and a fixed target point cloud, so the unique global minimum is exactly the
    target placement (RMSD 0).  This lets search tests assert that the engine
    recovers a *known* pose deterministically, independent of any empirical
    chemistry assumptions.
    """

    def __init__(self, target_coords: np.ndarray) -> None:
        self.target = np.asarray(target_coords, dtype=float)

    def score_table(self, receptor: AtomTable, ligand: AtomTable) -> ScoreBreakdown:
        rms = kabsch_rmsd(ligand.coords, self.target)
        total = float(rms**2)
        return ScoreBreakdown(
            vdw=0.0,
            electrostatic=0.0,
            hbond=0.0,
            hydrophobic=0.0,
            desolvation=0.0,
            total=total,
        )


def make_well_target(ligand: Structure) -> np.ndarray:
    """Return a target point cloud = ligand shifted to a known location.

    The target places the ligand centroid at ``(3.0, -2.0, 1.0)`` with its
    original orientation, so the global minimum is unambiguous.
    """
    coords = ligand.coordinates
    c = centroid(coords)
    shift = np.array([3.0, -2.0, 1.0], dtype=float) - c
    return coords + shift


# ---------------------------------------------------------------------------
# Minimal DockResult builders for validation tests
# ---------------------------------------------------------------------------


def make_pose_from_coords(
    coords: np.ndarray, score: float, transform: RigidTransform | None = None
) -> Pose:
    """Build a single :class:`Pose` from coordinates and a score."""
    if transform is None:
        transform = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
    return Pose(
        transform=transform,
        score=score,
        breakdown=ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, score),
        ligand_coords=np.asarray(coords, dtype=float),
    )


def make_box(center=(0.0, 0.0, 0.0), half=8.0) -> DockingBox:
    """Return a :class:`DockingBox` centered at *center* with half *half*."""
    return DockingBox(
        center=np.asarray(center, dtype=float),
        half_size=np.full(3, half, dtype=float),
    )
