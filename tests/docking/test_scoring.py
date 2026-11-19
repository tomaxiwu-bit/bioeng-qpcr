"""Tests for empirical scoring terms and score breakdowns."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.docking.scoring import (
    AtomTable,
    ScoringFunction,
    ScoringWeights,
    build_atom_table,
    score_pose,
    transformed_table,
)
from bioeng_qpcr.docking.structure import Atom, RigidTransform, Structure

from .conftest import make_ligand, make_receptor


def _single_atom_table(element: str, xyz, charge=None) -> AtomTable:
    """Build an :class:`AtomTable` for one atom at *xyz*."""
    xyz = np.asarray(xyz, dtype=float)
    atom = Atom(
        serial=1,
        name=element,
        altloc=" ",
        res_name="TST",
        chain_id="A",
        res_seq=1,
        ins_code=" ",
        xyz=xyz,
        occupancy=1.0,
        temp_factor=0.0,
        element=element,
        charge="",
        is_hetatm=True,
    )
    structure = Structure(atoms=(atom,), residues=(), source="<test>")
    table = build_atom_table(structure)
    if charge is not None:
        table.charges[0] = charge
    return table


class TestVdwTerm:
    def _pair_score(self, elt_a, pos_a, elt_b, pos_b, **weights):
        rec = _single_atom_table(elt_a, pos_a)
        lig = _single_atom_table(elt_b, pos_b)
        w = ScoringWeights(
            vdw=1.0, electrostatic=0.0, hbond=0.0,
            hydrophobic=0.0, desolvation=0.0,
        )
        sf = ScoringFunction(weights=w)
        return sf.score_table(rec, lig).total

    def test_far_apart_is_near_zero(self):
        assert abs(self._pair_score("O", [0, 0, 0], "O", [0, 0, 20])) < 1e-3

    def test_equilibrium_is_attractive_negative(self):
        # O-O equilibrium distance ~ 2^(1/6) * rmin with rmin = 3.05 A.
        from bioeng_qpcr.docking.scoring import _pair_sigma

        sigma, _ = _pair_sigma("O", "O")
        r_eq = (2.0 ** (1.0 / 6.0)) * sigma
        score = self._pair_score("O", [0, 0, 0], "O", [0, 0, r_eq])
        assert score < 0.0

    def test_severe_overlap_is_repulsive_positive(self):
        score = self._pair_score("O", [0, 0, 0], "O", [0, 0, 0.3])
        assert score > 0.0

    def test_soft_core_avoids_inf(self):
        score = self._pair_score("O", [0, 0, 0], "O", [0, 0, 0.0])
        assert np.isfinite(score)


class TestElectrostaticTerm:
    def _pair_score(self, elt_a, pos_a, charge_a, elt_b, pos_b, charge_b):
        rec = _single_atom_table(elt_a, pos_a)
        lig = _single_atom_table(elt_b, pos_b)
        rec.charges[0] = charge_a
        lig.charges[0] = charge_b
        w = ScoringWeights(
            vdw=0.0, electrostatic=1.0, hbond=0.0,
            hydrophobic=0.0, desolvation=0.0,
        )
        sf = ScoringFunction(weights=w)
        return sf.score_table(rec, lig).total

    def test_opposite_charges_attractive(self):
        score = self._pair_score("O", [0, 0, 0], -0.5, "H", [0, 0, 2], 0.15)
        assert score < 0.0

    def test_same_charges_repulsive(self):
        score = self._pair_score("O", [0, 0, 0], -0.5, "N", [0, 0, 2], -0.3)
        assert score > 0.0

    def test_far_charges_negligible(self):
        score = self._pair_score("O", [0, 0, 0], -0.5, "H", [0, 0, 30], 0.15)
        assert abs(score) < 1e-3


class TestHbondTerm:
    def test_linear_hbond_favourable(self):
        # acceptor O at (0,0,-1); donor-heavy N at (0,0,2); donor H at (0,0,1).
        receptor = _single_atom_table("O", [0, 0, -1.0])
        ligand = _build_donor_table()
        w = ScoringWeights(
            vdw=0.0, electrostatic=0.0, hbond=1.0,
            hydrophobic=0.0, desolvation=0.0,
        )
        sf = ScoringFunction(weights=w)
        score = sf.score_table(receptor, ligand).total
        assert score < 0.0

    def test_far_hbond_negligible(self):
        receptor = _single_atom_table("O", [0, 0, -20.0])
        ligand = _build_donor_table()
        w = ScoringWeights(
            vdw=0.0, electrostatic=0.0, hbond=1.0,
            hydrophobic=0.0, desolvation=0.0,
        )
        sf = ScoringFunction(weights=w)
        score = sf.score_table(receptor, ligand).total
        assert abs(score) < 1e-3


class TestHydrophobicTerm:
    def test_optimal_contact_favourable(self):
        rec = _single_atom_table("C", [0, 0, 0])
        lig = _single_atom_table("C", [0, 0, 4.2])
        w = ScoringWeights(
            vdw=0.0, electrostatic=0.0, hbond=0.0,
            hydrophobic=1.0, desolvation=0.0,
        )
        sf = ScoringFunction(weights=w)
        assert sf.score_table(rec, lig).total < 0.0

    def test_far_contact_negligible(self):
        rec = _single_atom_table("C", [0, 0, 0])
        lig = _single_atom_table("C", [0, 0, 15.0])
        w = ScoringWeights(
            vdw=0.0, electrostatic=0.0, hbond=0.0,
            hydrophobic=1.0, desolvation=0.0,
        )
        sf = ScoringFunction(weights=w)
        assert abs(sf.score_table(rec, lig).total) < 1e-3


class TestScoreBreakdown:
    def test_total_equals_weighted_sum(self):
        receptor = make_receptor()
        ligand = make_ligand()
        sf = ScoringFunction()
        rec_t = build_atom_table(receptor)
        lig_t = build_atom_table(ligand)
        bd = sf.score_table(rec_t, lig_t)
        expected = (
            bd.vdw + bd.electrostatic + bd.hbond + bd.hydrophobic + bd.desolvation
        )
        assert bd.total == pytest.approx(expected, abs=1e-9)

    def test_score_pose_helper(self):
        receptor = make_receptor()
        ligand = make_ligand()
        bd = score_pose(
            receptor, ligand.coordinates, tuple(a.element for a in ligand.atoms)
        )
        assert np.isfinite(bd.total)

    def test_transformed_table_updates_coordinates(self):
        ligand = make_ligand()
        lig_t = build_atom_table(ligand)
        tf = RigidTransform(rotation=np.eye(3), translation=np.array([5.0, 0.0, 0.0]))

        moved = transformed_table(lig_t, tf)
        assert moved.coords[0] == pytest.approx(
            lig_t.coords[0] + np.array([5.0, 0.0, 0.0])
        )

    def test_transformed_table_rotates_donor_vectors(self):
        ligand = _build_donor_table()
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        moved = transformed_table(
            ligand, RigidTransform(rotation=rotation, translation=np.zeros(3))
        )
        assert moved.donor_vec[0] == pytest.approx(
            ligand.donor_vec[0] @ rotation.T
        )


def _build_donor_table() -> AtomTable:
    """Construct a ligand with a single donor H bound to an N (at 180 deg)."""
    from bioeng_qpcr.docking.structure import Atom, Structure

    h = Atom(
        serial=1, name="HN", altloc=" ", res_name="LIG", chain_id="A", res_seq=1,
        ins_code=" ", xyz=np.array([0.0, 0.0, 1.0]), occupancy=1.0, temp_factor=0.0,
        element="H", charge="", is_hetatm=True,
    )
    n = Atom(
        serial=2, name="NN", altloc=" ", res_name="LIG", chain_id="A", res_seq=1,
        ins_code=" ", xyz=np.array([0.0, 0.0, 2.0]), occupancy=1.0, temp_factor=0.0,
        element="N", charge="", is_hetatm=True,
    )
    structure = Structure(atoms=(h, n), residues=(), source="<test>")
    return build_atom_table(structure)
