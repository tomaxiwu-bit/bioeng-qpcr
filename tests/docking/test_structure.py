"""Tests for PDB parsing, writing, geometry and RMSD utilities."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.docking.structure import (
    DockingBox,
    RigidTransform,
    centroid,
    identify_receptor_ligand,
    infer_element,
    kabsch_rmsd,
    parse_pdb_string,
    principal_axes,
    rmsd,
    transform_from_quaternion,
    write_pdb_string,
)

from .conftest import make_complex_pdb_string, make_receptor


class TestPdbParsing:
    def test_parse_recovers_atom_count(self):
        structure = parse_pdb_string(make_complex_pdb_string())
        assert structure.n_atoms == 21 + 12

    def test_parse_recovers_element(self):
        structure = parse_pdb_string(make_complex_pdb_string())
        elements = {a.element for a in structure.atoms}
        assert "C" in elements and "N" in elements and "O" in elements

    def test_parse_recovers_coordinates(self):
        structure = parse_pdb_string(make_complex_pdb_string())
        first = structure.atoms[0]
        assert first.xyz[0] == pytest.approx(0.0, abs=1e-6)
        assert first.xyz[1] == pytest.approx(0.0, abs=1e-6)
        assert first.xyz[2] == pytest.approx(8.0, abs=1e-3)

    def test_parse_recovers_residue_name_and_serial(self):
        structure = parse_pdb_string(make_complex_pdb_string())
        assert structure.atoms[0].res_name == "ALA"
        assert structure.atoms[0].serial == 1
        lig = [a for a in structure.atoms if a.is_hetatm]
        assert all(a.res_name == "LIG" for a in lig)

    def test_element_inference_when_blank(self):
        line = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        structure = parse_pdb_string(line)
        assert structure.atoms[0].element == "CA"

    def test_round_trip_write_then_read(self):
        original = make_receptor()
        text = write_pdb_string(original)
        reparsed = parse_pdb_string(text)
        assert reparsed.n_atoms == original.n_atoms
        for a, b in zip(original.atoms, reparsed.atoms, strict=True):
            assert a.element == b.element
            assert a.res_name == b.res_name
            assert a.serial == b.serial
            assert a.xyz == pytest.approx(b.xyz, abs=1e-6)

    def test_atom_limit_stops_parsing_early(self):
        text = make_complex_pdb_string()
        with pytest.raises(ValueError, match="more than 10 atoms"):
            parse_pdb_string(text, max_atoms=10)


class TestInferElement:
    def test_simple_carbon(self):
        assert infer_element("CA", "ALA") == "CA"

    def test_nitrogen(self):
        assert infer_element("N", "SER") == "N"

    def test_oxygen(self):
        assert infer_element("OG", "SER") == "O"

    def test_digit_prefixed_name(self):
        # "1HB" -> second character is the element (H).
        assert infer_element("1HB") == "H"

    def test_iron_is_two_letter(self):
        # Element column would normally carry "FE"; via name we still map it.
        assert infer_element("FE", "HEM") == "FE"


class TestGeometry:
    def test_centroid_of_origin_points(self):
        coords = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        assert centroid(coords) == pytest.approx([0.0, 2.0 / 3.0, 0.0])

    def test_centroid_empty(self):
        assert centroid(np.zeros((0, 3))).shape == (3,)

    def test_principal_axes_orthogonal(self):
        coords = np.array(
            [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        _, axes = principal_axes(coords)
        # The 3x3 axis matrix must be orthonormal.
        assert axes @ axes.T == pytest.approx(np.eye(3), abs=1e-9)

    def test_rmsd_identity_is_zero(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert rmsd(coords, coords) == pytest.approx(0.0)

    def test_kabsch_identity_is_zero(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert kabsch_rmsd(coords, coords) == pytest.approx(0.0)

    def test_kabsch_aligns_rotated_translated(self):
        q = np.array([np.cos(0.3), np.sin(0.3), 0.0, 0.0])
        tf = transform_from_quaternion(q, np.array([3.0, -2.0, 1.0]))
        q_coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]
        )
        p = tf.apply(q_coords)
        assert kabsch_rmsd(p, q_coords) == pytest.approx(0.0, abs=1e-9)

    def test_rmsd_differs_under_translation(self):
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        assert rmsd(a, b) == pytest.approx(5.0)


class TestRigidTransform:
    def test_apply_translation(self):
        tf = RigidTransform(rotation=np.eye(3), translation=np.array([1.0, 2.0, 3.0]))
        out = tf.apply(np.array([[0.0, 0.0, 0.0]]))
        assert out[0] == pytest.approx([1.0, 2.0, 3.0])

    def test_quaternion_round_trip_keeps_norm(self):
        q = np.array([0.8, 0.2, 0.4, 0.4472])
        q = q / np.linalg.norm(q)
        tf = transform_from_quaternion(q, np.zeros(3))
        # R @ R^T must be identity for a valid rotation.
        assert tf.rotation @ tf.rotation.T == pytest.approx(np.eye(3), abs=1e-9)


class TestDockingBox:
    def test_contains_center(self):
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 5.0))
        assert box.contains(np.zeros(3))

    def test_excludes_outside_point(self):
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 5.0))
        assert not box.contains(np.array([10.0, 0.0, 0.0]))

    def test_random_point_inside(self):
        rng = np.random.default_rng(0)
        box = DockingBox(center=np.array([1.0, 1.0, 1.0]), half_size=np.full(3, 2.0))
        for _ in range(20):
            assert box.contains(box.random_point(rng))

    def test_contains_all(self):
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 5.0))
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert box.contains_all(coords)
        assert not box.contains_all(np.array([[9.0, 0.0, 0.0]]))


class TestReceptorLigandSplit:
    def test_auto_identifies_ligand_as_hetatm(self):
        structure = parse_pdb_string(make_complex_pdb_string())
        receptor, ligand = identify_receptor_ligand(structure)
        assert ligand.n_atoms == 12
        assert all(a.is_hetatm for a in ligand.atoms)
        assert receptor.n_atoms == 21

    def test_explicit_ligand_residue_name(self):
        structure = parse_pdb_string(make_complex_pdb_string())
        receptor, ligand = identify_receptor_ligand(structure, ligand_res_name="LIG")
        assert ligand.n_atoms == 12

    def test_waters_are_stripped(self):
        water_line = (
            "HETATM  100  O   HOH A  99       0.000   0.000   0.000  1.00"
            "  0.00           O  \n"
        )
        structure = parse_pdb_string(make_complex_pdb_string() + water_line)
        receptor, ligand = identify_receptor_ligand(structure)
        assert all(a.res_name != "HOH" for a in receptor.atoms)
        assert all(a.res_name != "HOH" for a in ligand.atoms)
