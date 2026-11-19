"""Tests for the high-level docking engine and DockResult."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.docking.engine import DockResult, StructureMeta, dock
from bioeng_qpcr.docking.scoring import ScoringFunction
from bioeng_qpcr.docking.structure import identify_receptor_ligand, parse_pdb_string

from .conftest import make_complex_pdb_string


def _split():
    structure = parse_pdb_string(make_complex_pdb_string())
    return identify_receptor_ligand(structure)


class TestDock:
    def test_returns_dock_result_with_poses(self):
        receptor, ligand = _split()
        result = dock(receptor, ligand, n_runs=3, n_steps=200, seed=1)
        assert isinstance(result, DockResult)
        assert len(result.poses) > 0

    def test_aggregate_convergence_reflects_search(self):
        receptor, ligand = _split()
        result = dock(receptor, ligand, n_runs=1, n_steps=200, seed=1)
        assert "still improving" in result.stats.message
        assert not result.stats.converged

    def test_poses_sorted_ascending_by_score(self):
        receptor, ligand = _split()
        result = dock(receptor, ligand, n_runs=3, n_steps=200, seed=1)
        scores = [p.score for p in result.poses]
        assert scores == sorted(scores)

    def test_breakdown_total_equals_weighted_sum(self):
        receptor, ligand = _split()
        result = dock(receptor, ligand, n_runs=3, n_steps=200, seed=1)
        top = result.poses[0]
        expected = (
            top.breakdown.vdw
            + top.breakdown.electrostatic
            + top.breakdown.hbond
            + top.breakdown.hydrophobic
            + top.breakdown.desolvation
        )
        assert top.breakdown.total == pytest.approx(expected, abs=1e-9)

    def test_adjacent_poses_are_distinct(self):
        receptor, ligand = _split()
        result = dock(
            receptor, ligand, n_runs=4, n_steps=200, seed=1, rmsd_threshold=1.5
        )
        for a, b in zip(result.poses[:-1], result.poses[1:], strict=True):
            assert a.rmsd_to(b) > 1.5 - 1e-6

    def test_reproducible_with_same_seed(self):
        receptor, ligand = _split()
        r1 = dock(receptor, ligand, n_runs=3, n_steps=200, seed=123)
        r2 = dock(receptor, ligand, n_runs=3, n_steps=200, seed=123)
        assert r1.poses[0].score == pytest.approx(r2.poses[0].score, abs=1e-9)
        assert r1.poses[0].transform.translation == pytest.approx(
            r2.poses[0].transform.translation, abs=1e-9
        )

    def test_metadata_carried(self):
        receptor, ligand = _split()
        result = dock(receptor, ligand, n_runs=2, n_steps=100, seed=1)
        assert isinstance(result.receptor_meta, StructureMeta)
        assert result.receptor_meta.n_atoms == receptor.n_atoms
        assert result.ligand_meta.n_atoms == ligand.n_atoms

    def test_grid_mode_runs(self):
        receptor, ligand = _split()
        result = dock(
            receptor, ligand, use_grid=True, grid_translation_steps=2,
            grid_rotation_samples=3, seed=1,
        )
        assert len(result.poses) > 0
        assert result.stats.seed is None  # grid is deterministic, no RNG seed

    def test_custom_scoring_used(self):
        receptor, ligand = _split()
        result = dock(
            receptor, ligand, scoring=ScoringFunction(), n_runs=2, n_steps=100, seed=1
        )
        assert result.scoring_name == "ScoringFunction"

    def test_raises_on_empty_receptor(self):
        _, ligand = _split()
        empty = parse_pdb_string("")
        with pytest.raises(ValueError, match="no atoms"):
            dock(empty, ligand, seed=1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"top_n": 0},
            {"rmsd_threshold": -1.0},
            {"n_runs": 0},
            {"n_steps": 0},
            {"use_grid": True, "grid_translation_steps": 0},
            {"use_grid": True, "grid_rotation_samples": 0},
        ],
    )
    def test_rejects_invalid_search_parameters(self, kwargs):
        receptor, ligand = _split()
        with pytest.raises(ValueError):
            dock(receptor, ligand, **kwargs)

    def test_box_can_be_supplied(self):
        from bioeng_qpcr.docking.structure import DockingBox

        receptor, ligand = _split()
        box = DockingBox(center=np.zeros(3), half_size=np.full(3, 10.0))
        result = dock(receptor, ligand, box=box, n_runs=2, n_steps=100, seed=1)
        assert result.box is box
