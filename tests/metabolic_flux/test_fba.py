"""Tests for FBA solving and model loading."""

from __future__ import annotations

import pytest

from .conftest import skip_if_no_cobra


@skip_if_no_cobra
class TestLoadModel:
    def test_load_builtin_textbook(self):
        from bioeng_qpcr.metabolic_flux.fba import load_model

        model = load_model("textbook")
        assert model is not None
        assert len(model.reactions) > 0
        assert len(model.metabolites) > 0

    def test_load_nonexistent_file(self):
        from bioeng_qpcr.metabolic_flux.fba import load_model

        with pytest.raises(FileNotFoundError, match="not found"):
            load_model("nonexistent_model.xml")

    def test_unsupported_format(self, tmp_path):
        from bioeng_qpcr.metabolic_flux.fba import load_model

        bad_file = tmp_path / "model.txt"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_model(str(bad_file))


@skip_if_no_cobra
class TestRunFBA:
    def test_optimal_solution(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        result = run_fba(textbook_model)
        assert result.status == "optimal"
        assert result.objective_value > 0
        assert result.n_total > 0
        assert result.n_active > 0
        assert result.n_active < result.n_total

    def test_fluxes_dict_populated(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        result = run_fba(textbook_model)
        assert len(result.fluxes) == result.n_total
        # All values should be finite floats.
        for v in result.fluxes.values():
            assert isinstance(v, float)
            assert v == v  # not NaN

    def test_active_reactions_nonzero(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        result = run_fba(textbook_model)
        for rxn_id in result.active_reactions:
            assert abs(result.fluxes[rxn_id]) > 1e-10

    def test_model_summary(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        result = run_fba(textbook_model)
        assert result.model_summary is not None
        assert result.model_summary.n_reactions == result.n_total
        assert result.model_summary.n_metabolites > 0
        assert result.model_summary.n_genes > 0
        assert len(result.model_summary.compartments) > 0
        assert result.model_summary.objective_direction == "max"

    def test_custom_objective(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        # Use ATPM (ATP maintenance) as objective.
        result = run_fba(textbook_model, objective="ATPM")
        assert result.status == "optimal"
        assert result.objective_value > 0

    def test_invalid_objective(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        with pytest.raises(ValueError, match="not found"):
            run_fba(textbook_model, objective="NONEXISTENT")

    def test_invalid_direction(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        with pytest.raises(ValueError, match="Direction"):
            run_fba(textbook_model, direction="sideways")

    def test_minimize_objective(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        result = run_fba(textbook_model, direction="min")
        assert result.status == "optimal"
        # Minimised growth should be <= maximised growth.
        result_max = run_fba(textbook_model, direction="max")
        assert result.objective_value <= result_max.objective_value

    def test_model_not_mutated_by_objective(self, textbook_model):
        """run_fba must not permanently change the model's objective."""
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        original_obj = str(textbook_model.objective.expression)
        run_fba(textbook_model, objective="ATPM")
        assert str(textbook_model.objective.expression) == original_obj

    def test_model_not_mutated_by_direction(self, textbook_model):
        """run_fba must not permanently change the model's direction."""
        from bioeng_qpcr.metabolic_flux.fba import run_fba

        original_dir = textbook_model.objective_direction
        run_fba(textbook_model, direction="min")
        assert textbook_model.objective_direction == original_dir

    def test_infeasible_model_returns_status(self):
        """An infeasible model should return status='infeasible', not crash."""
        import copy

        from cobra.io import load_model

        from bioeng_qpcr.metabolic_flux.fba import run_fba

        # Use a fresh copy so we don't corrupt the session-scoped fixture.
        model = copy.deepcopy(load_model("textbook"))
        for rxn in model.exchanges:
            rxn.lower_bound = 0
            rxn.upper_bound = 0
        result = run_fba(model)
        assert result.status == "infeasible"

    def test_load_json_format(self, tmp_path):
        """JSON model format should be loadable."""
        from cobra.io import load_model as cobra_load
        from cobra.io import save_json_model

        from bioeng_qpcr.metabolic_flux.fba import load_model

        # Save textbook as JSON and reload.
        model = cobra_load("textbook")
        json_path = tmp_path / "model.json"
        save_json_model(model, str(json_path))
        reloaded = load_model(str(json_path))
        assert len(reloaded.reactions) == len(model.reactions)
        assert len(reloaded.metabolites) == len(model.metabolites)
