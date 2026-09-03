"""Tests for FBA validation rules."""

from __future__ import annotations

from .conftest import skip_if_no_cobra


@skip_if_no_cobra
class TestValidateObjectiveFeasibility:
    def test_optimal_passes(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_objective_feasibility,
        )

        result = run_fba(textbook_model)
        check = validate_objective_feasibility(result)
        assert check.rule_id == "FBA-001"
        assert check.passed is True
        assert "optimal" in check.message.lower()

    def test_evidence_fields(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_objective_feasibility,
        )

        result = run_fba(textbook_model)
        check = validate_objective_feasibility(result)
        assert "status" in check.evidence
        assert "objective_value" in check.evidence
        assert "objective_direction" in check.evidence

    def test_non_optimal_fails(self):
        from bioeng_qpcr.metabolic_flux.fba import FBAResult
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_objective_feasibility,
        )

        result = FBAResult(
            objective_value=0.0,
            status="infeasible",
        )
        check = validate_objective_feasibility(result)
        assert check.passed is False
        assert "infeasible" in check.message

    def test_zero_objective_fails_for_max(self):
        from bioeng_qpcr.metabolic_flux.fba import FBAResult, ModelSummary
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_objective_feasibility,
        )

        summary = ModelSummary(
            model_id="test",
            n_reactions=10,
            n_metabolites=5,
            n_genes=3,
            compartments=("c", "e"),
            objective_direction="max",
        )
        result = FBAResult(
            objective_value=0.0,
            status="optimal",
            model_summary=summary,
        )
        check = validate_objective_feasibility(result)
        assert check.passed is False
        assert "non-positive" in check.message

    def test_zero_objective_passes_for_min(self):
        from bioeng_qpcr.metabolic_flux.fba import FBAResult, ModelSummary
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_objective_feasibility,
        )

        summary = ModelSummary(
            model_id="test",
            n_reactions=10,
            n_metabolites=5,
            n_genes=3,
            compartments=("c", "e"),
            objective_direction="min",
        )
        result = FBAResult(
            objective_value=0.0,
            status="optimal",
            model_summary=summary,
        )
        check = validate_objective_feasibility(result)
        assert check.passed is True


@skip_if_no_cobra
class TestValidateMassBalance:
    def test_balanced_passes(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_mass_balance,
        )

        result = run_fba(textbook_model)
        check = validate_mass_balance(textbook_model, result)
        assert check.rule_id == "FBA-002"
        assert check.passed is True
        assert "mass balance" in check.message.lower()

    def test_evidence_fields(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_mass_balance,
        )

        result = run_fba(textbook_model)
        check = validate_mass_balance(textbook_model, result)
        assert "n_metabolites_checked" in check.evidence
        assert "n_imbalanced" in check.evidence
        assert "max_residual" in check.evidence
        assert "tolerance" in check.evidence
        assert check.evidence["n_imbalanced"] == 0

    def test_tolerance_effect(self, textbook_model):
        """With a very tight tolerance, small numerical residuals may appear."""
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_mass_balance,
        )

        result = run_fba(textbook_model)
        # With 1e-6 tolerance, textbook should pass.
        check_loose = validate_mass_balance(
            textbook_model, result, tolerance=1e-6
        )
        assert check_loose.passed is True

    def test_message_uses_actual_checked_count(self, textbook_model):
        """Success message should report internal metabolites, not total."""
        from bioeng_qpcr.metabolic_flux.fba import run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_mass_balance,
        )

        result = run_fba(textbook_model)
        check = validate_mass_balance(textbook_model, result)
        n_checked = check.evidence["n_metabolites_checked"]
        n_total = check.evidence["n_metabolites_total"]
        assert n_checked < n_total
        assert str(n_checked) in check.message
        assert str(n_total) not in check.message.split("internal")[0]

    def test_broken_balance_detected(self, textbook_model):
        """Deliberately corrupt a flux to trigger mass balance failure."""
        from bioeng_qpcr.metabolic_flux.fba import FBAResult, run_fba
        from bioeng_qpcr.metabolic_flux.validation import (
            validate_mass_balance,
        )

        result = run_fba(textbook_model)
        # Corrupt one flux value to break mass balance.
        corrupted_fluxes = dict(result.fluxes)
        first_rxn = next(iter(corrupted_fluxes))
        corrupted_fluxes[first_rxn] += 100.0
        corrupted = FBAResult(
            objective_value=result.objective_value,
            status=result.status,
            fluxes=corrupted_fluxes,
            model_summary=result.model_summary,
        )
        check = validate_mass_balance(textbook_model, corrupted)
        assert check.passed is False
        assert check.evidence["n_imbalanced"] > 0
