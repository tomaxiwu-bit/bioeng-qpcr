"""Tests for blocked reaction and dead-end metabolite detection."""

from __future__ import annotations

from .conftest import skip_if_no_cobra


@skip_if_no_cobra
class TestFindBlockedReactions:
    def test_returns_result(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.analysis import (
            find_blocked_reactions,
        )

        result = find_blocked_reactions(textbook_model)
        assert result.n_total > 0
        assert result.n_blocked >= 0
        assert 0.0 <= result.fraction_blocked <= 1.0
        assert len(result.blocked_reactions) == result.n_blocked

    def test_textbook_has_blocked(self, textbook_model):
        """The textbook model is known to have ~8 blocked reactions."""
        from bioeng_qpcr.metabolic_flux.analysis import (
            find_blocked_reactions,
        )

        result = find_blocked_reactions(textbook_model)
        assert result.n_blocked > 0
        assert result.n_blocked < result.n_total
        # Known blocked reactions in the textbook model.
        assert "EX_fru_e" in result.blocked_reactions

    def test_fraction_consistent(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.analysis import (
            find_blocked_reactions,
        )

        result = find_blocked_reactions(textbook_model)
        expected = result.n_blocked / result.n_total
        assert abs(result.fraction_blocked - expected) < 1e-10


@skip_if_no_cobra
class TestFindDeadEndMetabolites:
    def test_returns_result(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.analysis import (
            find_dead_end_metabolites,
        )

        result = find_dead_end_metabolites(textbook_model)
        assert result.n_total > 0
        assert result.n_dead_ends >= 0
        assert len(result.dead_end_ids) == result.n_dead_ends

    def test_details_populated(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.analysis import (
            find_dead_end_metabolites,
        )

        result = find_dead_end_metabolites(textbook_model)
        for met_id in result.dead_end_ids:
            assert met_id in result.details
            assert result.details[met_id]["type"] in (
                "sink",
                "source",
                "isolated",
            )

    def test_dead_ends_subset_of_total(self, textbook_model):
        from bioeng_qpcr.metabolic_flux.analysis import (
            find_dead_end_metabolites,
        )

        result = find_dead_end_metabolites(textbook_model)
        assert result.n_dead_ends <= result.n_total

    def test_isolated_metabolite_detected(self, tmp_path):
        """A metabolite with no internal reactions should be flagged 'isolated'."""
        from cobra import Metabolite, Model, Reaction

        from bioeng_qpcr.metabolic_flux.analysis import (
            find_dead_end_metabolites,
        )

        model = Model("test_isolated")
        met_b = Metabolite("B_c", compartment="c")
        # Only an exchange reaction for B — no internal reactions.
        ex_b = Reaction("EX_B_c")
        ex_b.add_metabolites({met_b: -1})
        model.add_reactions([ex_b])
        result = find_dead_end_metabolites(model)
        assert "B_c" in result.dead_end_ids
        assert result.details["B_c"]["type"] == "isolated"

    def test_reversible_reaction_supports_both_directions(self):
        """Coefficient signs alone must not make reversible metabolites dead ends."""
        from cobra import Metabolite, Model, Reaction

        from bioeng_qpcr.metabolic_flux.analysis import (
            find_dead_end_metabolites,
        )

        model = Model("reversible")
        met_a = Metabolite("A_c", compartment="c")
        met_b = Metabolite("B_c", compartment="c")
        reaction = Reaction("R_REV", lower_bound=-1000.0, upper_bound=1000.0)
        reaction.add_metabolites({met_a: -1.0, met_b: 1.0})
        model.add_reactions([reaction])

        result = find_dead_end_metabolites(model)

        assert "A_c" not in result.dead_end_ids
        assert "B_c" not in result.dead_end_ids

    def test_boundary_property_not_reaction_prefix_is_used(self):
        """A custom-named one-metabolite boundary must be excluded."""
        from cobra import Metabolite, Model, Reaction

        from bioeng_qpcr.metabolic_flux.analysis import (
            find_dead_end_metabolites,
        )

        model = Model("custom_boundary")
        metabolite = Metabolite("A_c", compartment="c")
        boundary = Reaction("CUSTOM_BOUNDARY", lower_bound=-10.0, upper_bound=10.0)
        boundary.add_metabolites({metabolite: -1.0})
        model.add_reactions([boundary])

        result = find_dead_end_metabolites(model)

        assert "A_c" in result.dead_end_ids
        assert result.details["A_c"]["type"] == "isolated"
