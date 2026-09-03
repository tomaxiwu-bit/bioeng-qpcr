"""Tests for the bioeng-qpcr MCP server tools.

Each test calls the async tool function directly with a Pydantic input
model, avoiding the need for a full MCP protocol round-trip while still
exercising the real validation, calculation and plotting logic.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from bioeng_qpcr.mcp.server import (
    EnzymeClassifyInhibitionInput,
    EnzymeFitKineticsInput,
    EnzymePlotKineticsInput,
    FbaAnalyzeInput,
    FermentFitMonodInput,
    FermentPlotKineticsInput,
    QpcrAnalyzeInput,
    QpcrCompareGroupsInput,
    QpcrFitEfficiencyInput,
    QpcrGenerateReportInput,
    QpcrGetProvenanceInput,
    QpcrPlotFoldChangeInput,
    QpcrPlotHeatmapInput,
    QpcrReferenceStabilityInput,
    QpcrValidateInput,
    enzyme_classify_inhibition,
    enzyme_fit_kinetics,
    enzyme_plot_kinetics,
    ferment_fit_monod,
    ferment_plot_kinetics,
    main,
    mcp,
    qpcr_analyze,
    qpcr_compare_groups,
    qpcr_fit_efficiency,
    qpcr_generate_report,
    qpcr_get_provenance,
    qpcr_plot_fold_change,
    qpcr_plot_heatmap,
    qpcr_reference_stability,
    qpcr_validate,
)

from .conftest import (
    ENZYME_INHIBITION_CSV,
    ENZYME_MM_CSV,
    EXAMPLE_CSV,
    FERMENTATION_CSV,
)

try:
    import cobra  # noqa: F401

    COBRA_AVAILABLE = True
except ImportError:
    COBRA_AVAILABLE = False

skip_if_no_cobra = pytest.mark.skipif(
    not COBRA_AVAILABLE,
    reason="COBRApy not installed; skipping FBA MCP tests.",
)

if COBRA_AVAILABLE:
    from bioeng_qpcr.mcp.server import fba_analyze

_REFS = ["GAPDH", "RPLP0"]
_TARGET = "IL6"


def _run(coro):
    """Execute *coro* synchronously (pytest is not async here)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Server metadata
# ---------------------------------------------------------------------------


class TestServerMetadata:
    def test_server_name(self):
        assert mcp.name == "bioeng_qpcr_mcp"

    def test_main_is_callable(self):
        assert callable(main)

    def test_tool_count(self):
        """Core toolset stays under 15; FBA adds 1 when cobra is present."""
        tools = getattr(mcp._tool_manager, "_tools", {})
        n_core = 14
        n_fba = 1 if COBRA_AVAILABLE else 0
        expected = n_core + n_fba
        assert len(tools) == expected
        # Total tools must satisfy the <=15 audit constraint.
        assert expected <= 15


# ---------------------------------------------------------------------------
# qpcr_validate
# ---------------------------------------------------------------------------


class TestQpcrValidate:
    def test_validate_pass(self):
        result = _run(qpcr_validate(QpcrValidateInput(csv_path=EXAMPLE_CSV)))
        assert "results" in result
        assert "overall_status" in result
        assert "rule_coverage" in result

    def test_validate_with_explicit_refs(self):
        result = _run(
            qpcr_validate(
                QpcrValidateInput(
                    csv_path=EXAMPLE_CSV,
                    reference_targets=_REFS,
                )
            )
        )
        assert "results" in result

    def test_validate_file_not_found(self):
        result = _run(qpcr_validate(QpcrValidateInput(csv_path="nonexistent.csv")))
        assert "error" in result
        assert "error_type" in result

    def test_validate_lenient_mode(self):
        result = _run(
            qpcr_validate(
                QpcrValidateInput(csv_path=EXAMPLE_CSV, lenient=True)
            )
        )
        assert "results" in result


# ---------------------------------------------------------------------------
# qpcr_analyze
# ---------------------------------------------------------------------------


class TestQpcrAnalyze:
    def test_analyze_normalized_quantity(self):
        result = _run(
            qpcr_analyze(
                QpcrAnalyzeInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    method="normalized_quantity",
                    reference_targets=_REFS,
                )
            )
        )
        assert "results" in result
        assert "exclusions" in result
        assert "parameters" in result
        assert result["parameters"]["method"] == "normalized_quantity"

    def test_analyze_pfaffl(self):
        result = _run(
            qpcr_analyze(
                QpcrAnalyzeInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    method="pfaffl",
                    control="Control_1",
                    reference_targets=["GAPDH"],
                )
            )
        )
        assert "results" in result
        assert result["parameters"]["method"] == "pfaffl"

    def test_analyze_delta_delta_cq(self):
        result = _run(
            qpcr_analyze(
                QpcrAnalyzeInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    method="delta_delta_cq",
                    control="Control_1",
                    reference_targets=["GAPDH"],
                )
            )
        )
        assert "results" in result
        assert result["parameters"]["method"] == "delta_delta_cq"

    def test_analyze_missing_control_raises_error(self):
        result = _run(
            qpcr_analyze(
                QpcrAnalyzeInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    method="pfaffl",
                )
            )
        )
        assert "error" in result

    def test_analyze_too_many_refs_for_pfaffl(self):
        result = _run(
            qpcr_analyze(
                QpcrAnalyzeInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    method="pfaffl",
                    control="Control_1",
                    reference_targets=_REFS,
                )
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# qpcr_compare_groups
# ---------------------------------------------------------------------------


class TestQpcrCompareGroups:
    def test_compare_groups_success(self):
        result = _run(
            qpcr_compare_groups(
                QpcrCompareGroupsInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    treated_group="treated",
                    control_group="control",
                    reference_targets=_REFS,
                )
            )
        )
        assert "comparison" in result
        assert "analysis_audit" in result
        assert "provenance" in result
        comp = result["comparison"]
        assert comp["treated_group"] == "treated"
        assert comp["control_group"] == "control"
        assert "fold_difference" in comp

    def test_compare_groups_missing_group(self):
        result = _run(
            qpcr_compare_groups(
                QpcrCompareGroupsInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    treated_group="nonexistent",
                    control_group="control",
                    reference_targets=_REFS,
                )
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# qpcr_generate_report
# ---------------------------------------------------------------------------


class TestQpcrGenerateReport:
    def test_report_returns_html(self):
        result = _run(
            qpcr_generate_report(
                QpcrGenerateReportInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    reference_targets=_REFS,
                )
            )
        )
        assert isinstance(result, str)
        assert "<html" in result.lower()
        assert "bioeng-qpcr" in result

    def test_report_file_not_found(self):
        result = _run(
            qpcr_generate_report(
                QpcrGenerateReportInput(
                    csv_path="missing.csv",
                    target=_TARGET,
                )
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# qpcr_get_provenance
# ---------------------------------------------------------------------------


class TestQpcrGetProvenance:
    def test_provenance_has_required_fields(self):
        result = _run(
            qpcr_get_provenance(
                QpcrGetProvenanceInput(csv_path=EXAMPLE_CSV)
            )
        )
        assert "input_hash" in result
        assert "package_versions" in result
        assert "ruleset_versions" in result
        assert "created_at" in result

    def test_provenance_with_parameters(self):
        result = _run(
            qpcr_get_provenance(
                QpcrGetProvenanceInput(
                    csv_path=EXAMPLE_CSV,
                    parameters={"method": "normalized_quantity"},
                )
            )
        )
        assert result["parameters"]["method"] == "normalized_quantity"


# ---------------------------------------------------------------------------
# qpcr_reference_stability
# ---------------------------------------------------------------------------


class TestQpcrReferenceStability:
    def test_stability_genorm(self):
        result = _run(
            qpcr_reference_stability(
                QpcrReferenceStabilityInput(
                    csv_path=EXAMPLE_CSV,
                    reference_targets=_REFS,
                    method="genorm",
                )
            )
        )
        assert result["method"] == "genorm"

    def test_stability_normfinder(self):
        result = _run(
            qpcr_reference_stability(
                QpcrReferenceStabilityInput(
                    csv_path=EXAMPLE_CSV,
                    reference_targets=[*_REFS, _TARGET],
                    method="normfinder",
                )
            )
        )
        assert result["method"] == "normfinder"

    def test_stability_consensus(self):
        result = _run(
            qpcr_reference_stability(
                QpcrReferenceStabilityInput(
                    csv_path=EXAMPLE_CSV,
                    reference_targets=[*_REFS, _TARGET],
                    method="consensus",
                )
            )
        )
        assert result["method"] == "consensus"


# ---------------------------------------------------------------------------
# qpcr_fit_efficiency
# ---------------------------------------------------------------------------


class TestQpcrFitEfficiency:
    def test_fit_from_slope(self):
        result = _run(
            qpcr_fit_efficiency(
                QpcrFitEfficiencyInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                )
            )
        )
        assert "efficiency" in result
        assert "slope" in result
        assert result["slope"] is not None
        assert result["source"] == "assay_slope_metadata"

    def test_fit_unknown_target(self):
        result = _run(
            qpcr_fit_efficiency(
                QpcrFitEfficiencyInput(
                    csv_path=EXAMPLE_CSV,
                    target="NONEXISTENT",
                )
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# qpcr_plot_fold_change
# ---------------------------------------------------------------------------


class TestQpcrPlotFoldChange:
    def test_plot_returns_base64_png(self):
        result = _run(
            qpcr_plot_fold_change(
                QpcrPlotFoldChangeInput(
                    csv_path=EXAMPLE_CSV,
                    target=_TARGET,
                    control="Control_1",
                    reference_target="GAPDH",
                )
            )
        )
        assert "image_base64" in result
        assert result["format"] == "png"
        assert result["mime_type"] == "image/png"
        # Verify it's valid base64
        decoded = base64.b64decode(result["image_base64"])
        assert len(decoded) > 0
        # PNG magic number
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# qpcr_plot_heatmap
# ---------------------------------------------------------------------------


class TestQpcrPlotHeatmap:
    def test_heatmap_returns_base64_png(self):
        result = _run(
            qpcr_plot_heatmap(QpcrPlotHeatmapInput(csv_path=EXAMPLE_CSV))
        )
        assert "image_base64" in result
        assert result["format"] == "png"
        decoded = base64.b64decode(result["image_base64"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# enzyme_fit_kinetics
# ---------------------------------------------------------------------------


class TestEnzymeFitKinetics:
    def test_fit_returns_parameters(self):
        result = _run(
            enzyme_fit_kinetics(
                EnzymeFitKineticsInput(csv_path=ENZYME_MM_CSV)
            )
        )
        assert "fit" in result
        fit = result["fit"]
        assert "vmax" in fit
        assert "km" in fit
        assert "r_squared" in fit
        assert "n_points" in fit
        # The fixture data uses Vmax=100, Km=10
        assert fit["vmax"] == pytest.approx(100.0, rel=0.05)
        assert fit["km"] == pytest.approx(10.0, rel=0.1)
        assert fit["n_points"] == 7

    def test_fit_returns_validation(self):
        result = _run(
            enzyme_fit_kinetics(
                EnzymeFitKineticsInput(csv_path=ENZYME_MM_CSV)
            )
        )
        assert "validation" in result
        validation = result["validation"]
        assert "ENZ-001" in validation
        assert "ENZ-002" in validation
        assert validation["ENZ-001"]["rule_id"] == "ENZ-001"
        assert validation["ENZ-002"]["rule_id"] == "ENZ-002"
        # Km=10 is within valid range
        assert validation["ENZ-001"]["passed"] is True
        # Substrate [1..100] brackets Km=10
        assert validation["ENZ-002"]["passed"] is True

    def test_fit_file_not_found(self):
        result = _run(
            enzyme_fit_kinetics(
                EnzymeFitKineticsInput(csv_path="nonexistent_kinetics.csv")
            )
        )
        assert "error" in result
        assert "error_type" in result


# ---------------------------------------------------------------------------
# enzyme_classify_inhibition
# ---------------------------------------------------------------------------


class TestEnzymeClassifyInhibition:
    def test_classify_returns_type(self):
        result = _run(
            enzyme_classify_inhibition(
                EnzymeClassifyInhibitionInput(
                    csv_path=ENZYME_INHIBITION_CSV
                )
            )
        )
        assert "inhibition_type" in result
        # The fixture data uses competitive inhibition
        assert result["inhibition_type"] == "competitive"
        assert "km_cv" in result
        assert "vmax_cv" in result
        assert "confidence" in result
        assert "message" in result

    def test_classify_returns_datasets(self):
        result = _run(
            enzyme_classify_inhibition(
                EnzymeClassifyInhibitionInput(
                    csv_path=ENZYME_INHIBITION_CSV
                )
            )
        )
        assert "datasets" in result
        datasets = result["datasets"]
        assert len(datasets) == 3  # [I] = 0, 5, 10
        # Verify sorted by inhibitor concentration
        concs = [d["inhibitor_concentration"] for d in datasets]
        assert concs == sorted(concs)
        # Each dataset should have fit parameters
        for ds in datasets:
            assert "km_app" in ds
            assert "vmax_app" in ds
            assert "r_squared" in ds
            assert "lb_slope" in ds
            assert "lb_intercept" in ds

    def test_classify_competitive_km_increases(self):
        result = _run(
            enzyme_classify_inhibition(
                EnzymeClassifyInhibitionInput(
                    csv_path=ENZYME_INHIBITION_CSV
                )
            )
        )
        km_apps = [d["km_app"] for d in result["datasets"]]
        # Competitive: Km should increase with [I]
        assert km_apps[-1] > km_apps[0]

    def test_classify_competitive_vmax_unchanged(self):
        result = _run(
            enzyme_classify_inhibition(
                EnzymeClassifyInhibitionInput(
                    csv_path=ENZYME_INHIBITION_CSV
                )
            )
        )
        # Vmax CV should be low for competitive inhibition
        assert result["vmax_cv"] < 0.05

    def test_classify_file_not_found(self):
        result = _run(
            enzyme_classify_inhibition(
                EnzymeClassifyInhibitionInput(
                    csv_path="nonexistent_inhibition.csv"
                )
            )
        )
        assert "error" in result

    def test_classify_missing_inhibitor_column(self):
        """Using a CSV without inhibitor_concentration should error."""
        result = _run(
            enzyme_classify_inhibition(
                EnzymeClassifyInhibitionInput(csv_path=ENZYME_MM_CSV)
            )
        )
        assert "error" in result
        assert "inhibitor_concentration" in result["error"]


# ---------------------------------------------------------------------------
# enzyme_plot_kinetics
# ---------------------------------------------------------------------------


class TestEnzymePlotKinetics:
    def test_plot_mm_returns_base64_png(self):
        result = _run(
            enzyme_plot_kinetics(
                EnzymePlotKineticsInput(
                    csv_path=ENZYME_MM_CSV,
                    plot_type="michaelis_menten",
                )
            )
        )
        assert "image_base64" in result
        assert result["format"] == "png"
        assert result["mime_type"] == "image/png"
        decoded = base64.b64decode(result["image_base64"])
        assert len(decoded) > 0
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_plot_lb_returns_base64_png(self):
        result = _run(
            enzyme_plot_kinetics(
                EnzymePlotKineticsInput(
                    csv_path=ENZYME_INHIBITION_CSV,
                    plot_type="lineweaver_burk",
                )
            )
        )
        assert "image_base64" in result
        assert result["format"] == "png"
        decoded = base64.b64decode(result["image_base64"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_plot_default_is_mm(self):
        """When plot_type is omitted, MM plot is generated."""
        result = _run(
            enzyme_plot_kinetics(
                EnzymePlotKineticsInput(csv_path=ENZYME_MM_CSV)
            )
        )
        assert "image_base64" in result

    def test_plot_lb_missing_inhibitor_column(self):
        """LB plot on a CSV without inhibitor_concentration should error."""
        result = _run(
            enzyme_plot_kinetics(
                EnzymePlotKineticsInput(
                    csv_path=ENZYME_MM_CSV,
                    plot_type="lineweaver_burk",
                )
            )
        )
        assert "error" in result
        assert "inhibitor_concentration" in result["error"]

    def test_plot_file_not_found(self):
        result = _run(
            enzyme_plot_kinetics(
                EnzymePlotKineticsInput(csv_path="nonexistent_plot.csv")
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# ferment_fit_monod
# ---------------------------------------------------------------------------


class TestFermentFitMonod:
    def test_fit_returns_parameters(self):
        result = _run(
            ferment_fit_monod(FermentFitMonodInput(csv_path=FERMENTATION_CSV))
        )
        assert "fit" in result
        fit = result["fit"]
        assert "mu_max" in fit
        assert "ks" in fit
        assert "yxs" in fit
        assert "r_squared" in fit
        assert "n_points" in fit
        assert fit["n_points"] == 12
        # μmax should be close to 0.5 h⁻¹ (ground truth).
        assert 0.1 < fit["mu_max"] < 1.0
        # R² should be high for well-behaved synthetic data.
        assert fit["r_squared"] > 0.95

    def test_fit_returns_identifiability(self):
        result = _run(
            ferment_fit_monod(FermentFitMonodInput(csv_path=FERMENTATION_CSV))
        )
        assert "identifiability" in result
        ident = result["identifiability"]
        assert "condition_number" in ident
        assert "is_identifiable" in ident
        assert "confidence" in ident
        assert "message" in ident
        assert "correlation_matrix" in ident
        assert "eigenvalues" in ident
        assert len(ident["correlation_matrix"]) == 3
        assert len(ident["eigenvalues"]) > 0

    def test_fit_returns_validation(self):
        result = _run(
            ferment_fit_monod(FermentFitMonodInput(csv_path=FERMENTATION_CSV))
        )
        assert "validation" in result
        val = result["validation"]
        assert "FERM-001" in val
        assert "FERM-002" in val
        assert val["FERM-001"]["passed"] is True
        assert val["FERM-002"]["passed"] is True

    def test_fit_file_not_found(self):
        result = _run(
            ferment_fit_monod(FermentFitMonodInput(csv_path="nonexistent.csv"))
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# ferment_plot_kinetics
# ---------------------------------------------------------------------------


class TestFermentPlotKinetics:
    def test_plot_returns_base64_png(self):
        result = _run(
            ferment_plot_kinetics(
                FermentPlotKineticsInput(csv_path=FERMENTATION_CSV)
            )
        )
        assert "image_base64" in result
        assert result["format"] == "png"
        assert result["mime_type"] == "image/png"
        decoded = base64.b64decode(result["image_base64"])
        assert len(decoded) > 0
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_plot_file_not_found(self):
        result = _run(
            ferment_plot_kinetics(
                FermentPlotKineticsInput(csv_path="nonexistent.csv")
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# fba_analyze (conditional on cobra)
# ---------------------------------------------------------------------------


@skip_if_no_cobra
class TestFbaAnalyze:
    def test_analyze_returns_fba_result(self):
        result = _run(
            fba_analyze(FbaAnalyzeInput(model_source="textbook"))
        )
        assert "fba" in result
        fba = result["fba"]
        assert "objective_value" in fba
        assert "status" in fba
        assert fba["status"] == "optimal"
        assert fba["objective_value"] > 0
        assert "n_active" in fba
        assert "n_total" in fba

    def test_analyze_returns_model_summary(self):
        result = _run(
            fba_analyze(FbaAnalyzeInput(model_source="textbook"))
        )
        fba = result["fba"]
        assert fba["model_summary"] is not None
        summary = fba["model_summary"]
        assert summary["model_id"] == "e_coli_core"
        assert summary["n_reactions"] > 0
        assert summary["n_metabolites"] > 0
        assert "compartments" in summary
        assert "objective_direction" in summary

    def test_analyze_returns_blocked_reactions(self):
        result = _run(
            fba_analyze(FbaAnalyzeInput(model_source="textbook"))
        )
        assert "blocked_reactions" in result
        blocked = result["blocked_reactions"]
        assert "n_blocked" in blocked
        assert "n_total" in blocked
        assert "fraction_blocked" in blocked
        assert blocked["n_total"] > 0
        assert 0.0 <= blocked["fraction_blocked"] <= 1.0

    def test_analyze_returns_dead_end_metabolites(self):
        result = _run(
            fba_analyze(FbaAnalyzeInput(model_source="textbook"))
        )
        assert "dead_end_metabolites" in result
        dead_ends = result["dead_end_metabolites"]
        assert "n_dead_ends" in dead_ends
        assert "n_total" in dead_ends
        assert dead_ends["n_total"] > 0

    def test_analyze_returns_validation(self):
        result = _run(
            fba_analyze(FbaAnalyzeInput(model_source="textbook"))
        )
        assert "validation" in result
        val = result["validation"]
        assert "FBA-001" in val
        assert "FBA-002" in val
        assert val["FBA-001"]["rule_id"] == "FBA-001"
        assert val["FBA-002"]["rule_id"] == "FBA-002"
        # Textbook model should pass both checks.
        assert val["FBA-001"]["passed"] is True
        assert val["FBA-002"]["passed"] is True

    def test_analyze_with_custom_objective(self):
        result = _run(
            fba_analyze(
                FbaAnalyzeInput(
                    model_source="textbook",
                    objective="ATPM",
                    direction="max",
                )
            )
        )
        assert "fba" in result
        assert result["fba"]["status"] == "optimal"

    def test_analyze_invalid_objective(self):
        result = _run(
            fba_analyze(
                FbaAnalyzeInput(
                    model_source="textbook",
                    objective="NONEXISTENT",
                )
            )
        )
        assert "error" in result

    def test_analyze_file_not_found(self):
        result = _run(
            fba_analyze(
                FbaAnalyzeInput(model_source="nonexistent_model.xml")
            )
        )
        assert "error" in result

    def test_analyze_no_plot_by_default(self):
        """Without include_plot, no 'plot' key in response."""
        result = _run(
            fba_analyze(FbaAnalyzeInput(model_source="textbook"))
        )
        assert "plot" not in result

    def test_analyze_with_plot_returns_base64_png(self):
        result = _run(
            fba_analyze(
                FbaAnalyzeInput(
                    model_source="textbook",
                    include_plot=True,
                )
            )
        )
        assert "plot" in result
        plot = result["plot"]
        assert "image_base64" in plot
        assert plot["format"] == "png"
        assert plot["mime_type"] == "image/png"
        decoded = base64.b64decode(plot["image_base64"])
        assert len(decoded) > 0
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_analyze_with_plot_custom_top_n(self):
        result = _run(
            fba_analyze(
                FbaAnalyzeInput(
                    model_source="textbook",
                    include_plot=True,
                    top_n=10,
                )
            )
        )
        assert "plot" in result
        assert "image_base64" in result["plot"]


# ---------------------------------------------------------------------------
# Pydantic input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_extra_fields_forbidden(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QpcrValidateInput(
                csv_path=EXAMPLE_CSV,
                unexpected_field="bad",
            )

    def test_strings_are_stripped(self):
        params = QpcrValidateInput(csv_path=f"  {EXAMPLE_CSV}  ")
        assert params.csv_path == EXAMPLE_CSV
