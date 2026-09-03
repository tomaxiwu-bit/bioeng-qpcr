"""MCP server exposing the bioeng-qpcr analysis engine to AI assistants.

This module wraps the core qPCR analysis library behind Model Context
Protocol tools so that an MCP-compatible AI assistant can run MIQE readiness
assessments, relative-quantification analyses, group comparisons, reference
gene stability checks, efficiency fitting, provenance capture, report
generation, plotting, enzyme kinetics fitting/inhibition classification,
fermentation kinetics parameter fitting/plotting, and (when COBRApy is
installed) FBA metabolic flux analysis and flux distribution plotting
without leaving the conversation.

The server uses the MCPServer framework (MCP Python SDK v2).  Every tool
accepts a Pydantic ``BaseModel`` for strict input validation and returns
JSON-serialisable results (or a base64-encoded PNG payload for the
plotting tools).
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict
from typing import Any, Literal

import matplotlib
import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from bioeng_qpcr.calculations import (
    compare_groups,
    delta_delta_cq_batch,
    normalized_quantity_all,
    pfaffl_batch,
)
from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.enzyme_kinetics import (
    classify_inhibition,
    fit_michaelis_menten,
    read_kinetics_csv,
    validate_km_range,
    validate_substrate_span,
)
from bioeng_qpcr.fermentation_kinetics import (
    analyze_identifiability,
    fit_monod,
    read_fermentation_csv,
    validate_data_sufficiency,
    validate_mu_max_range,
)
from bioeng_qpcr.io import CSVValidationError, read_tidy_csv
from bioeng_qpcr.provenance import ProvenanceManifest
from bioeng_qpcr.reference_genes import consensus_ranking, genorm, normfinder
from bioeng_qpcr.reporting import build_research_report
from bioeng_qpcr.validation import RuleExecutionError, ValidationEngine

matplotlib.use("Agg")  # headless backend; must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from bioeng_qpcr.plotting import (  # noqa: E402
    plot_cq_heatmap,
    plot_fermentation_kinetics,
    plot_flux_distribution,
    plot_fold_change,
    plot_lineweaver_burk,
    plot_michaelis_menten,
)

__all__ = ["main", "mcp"]

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = MCPServer("bioeng_qpcr_mcp")

# Exceptions surfaced to the caller as structured error dictionaries.  Note
# that ``CSVValidationError`` is a subclass of ``ValueError``; it is listed
# explicitly to document intent and is caught by the ``ValueError`` entry.
_TOOL_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    CSVValidationError,
    ValueError,
    RuleExecutionError,
)

_READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _error_response(exc: BaseException) -> dict[str, Any]:
    """Build a structured error dictionary from a caught exception."""
    return {"error": str(exc), "error_type": type(exc).__name__}


def _load_experiment(
    csv_path: str,
    reference_targets: list[str] | None = None,
    *,
    lenient: bool = False,
) -> Experiment:
    """Read a tidy CSV into an :class:`Experiment`.

    ``lenient`` toggles the reader's strict mode: when ``False`` (the default)
    recoverable input problems raise :class:`CSVValidationError`; when ``True``
    they are retained as diagnostics and the experiment is still returned.
    """
    return read_tidy_csv(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )


def _figure_to_base64(fig: Figure) -> dict[str, Any]:
    """Encode a matplotlib figure as a base64 PNG payload and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi, bbox_inches="tight")
    plt.close(fig)
    image_bytes = buffer.getvalue()
    return {
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "format": "png",
        "mime_type": "image/png",
    }


def _efficiency_from_experiment(experiment: Experiment, target: str) -> dict[str, Any]:
    """Derive amplification efficiency for *target* from the experiment.

    If the assay carries a standard-curve slope the efficiency is computed
    directly as ``10**(-1/slope) - 1`` (the same formula used internally by
    :func:`fit_efficiency_from_standard_curve`).  A pre-recorded efficiency
    is returned verbatim.  Otherwise a clear error is raised because the
    Level A tidy CSV format does not carry the concentration column required
    for an in-place standard-curve fit.
    """
    assay = experiment.assays.get(target)
    if assay is None:
        raise ValueError(f"Target '{target}' is not present in the experiment.")

    if assay.slope is not None:
        slope = float(assay.slope)
        efficiency = 10.0 ** (-1.0 / slope) - 1.0
        return {
            "target_name": target,
            "efficiency": efficiency,
            "r_squared": assay.r2,
            "slope": slope,
            "intercept": None,
            "n_points": None,
            "source": "assay_slope_metadata",
        }
    if assay.efficiency is not None:
        return {
            "target_name": target,
            "efficiency": float(assay.efficiency),
            "r_squared": assay.r2,
            "slope": None,
            "intercept": None,
            "n_points": None,
            "source": "assay_efficiency_metadata",
        }
    raise ValueError(
        f"Cannot fit efficiency for target '{target}': the assay has no "
        "slope, r2 or efficiency metadata, and the Level A tidy CSV format "
        "does not include standard-curve concentration data required for a "
        "fit_efficiency_from_standard_curve fit."
    )


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------


class QpcrValidateInput(BaseModel):
    """Input for the MIQE readiness assessment."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(
        description="Absolute path to the Level A tidy qPCR CSV file."
    )
    reference_targets: list[str] | None = Field(
        default=None,
        description=(
            "Optional reference (housekeeping) gene names. When omitted the "
            "assays flagged with is_reference are used."
        ),
    )
    lenient: bool = Field(
        default=False,
        description=(
            "Accept recoverable input problems and retain them as "
            "diagnostics instead of rejecting the file."
        ),
    )


class QpcrAnalyzeInput(BaseModel):
    """Input for relative-quantification analysis."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    target: str = Field(description="Target gene of interest.")
    method: Literal["normalized_quantity", "pfaffl", "delta_delta_cq"] = Field(
        default="normalized_quantity",
        description=(
            "Quantification method. 'normalized_quantity' is the "
            "efficiency-corrected default; 'pfaffl' and 'delta_delta_cq' "
            "require a control sample."
        ),
    )
    control: str | None = Field(
        default=None,
        description=(
            "Control (calibrator) sample name; required for the 'pfaffl' "
            "and 'delta_delta_cq' methods."
        ),
    )
    reference_targets: list[str] | None = Field(
        default=None,
        description=(
            "Reference gene names. 'normalized_quantity' accepts several; "
            "'pfaffl' and 'delta_delta_cq' accept at most one."
        ),
    )
    allow_exclusions: bool = Field(
        default=False,
        description=(
            "Return partial results with excluded samples reported instead "
            "of failing when samples are omitted."
        ),
    )
    lenient: bool = Field(
        default=False,
        description="Accept recoverable input problems as diagnostics.",
    )


class QpcrCompareGroupsInput(BaseModel):
    """Input for biological group comparison."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    target: str = Field(description="Target gene of interest.")
    treated_group: str = Field(description="Name of the treated sample group.")
    control_group: str = Field(description="Name of the control sample group.")
    reference_targets: list[str] | None = Field(
        default=None,
        description="Optional reference gene names used for normalization.",
    )
    allow_exclusions: bool = Field(
        default=False,
        description="Return partial results with excluded samples reported.",
    )
    lenient: bool = Field(
        default=False,
        description="Accept recoverable input problems as diagnostics.",
    )


class QpcrGenerateReportInput(BaseModel):
    """Input for HTML report generation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    target: str = Field(description="Target gene of interest.")
    reference_targets: list[str] | None = Field(
        default=None,
        description="Optional reference gene names used for normalization.",
    )
    lenient: bool = Field(
        default=False,
        description="Accept recoverable input problems as diagnostics.",
    )


class QpcrGetProvenanceInput(BaseModel):
    """Input for provenance manifest retrieval."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Optional analysis parameters to record on the manifest.",
    )


class QpcrReferenceStabilityInput(BaseModel):
    """Input for reference-gene stability analysis."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    reference_targets: list[str] | None = Field(
        default=None,
        description=(
            "Candidate reference genes to evaluate. When omitted the "
            "experiment's registered reference genes are used. Model-based "
            "NormFinder and consensus require at least three candidates."
        ),
    )
    method: Literal["genorm", "normfinder", "consensus"] = Field(
        default="consensus",
        description=(
            "Stability algorithm. 'consensus' aggregates geNorm and "
            "NormFinder rankings into a single recommendation."
        ),
    )


class QpcrFitEfficiencyInput(BaseModel):
    """Input for standard-curve efficiency fitting."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    target: str = Field(description="Target assay whose efficiency is estimated.")


class QpcrPlotFoldChangeInput(BaseModel):
    """Input for the fold-change bar plot."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")
    target: str = Field(description="Target gene of interest.")
    control: str = Field(description="Control (calibrator) sample name.")
    reference_target: str | None = Field(
        default=None,
        description=(
            "Reference gene used for normalization. When omitted the first "
            "registered reference gene is used."
        ),
    )


class QpcrPlotHeatmapInput(BaseModel):
    """Input for the Cq heatmap."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Absolute path to the tidy qPCR CSV file.")


class EnzymeFitKineticsInput(BaseModel):
    """Input for Michaelis-Menten parameter fitting."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(
        description=(
            "Path to a tidy enzyme-kinetics CSV with columns "
            "'substrate_concentration' and 'velocity'."
        )
    )


class EnzymeClassifyInhibitionInput(BaseModel):
    """Input for inhibition type classification."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(
        description=(
            "Path to a tidy enzyme-kinetics CSV with columns "
            "'substrate_concentration', 'velocity', and "
            "'inhibitor_concentration'."
        )
    )


class EnzymePlotKineticsInput(BaseModel):
    """Input for enzyme kinetics plotting."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(description="Path to a tidy enzyme-kinetics CSV.")
    plot_type: Literal["michaelis_menten", "lineweaver_burk"] = Field(
        default="michaelis_menten",
        description=(
            "Plot type: 'michaelis_menten' for a fitted MM curve, "
            "'lineweaver_burk' for a double-reciprocal plot (requires "
            "an inhibitor_concentration column)."
        ),
    )


class FermentFitMonodInput(BaseModel):
    """Input for Monod fermentation parameter fitting."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(
        description=(
            "Path to a tidy fermentation-kinetics CSV with columns "
            "'time', 'biomass', and 'substrate'."
        )
    )


class FermentPlotKineticsInput(BaseModel):
    """Input for fermentation kinetics plotting."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_path: str = Field(
        description=(
            "Path to a tidy fermentation-kinetics CSV with columns "
            "'time', 'biomass', and 'substrate'."
        )
    )


class FbaAnalyzeInput(BaseModel):
    """Input for FBA metabolic flux analysis."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_source: str = Field(
        description=(
            "Path to an SBML/JSON/MAT model file, or a built-in model "
            "name (e.g. 'textbook' for the E. coli core model)."
        )
    )
    objective: str | None = Field(
        default=None,
        description=(
            "Optional reaction ID to use as the objective. When omitted "
            "the model's current objective is used."
        ),
    )
    direction: Literal["max", "min"] | None = Field(
        default=None,
        description=(
            "Optimisation direction. When omitted the model's current "
            "direction is used."
        ),
    )
    include_plot: bool = Field(
        default=False,
        description=(
            "When True, also render a flux distribution bar chart "
            "(top-20 reactions by absolute flux) as a base64-encoded "
            "PNG and include it in the response under 'plot'."
        ),
    )
    top_n: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of reactions to display in the chart "
        "(only used when include_plot is True).",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="qpcr_validate",
    title="MIQE readiness assessment",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_validate(params: QpcrValidateInput) -> dict[str, Any]:
    """Assess MIQE readiness of a tidy qPCR CSV file.

    Runs the full validation rule set against the experiment and returns the
    structured readiness report (per-rule status, summary counts and rule
    coverage).
    """
    try:
        experiment = _load_experiment(
            params.csv_path,
            params.reference_targets,
            lenient=params.lenient,
        )
        report = ValidationEngine().run(experiment)
        return report.to_dict()
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_analyze",
    title="Relative quantification analysis",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_analyze(params: QpcrAnalyzeInput) -> dict[str, Any]:
    """Run an auditable relative-quantification analysis.

    Supports efficiency-corrected normalized quantities (default), Pfaffl and
    Livak delta-delta-Cq methods.  Results, exclusions, warnings and effective
    parameters are returned as a batch dictionary.
    """
    try:
        experiment = _load_experiment(
            params.csv_path,
            params.reference_targets,
            lenient=params.lenient,
        )
        if params.method == "normalized_quantity":
            batch = normalized_quantity_all(
                experiment,
                params.target,
                reference_targets=params.reference_targets,
            )
        else:
            if params.control is None:
                raise ValueError(
                    f"control sample is required for method '{params.method}'"
                )
            if params.reference_targets and len(params.reference_targets) > 1:
                raise ValueError(
                    f"method '{params.method}' accepts at most one reference target"
                )
            reference_target = (
                params.reference_targets[0] if params.reference_targets else None
            )
            if params.method == "delta_delta_cq":
                batch = delta_delta_cq_batch(
                    experiment,
                    params.target,
                    params.control,
                    reference_target=reference_target,
                )
            else:  # pfaffl
                batch = pfaffl_batch(
                    experiment,
                    params.target,
                    params.control,
                    reference_target=reference_target,
                )
        if batch.exclusions and not params.allow_exclusions:
            batch.raise_for_exclusions()
        return batch.to_dict()
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_compare_groups",
    title="Biological group comparison",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_compare_groups(
    params: QpcrCompareGroupsInput,
) -> dict[str, Any]:
    """Compare group means of per-sample normalized quantities.

    Returns the group fold difference, a Welch's t-test (when both groups have
    at least two biological samples), the per-sample analysis audit and a
    provenance manifest.
    """
    try:
        experiment = _load_experiment(
            params.csv_path,
            params.reference_targets,
            lenient=params.lenient,
        )
        result, batch = compare_groups(
            experiment,
            params.target,
            params.treated_group,
            params.control_group,
            params.reference_targets,
        )
        if batch.exclusions and not params.allow_exclusions:
            batch.raise_for_exclusions()
        comparison_parameters = {
            **batch.parameters,
            "treated_group": params.treated_group,
            "control_group": params.control_group,
            "group_summary": "geometric_mean",
            "inference_scale": "log2_normalized_quantity",
        }
        manifest = ProvenanceManifest.from_experiment(
            experiment,
            parameters=comparison_parameters,
        )
        manifest.exclusions = batch.to_dict()["exclusions"]
        return {
            "comparison": asdict(result),
            "analysis_audit": batch.to_dict(),
            "provenance": manifest.to_dict(),
        }
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_generate_report",
    title="HTML research report",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_generate_report(
    params: QpcrGenerateReportInput,
) -> str | dict[str, Any]:
    """Generate a self-contained HTML readiness, analysis and provenance report.

    Returns the HTML document as a string.  On error a structured dictionary
    with ``error`` and ``error_type`` keys is returned instead.
    """
    try:
        experiment = _load_experiment(
            params.csv_path,
            params.reference_targets,
            lenient=params.lenient,
        )
        report = build_research_report(
            experiment, params.target, params.reference_targets
        )
        return report.html
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_get_provenance",
    title="Provenance manifest",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_get_provenance(
    params: QpcrGetProvenanceInput,
) -> dict[str, Any]:
    """Build a reproducibility manifest for a dataset.

    Records the input content hash, package versions, validation ruleset
    versions, optional analysis parameters and a UTC creation timestamp.
    """
    try:
        experiment = _load_experiment(params.csv_path)
        manifest = ProvenanceManifest.from_experiment(
            experiment,
            parameters=params.parameters,
        )
        return manifest.to_dict()
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_reference_stability",
    title="Reference gene stability",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_reference_stability(
    params: QpcrReferenceStabilityInput,
) -> dict[str, Any]:
    """Evaluate reference-gene stability with geNorm, NormFinder or consensus.

    Returns the stability ranking and per-gene measures for the selected
    algorithm; the consensus method also reports the underlying geNorm and
    NormFinder rankings.
    """
    try:
        experiment = _load_experiment(params.csv_path, params.reference_targets)
        if params.method == "genorm":
            result = genorm(experiment, params.reference_targets)
        elif params.method == "normfinder":
            result = normfinder(experiment, params.reference_targets)
        else:  # consensus
            result = consensus_ranking(experiment, params.reference_targets)
        return {**asdict(result), "method": params.method}
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_fit_efficiency",
    title="Standard-curve efficiency fit",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_fit_efficiency(
    params: QpcrFitEfficiencyInput,
) -> dict[str, Any]:
    """Estimate amplification efficiency for a target assay.

    Returns the efficiency derived from the assay's standard-curve slope when
    available, or a pre-recorded efficiency.  Raises a descriptive error when
    neither is present because the Level A tidy CSV carries no concentration
    data for an in-place fit.
    """
    try:
        experiment = _load_experiment(params.csv_path)
        return _efficiency_from_experiment(experiment, params.target)
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_plot_fold_change",
    title="Fold-change bar plot",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_plot_fold_change(
    params: QpcrPlotFoldChangeInput,
) -> dict[str, Any]:
    """Render a delta-delta-Cq fold-change bar plot as a base64-encoded PNG.

    Runs the Livak method (assuming ~100% efficiency) relative to the control
    sample and plots per-sample fold changes.  Bars are coloured by direction
    (green up, red down).
    """
    try:
        reference_list = [params.reference_target] if params.reference_target else None
        experiment = _load_experiment(params.csv_path, reference_list)
        batch = delta_delta_cq_batch(
            experiment,
            params.target,
            params.control,
            reference_target=params.reference_target,
        )
        batch.raise_for_exclusions()
        results = batch.results
        if not results:
            raise ValueError("no samples with valid fold-change data to plot")
        fig, _ax = plot_fold_change(results)
        return _figure_to_base64(fig)
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="qpcr_plot_heatmap",
    title="Cq heatmap",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def qpcr_plot_heatmap(params: QpcrPlotHeatmapInput) -> dict[str, Any]:
    """Render a samples-by-targets Cq heatmap as a base64-encoded PNG.

    Low Cq (high expression) is drawn bright and high Cq (low expression) is
    drawn dark; undetermined wells are masked and labelled 'N/A'.
    """
    try:
        experiment = _load_experiment(params.csv_path)
        fig, _ax = plot_cq_heatmap(experiment)
        return _figure_to_base64(fig)
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Enzyme kinetics tools (Phase 2)
# ---------------------------------------------------------------------------


def _group_by_inhibitor(
    substrate: np.ndarray,
    velocity: np.ndarray,
    inhibitor: np.ndarray,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Split kinetics arrays into per-inhibitor-concentration groups.

    Returns a list of ``(inhibitor_concentration, substrate, velocity)``
    tuples sorted by inhibitor concentration.
    """
    datasets: list[tuple[float, np.ndarray, np.ndarray]] = []
    for conc in np.unique(inhibitor):
        mask = inhibitor == conc
        datasets.append((float(conc), substrate[mask], velocity[mask]))
    return datasets


@mcp.tool(
    name="enzyme_fit_kinetics",
    title="Michaelis-Menten parameter fit",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def enzyme_fit_kinetics(
    params: EnzymeFitKineticsInput,
) -> dict[str, Any]:
    """Fit Michaelis-Menten parameters from a kinetics CSV file.

    Reads substrate-velocity data, fits Vmax and Km via nonlinear
    regression, and runs ENZ-001 (Km physical range) and ENZ-002
    (substrate concentration span) validation checks.
    """
    try:
        substrate, velocity, _ = read_kinetics_csv(params.csv_path)
        result = fit_michaelis_menten(substrate, velocity)
        km_check = validate_km_range(result.km)
        span_check = validate_substrate_span(substrate, result.km)
        return {
            "fit": {
                "vmax": result.vmax,
                "km": result.km,
                "r_squared": result.r_squared,
                "warnings": list(result.warnings),
                "n_points": result.n_points,
                "vmax_ci": (
                    list(result.vmax_ci) if result.vmax_ci else None
                ),
                "km_ci": (
                    list(result.km_ci) if result.km_ci else None
                ),
            },
            "validation": {
                "ENZ-001": {
                    "rule_id": km_check.rule_id,
                    "passed": km_check.passed,
                    "message": km_check.message,
                },
                "ENZ-002": {
                    "rule_id": span_check.rule_id,
                    "passed": span_check.passed,
                    "message": span_check.message,
                },
            },
        }
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="enzyme_classify_inhibition",
    title="Enzyme inhibition type classification",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def enzyme_classify_inhibition(
    params: EnzymeClassifyInhibitionInput,
) -> dict[str, Any]:
    """Classify reversible inhibition mechanism from multi-[I] data.

    Reads a CSV with an ``inhibitor_concentration`` column, groups by [I],
    fits Michaelis-Menten for each group, and classifies the inhibition
    type (competitive, non-competitive, uncompetitive, mixed, or none)
    based on Km/Vmax trends.
    """
    try:
        substrate, velocity, inhibitor = read_kinetics_csv(params.csv_path)
        if inhibitor is None:
            raise ValueError(
                "CSV must contain an 'inhibitor_concentration' column "
                "for inhibition classification."
            )
        datasets = _group_by_inhibitor(substrate, velocity, inhibitor)
        result = classify_inhibition(datasets)
        return {
            "inhibition_type": result.inhibition_type.value,
            "km_cv": result.km_cv,
            "vmax_cv": result.vmax_cv,
            "confidence": result.confidence,
            "message": result.message,
            "datasets": [
                {
                    "inhibitor_concentration": d.inhibitor_concentration,
                    "km_app": d.mm_result.km,
                    "vmax_app": d.mm_result.vmax,
                    "r_squared": d.mm_result.r_squared,
                    "lb_slope": d.lb_result.slope,
                    "lb_intercept": d.lb_result.intercept,
                }
                for d in result.datasets
            ],
        }
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="enzyme_plot_kinetics",
    title="Enzyme kinetics plot",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def enzyme_plot_kinetics(
    params: EnzymePlotKineticsInput,
) -> dict[str, Any]:
    """Render an enzyme kinetics plot as a base64-encoded PNG.

    For ``michaelis_menten``: fits MM to the data and plots observed
    points with the fitted curve.

    For ``lineweaver_burk``: groups by ``inhibitor_concentration`` and
    plots double-reciprocal lines for each [I] to visualise the
    inhibition pattern.
    """
    try:
        substrate, velocity, inhibitor = read_kinetics_csv(params.csv_path)
        if params.plot_type == "michaelis_menten":
            result = fit_michaelis_menten(substrate, velocity)
            fig, _ax = plot_michaelis_menten(result, substrate, velocity)
        else:  # lineweaver_burk
            if inhibitor is None:
                raise ValueError(
                    "CSV must contain an 'inhibitor_concentration' "
                    "column for Lineweaver-Burk plot."
                )
            datasets = _group_by_inhibitor(substrate, velocity, inhibitor)
            result = classify_inhibition(datasets)
            fig, _ax = plot_lineweaver_burk(result.datasets)
        return _figure_to_base64(fig)
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Fermentation kinetics tools (Phase 3)
# ---------------------------------------------------------------------------


@mcp.tool(
    name="ferment_fit_monod",
    title="Monod fermentation parameter fit",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def ferment_fit_monod(
    params: FermentFitMonodInput,
) -> dict[str, Any]:
    """Fit Monod parameters from batch fermentation data.

    Reads time/biomass/substrate data from a CSV, fits μmax, Ks, and Yxs
    via nonlinear ODE regression, runs parameter identifiability analysis
    (Fisher Information Matrix), and validates FERM-001 (μmax physical
    range) and FERM-002 (data sufficiency) rules.
    """
    try:
        time, biomass, substrate = read_fermentation_csv(params.csv_path)
        result = fit_monod(time, biomass, substrate)
        ident = analyze_identifiability(result, time, biomass, substrate)
        mu_max_check = validate_mu_max_range(result.mu_max)
        data_check = validate_data_sufficiency(time, biomass, substrate)
        return {
            "fit": {
                "mu_max": result.mu_max,
                "ks": result.ks,
                "yxs": result.yxs,
                "r_squared": result.r_squared,
                "warnings": list(result.warnings),
                "n_points": result.n_points,
                "x0": result.x0,
                "s0": result.s0,
                "mu_max_ci": (
                    list(result.mu_max_ci) if result.mu_max_ci else None
                ),
                "ks_ci": (
                    list(result.ks_ci) if result.ks_ci else None
                ),
                "yxs_ci": (
                    list(result.yxs_ci) if result.yxs_ci else None
                ),
            },
            "identifiability": {
                "condition_number": ident.condition_number,
                "is_identifiable": ident.is_identifiable,
                "confidence": ident.confidence,
                "message": ident.message,
                "correlation_matrix": [
                    list(row) for row in ident.correlation_matrix
                ],
                "eigenvalues": list(ident.eigenvalues),
            },
            "validation": {
                "FERM-001": {
                    "rule_id": mu_max_check.rule_id,
                    "passed": mu_max_check.passed,
                    "message": mu_max_check.message,
                },
                "FERM-002": {
                    "rule_id": data_check.rule_id,
                    "passed": data_check.passed,
                    "message": data_check.message,
                },
            },
        }
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


@mcp.tool(
    name="ferment_plot_kinetics",
    title="Fermentation kinetics plot",
    annotations=_READ_ONLY_ANNOTATIONS,
)
async def ferment_plot_kinetics(
    params: FermentPlotKineticsInput,
) -> dict[str, Any]:
    """Render a fermentation kinetics plot as a base64-encoded PNG.

    Fits Monod parameters to the CSV data and plots a dual y-axis figure:
    left axis shows biomass (scatter + fitted curve), right axis shows
    substrate (scatter + fitted curve).
    """
    try:
        time, biomass, substrate = read_fermentation_csv(params.csv_path)
        result = fit_monod(time, biomass, substrate)
        fig, _ax = plot_fermentation_kinetics(result, time, biomass, substrate)
        return _figure_to_base64(fig)
    except _TOOL_ERRORS as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Metabolic flux tools (Phase 4, conditional on cobra availability)
# ---------------------------------------------------------------------------

try:
    import importlib.util

    _COBRA_AVAILABLE = importlib.util.find_spec("cobra") is not None
except ImportError:
    _COBRA_AVAILABLE = False

if _COBRA_AVAILABLE:
    from bioeng_qpcr.metabolic_flux import (
        find_blocked_reactions,
        find_dead_end_metabolites,
        load_model,
        run_fba,
        validate_mass_balance,
        validate_objective_feasibility,
    )

    @mcp.tool(
        name="fba_analyze",
        title="FBA metabolic flux analysis",
        annotations=_READ_ONLY_ANNOTATIONS,
    )
    async def fba_analyze(params: FbaAnalyzeInput) -> dict[str, Any]:
        """Run FBA, detect blocked reactions, dead-end metabolites, and
        optionally render a flux distribution plot.

        Loads a metabolic model, solves the FBA optimisation problem,
        detects blocked reactions and dead-end metabolites, and validates
        FBA-001 (objective feasibility) and FBA-002 (mass balance) rules.

        When ``include_plot`` is True the response also contains a
        ``plot`` key with a base64-encoded PNG of the top-N fluxes.
        """
        try:
            model = load_model(params.model_source)
            result = run_fba(
                model,
                objective=params.objective,
                direction=params.direction,
            )
            blocked = find_blocked_reactions(model)
            dead_ends = find_dead_end_metabolites(model)
            obj_check = validate_objective_feasibility(result)
            mass_check = validate_mass_balance(model, result)
            response: dict[str, Any] = {
                "fba": {
                    "objective_value": result.objective_value,
                    "status": result.status,
                    "n_active": result.n_active,
                    "n_total": result.n_total,
                    "model_summary": (
                        {
                            "model_id": result.model_summary.model_id,
                            "n_reactions": result.model_summary.n_reactions,
                            "n_metabolites": result.model_summary.n_metabolites,
                            "n_genes": result.model_summary.n_genes,
                            "compartments": list(
                                result.model_summary.compartments
                            ),
                            "objective_direction": (
                                result.model_summary.objective_direction
                            ),
                        }
                        if result.model_summary
                        else None
                    ),
                },
                "blocked_reactions": {
                    "n_blocked": blocked.n_blocked,
                    "n_total": blocked.n_total,
                    "fraction_blocked": blocked.fraction_blocked,
                    "blocked_ids": list(blocked.blocked_reactions[:20]),
                },
                "dead_end_metabolites": {
                    "n_dead_ends": dead_ends.n_dead_ends,
                    "n_total": dead_ends.n_total,
                    "dead_end_ids": list(dead_ends.dead_end_ids[:20]),
                    "details": dict(
                        list(dead_ends.details.items())[:20]
                    ),
                },
                "validation": {
                    "FBA-001": {
                        "rule_id": obj_check.rule_id,
                        "passed": obj_check.passed,
                        "message": obj_check.message,
                    },
                    "FBA-002": {
                        "rule_id": mass_check.rule_id,
                        "passed": mass_check.passed,
                        "message": mass_check.message,
                    },
                },
            }
            if params.include_plot:
                fig, _ax = plot_flux_distribution(
                    result, top_n=params.top_n
                )
                response["plot"] = _figure_to_base64(fig)
            return response
        except _TOOL_ERRORS as exc:
            return _error_response(exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry point for the bioeng-qpcr MCP server."""
    mcp.run()


if __name__ == "__main__":
    mcp.run()
