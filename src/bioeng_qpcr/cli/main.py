"""Command-line interface entry point for bioeng-qpcr.

This module provides a durable, non-Jupyter command-line interface to the
bioeng-qpcr engine.  It wraps the IO, validation, calculations, provenance
and plotting layers behind these sub-commands:

* ``validate``   -- run a MIQE readiness assessment on a tidy CSV.
* ``outliers``   -- detect suspect technical-replicate wells (Chauvenet).
* ``analyze``    -- run relative-quantification analysis (ΔΔCq / Pfaffl).
* ``report``     -- guided HTML readiness + MIQE 2.0 + quantities + provenance.
* ``miqe``       -- emit a standalone MIQE 2.0 (2025) reporting checklist.
* ``provenance`` -- emit a JSON provenance manifest for a dataset.
* ``plot``       -- render fold-change bar plots or Cq heatmaps.
* ``preview``    -- describe an arbitrary CSV and check it against a format.
* ``schema``     -- list the import formats, show one, or write a template.
* ``potency``    -- fit dose-response curves and report IC50/EC50.
* ``dock``       -- rigid-body molecular docking with ranked poses.
* ``align``      -- sequencing-read alignment, variants and audit rules.
* ``standard-curve`` -- fit efficiency and empirical LOD/quantifiable range.

Every command that ingests data accepts both tidy CSV files and RDML
files (``.rdml``/``.rdm`` archives or the XML they contain); RDML sample
groups, reference-target types, and raw amplification curves are mapped
onto the same validation and calculation pipeline.

All commands are designed to be testable with :class:`click.testing.CliRunner`
and use exit codes consistently:

* ``0`` -- success.
* ``1`` -- analysis/validation error or non-compliance.
* ``2`` -- click usage error (missing required option, bad choice, etc.).

The ``click`` library is declared as an optional dependency (``[cli]``) in
``pyproject.toml``; the ``matplotlib``-dependent ``plot`` command imports the
plotting module lazily so that ``validate``, ``analyze`` and ``provenance``
work without matplotlib installed.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import click

from bioeng_qpcr import __version__
from bioeng_qpcr.calculations import (
    assess_standard_curve_range,
    compare_groups,
    delta_delta_cq_batch,
    fit_efficiency_from_standard_curve,
    normalized_quantity_all,
    pfaffl_batch,
)
from bioeng_qpcr.io import CSVValidationError, read_rdml, read_tidy_csv
from bioeng_qpcr.provenance import ProvenanceManifest
from bioeng_qpcr.reporting import build_research_report
from bioeng_qpcr.validation import (
    RuleExecutionError,
    ValidationEngine,
    miqe2_checklist,
)

__all__ = ["cli"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split_reference_targets(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> list[str] | None:
    """Click callback: split a comma-separated string into a list.

    ``None`` and empty strings yield ``None`` so that ``read_tidy_csv`` keeps
    its default behaviour (no reference-target override).
    """
    if not value:
        return None
    targets = [v.strip() for v in value.split(",") if v.strip()]
    return targets or None


def _emit(content: str, output_file: str | None) -> None:
    """Write *content* to *output_file* or echo it to stdout.

    When *output_file* is ``None`` the content is printed to stdout via
    :func:`click.echo` (which appends a trailing newline).  When a file path
    is given the content is written verbatim with a single trailing newline
    so the file is well-formed on every platform.
    """
    if output_file:
        Path(output_file).write_text(content + "\n", encoding="utf-8")
    else:
        click.echo(content)


def _fail(message: str) -> None:
    """Print *message* to stderr and exit with code 1.

    Raises :class:`click.exceptions.Exit` so that :class:`CliRunner` captures
    the exit code cleanly without printing a traceback.
    """
    click.echo(message, err=True)
    raise click.exceptions.Exit(1)


def _load_miqe_answers(path: str | None) -> dict[str, bool | str] | None:
    """Load a MIQE evidence mapping from a JSON object."""
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Error reading MIQE answers: {exc}")
    if not isinstance(payload, dict):
        _fail("Error reading MIQE answers: top-level JSON value must be an object")
    return payload


#: Extensions parsed as RDML (ZIP archive, gzip, or the XML itself).
#: A non-RDML XML file produces a clear "expected an <rdml> root element"
#: error from the reader, so routing every ``.xml`` to the RDML reader
#: cannot silently mis-parse anything.
_RDML_SUFFIXES = frozenset({".rdml", ".rdm", ".xml"})


def _load_experiment(
    csv_path: str,
    reference_targets: list[str] | None = None,
    *,
    strict: bool = True,
):
    """Read a tidy CSV or RDML file into an :class:`~bioeng_qpcr.domain.Experiment`.

    Files with an ``.rdml``/``.rdm`` extension are parsed as RDML archives
    (ZIP, gzip, or plain XML); everything else is read as a tidy CSV.  On
    :class:`FileNotFoundError` or :class:`ValueError` an error message is
    printed and the process exits with code 1.
    """
    reader = (
        read_rdml if Path(csv_path).suffix.lower() in _RDML_SUFFIXES else read_tidy_csv
    )
    try:
        return reader(
            csv_path,
            reference_targets=reference_targets,
            strict=strict,
        )
    except FileNotFoundError:
        _fail(f"Error: File not found: {csv_path}")
    except CSVValidationError as exc:
        _fail(f"Error: {exc}")
    except ValueError as exc:
        _fail(f"Error: {exc}")


def _results_to_dicts(results: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of dataclass result objects to plain dictionaries."""
    return [asdict(r) for r in results if is_dataclass(r)]


def _dicts_to_csv(dicts: list[dict[str, Any]], *, escape_formulas: bool = True) -> str:
    """Serialise a list of homogeneous dictionaries to CSV text.

    Returns an empty string when *dicts* is empty (no rows to write).

    ``lineterminator`` is ``"\\n"`` rather than CSV's conventional ``"\\r\\n"``
    on purpose: the text then survives both the stdout and the file paths
    unchanged, whereas a literal ``\\r\\n`` gets its ``\\n`` translated a second
    time on Windows and prints a blank line between every row.
    """
    if not dicts:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(dicts[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            key: (
                "'" + value
                if escape_formulas
                and isinstance(value, str)
                and value.lstrip(" ").startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
                else value
            )
            for key, value in row.items()
        }
        for row in dicts
    )
    return buf.getvalue().rstrip("\r\n")


def _parse_parameters(raw: tuple[str, ...]) -> dict[str, str]:
    """Parse a tuple of ``KEY=VALUE`` strings into a dictionary.

    Each item must contain at least one ``=``.  Whitespace around the key
    and value is stripped.  On malformed input the process exits with code 1.
    """
    params: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            _fail(f"Error: invalid parameter {item!r}, expected KEY=VALUE")
        key, _, val = item.partition("=")
        key = key.strip()
        val = val.strip()
        if not key:
            _fail(f"Error: empty parameter key in {item!r}")
        params[key] = val
    return params


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="bioeng-qpcr")
def cli() -> None:
    """bioeng-qpcr: auditable qPCR analysis engine.

    Run MIQE validation, relative-quantification analysis, provenance
    tracking and plotting from the command line.
    """


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write the report to this file instead of stdout.",
)
@click.option(
    "--reference-targets",
    callback=_split_reference_targets,
    default=None,
    help="Comma-separated list of reference (housekeeping) gene names.",
)
@click.option(
    "--lenient",
    is_flag=True,
    help="Accept recoverable input problems and record them as diagnostics.",
)
def validate(
    csv_path: str,
    output_format: str,
    output_file: str | None,
    reference_targets: list[str] | None,
    lenient: bool,
) -> None:
    """Run a MIQE readiness assessment on a tidy CSV or RDML file.

    Exits with code 1 only when a rule fails or the input cannot be read.
    Warnings and unavailable evidence remain visible in the report but do not
    make a completed assessment look like a command failure.
    """
    experiment = _load_experiment(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )

    engine = ValidationEngine()
    try:
        report = engine.run(experiment)
    except RuleExecutionError as exc:
        _fail(f"Internal validation error: {exc}")

    content = report.to_json(indent=2) if output_format == "json" else report.to_text()

    _emit(content, output_file)

    ctx = click.get_current_context()
    ctx.exit(1 if report.failures else 0)


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------


def _format_outlier_report(result: Any) -> str:
    """Render a :class:`ReplicateOutlierResult` as human-readable text."""
    lines = ["Technical-replicate outlier report (Chauvenet criterion)", ""]
    lines.append(
        f"Parameters: sd-trigger={result.sd_trigger}, "
        f"min-group-size={result.min_group_size}"
    )
    lines.append(f"{result.summary()}")
    lines.append("")

    if result.chauvenet_groups:
        lines.append("Chauvenet-rejected wells (statistical evidence):")
        for group in result.chauvenet_groups:
            lines.extend(
                f"  {group.label}  well {flag.position}  "
                f"Cq={flag.cq:.2f}  z={flag.z_score:.2f}  "
                f"P={flag.tail_probability:.2e} < "
                f"{flag.threshold_probability:.3f}"
                for flag in group.chauvenet_flags
            )
        lines.append("")

    if result.review_groups:
        lines.append(
            "Review recommended (spread exceeds trigger; heuristic, "
            "no statistical flag):"
        )
        for group in result.review_groups:
            candidate = (
                f"  candidate well {group.review_candidate_position} "
                f"({group.review_candidate_deviation:.2f} Cq from median)"
                if group.review_candidate_position
                else ""
            )
            lines.append(
                f"  {group.label}  SD={group.std_cq:.2f}  n={group.n_wells}{candidate}"
            )
        lines.append("")

    if not result.chauvenet_groups and not result.review_groups:
        lines.append("No Chauvenet flags and no review recommendations.")
    else:
        lines.append(f"Groups passing: {result.n_pass}")
    return "\n".join(lines)


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--sd-trigger",
    type=click.FloatRange(min=0),
    default=0.3,
    show_default=True,
    help=(
        "Cq standard deviation above which an unflagged replicate group "
        "is marked for review."
    ),
)
@click.option(
    "--min-group-size",
    type=click.IntRange(min=3),
    default=3,
    show_default=True,
    help="Smallest technical-replicate group size assessed by Chauvenet.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write the report to this file instead of stdout.",
)
@click.option(
    "--lenient",
    is_flag=True,
    help="Accept recoverable input problems and record them as diagnostics.",
)
def outliers(
    csv_path: str,
    sd_trigger: float,
    min_group_size: int,
    output_format: str,
    output_file: str | None,
    lenient: bool,
) -> None:
    """Identify suspect technical-replicate wells (recommendations only).

    Applies Chauvenet's criterion iteratively to each technical-replicate
    group and reports rejected wells with full statistical evidence.
    Groups whose Cq spread exceeds the SD trigger without a statistical
    flag are listed for manual review with their most deviant well.  The
    input file is never modified and detected wells are never excluded
    automatically.  The command exits 0 even when outliers are found:
    findings are recommendations, not command failures.
    """
    from bioeng_qpcr.calculations import detect_replicate_outliers

    experiment = _load_experiment(csv_path, strict=not lenient)
    result = detect_replicate_outliers(
        experiment,
        min_group_size=min_group_size,
        sd_trigger=sd_trigger,
    )

    if output_format == "json":
        content = json.dumps(result.to_dict(), indent=2)
    else:
        content = _format_outlier_report(result)
    _emit(content, output_file)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--target",
    required=True,
    help="Target gene of interest.",
)
@click.option(
    "--control",
    default=None,
    help="Control sample; required for delta_delta_cq and pfaffl.",
)
@click.option(
    "--method",
    type=click.Choice(["normalized_quantity", "pfaffl", "delta_delta_cq"]),
    default="normalized_quantity",
    show_default=True,
    help="Relative-quantification method.",
)
@click.option(
    "--reference-targets",
    "--reference-target",
    "reference_targets",
    callback=_split_reference_targets,
    default=None,
    help="Comma-separated reference genes; defaults to all designated references.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default="json",
    show_default=True,
    help="Output format for the results.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write results to this file instead of stdout.",
)
@click.option(
    "--unsafe-raw-csv",
    is_flag=True,
    help=(
        "Preserve formula-leading CSV cells for machine round-trips; "
        "unsafe in spreadsheets."
    ),
)
@click.option(
    "--provenance-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write a provenance manifest paired with this analysis.",
)
@click.option(
    "--allow-exclusions",
    is_flag=True,
    help="Return partial results while prominently reporting excluded samples.",
)
@click.option(
    "--lenient",
    is_flag=True,
    help="Accept recoverable input problems and retain audit diagnostics.",
)
def analyze(
    csv_path: str,
    target: str,
    control: str | None,
    method: str,
    reference_targets: list[str] | None,
    output_format: str,
    output_file: str | None,
    unsafe_raw_csv: bool,
    provenance_file: str | None,
    allow_exclusions: bool,
    lenient: bool,
) -> None:
    """Run an auditable relative-quantification analysis.

    Efficiency-corrected normalized quantities are the default and require
    an ``efficiency`` column in the CSV.  When efficiency data is unavailable
    use ``--method delta_delta_cq`` instead (assumes ~100% efficiency).
    """
    experiment = _load_experiment(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )

    try:
        if method == "normalized_quantity":
            try:
                batch = normalized_quantity_all(
                    experiment,
                    target,
                    reference_targets=reference_targets,
                )
            except ValueError as exc:
                msg = str(exc)
                if "efficiency" in msg.lower():
                    msg += (
                        "\n\nTip: normalized_quantity requires measured amplification "
                        "efficiencies for every assay.\n"
                        "Options:\n"
                        "  1. Add an 'efficiency' column to your CSV\n"
                        "  2. Use --method delta_delta_cq instead (assumes 100% "
                        "efficiency)"
                    )
                _fail(f"Error: {msg}")
        else:
            if control is None:
                _fail(f"Error: --control is required for method {method}")
            if reference_targets and len(reference_targets) > 1:
                _fail(f"Error: method {method} accepts exactly one reference target")
            reference_target = reference_targets[0] if reference_targets else None
            if method == "delta_delta_cq":
                batch = delta_delta_cq_batch(
                    experiment,
                    target,
                    control,
                    reference_target=reference_target,
                )
            else:
                batch = pfaffl_batch(
                    experiment,
                    target,
                    control,
                    reference_target=reference_target,
                )
    except ValueError as exc:
        _fail(f"Error: {exc}")

    if batch.exclusions and not allow_exclusions:
        try:
            batch.raise_for_exclusions()
        except ValueError as exc:
            _fail(
                f"Error: {exc}. Re-run with --allow-exclusions "
                "to audit partial results."
            )

    dicts = _results_to_dicts(batch.results)

    if output_format == "json":
        content = json.dumps(batch.to_dict(), indent=2, ensure_ascii=False)
    else:
        content = _dicts_to_csv(dicts, escape_formulas=not unsafe_raw_csv)
        for warning in batch.warnings:
            click.echo(f"Warning: {warning}", err=True)
        for exclusion in batch.exclusions:
            click.echo(
                f"Excluded {exclusion.sample_name}: {exclusion.reason}", err=True
            )

    _emit(content, output_file)

    if provenance_file:
        manifest = ProvenanceManifest.from_experiment(
            experiment, parameters=batch.parameters
        )
        manifest.exclusions = batch.to_dict()["exclusions"]
        _emit(manifest.to_json(), provenance_file)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("csv_path", type=click.Path(dir_okay=False, path_type=str))
@click.option("--target", required=True, help="Target gene of interest.")
@click.option("--treated-group", required=True, help="Treated sample-group name.")
@click.option("--control-group", required=True, help="Control sample-group name.")
@click.option(
    "--reference-targets",
    callback=_split_reference_targets,
    default=None,
    help="Comma-separated reference genes.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
)
@click.option("--allow-exclusions", is_flag=True)
@click.option("--lenient", is_flag=True)
def compare(
    csv_path: str,
    target: str,
    treated_group: str,
    control_group: str,
    reference_targets: list[str] | None,
    output_file: str | None,
    allow_exclusions: bool,
    lenient: bool,
) -> None:
    """Compare group means of per-sample normalized quantities."""
    experiment = _load_experiment(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )
    try:
        result, batch = compare_groups(
            experiment,
            target,
            treated_group,
            control_group,
            reference_targets,
        )
        if batch.exclusions and not allow_exclusions:
            batch.raise_for_exclusions()
    except ValueError as exc:
        _fail(f"Error: {exc}")
    comparison_parameters = {
        **batch.parameters,
        "treated_group": treated_group,
        "control_group": control_group,
        "group_summary": "geometric_mean",
        "inference_scale": "log2_normalized_quantity",
    }
    manifest = ProvenanceManifest.from_experiment(
        experiment,
        parameters=comparison_parameters,
    )
    manifest.exclusions = batch.to_dict()["exclusions"]
    payload = {
        "comparison": asdict(result),
        "analysis_audit": batch.to_dict(),
        "provenance": manifest.to_dict(),
    }
    _emit(json.dumps(payload, indent=2, ensure_ascii=False), output_file)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("csv_path", type=click.Path(dir_okay=False, path_type=str))
@click.option("--target", required=True, help="Target gene of interest.")
@click.option(
    "--reference-targets",
    callback=_split_reference_targets,
    default=None,
    help="Comma-separated reference genes.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default="bioeng-qpcr-report.html",
    show_default=True,
)
@click.option(
    "--json-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Optionally write the complete machine-readable report.",
)
@click.option(
    "--miqe-output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Optionally write the embedded MIQE 2.0 checklist as standalone JSON.",
)
@click.option(
    "--package-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Optionally write a ZIP research package with input, report and checksums.",
)
@click.option(
    "--miqe-answers-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="JSON object mapping MIQE field IDs to evidence strings or booleans.",
)
@click.option("--lenient", is_flag=True)
def report(
    csv_path: str,
    target: str,
    reference_targets: list[str] | None,
    output_file: str,
    json_file: str | None,
    miqe_output_file: str | None,
    package_file: str | None,
    miqe_answers_file: str | None,
    lenient: bool,
) -> None:
    """Generate a guided HTML readiness, analysis, MIQE, and provenance report."""
    experiment = _load_experiment(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )
    try:
        researcher_report = build_research_report(
            experiment,
            target,
            reference_targets,
            miqe_answers=_load_miqe_answers(miqe_answers_file),
        )
    except (RuleExecutionError, ValueError) as exc:
        _fail(f"Error: {exc}")
    Path(output_file).write_text(researcher_report.html, encoding="utf-8")
    if json_file:
        Path(json_file).write_text(researcher_report.to_json() + "\n", encoding="utf-8")
    if miqe_output_file:
        miqe_payload = json.dumps(
            researcher_report.payload["miqe2"], indent=2, ensure_ascii=False
        )
        Path(miqe_output_file).write_text(miqe_payload + "\n", encoding="utf-8")
    if package_file:
        researcher_report.write_package(package_file, source_file=csv_path)
    click.echo(f"Research report saved to {output_file}")


# ---------------------------------------------------------------------------
# standard-curve
# ---------------------------------------------------------------------------


@cli.command("standard-curve")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--target", default="unknown", show_default=True)
@click.option("--min-detection-rate", default=0.95, show_default=True, type=float)
@click.option("--max-cq-sd", default=0.5, show_default=True, type=float)
@click.option("--min-replicates", default=3, show_default=True, type=int)
@click.option(
    "--predict-concentration",
    multiple=True,
    type=float,
    help="Add a Cq prediction interval at this concentration; may be repeated.",
)
@click.option("--confidence", default=0.95, show_default=True, type=float)
@click.option("--output-file", default=None)
def standard_curve(
    csv_path: str,
    target: str,
    min_detection_rate: float,
    max_cq_sd: float,
    min_replicates: int,
    predict_concentration: tuple[float, ...],
    confidence: float,
    output_file: str | None,
) -> None:
    """Fit a replicated qPCR standard curve and report its usable range."""
    import numpy as np

    from bioeng_qpcr.io import read_csv_table, schema_for, validate_rows_against_schema

    try:
        table = read_csv_table(csv_path)
        schema = schema_for("qpcr_standard_curve")
        issues = [*table.issues, *validate_rows_against_schema(table, schema)]
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise CSVValidationError(errors)
        concentration_spec = schema.find("concentration")
        cq_spec = schema.find("cq")
        concentration_index = next(
            index
            for index, header in enumerate(table.header)
            if schema.find(header) is concentration_spec
        )
        cq_index = next(
            index
            for index, header in enumerate(table.header)
            if schema.find(header) is cq_spec
        )
        concentrations = np.asarray(
            [float(row[concentration_index]) for row in table.rows], dtype=float
        )
        cq_values = np.asarray(
            [
                float(row[cq_index])
                if row[cq_index].strip().lower()
                not in {
                    "",
                    "na",
                    "n/a",
                    "nan",
                    "null",
                    "none",
                    "-",
                    "nd",
                    "undetermined",
                }
                else np.nan
                for row in table.rows
            ],
            dtype=float,
        )
        performance = assess_standard_curve_range(
            concentrations,
            cq_values,
            min_detection_rate=min_detection_rate,
            max_cq_sd=max_cq_sd,
            min_replicates=min_replicates,
        )
        qualified_levels = np.asarray(
            [
                level.concentration
                for level in performance.levels
                if level.quantification_qualified
            ]
        )
        detected = np.isfinite(cq_values)
        fit_mask = detected & np.isin(concentrations, qualified_levels)
        fit = fit_efficiency_from_standard_curve(
            np.log10(concentrations[fit_mask]), cq_values[fit_mask], target
        )
        predictions = []
        for concentration in predict_concentration:
            if not np.isfinite(concentration) or concentration <= 0:
                raise ValueError(
                    "Prediction concentrations must be finite and positive."
                )
            estimate, lower, upper = fit.predict_cq_interval(
                float(np.log10(concentration)), confidence=confidence
            )
            predictions.append(
                {
                    "concentration": concentration,
                    "estimated_cq": estimate,
                    "lower_cq": lower,
                    "upper_cq": upper,
                    "confidence": confidence,
                    "extrapolated": bool(
                        performance.lloq is None
                        or performance.uloq is None
                        or concentration < performance.lloq
                        or concentration > performance.uloq
                    ),
                }
            )
    except (CSVValidationError, ValueError, OSError) as exc:
        _fail(f"Error: {exc}")

    payload = {
        "target": target,
        "fit": asdict(fit),
        "performance": asdict(performance),
        "predictions": predictions,
        "input_warnings": [
            issue.to_dict() for issue in issues if issue.severity != "error"
        ],
    }
    _emit(json.dumps(payload, indent=2, ensure_ascii=False), output_file)


# ---------------------------------------------------------------------------
# miqe: MIQE 2.0 reporting-completeness checklist
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--reference-targets",
    callback=_split_reference_targets,
    default=None,
    help="Comma-separated reference (housekeeping) gene names.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    show_default=True,
    help="Checklist format.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write the checklist to this file instead of stdout.",
)
@click.option(
    "--answers-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="JSON object mapping MIQE field IDs to evidence strings or booleans.",
)
@click.option("--lenient", is_flag=True)
def miqe(
    csv_path: str,
    reference_targets: list[str] | None,
    output_format: str,
    output_file: str | None,
    answers_file: str | None,
    lenient: bool,
) -> None:
    """Emit a MIQE 2.0 (2025) reporting-completeness checklist.

    The alpha report is not a certification: it records which of the
    MIQE 2.0 items are attested by the data (``reported``), still expected
    yet absent (``missing``), or not applicable (``na``).
    """
    experiment = _load_experiment(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )
    try:
        checklist = miqe2_checklist(
            experiment,
            answers=_load_miqe_answers(answers_file),
            experiment_name=experiment.name,
        )
    except ValueError as exc:
        _fail(f"Error: {exc}")
    content = checklist.to_json() if output_format == "json" else checklist.to_text()
    _emit(content, output_file)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write the manifest to this file instead of stdout.",
)
@click.option(
    "--parameters",
    "parameters_raw",
    multiple=True,
    help="Analysis parameters as KEY=VALUE pairs (may be repeated).",
)
def provenance(
    csv_path: str,
    output_file: str | None,
    parameters_raw: tuple[str, ...],
) -> None:
    """Emit a JSON provenance manifest for a dataset.

    The manifest records the input content hash, package versions, analysis
    parameters and a UTC creation timestamp for full reproducibility.
    """
    experiment = _load_experiment(csv_path)
    parameters = _parse_parameters(parameters_raw)
    manifest = ProvenanceManifest.from_experiment(
        experiment,
        parameters=parameters or None,
    )
    content = manifest.to_json()
    _emit(content, output_file)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--type",
    "plot_type",
    type=click.Choice(["fold_change", "heatmap"]),
    required=True,
    help="Type of plot to generate.",
)
@click.option(
    "--target",
    default=None,
    help="Target gene of interest (required for fold_change).",
)
@click.option(
    "--control",
    default=None,
    help="Control (calibrator) sample name (required for fold_change).",
)
@click.option(
    "--reference-target",
    default=None,
    help="Reference (housekeeping) gene name (for fold_change).",
)
@click.option(
    "--output-file",
    default="plot.png",
    show_default=True,
    help="Path where the image will be saved.",
)
def plot(
    csv_path: str,
    plot_type: str,
    target: str | None,
    control: str | None,
    reference_target: str | None,
    output_file: str,
) -> None:
    """Render a fold-change bar plot or a Cq heatmap.

    For ``fold_change`` the ΔΔCq method is run first (requires --target and
    --control) and the resulting fold-change values are plotted as bars.
    For ``heatmap`` the per-well Cq matrix is drawn directly from the
    experiment.
    """
    # Lazy imports: matplotlib is an optional dependency.  Setting the Agg
    # backend guarantees the command works headless (file output only).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bioeng_qpcr.plotting import plot_cq_heatmap, plot_fold_change, save_figure

    ref_list = [reference_target] if reference_target else None
    experiment = _load_experiment(csv_path, reference_targets=ref_list)

    if plot_type == "fold_change":
        if not target or not control:
            _fail("Error: --target and --control are required for fold_change plots.")
        try:
            batch = delta_delta_cq_batch(
                experiment, target, control, reference_target=reference_target
            )
            batch.raise_for_exclusions()
            results = batch.results
        except ValueError as exc:
            _fail(f"Error: {exc}")
        if not results:
            _fail("Error: no results to plot (no samples with valid data).")
        fig, _ax = plot_fold_change(results)
    else:
        fig, _ax = plot_cq_heatmap(experiment)

    save_figure(fig, output_file)
    plt.close(fig)
    click.echo(f"Plot saved to {output_file}")


# ---------------------------------------------------------------------------
# curves: per-well amplification-curve analysis
# ---------------------------------------------------------------------------


def _load_curve_experiment(
    csv_path: str,
    reference_targets: list[str] | None = None,
):
    """Load an experiment that carries raw amplification traces.

    A long-format trace CSV (``position, cycle, fluorescence``) is read with
    :func:`read_amplification_csv`; anything else (including RDML) is routed
    through :func:`_load_experiment`, which maps RDML ``<adp>`` data onto
    ``Experiment.traces``.
    """
    from bioeng_qpcr.io import NotAmplificationFile, read_amplification_csv

    suffix = Path(csv_path).suffix.lower()
    if suffix not in _RDML_SUFFIXES:
        try:
            return read_amplification_csv(csv_path, reference_targets=reference_targets)
        except (NotAmplificationFile, FileNotFoundError):
            pass
    return _load_experiment(csv_path, reference_targets)


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(dir_okay=False, path_type=str),
)
@click.option(
    "--baseline-method",
    type=click.Choice(["constant", "linear"]),
    default="constant",
    show_default=True,
    help="Background model used to correct each trace before fitting.",
)
@click.option(
    "--reference-targets",
    default=None,
    help="Comma-separated reference (housekeeping) gene names.",
    callback=_split_reference_targets,
)
@click.option(
    "--output-file",
    default=None,
    help="Write the per-well and per-assay JSON summary to this path.",
)
def curves(
    csv_path: str,
    baseline_method: str,
    reference_targets: list[str] | None,
    output_file: str | None,
) -> None:
    """Fit every raw amplification curve and report per-well metrics.

    Inputs may be a long-format trace CSV (``position, cycle,
    fluorescence``, plus optional ``target_name``/``sample_name``/``cq``) or
    an RDML file containing ``<adp>`` amplification data.  Each well is
    baseline-corrected and fit to a sigmoid to derive a curve *Cq* and a
    per-well efficiency from the early exponential window; per-assay median
    efficiencies are aggregated, and every well that could not be fitted is
    retained in the report rather than silently dropped.
    """
    from bioeng_qpcr.calculations import analyze_amplification_curves

    experiment = _load_curve_experiment(csv_path, reference_targets)
    if not experiment.has_traces:
        _fail("Error: no raw amplification traces were found in the input.")

    try:
        result = analyze_amplification_curves(
            experiment, baseline_method=baseline_method
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Error: {exc}")

    rows = []
    for position in sorted(result.per_well):
        fit = result.per_well[position]
        rows.append(
            {
                "position": position,
                "target": result.trace_target.get(position, ""),
                "status": fit.status,
                "cq": fit.cq,
                "cq_sdm": fit.cq_sdm,
                "efficiency": fit.eff,
                "delta_rn": fit.delta_rn,
                "r_squared": fit.r_squared,
                "baseline": fit.baseline,
            }
        )
    assay_rows = [
        {
            "target": summary.target,
            "median_efficiency": summary.median,
            "n_wells": summary.n_wells,
            "min_efficiency": summary.min,
            "max_efficiency": summary.max,
        }
        for summary in result.per_assay.values()
    ]
    summary_dict = {
        "per_well": rows,
        "per_assay": assay_rows,
        "wells_without_trace": list(result.wells_without_trace),
        "traces_without_well": list(result.traces_without_well),
        "non_amplifying": list(result.non_amplifying),
        "warnings": list(result.warnings),
    }

    if output_file:
        Path(output_file).write_text(
            json.dumps(summary_dict, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        click.echo(f"Amplification-curve summary written to {output_file}")
        return

    text_lines = ["Amplification-curve summary", "=" * 30]
    text_lines.extend(
        f"{row['position']:>4}  {row['target']:<10} {row['status']:<14} "
        f"Cq={_fmt(row['cq'])}  Eff={_fmt(row['efficiency'])}  "
        f"ΔRn={_fmt(row['delta_rn'])}  R²={_fmt(row['r_squared'])}"
        for row in rows
    )
    text_lines.append("")
    text_lines.append("Per-assay median efficiency:")
    text_lines.extend(
        f"  {a['target']:<10} n={a['n_wells']}  median={_fmt(a['median_efficiency'])}"
        for a in assay_rows
    )
    for label, items in (
        ("wells without a trace", result.wells_without_trace),
        ("traces without a well", result.traces_without_well),
        ("non-amplifying wells", result.non_amplifying),
    ):
        if items:
            text_lines.append(f"{label}: {', '.join(items)}")
    _emit("\n".join(text_lines).rstrip("\n"), None)


def _fmt(value: float | None) -> str:
    """Plain-text formatting for an optional float."""
    if value is None:
        return "-"
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.option(
    "--format",
    "csv_format",
    default=None,
    help="Force a registered import format instead of auto-detecting one.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--rows",
    default=20,
    show_default=True,
    help="Data rows shown in the preview table.",
)
@click.option(
    "--scan-rows",
    default=200,
    show_default=True,
    help="Data rows profiled when computing column statistics.",
)
@click.option(
    "--output-file",
    default=None,
    help="Write the report to a file instead of stdout.",
)
def preview(
    csv_path: str,
    csv_format: str | None,
    output_format: str,
    rows: int,
    scan_rows: int,
    output_file: str | None,
) -> None:
    """Describe a CSV file and check it against its import format.

    Nothing is repaired: ragged rows, duplicate headers, unexpected columns,
    unparsable numbers and out-of-range values are all reported with the row
    and column they came from, so the file can be fixed at the source.
    """
    from bioeng_qpcr.io import preview_csv

    try:
        result = preview_csv(
            csv_path, schema=csv_format, preview_rows=scan_rows, display_rows=rows
        )
    except KeyError as exc:
        _fail(f"Error: {exc}")
    except (ValueError, OSError) as exc:
        _fail(f"Error: {exc}")

    if output_format == "json":
        import json as _json

        payload = {
            "path": result.table.path,
            "encoding": result.table.encoding,
            "delimiter": result.table.delimiter,
            "rows_total": result.rows_total,
            "rows_scanned": result.rows_scanned,
            "truncated": result.truncated,
            "size_bytes": result.size_bytes,
            "sha256": result.sha256,
            "schema": result.schema_key,
            "schema_score": result.schema_score,
            "columns": [asdict(column) for column in result.columns],
            "issues": [issue.to_dict() for issue in result.issues],
        }
        _emit(_json.dumps(payload, indent=2, ensure_ascii=False), output_file)
    else:
        _emit(result.to_text(max_table_rows=rows), output_file)

    # Errors mean the file cannot be analysed as it stands.
    if result.errors:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--list",
    "list_formats",
    is_flag=True,
    help="List every registered import format.",
)
@click.option(
    "--format",
    "csv_format",
    default=None,
    help="Show one format in full.",
)
@click.option(
    "--write-template",
    default=None,
    help="Write a template CSV for --format to this path.",
)
@click.option(
    "--write-import-spec",
    default=None,
    help="Write the generated import specification (Markdown) to this path.",
)
@click.option(
    "--rows",
    default=0,
    show_default=True,
    help="Data rows in a generated template (0 = the format's own example).",
)
def schema(
    list_formats: bool,
    csv_format: str | None,
    write_template: str | None,
    write_import_spec: str | None,
    rows: int,
) -> None:
    """List, document or template the CSV import formats.

    The formats are declared in code and this command renders the same
    registry, so the documentation cannot describe a format the reader does
    not implement.
    """
    from bioeng_qpcr.io import (
        SCHEMAS,
        import_spec_markdown,
        schema_for,
        write_template_csv,
    )

    produced_something = False

    if list_formats or (csv_format is None and not write_import_spec):
        lines = ["Registered import formats", "=" * 52]
        for key in sorted(SCHEMAS):
            entry = SCHEMAS[key]
            lines.append(f"  {key:<24} {entry.title}")
        lines.append("")
        lines.append("Use --format KEY to see one in full.")
        _emit("\n".join(lines), None)
        produced_something = True

    if csv_format is not None:
        try:
            resolved = schema_for(csv_format)
        except KeyError as exc:
            _fail(f"Error: {exc}")
        if write_template is not None:
            try:
                destination = write_template_csv(write_template, resolved, n_rows=rows)
            except (ValueError, OSError) as exc:
                _fail(f"Error: {exc}")
            click.echo(f"Wrote template: {destination}")
        else:
            _emit(resolved.to_markdown(), None)
        produced_something = True

    if write_import_spec is not None:
        Path(write_import_spec).write_text(import_spec_markdown(), encoding="utf-8")
        click.echo(f"Wrote import specification: {write_import_spec}")
        produced_something = True

    if not produced_something:
        _fail("Error: nothing to do; see --help.")


# ---------------------------------------------------------------------------
# potency
# ---------------------------------------------------------------------------


def _parse_models(raw: str | None):
    """Parse a comma-separated model list into :class:`DoseResponseModel`s."""
    from bioeng_qpcr.potency import DoseResponseModel

    if not raw:
        return None
    models = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            models.append(DoseResponseModel(token))
        except ValueError:
            _fail(
                f"Error: unknown model {token!r}; expected one of "
                + ", ".join(model.value for model in DoseResponseModel)
            )
    return models or None


@cli.command()
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.option(
    "--kind",
    type=click.Choice(["auto", "IC50", "EC50"]),
    default="auto",
    show_default=True,
    help="Potency to report; 'auto' follows the fitted curve direction.",
)
@click.option(
    "--layout",
    type=click.Choice(["auto", "long", "matrix"]),
    default="auto",
    show_default=True,
    help="Dose-response table layout.",
)
@click.option(
    "--normalize",
    type=click.Choice(["none", "percent_of_control"]),
    default="none",
    show_default=True,
    help="Optional rescaling of the response column.",
)
@click.option(
    "--model",
    "model_list",
    default=None,
    help="Comma-separated candidate models (4pl,5pl,3pl,gompertz).",
)
@click.option(
    "--ci-method",
    type=click.Choice(["asymptotic", "bootstrap"]),
    default="asymptotic",
    show_default=True,
    help="How confidence intervals are computed.",
)
@click.option(
    "--n-bootstrap",
    default=500,
    show_default=True,
    help="Bootstrap replicates when --ci-method=bootstrap.",
)
@click.option(
    "--confidence",
    default=0.95,
    show_default=True,
    help="Confidence level for every interval.",
)
@click.option(
    "--control-fraction",
    default=0.5,
    show_default=True,
    help="Fraction of the control defining the absolute potency.",
)
@click.option(
    "--control-response",
    type=float,
    default=None,
    help="Measured untreated/vehicle response; defaults to the lowest dose.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Seed for the bootstrap, so results are reproducible.",
)
@click.option(
    "--compare",
    default=None,
    help="Two compound names (comma-separated) for a relative-potency ratio.",
)
@click.option(
    "--no-validation",
    is_flag=True,
    help="Skip the POT-001..POT-011 audit rules.",
)
@click.option(
    "--plot",
    "plot_path",
    default=None,
    help="Save the fitted curves as an image (per compound when several).",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json", "csv"]),
    default="text",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output-file",
    default=None,
    help="Write the report to a file instead of stdout.",
)
@click.option(
    "--unsafe-raw-csv",
    is_flag=True,
    help=(
        "Preserve formula-leading CSV cells for machine round-trips; "
        "unsafe in spreadsheets."
    ),
)
def potency(
    csv_path: str,
    kind: str,
    layout: str,
    normalize: str,
    model_list: str | None,
    ci_method: str,
    n_bootstrap: int,
    confidence: float,
    control_fraction: float,
    control_response: float | None,
    seed: int,
    compare: str | None,
    no_validation: bool,
    plot_path: str | None,
    output_format: str,
    output_file: str | None,
    unsafe_raw_csv: bool,
) -> None:
    """Fit dose-response curves and report IC50/EC50 with audit rules.

    Both the *relative* potency (the midpoint of the fitted window) and the
    *absolute* potency (the concentration giving ``--control-fraction`` of the
    control response) are reported, because they differ whenever inhibition is
    incomplete and only the second one means "50 % inhibition".

    Exit code 1 is returned when a validation rule reports a failure, so a
    batch script can gate on it.
    """
    from bioeng_qpcr.io import read_dose_response_csv
    from bioeng_qpcr.potency import (
        PotencyKind,
        fit_dose_response,
        relative_potency,
        run_potency_validation,
    )

    models = _parse_models(model_list)
    try:
        datasets = read_dose_response_csv(csv_path, layout=layout, normalize=normalize)
    except (ValueError, OSError) as exc:
        _fail(f"Error: {exc}")

    results = []
    for dataset in datasets:
        try:
            results.append(
                fit_dose_response(
                    dataset,
                    kind=None if kind == "auto" else PotencyKind(kind),
                    models=models,
                    ci_method=ci_method,
                    confidence=confidence,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    control_fraction=control_fraction,
                    control_response=control_response,
                )
            )
        except ValueError as exc:
            _fail(f"Error: {dataset.compound}: {exc}")

    validation: dict[str, list] = {}
    if not no_validation:
        for dataset, result in zip(datasets, results, strict=True):
            validation[result.compound] = run_potency_validation(result, dataset)

    comparison = None
    if compare:
        names = [name.strip() for name in compare.split(",") if name.strip()]
        if len(names) != 2:
            _fail("Error: --compare expects exactly two compound names.")
        by_name = {result.compound: result for result in results}
        missing = [name for name in names if name not in by_name]
        if missing:
            _fail(
                "Error: --compare names not present in the file: " + ", ".join(missing)
            )
        try:
            comparison = relative_potency(by_name[names[0]], by_name[names[1]])
        except ValueError as exc:
            _fail(f"Error: {exc}")

    if output_format == "json":
        import json as _json

        payload = {
            "source_file": str(csv_path),
            "source_sha256": _sha256(csv_path),
            "ci_method": ci_method,
            "confidence": confidence,
            "results": [
                _potency_to_dict(result, validation.get(result.compound, []))
                for result in results
            ],
            "comparison": (
                {
                    "compound_a": comparison.compound_a,
                    "compound_b": comparison.compound_b,
                    "kind": comparison.kind.value,
                    "ratio": comparison.ratio,
                    "ci": list(comparison.ci) if comparison.ci else None,
                    "fold_difference": comparison.fold_difference,
                    "message": comparison.message,
                }
                if comparison is not None
                else None
            ),
        }
        _emit(_json.dumps(payload, indent=2, ensure_ascii=False), output_file)
    elif output_format == "csv":
        rows = [
            {
                "compound": result.compound,
                "kind": result.kind.value,
                "model": result.best_fit.model.value,
                "relative": result.relative_value,
                "relative_low": result.relative_ci[0] if result.relative_ci else "",
                "relative_high": result.relative_ci[1] if result.relative_ci else "",
                "absolute": (
                    result.absolute_value if result.absolute_value is not None else ""
                ),
                "hill_slope": result.best_fit.hill_slope,
                "r_squared": result.best_fit.r_squared,
                "n_points": result.best_fit.n_points,
                "extrapolated": result.extrapolated,
            }
            for result in results
        ]
        _emit(_dicts_to_csv(rows, escape_formulas=not unsafe_raw_csv), output_file)
    else:
        _emit(_potency_text(results, validation, comparison), output_file)

    if plot_path:
        _render_potency_plots(results, datasets, plot_path)

    failed = any(
        check.severity == "failure"
        for checks in validation.values()
        for check in checks
    )
    if failed:
        raise click.exceptions.Exit(1)


def _sha256(path: str) -> str:
    """Return the SHA-256 digest of a file."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _alignment_block(
    aligned_query: str, aligned_reference: str, width: int = 60
) -> list[str]:
    """Render an aligned pair as wrapped query / match / reference triplets.

    The match row uses ``|`` for an identity and a space otherwise, which is
    what makes a mismatch visible at a glance in a text report.
    """
    rendered: list[str] = []
    for start in range(0, len(aligned_query), width):
        query_slice = aligned_query[start : start + width]
        reference_slice = aligned_reference[start : start + width]
        marks = "".join(
            "|" if left == right else " "
            for left, right in zip(query_slice, reference_slice, strict=True)
        )
        rendered.append(f"q {query_slice}")
        rendered.append(f"  {marks}")
        rendered.append(f"r {reference_slice}")
    return rendered


def _potency_to_dict(result, checks) -> dict[str, Any]:
    """Serialise a potency result plus its validation outcomes."""
    return {
        "compound": result.compound,
        "kind": result.kind.value,
        "model": result.best_fit.model.value,
        "message": result.message,
        "relative_value": result.relative_value,
        "relative_ci": list(result.relative_ci) if result.relative_ci else None,
        "absolute_value": result.absolute_value,
        "absolute_ci": list(result.absolute_ci) if result.absolute_ci else None,
        "control_response": result.control_response,
        "control_fraction": result.control_fraction,
        "hill_slope": result.best_fit.hill_slope,
        "bottom": result.best_fit.bottom,
        "top": result.best_fit.top,
        "r_squared": result.best_fit.r_squared,
        "n_points": result.best_fit.n_points,
        "aicc": result.best_fit.aicc,
        "extrapolated": result.extrapolated,
        "fold_range_covered": result.fold_range_covered,
        "warnings": list(result.warnings),
        "model_selection": {
            "ambiguous": result.comparison.ambiguous,
            "message": result.comparison.message,
            "models": [
                {"model": fit.model.value, "aicc": fit.aicc}
                for fit in result.comparison.fits
            ],
        },
        "validation": [
            {
                "rule_id": check.rule_id,
                "passed": check.passed,
                "severity": check.severity,
                "message": check.message,
            }
            for check in checks
        ],
    }


def _potency_text(results, validation, comparison) -> str:
    """Render the human-readable potency report."""
    lines: list[str] = []
    for result in results:
        fit = result.best_fit
        lines.append(f"{result.compound}  —  {result.kind.value}")
        lines.append("=" * 56)
        lines.append(f"  相对 {result.kind.value}: {result.relative_value:.4g}")
        if result.relative_ci:
            lines.append(
                f"    {result.confidence:.0%} CI: "
                f"{result.relative_ci[0]:.4g} .. {result.relative_ci[1]:.4g}"
            )
        if result.absolute_value is not None:
            lines.append(
                f"  绝对 {result.kind.value}（对照的 "
                f"{result.control_fraction:.0%}）: {result.absolute_value:.4g}"
            )
            if result.absolute_ci:
                lines.append(
                    f"    {result.confidence:.0%} CI: "
                    f"{result.absolute_ci[0]:.4g} .. {result.absolute_ci[1]:.4g}"
                )
        else:
            lines.append(f"  绝对 {result.kind.value}: 不可达（曲线未覆盖该响应水平）")
        lines.append(
            f"  模型: {fit.model.value}   Hill 斜率: {fit.hill_slope:.3g}   "
            f"R²: {fit.r_squared:.5f}   n: {fit.n_points}"
        )
        lines.append(
            f"  窗口: {fit.bottom:.4g} → {fit.top:.4g}   "
            f"浓度跨度: {result.fold_range_covered:.2f} 个数量级   "
            f"CI 方法: {result.ci_method}"
        )
        if result.extrapolated:
            lines.append("  [!] 效价落在测试浓度范围之外（外推）")
        for warning in result.warnings:
            lines.append(f"  [!] {warning}")
        lines.append(f"  {result.comparison.message}")
        for check in validation.get(result.compound, []):
            if check.passed:
                continue
            tag = "[错误]" if check.severity == "failure" else "[警告]"
            lines.append(f"  {tag} {check.rule_id} {check.message}")
        lines.append("")

    if comparison is not None:
        lines.append(comparison.message)
        if comparison.ci:
            lines.append(
                f"  比值 {comparison.confidence:.0%} CI: "
                f"{comparison.ci[0]:.4g} .. {comparison.ci[1]:.4g}"
            )
    return "\n".join(lines).rstrip("\n")


def _render_potency_plots(results, datasets, plot_path: str) -> None:
    """Write one dose-response figure per compound."""
    import matplotlib

    matplotlib.use("Agg")

    from bioeng_qpcr.plotting import plot_dose_response, save_figure

    target = Path(plot_path)
    suffix = target.suffix or ".png"
    stem = target.with_suffix("")
    for result, dataset in zip(results, datasets, strict=True):
        if not result.best_fit.converged:
            continue
        figure, _ax = plot_dose_response(result.best_fit, dataset)
        if len(results) == 1:
            destination = target
        else:
            safe = "".join(
                character if character.isalnum() else "_"
                for character in result.compound
            )
            destination = stem.with_name(f"{stem.name}_{safe}{suffix}")
        save_figure(figure, str(destination))
        import matplotlib.pyplot as plt

        plt.close(figure)
        click.echo(f"Wrote plot: {destination}")


# ---------------------------------------------------------------------------
# dock
# ---------------------------------------------------------------------------


@cli.command()
@click.argument(
    "receptor_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.argument(
    "ligand_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=False,
)
@click.option(
    "--split-from-receptor",
    is_flag=True,
    help="Take the ligand out of the receptor file instead of a second file.",
)
@click.option(
    "--box-center",
    default=None,
    help="Search-box centre as x,y,z (default: the ligand's own centroid).",
)
@click.option(
    "--box-size",
    default=None,
    type=float,
    help="Half-size of the cubic search box in angstrom (default: automatic).",
)
@click.option(
    "--n-runs",
    default=4,
    show_default=True,
    help="Independent Monte Carlo runs.",
)
@click.option(
    "--n-steps",
    default=1500,
    show_default=True,
    help="Monte Carlo steps per run.",
)
@click.option(
    "--seed",
    default=1,
    show_default=True,
    help="RNG seed, so a run is reproducible.",
)
@click.option(
    "--top",
    "top_n",
    default=5,
    show_default=True,
    help="Number of poses reported.",
)
@click.option(
    "--rmsd-threshold",
    default=2.0,
    show_default=True,
    help="Minimum RMSD (A) between reported poses.",
)
@click.option(
    "--grid",
    "use_grid",
    is_flag=True,
    help="Use the deterministic grid search instead of simulated annealing.",
)
@click.option("--grid-translation-steps", default=3, show_default=True)
@click.option("--grid-rotation-samples", default=6, show_default=True)
@click.option(
    "--write-pose",
    default=None,
    help="Write the best pose back out as a PDB file.",
)
@click.option(
    "--no-validation",
    is_flag=True,
    help="Skip the DCK audit rules.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output-file",
    default=None,
    help="Write the report to a file instead of stdout.",
)
def dock(
    receptor_path: str,
    ligand_path: str | None,
    split_from_receptor: bool,
    box_center: str | None,
    box_size: float | None,
    n_runs: int,
    n_steps: int,
    seed: int,
    top_n: int,
    rmsd_threshold: float,
    use_grid: bool,
    grid_translation_steps: int,
    grid_rotation_samples: int,
    write_pose: str | None,
    no_validation: bool,
    output_format: str,
    output_file: str | None,
) -> None:
    """Dock a rigid ligand into a receptor and report ranked poses.

    Lower score is better throughout.  Every pose reports its per-term energy
    breakdown so the ranking can be audited rather than trusted.

    Exit code 1 is returned when a DCK validation rule fails.
    """
    import numpy as np

    from bioeng_qpcr.docking import (
        SCORE_INTERPRETATION,
        DockingBox,
        identify_receptor_ligand,
        parse_pdb,
        summarize_docking,
        validate_docking,
        write_pdb,
    )
    from bioeng_qpcr.docking.engine import dock as run_dock

    if split_from_receptor:
        if ligand_path:
            _fail("Error: give either a ligand file or --split-from-receptor.")
        try:
            receptor, ligand = identify_receptor_ligand(parse_pdb(receptor_path))
        except ValueError as exc:
            _fail(f"Error: {exc}")
    else:
        if not ligand_path:
            _fail("Error: a ligand PDB is required (or use --split-from-receptor).")
        try:
            receptor = parse_pdb(receptor_path)
            ligand = parse_pdb(ligand_path)
        except (ValueError, OSError) as exc:
            _fail(f"Error: {exc}")

    box = None
    if box_center is not None or box_size is not None:
        if box_center is None or box_size is None:
            _fail("Error: --box-center and --box-size must be given together.")
        try:
            centre = np.array(
                [float(part) for part in box_center.split(",")], dtype=float
            )
        except ValueError:
            _fail("Error: --box-center expects three comma-separated numbers.")
        if centre.size != 3:
            _fail("Error: --box-center expects exactly three numbers.")
        box = DockingBox(center=centre, half_size=np.full(3, float(box_size)))

    try:
        result = run_dock(
            receptor,
            ligand,
            box,
            n_runs=n_runs,
            seed=seed,
            n_steps=n_steps,
            use_grid=use_grid,
            grid_translation_steps=grid_translation_steps,
            grid_rotation_samples=grid_rotation_samples,
            top_n=top_n,
            rmsd_threshold=rmsd_threshold,
        )
    except ValueError as exc:
        _fail(f"Error: {exc}")

    checks = [] if no_validation else validate_docking(result, receptor, ligand)
    review = summarize_docking(result, checks)

    if output_format == "json":
        import json as _json

        poses = []
        for index, pose in enumerate(result.poses):
            deviation = 0.0 if index == 0 else pose.rmsd_to(result.poses[0])
            poses.append(
                {
                    "rank": index + 1,
                    "score": pose.score,
                    "vdw": pose.breakdown.vdw,
                    "electrostatic": pose.breakdown.electrostatic,
                    "hbond": pose.breakdown.hbond,
                    "hydrophobic": pose.breakdown.hydrophobic,
                    "desolvation": pose.breakdown.desolvation,
                    "rmsd_to_best": deviation,
                    "translation": pose.transform.translation.tolist(),
                }
            )

        payload = {
            "source_file": str(receptor_path),
            "ligand_file": str(ligand_path) if ligand_path else None,
            "scoring": result.scoring_name,
            "score_interpretation": SCORE_INTERPRETATION,
            "weights": list(result.weights),
            "receptor": {
                "n_atoms": result.receptor_meta.n_atoms,
                "n_residues": result.receptor_meta.n_residues,
            },
            "ligand": {
                "n_atoms": result.ligand_meta.n_atoms,
                "elements": sorted({atom.element for atom in ligand.atoms}),
                "residue_names": list(result.ligand_meta.residue_names),
            },
            "box": {
                "center": result.box.center.tolist(),
                "half_size": result.box.half_size.tolist(),
            },
            "stats": asdict(result.stats),
            "review": review,
            "poses": poses,
            "validation": [
                {
                    "rule_id": check.rule_id,
                    "passed": check.passed,
                    "severity": getattr(check.severity, "value", check.severity),
                    "message": check.message,
                    "evidence": check.evidence,
                }
                for check in checks
            ],
        }
        _emit(_json.dumps(payload, indent=2, ensure_ascii=False), output_file)
    else:
        lines = ["分子对接结果", "=" * 56]
        lines.append(f"受体: {receptor_path}  ({result.receptor_meta.n_atoms} 原子)")
        if ligand_path:
            lines.append(f"配体: {ligand_path}  ({result.ligand_meta.n_atoms} 原子)")
        else:
            lines.append(f"配体: 从受体文件提取  ({result.ligand_meta.n_atoms} 原子)")
        lines.append(f"打分函数: {result.scoring_name}")
        lines.append(
            "搜索盒中心: "
            + ", ".join(f"{value:.2f}" for value in result.box.center)
            + "   半径: "
            + f"{float(result.box.half_size[0]):.2f} A"
        )
        lines.append(
            f"采样: {result.stats.iterations} 步, "
            f"接受 {result.stats.accepted} 次 "
            f"({result.stats.acceptance_rate:.1%}), "
            f"seed={result.stats.seed}"
        )
        lines.append(
            f"搜索诊断: {review['search_label']}  {result.stats.message}"
        )
        lines.append(f"使用判断: {review['summary']}")
        lines.extend(f"复核建议: {item}" for item in review["next_steps"])
        lines.append("")
        lines.append("排名   总分     vdW     静电    氢键    疏水  去溶剂   RMSD")
        lines.append("-" * 76)
        for index, pose in enumerate(result.poses):
            deviation = 0.0 if index == 0 else pose.rmsd_to(result.poses[0])
            lines.append(
                f"{index + 1:>3}   {pose.score:9.3f} "
                f"{pose.breakdown.vdw:8.3f} {pose.breakdown.electrostatic:8.3f} "
                f"{pose.breakdown.hbond:8.3f} {pose.breakdown.hydrophobic:8.3f} "
                f"{pose.breakdown.desolvation:8.3f} {deviation:7.2f}"
            )
        lines.append("")
        lines.append(SCORE_INTERPRETATION)
        for check in checks:
            if check.passed:
                continue
            tag = (
                "[错误]"
                if str(getattr(check.severity, "value", check.severity)) == "error"
                else "[警告]"
                if str(getattr(check.severity, "value", check.severity)) == "warning"
                else "[提示]"
            )
            lines.append(f"{tag} {check.rule_id} {check.message}")
        _emit("\n".join(lines), output_file)

    if write_pose and result.poses:
        from dataclasses import replace as _replace

        pose = result.poses[0]
        original = np.array([atom.xyz for atom in ligand.atoms], dtype=float)
        posed_atoms = tuple(
            _replace(atom, xyz=coordinate)
            for atom, coordinate in zip(
                ligand.atoms, pose.transform.apply(original), strict=True
            )
        )
        write_pdb(_replace(ligand, atoms=posed_atoms), write_pose)
        click.echo(f"Wrote top pose: {write_pose}")

    failed = any(
        str(getattr(check.severity, "value", check.severity)) == "error"
        and not check.passed
        for check in checks
    )
    if failed:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


def _read_first_sequence(path: str, index: int) -> tuple[str, list[int] | None]:
    """Read one record from a sequence file, returning its bases and qualities."""
    from bioeng_qpcr.sequence import read_sequence_file

    try:
        records = read_sequence_file(path)
    except (ValueError, OSError) as exc:
        _fail(f"Error: {exc}")
    if not records:
        _fail(f"Error: no sequences found in {path}.")
    if not 0 <= index < len(records):
        _fail(
            f"Error: {path} holds {len(records)} record(s); index {index} is "
            "out of range."
        )
    record = records[index]
    return record.sequence, list(record.quality) if record.quality else None


@cli.command()
@click.argument(
    "query_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.argument(
    "reference_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.option(
    "--mode",
    type=click.Choice(["glocal", "global", "local"]),
    default="glocal",
    show_default=True,
    help="glocal maps a read onto a template; global/local are classic alignments.",
)
@click.option(
    "--matrix",
    type=click.Choice(["dna", "blosum62"]),
    default="dna",
    show_default=True,
    help="Substitution matrix.",
)
@click.option("--gap-open", default=5, show_default=True)
@click.option("--gap-extend", default=2, show_default=True)
@click.option("--read-index", default=0, show_default=True, help="Query record index.")
@click.option("--reference-index", default=0, show_default=True)
@click.option(
    "--primer-5",
    default=None,
    help="5' primer to excise from the read before aligning.",
)
@click.option(
    "--primer-3",
    default=None,
    help="3' primer to excise from the read before aligning.",
)
@click.option(
    "--primer-mismatch",
    default=2,
    show_default=True,
    help="Allowed mismatches when locating a primer.",
)
@click.option(
    "--trim-quality",
    is_flag=True,
    help="Trim low-quality tails (requires FASTQ input).",
)
@click.option(
    "--min-quality",
    "min_q",
    default=20,
    show_default=True,
    help="Mean Phred threshold for the quality-trimming window.",
)
@click.option(
    "--quality-window",
    "quality_window",
    default=10,
    show_default=True,
    help="Window length for quality trimming.",
)
@click.option(
    "--variants",
    is_flag=True,
    help="Call and list variants against the reference.",
)
@click.option(
    "--coding",
    is_flag=True,
    help="Treat the amplicon as coding (enables the frameshift rule).",
)
@click.option(
    "--min-identity",
    default=0.9,
    show_default=True,
    help="Identity threshold for the SEQ-001 rule.",
)
@click.option(
    "--consensus",
    is_flag=True,
    help="Build a consensus from all query records instead of aligning one.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output-file",
    default=None,
    help="Write the report to a file instead of stdout.",
)
def align(
    query_path: str,
    reference_path: str,
    mode: str,
    matrix: str,
    gap_open: int,
    gap_extend: int,
    read_index: int,
    reference_index: int,
    primer_5: str | None,
    primer_3: str | None,
    primer_mismatch: int,
    trim_quality: bool,
    coding: bool,
    min_identity: float,
    variants: bool,
    consensus: bool,
    min_q: int,
    quality_window: int,
    output_format: str,
    output_file: str | None,
) -> None:
    """Align a sequencing read to a reference and audit the result.

    The default ``glocal`` mode treats the query as a sub-region of the
    reference, which is what a Sanger read against an amplicon actually is.
    Primers and low-quality tails are trimmed before aligning, and every
    reported coordinate is in reference space.

    Exit code 1 is returned when a SEQ rule reports an error.
    """
    from bioeng_qpcr.sequence import (
        BLOSUM62,
        build_consensus,
        compare_to_reference,
        detect_variants,
        dna_matrix,
        needleman_wunsch_affine,
        read_sequence_file,
        smith_waterman_affine,
        summarize_sequencing_quality,
        trim_low_quality,
        trim_primers,
        validate_sequencing_result,
    )

    substitution = dna_matrix() if matrix == "dna" else BLOSUM62

    query, quality = _read_first_sequence(query_path, read_index)
    reference, _unused = _read_first_sequence(reference_path, reference_index)

    notes: list[str] = []
    primer_result = None
    if primer_5 or primer_3:
        try:
            primer_result = trim_primers(
                query,
                primer_5=primer_5,
                primer_3=primer_3,
                max_mismatch=primer_mismatch,
            )
        except ValueError as exc:
            _fail(f"Error: {exc}")
        query = primer_result.trimmed
        notes.append(
            f"引物裁剪: 5' 去掉 {primer_result.n_trimmed_5} 个碱基, "
            f"3' 去掉 {primer_result.n_trimmed_3} 个"
        )
        if quality is not None:
            quality = quality[
                primer_result.n_trimmed_5 : len(quality) - primer_result.n_trimmed_3
            ]

    if trim_quality:
        if quality is None:
            _fail("Error: --trim-quality needs quality values (FASTQ input).")
        quality_trim = trim_low_quality(
            query, quality, window=quality_window, min_q=min_q
        )
        query = quality_trim.trimmed
        quality = quality[quality_trim.kept_start : quality_trim.kept_end]
        notes.append(
            f"质量裁剪: 保留 {quality_trim.kept_start}..{quality_trim.kept_end} "
            f"(窗口 {quality_window}, 阈值 Q{min_q})"
        )

    if consensus:
        records = read_sequence_file(query_path)
        consensus_result = build_consensus([r.sequence for r in records], reference)
        if output_format == "json":
            import json as _json

            _emit(
                _json.dumps(
                    {
                        "consensus": consensus_result.consensus,
                        "n_ambiguous": consensus_result.n_ambiguous,
                        "method": consensus_result.method,
                        "threshold": consensus_result.threshold,
                        "depth": consensus_result.depth,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                output_file,
            )
        else:
            lines = ["共识序列", "=" * 56]
            lines.append(f"读长数: {len(records)}  方法: {consensus_result.method}")
            lines.append(f"模糊位点: {consensus_result.n_ambiguous}")
            lines.append("")
            lines.append(consensus_result.consensus)
            _emit("\n".join(lines), output_file)
        return

    if mode == "global":
        result = needleman_wunsch_affine(
            query, reference, substitution, gap_open=gap_open, gap_extend=gap_extend
        )
    elif mode == "local":
        result = smith_waterman_affine(
            query, reference, substitution, gap_open=gap_open, gap_extend=gap_extend
        )
    else:
        result = compare_to_reference(
            query, reference, substitution, gap_open=gap_open, gap_extend=gap_extend
        )

    variants_found = (
        detect_variants(result, reference, query_quality=quality) if variants else []
    )

    checks = validate_sequencing_result(
        result,
        query,
        reference,
        variants_found,
        quality=quality,
        coding=coding,
        primer_trim=primer_result,
        primer_5=primer_5,
        primer_3=primer_3,
        threshold_identity=min_identity,
        min_q=min_q,
    )

    if output_format == "json":
        import json as _json

        payload = {
            "query_file": str(query_path),
            "reference_file": str(reference_path),
            "mode": mode,
            "matrix": matrix,
            "query_sha256": _sha256(query_path),
            "reference_sha256": _sha256(reference_path),
            "notes": notes,
            "alignment": {
                "score": result.score,
                "identity": result.identity,
                "similarity": result.similarity,
                "gaps": result.gaps,
                "gap_opens": result.gap_opens,
                "length": result.length,
                "query_start": result.query_start,
                "query_end": result.query_end,
                "ref_start": result.ref_start,
                "ref_end": result.ref_end,
                "cigar": result.cigar,
                "aligned_query": result.aligned_query,
                "aligned_reference": result.aligned_reference,
                "query_orientation": result.query_orientation,
            },
            "variants": [
                {
                    "type": variant.type.value,
                    "ref_position": variant.ref_position,
                    "ref_allele": variant.ref_allele,
                    "alt_allele": variant.alt_allele,
                    "length": variant.length,
                    "quality": variant.quality,
                    "in_low_complexity": variant.in_low_complexity,
                }
                for variant in variants_found
            ],
            "validation": [
                {
                    "rule_id": check.rule_id,
                    "passed": check.passed,
                    "severity": check.severity.value,
                    "message": check.message,
                }
                for check in checks.results
            ],
            "passed": checks.passed,
        }
        if quality is not None:
            summary = summarize_sequencing_quality(
                read_sequence_file(query_path), min_q=min_q
            )
            payload["read_quality"] = asdict(summary)
        _emit(_json.dumps(payload, indent=2, ensure_ascii=False), output_file)
    else:
        lines = ["测序比对结果", "=" * 56]
        lines.append(f"读长文件: {query_path}      参考: {reference_path}")
        lines.append(f"模式: {mode}   矩阵: {matrix}   gap: {gap_open}/{gap_extend}")
        lines.append(
            "读长方向: "
            + (
                "反向互补"
                if result.query_orientation == "reverse_complement"
                else "正向"
            )
        )
        for note in notes:
            lines.append(note)
        lines.append("")
        lines.append(
            f"得分: {result.score:.1f}   一致性: {result.identity:.2%}   "
            f"相似性: {result.similarity:.2%}"
        )
        lines.append(
            f"比对长度: {result.length}   gap 数: {result.gaps} "
            f"(开启 {result.gap_opens})"
        )
        lines.append(
            f"参考坐标: {result.ref_start}..{result.ref_end}   "
            f"读长坐标: {result.query_start}..{result.query_end}"
        )
        lines.append(f"CIGAR: {result.cigar}")
        lines.append("")
        lines.append("比对（每 60 列一行）")
        lines.append("-" * 56)
        lines.extend(_alignment_block(result.aligned_query, result.aligned_reference))
        if variants:
            lines.append("")
            lines.append(f"变异（{len(variants_found)}）")
            lines.append("-" * 56)
            if not variants_found:
                lines.append("  未检出变异。")
            for variant in variants_found:
                lines.append(
                    f"  {variant.type.value:<10} 参考位置 {variant.ref_position:<6} "
                    f"{variant.ref_allele or '-'} → {variant.alt_allele or '-'}"
                    + (f"   Q={variant.quality}" if variant.quality else "")
                )
        lines.append("")
        lines.append("校验")
        lines.append("-" * 56)
        for check in checks.results:
            status = "通过" if check.passed else "未通过"
            lines.append(
                f"  {check.rule_id} ({check.severity.value}): {status} — "
                f"{check.message}"
            )
        lines.append("")
        lines.append(f"总体: {'通过' if checks.passed else '存在未通过项'}")
        _emit("\n".join(lines), output_file)

    if not checks.passed:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
