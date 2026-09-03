"""Command-line interface entry point for bioeng-qpcr.

This module provides a durable, non-Jupyter command-line interface to the
bioeng-qpcr engine.  It wraps the IO, validation, calculations, provenance
and plotting layers behind four sub-commands:

* ``validate``   -- run a MIQE readiness assessment on a tidy CSV.
* ``analyze``    -- run relative-quantification analysis (ΔΔCq / Pfaffl).
* ``provenance`` -- emit a JSON provenance manifest for a dataset.
* ``plot``       -- render fold-change bar plots or Cq heatmaps.

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
    compare_groups,
    delta_delta_cq_batch,
    normalized_quantity_all,
    pfaffl_batch,
)
from bioeng_qpcr.io import CSVValidationError, read_tidy_csv
from bioeng_qpcr.provenance import ProvenanceManifest
from bioeng_qpcr.reporting import build_research_report
from bioeng_qpcr.validation import RuleExecutionError, ValidationEngine

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


def _load_experiment(
    csv_path: str,
    reference_targets: list[str] | None = None,
    *,
    strict: bool = True,
):
    """Read a tidy CSV into an :class:`~bioeng_qpcr.domain.Experiment`.

    On :class:`FileNotFoundError` or :class:`ValueError` (e.g. missing
    required columns) an error message is printed and the process exits
    with code 1.
    """
    try:
        return read_tidy_csv(
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


def _dicts_to_csv(dicts: list[dict[str, Any]]) -> str:
    """Serialise a list of homogeneous dictionaries to CSV text.

    Returns an empty string when *dicts* is empty (no rows to write).
    """
    if not dicts:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(dicts[0].keys()))
    writer.writeheader()
    writer.writerows(dicts)
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
    """Run a MIQE readiness assessment on a tidy CSV file.

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

    if output_format == "json":
        content = report.to_json(indent=2)
    else:
        content = report.to_text()

    _emit(content, output_file)

    ctx = click.get_current_context()
    ctx.exit(1 if report.failures else 0)


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
        content = _dicts_to_csv(dicts)
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
@click.option("--lenient", is_flag=True)
def report(
    csv_path: str,
    target: str,
    reference_targets: list[str] | None,
    output_file: str,
    json_file: str | None,
    lenient: bool,
) -> None:
    """Generate a guided HTML readiness, analysis, and provenance report."""
    experiment = _load_experiment(
        csv_path,
        reference_targets=reference_targets,
        strict=not lenient,
    )
    try:
        researcher_report = build_research_report(experiment, target, reference_targets)
    except (RuleExecutionError, ValueError) as exc:
        _fail(f"Error: {exc}")
    Path(output_file).write_text(researcher_report.html, encoding="utf-8")
    if json_file:
        Path(json_file).write_text(researcher_report.to_json() + "\n", encoding="utf-8")
    click.echo(f"Research report saved to {output_file}")


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
    import matplotlib.pyplot as plt  # noqa: F401  (ensures pyplot is loaded)

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
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
