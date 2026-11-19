"""Server-side web application for the bioeng-qpcr analysis engine.

This module exposes the auditable qPCR analysis pipeline as a small,
self-contained local web application.  It reuses the same validated IO,
validation, calculation and plotting layers as the CLI and the MCP server,
so results are byte-for-byte consistent across every surface.

Workflows exposed:
    * **Validate**  -- MIQE readiness assessment plus a Cq heatmap.
    * **Analyze**   -- relative-quantification (normalized quantity, ΔΔCq,
      Pfaffl or group comparison) with fold-change and heatmap plots.
    * **Curves**    -- per-well amplification-curve fitting (baseline
      correction, sigmoid Cq, per-well efficiency) with an amplification
      plot.

The application is deliberately synchronous and Python-only: the web layer
owns the HTTP/HTML concerns while all science stays in the reusable
:mod:`~bioeng_qpcr.calculations` and :mod:`~bioeng_qpcr.io` packages.  All
plotting is rendered to in-memory PNG and returned as ``data:`` URIs so the
browser needs no image-serving round trip.

Run locally with::

    python -m bioeng_qpcr.web  # or ``bioeng-qpcr-web``
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file

from bioeng_qpcr.io import (
    NotAmplificationFile,
    read_amplification_csv,
    read_rdml,
    read_tidy_csv,
)
from bioeng_qpcr.validation import ValidationEngine

__all__ = ["create_app"]

#: Looked at before treating a file as a plain tidy/long-format CSV.
_RDML_SUFFIXES = frozenset({".rdml", ".rdm", ".xml"})

#: File extensions we accept as uploads.
_ALLOWED_SUFFIXES = frozenset({".csv", ".tsv", ".rdml", ".rdm", ".xml", ".txt"})

#: Extensions accepted by the workspace preview endpoints.
_TABLE_SUFFIXES = frozenset({".csv", ".tsv", ".txt"})

_WORKSPACE_EXAMPLES = {
    # DEMO: CSVs are the original teaching examples; FBA is COBRApy 0.31.1's
    # public textbook model. Sources and hashes: web/examples/KINETICS_SOURCES.md.
    # REAL DATA: replace via the module's file upload, never edit the demo files.
    "enzyme": {"file": "inhibition_example.csv"},
    "ferment": {"file": "fermentation_example.csv"},
    "fba": {"file": "e_coli_core.xml"},
    "preview_image": {"file": "image_preview_example.ppm"},
    "potency": {"file": "dose_response_example.csv"},
    "docking": {
        "receptor": "docking_receptor.pdb",
        "ligand": "docking_ligand.pdb",
    },
    "sequence": {
        "query": "sequencing_reads.fastq",
        "reference": "sequencing_reference.fasta",
    },
}
_EXAMPLES_DIR = Path(__file__).with_name("examples")

#: Image extensions accepted by the workspace preview endpoint.  The list is
#: deliberately generous: the inspector reads the container from the bytes, so
#: an extension mismatch is reported rather than silently mis-parsed.
_IMAGE_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
        ".ico",
        ".psd",
        ".tga",
        ".ppm",
        ".svg",
    }
)

#: Sequence-file extensions accepted by the alignment endpoint.
_SEQUENCE_SUFFIXES = frozenset(
    {".ab1", ".abi", ".fasta", ".fa", ".fna", ".fastq", ".fq", ".seq", ".txt"}
)

#: PDB extensions accepted by the docking endpoint.
_PDB_SUFFIXES = frozenset({".pdb", ".ent"})

_MAX_POTENCY_BOOTSTRAPS = 5000
_MAX_POTENCY_BOOTSTRAP_FITS = 5000
_MAX_DOCKING_RUNS = 50
_MAX_DOCKING_STEPS = 20_000
_MAX_DOCKING_EVALUATIONS = 200_000
_MAX_DOCKING_RECEPTOR_ATOMS = 100_000
_MAX_DOCKING_LIGAND_ATOMS = 10_000
_MAX_DOCKING_ATOM_PAIRS = 1_000_000
_MAX_DOCKING_PAIR_EVALUATIONS = 200_000_000
_DEFAULT_GRID_EVALUATIONS = 3**3 * (1 + 3 * 6)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_reference_targets(raw: str | None) -> list[str] | None:
    """Split a comma-separated reference-gene string into a list (or ``None``)."""
    if not raw:
        return None
    targets = [v.strip() for v in raw.split(",") if v.strip()]
    return targets or None


def _bounded_form_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read one integer form field and enforce its public HTTP work budget."""
    raw = request.form.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value}."
        )
    return value


def _save_upload(
    allowed: frozenset[str] = _ALLOWED_SUFFIXES, field: str = "file"
) -> str:
    """Persist an uploaded file to a temp location and return its path.

    Parameters
        allowed:
            Suffixes the caller is willing to handle.  Each endpoint passes its
            own set, so a PNG cannot reach the CSV reader by accident.
        field:
            Name of the multipart field carrying the file.
    """
    if field not in request.files:
        raise ValueError("No file was uploaded.")
    upload = request.files[field]
    if not upload or not upload.filename:
        raise ValueError("No file was uploaded.")

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in allowed:
        raise ValueError(
            f"Unsupported file type '{suffix or '<none>'}'. Accepted: "
            + ", ".join(sorted(allowed))
        )

    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        upload.save(handle.name)
    finally:
        handle.close()
    return handle.name


def _fig_to_data_uri(fig: Any) -> str:
    """Render a matplotlib figure to a base64 ``data:image/png`` URI."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _publication_plot(title: str, fig: Any, stem: str) -> dict[str, Any]:
    """Return a preview plus manuscript-ready vector and raster downloads."""
    import matplotlib.pyplot as plt

    safe_stem = (
        "-".join(
            part
            for part in "".join(
                char.lower() if char.isascii() and char.isalnum() else "-"
                for char in stem
            ).split("-")
            if part
        )
        or "figure"
    )
    width = 7.2
    current_width, current_height = fig.get_size_inches()
    height = min(max(width * current_height / current_width, 2.4), 8.0)
    fig.set_size_inches(width, height, forward=True)
    fig.patch.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")

    downloads = []
    formats = (
        ("svg", "image/svg+xml", "SVG 矢量", None),
        ("pdf", "application/pdf", "PDF 矢量", None),
        ("png", "image/png", "PNG 300 dpi", 300),
    )
    try:
        preview = io.BytesIO()
        fig.savefig(preview, format="png", bbox_inches="tight", dpi=110)
        preview_uri = "data:image/png;base64," + base64.b64encode(
            preview.getvalue()
        ).decode("ascii")
        for fmt, mime, label, dpi in formats:
            metadata = {
                "svg": {
                    "Title": title,
                    "Description": "Publication figure exported by bioeng-qpcr",
                    "Creator": "bioeng-qpcr",
                },
                "pdf": {
                    "Title": title,
                    "Subject": "Publication figure exported by bioeng-qpcr",
                    "Creator": "bioeng-qpcr",
                },
                "png": {
                    "Title": title,
                    "Description": "Publication figure exported by bioeng-qpcr",
                    "Software": "bioeng-qpcr",
                },
            }[fmt]
            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format=fmt,
                bbox_inches="tight",
                dpi=dpi or fig.dpi,
                facecolor="white",
                metadata=metadata,
            )
            downloads.append(
                {
                    "format": fmt,
                    "label": label,
                    "filename": f"{safe_stem}.{fmt}",
                    "data_uri": (
                        f"data:{mime};base64,"
                        + base64.b64encode(buffer.getvalue()).decode("ascii")
                    ),
                }
            )
    finally:
        plt.close(fig)
    return {
        "title": title,
        "data_uri": preview_uri,
        "downloads": downloads,
        "publication": {"width_in": width, "height_in": height, "raster_dpi": 300},
    }


def _plot_payload(
    title: str, fig: Any, stem: str, *, publication: bool
) -> dict[str, Any]:
    if publication:
        return _publication_plot(title, fig, stem)
    return {"title": title, "data_uri": _fig_to_data_uri(fig), "downloads": []}


def _results_to_dicts(results: list[Any]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results if is_dataclass(r)]


def _reader_for(path: str):
    """Pick the correct IO reader for a path by its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in _RDML_SUFFIXES:
        return read_rdml
    return read_tidy_csv


def _load_experiment(
    path: str, reference_targets: list[str] | None = None, *, strict: bool = True
):
    """Read a tidy CSV or RDML file into an ``Experiment``."""
    return _reader_for(path)(path, reference_targets=reference_targets, strict=strict)


def _load_curve_experiment(path: str, reference_targets: list[str] | None = None):
    """Load an experiment that carries raw amplification traces.

    A long-format trace CSV is preferred; files that structurally are not
    trace CSVs (and missing files) route through :func:`_load_experiment`.
    A file that *looks* like a trace CSV but is malformed raises instead of
    being silently reinterpreted.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _RDML_SUFFIXES:
        try:
            return read_amplification_csv(path, reference_targets=reference_targets)
        except (NotAmplificationFile, FileNotFoundError):
            pass
    return _load_experiment(path, reference_targets)


def _message(message: str, status: int, details: Any = None):
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status


def _ensure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Report serialisation helpers
# ---------------------------------------------------------------------------


def _validation_payload(report) -> dict[str, Any]:
    return {
        "ok": not report.failures,
        "n_rules": len(report.results),
        "n_failures": len(report.failures),
        "n_warnings": len(report.warnings),
        "failures": [
            {
                "rule": r.rule_id,
                "message": (r.message or r.passes),
            }
            for r in report.failures
        ],
        "text": report.to_text(),
        "json": json.loads(report.to_json()),
    }


def _analysis_payload(batch, *, method: str) -> dict[str, Any]:
    return {
        "method": method,
        "results": _results_to_dicts(batch.results),
        "warnings": list(batch.warnings),
        "exclusions": [
            {"sample_name": e.sample_name, "reason": e.reason} for e in batch.exclusions
        ],
        "parameters": getattr(batch, "parameters", {}),
    }


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Build a configured :class:`flask.Flask` application."""
    template_dir = Path(__file__).parent / "templates"
    app = Flask("bioeng_qpcr", template_folder=str(template_dir))
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1"]

    @app.before_request
    def local_writes():
        origin = request.headers.get("Origin")
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin
            and origin != request.host_url.rstrip("/")
        ):
            return jsonify(ok=False, error="只接受当前本地工作台的修改请求"), 403
        return None

    from bioeng_qpcr.web.workbench import blueprint as workbench_blueprint

    app.register_blueprint(workbench_blueprint)

    from bioeng_qpcr.web.lab import blueprint as lab_blueprint

    app.register_blueprint(lab_blueprint)

    from bioeng_qpcr.web.records import blueprint as records_blueprint

    app.register_blueprint(records_blueprint)

    from bioeng_qpcr.web.potency_review import blueprint as potency_review_blueprint

    app.register_blueprint(potency_review_blueprint)

    from bioeng_qpcr.web.kinetics import blueprint as kinetics_blueprint

    app.register_blueprint(kinetics_blueprint)

    @app.get("/")
    def index():
        return app.jinja_env.get_template("index.html").render()

    @app.get("/api/health")
    def health():
        from bioeng_qpcr import __version__

        return jsonify({"status": "ok", "version": __version__})

    @app.get("/api/workspace/examples/<module_key>")
    def api_workspace_example(module_key: str):
        """Return a small, whitelisted example set for one workspace module."""
        example = _WORKSPACE_EXAMPLES.get(module_key)
        if example is None:
            return _message("No built-in example is available for this module.", 404)
        files = []
        for role, name in example.items():
            path = _EXAMPLES_DIR / name
            if not path.is_file():
                return _message(f"Built-in example is missing: {name}", 500)
            files.append(
                {
                    "role": role,
                    "name": name,
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            )
        note = {
            "enzyme": "已载入项目教学示例（非实测速率），请上传 CSV 替换真实数据。",
            "ferment": "已载入项目教学示例（非实测发酵），请上传 CSV 替换真实数据。",
            "fba": "已载入 COBRApy 的公开 E. coli core 模型；上传模型可替换研究对象。",
        }.get(module_key)
        return jsonify({"ok": True, "files": files, "note": note})

    @app.get("/api/workspace/examples/<module_key>/download")
    def download_workspace_example(module_key):
        example = _WORKSPACE_EXAMPLES.get(module_key)
        if not example:
            return _message("未知示例", 404)
        # A multi-file module exports a ZIP so its paired inputs stay together.
        if len(example) > 1:
            import zipfile

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in example.values():
                    archive.write(_EXAMPLES_DIR / name, name)
            buffer.seek(0)
            return send_file(
                buffer, as_attachment=True, download_name=f"{module_key}-example.zip"
            )
        name = next(iter(example.values()))
        return send_file(_EXAMPLES_DIR / name, as_attachment=True, download_name=name)

    @app.post("/api/validate")
    def api_validate():
        _ensure_matplotlib()
        path: str | None = None
        try:
            path = _save_upload()
            experiment = _load_experiment(path)
            report = ValidationEngine().run(experiment)
            from bioeng_qpcr.plotting import plot_cq_heatmap

            heatmap = ""
            try:
                fig, _ax = plot_cq_heatmap(experiment)
                heatmap = _fig_to_data_uri(fig)
            except ValueError:
                heatmap = ""
            payload = _validation_payload(report)
            payload["heatmap"] = heatmap
            payload["ok"] = True
            return jsonify(payload)
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            if path:
                _remove(path)

    @app.post("/api/analyze")
    def api_analyze():
        from bioeng_qpcr.calculations import (
            compare_groups,
            delta_delta_cq_batch,
            normalized_quantity_all,
            pfaffl_batch,
        )

        _ensure_matplotlib()
        path: str | None = None
        try:
            path = _save_upload()
            data = request.form
            method = data.get("method", "normalized_quantity")
            target = data.get("target") or ""
            control = data.get("control") or None
            reference_targets = _parse_reference_targets(data.get("reference_targets"))
            treated = data.get("treated_group") or None
            control_group = data.get("control_group") or None

            if method not in {
                "normalized_quantity",
                "delta_delta_cq",
                "pfaffl",
                "compare",
            }:
                raise ValueError(f"Unknown method '{method}'.")
            if not target:
                raise ValueError("A target gene is required.")

            experiment = _load_experiment(
                path, reference_targets=reference_targets, strict=True
            )

            if method == "compare":
                if not treated or not control_group:
                    raise ValueError("compare requires a treated and control group.")
                result, batch = compare_groups(
                    experiment, target, treated, control_group, reference_targets
                )
                batch.raise_for_exclusions()
            elif method == "normalized_quantity":
                batch = normalized_quantity_all(
                    experiment, target, reference_targets=reference_targets
                )
            else:
                if not control:
                    raise ValueError(f"Method '{method}' requires a control sample.")
                if reference_targets and len(reference_targets) > 1:
                    raise ValueError(f"Method '{method}' accepts one reference target.")
                ref = reference_targets[0] if reference_targets else None
                if method == "delta_delta_cq":
                    batch = delta_delta_cq_batch(
                        experiment, target, control, reference_target=ref
                    )
                else:
                    batch = pfaffl_batch(
                        experiment, target, control, reference_target=ref
                    )

            from bioeng_qpcr.plotting import plot_cq_heatmap, plot_fold_change

            heatmap = ""
            foldchange = ""
            try:
                fig, _ax = plot_cq_heatmap(experiment)
                heatmap = _fig_to_data_uri(fig)
            except ValueError:
                pass
            if method == "delta_delta_cq":
                try:
                    fig, _ax = plot_fold_change(batch.results)
                    foldchange = _fig_to_data_uri(fig)
                except ValueError:
                    pass

            payload = _analysis_payload(batch, method=method)
            payload.update({"ok": True, "heatmap": heatmap, "fold_change": foldchange})
            return jsonify(payload)
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            if path:
                _remove(path)

    @app.post("/api/curves")
    def api_curves():
        from bioeng_qpcr.calculations import analyze_amplification_curves

        _ensure_matplotlib()
        path: str | None = None
        try:
            path = _save_upload()
            baseline_method = request.form.get("baseline_method", "constant")
            if baseline_method not in {"constant", "linear"}:
                raise ValueError("baseline_method must be 'constant' or 'linear'.")
            reference_targets = _parse_reference_targets(
                request.form.get("reference_targets")
            )

            experiment = _load_curve_experiment(path, reference_targets)
            if not experiment.has_traces:
                raise ValueError("No raw amplification traces were found in the input.")

            result = analyze_amplification_curves(
                experiment, baseline_method=baseline_method
            )

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
                    "target": s.target,
                    "median_efficiency": s.median,
                    "n_wells": s.n_wells,
                    "min_efficiency": s.min,
                    "max_efficiency": s.max,
                }
                for s in result.per_assay.values()
            ]

            plot = ""
            try:
                from bioeng_qpcr.plotting import plot_amplification_curves

                fig, _ax = plot_amplification_curves(experiment)
                plot = _fig_to_data_uri(fig)
            except ValueError:
                pass

            return jsonify(
                {
                    "ok": True,
                    "per_well": rows,
                    "per_assay": assay_rows,
                    "wells_without_trace": list(result.wells_without_trace),
                    "traces_without_well": list(result.traces_without_well),
                    "non_amplifying": list(result.non_amplifying),
                    "warnings": list(result.warnings),
                    "plot": plot,
                }
            )
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            if path:
                _remove(path)

    # ------------------------------------------------------------------
    # Workspace: preview, formats and the Phase 5 analysis modules
    # ------------------------------------------------------------------

    @app.get("/workspace")
    def workspace():
        return app.jinja_env.get_template("workspace.html").render()

    @app.get("/api/schemas")
    def api_schemas():
        from bioeng_qpcr.io import list_schemas

        return jsonify(
            {
                "ok": True,
                "schemas": [
                    {
                        "key": schema.key,
                        "title": schema.title,
                        "summary": schema.summary,
                        "layout": schema.layout,
                        "min_rows": schema.min_rows,
                        "header": list(schema.header()),
                        "notes": list(schema.notes),
                        "markdown": schema.to_markdown(),
                        "columns": [
                            {
                                "name": column.name,
                                "type": column.type,
                                "required": column.required,
                                "unit": column.unit,
                                "description": column.description,
                                "aliases": list(column.aliases),
                                "choices": list(column.choices),
                                "allow_missing": column.allow_missing,
                            }
                            for column in schema.columns
                        ],
                    }
                    for schema in list_schemas()
                ],
            }
        )

    @app.get("/api/schemas/<schema_key>/template")
    def api_schema_template(schema_key: str):
        """Download a template CSV for one registered format."""
        from flask import Response

        from bioeng_qpcr.io import schema_for, write_template_csv

        try:
            resolved = schema_for(schema_key)
        except KeyError as exc:
            return _message(str(exc), 404)

        handle = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix=f"{schema_key}_template_"
        )
        handle.close()
        try:
            write_template_csv(handle.name, resolved, overwrite=True)
            body = Path(handle.name).read_text(encoding=resolved.encoding)
        finally:
            _remove(handle.name)
        return Response(
            body,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{schema_key}_template.csv"'
                )
            },
        )

    @app.post("/api/preview/example")
    def api_preview_example():
        """Return a filled-in example table so the UI can show the format."""
        from bioeng_qpcr.io import schema_for

        key = request.form.get("format")
        if not key:
            return _message("A 'format' key is required.", 400)
        try:
            resolved = schema_for(key)
        except KeyError as exc:
            return _message(str(exc), 404)

        if not resolved.example_rows:
            return _message(f"Format {key!r} has no example rows.", 400)
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=resolved.delimiter, lineterminator="\n")
        writer.writerow(resolved.header())
        writer.writerows(resolved.example_rows)
        return jsonify(
            {
                "ok": True,
                "format": resolved.key,
                "title": resolved.title,
                "header": list(resolved.header()),
                "rows": [list(row) for row in resolved.example_rows],
                "notes": list(resolved.notes),
                "csv": buffer.getvalue(),
            }
        )

    @app.post("/api/preview")
    def api_preview():
        """Preview an uploaded CSV and report every structural problem."""
        from bioeng_qpcr.io import preview_csv

        path: str | None = None
        try:
            path = _save_upload(_TABLE_SUFFIXES)
            schema_key = request.form.get("format") or None
            if schema_key in (None, "", "auto"):
                schema_key = None
            preview = preview_csv(
                path,
                schema=schema_key,
                preview_rows=int(request.form.get("scan_rows", 200)),
                display_rows=int(request.form.get("display_rows", 20)),
            )
        except KeyError as exc:
            return _message(str(exc), 400)
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            if path:
                _remove(path)

        payload = _preview_payload(preview)
        payload["ok"] = True
        return jsonify(payload)

    @app.get("/api/image/formats")
    def api_image_formats():
        from bioeng_qpcr.io import list_supported_formats

        return jsonify({"ok": True, "formats": list_supported_formats()})

    @app.post("/api/image")
    def api_image():
        """Inspect an image, and optionally return a processed preview."""
        from bioeng_qpcr.io import inspect_image, load_preview

        path: str | None = None
        payload: dict[str, Any] = {"ok": True, "rendered": False}
        try:
            path = _save_upload(_IMAGE_SUFFIXES)
            info = inspect_image(path)
            payload["info"] = asdict(info)
            if request.form.get("render", "1") != "0" and not info.is_vector:
                original = load_preview(
                    path,
                    frame=int(request.form.get("frame", 0)),
                    invert=False,
                    auto_contrast=False,
                    gamma=1.0,
                    palette="gray",
                )
                preview = load_preview(
                    path,
                    frame=int(request.form.get("frame", 0)),
                    invert=request.form.get("invert") == "1",
                    auto_contrast=request.form.get("auto_contrast", "1") != "0",
                    gamma=float(request.form.get("gamma", 1.0)),
                    palette=request.form.get("palette") or "gray",
                )
                payload.update(
                    {
                        "rendered": True,
                        "data_uri": preview.data_uri,
                        "thumbnail_size": list(preview.thumbnail_size),
                        "palette": preview.palette,
                        "enhancements": list(preview.enhancements),
                        "warnings": list(preview.warnings),
                        "frame": preview.frame,
                        "plots": [
                            {"title": "原始像素预览", "data_uri": original.data_uri},
                            {"title": "处理后预览", "data_uri": preview.data_uri},
                        ],
                    }
                )
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            if path:
                _remove(path)
        return jsonify(payload)

    @app.post("/api/potency")
    def api_potency():
        """Fit dose-response curves and report IC50/EC50 with audit rules."""
        from bioeng_qpcr.io import read_dose_response_csv
        from bioeng_qpcr.potency import PotencyKind, fit_dose_response

        path: str | None = None
        results: list[dict[str, Any]] = []
        plot_inputs: list[tuple[Any, Any]] = []
        try:
            ci_method = request.form.get("ci_method", "asymptotic")
            n_bootstrap = _bounded_form_int(
                "n_bootstrap",
                500,
                minimum=1,
                maximum=_MAX_POTENCY_BOOTSTRAPS,
            )
            path = _save_upload(_TABLE_SUFFIXES)
            kind_raw = request.form.get("kind", "auto")
            kind = None if kind_raw == "auto" else PotencyKind(kind_raw)
            datasets = read_dose_response_csv(
                path,
                layout=request.form.get("layout", "auto"),
                normalize=request.form.get("normalize", "none"),
            )
            if (
                ci_method == "bootstrap"
                and n_bootstrap * len(datasets) > _MAX_POTENCY_BOOTSTRAP_FITS
            ):
                raise ValueError(
                    "Requested bootstrap workload exceeds the 5000-fit "
                    "HTTP budget; reduce n_bootstrap or the number of compounds."
                )
            run_validation = request.form.get("validate", "1") != "0"
            control_raw = (request.form.get("control_response") or "").strip()
            control_response = float(control_raw) if control_raw else None
            for dataset in datasets:
                try:
                    fitted = fit_dose_response(
                        dataset,
                        kind=kind,
                        ci_method=ci_method,
                        confidence=float(request.form.get("confidence", 0.95)),
                        n_bootstrap=n_bootstrap,
                        control_fraction=float(
                            request.form.get("control_fraction", 0.5)
                        ),
                        control_response=control_response,
                    )
                except ValueError as exc:
                    results.append({"compound": dataset.compound, "error": str(exc)})
                    continue
                results.append(
                    _potency_payload(fitted, dataset, run_validation=run_validation)
                )
                plot_inputs.append((fitted, dataset))
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            if path:
                _remove(path)

        plots: list[dict[str, Any]] = []
        plot_warnings = []
        try:
            _ensure_matplotlib()
            publication = request.form.get("publication") == "1"
            summary = _potency_summary_plot(
                [entry for entry in results if not entry.get("error")]
            )
            if summary:
                plots.append(
                    _plot_payload(
                        "效价比较",
                        summary,
                        "potency-comparison",
                        publication=publication,
                    )
                )
            from bioeng_qpcr.plotting import (
                plot_dose_response,
                plot_dose_response_residuals,
            )

            for fitted, dataset in plot_inputs[:3]:
                figure, _ax = plot_dose_response(fitted.best_fit, dataset)
                plots.append(
                    _plot_payload(
                        f"{dataset.compound} 剂量-反应曲线",
                        figure,
                        f"{dataset.compound}-dose-response",
                        publication=publication,
                    )
                )
                figure, _ax = plot_dose_response_residuals(fitted.best_fit, dataset)
                plots.append(
                    _plot_payload(
                        f"{dataset.compound} 残差诊断",
                        figure,
                        f"{dataset.compound}-residuals",
                        publication=publication,
                    )
                )
        except Exception:
            app.logger.exception("Potency figure generation failed")
            plot_warnings.append("效价图表未完整生成；数值结果已保留，请重试绘图并核对日志。")
        return jsonify(
            {
                "ok": True,
                "results": results,
                "plot_warnings": plot_warnings,
                "plot": plots[0]["data_uri"] if plots else "",
                "plots": plots,
            }
        )

    @app.post("/api/potency/compare")
    def api_potency_compare():
        """Relative potency between two compounds in the same file."""
        from bioeng_qpcr.io import read_dose_response_csv
        from bioeng_qpcr.potency import fit_dose_response, relative_potency

        path: str | None = None
        try:
            path = _save_upload(_TABLE_SUFFIXES)
            names = [
                name.strip()
                for name in (request.form.get("compounds") or "").split(",")
                if name.strip()
            ]
            if len(names) != 2:
                raise ValueError("Exactly two compound names are required.")
            datasets = {
                dataset.compound: dataset
                for dataset in read_dose_response_csv(
                    path, layout=request.form.get("layout", "auto")
                )
            }
            missing = [name for name in names if name not in datasets]
            if missing:
                raise ValueError(
                    "Compound(s) not present in the file: " + ", ".join(missing)
                )
            comparison = relative_potency(
                fit_dose_response(datasets[names[0]]),
                fit_dose_response(datasets[names[1]]),
            )
        except ValueError as exc:
            return _message(str(exc), 400)
        finally:
            if path:
                _remove(path)
        return jsonify(
            {
                "ok": True,
                "compound_a": comparison.compound_a,
                "compound_b": comparison.compound_b,
                "kind": comparison.kind.value,
                "ratio": comparison.ratio,
                "ci": list(comparison.ci) if comparison.ci else None,
                "fold_difference": comparison.fold_difference,
                "confidence": comparison.confidence,
                "message": comparison.message,
            }
        )

    @app.post("/api/dock")
    def api_dock():
        """Rigid-body docking with ranked poses and the DCK audit rules."""
        import numpy as np

        from bioeng_qpcr.docking import (
            DockingBox,
            identify_receptor_ligand,
            parse_pdb,
            summarize_docking,
            validate_docking,
        )
        from bioeng_qpcr.docking.engine import SCORE_INTERPRETATION, dock

        receptor_path: str | None = None
        ligand_path: str | None = None
        try:
            n_runs = _bounded_form_int(
                "n_runs", 4, minimum=1, maximum=_MAX_DOCKING_RUNS
            )
            n_steps = _bounded_form_int(
                "n_steps", 1500, minimum=50, maximum=_MAX_DOCKING_STEPS
            )
            top_n = _bounded_form_int("top_n", 5, minimum=1, maximum=100)
            monte_carlo_evaluations = n_runs * (n_steps + 1)
            if monte_carlo_evaluations > _MAX_DOCKING_EVALUATIONS:
                raise ValueError(
                    "n_runs * (n_steps + 1) exceeds the 200000-evaluation HTTP budget."
                )
            receptor_path = _save_upload(_PDB_SUFFIXES, field="receptor")
            if request.form.get("split_from_receptor") == "1":
                structure = parse_pdb(
                    receptor_path,
                    max_atoms=(_MAX_DOCKING_RECEPTOR_ATOMS + _MAX_DOCKING_LIGAND_ATOMS),
                )
                receptor, ligand = identify_receptor_ligand(structure)
            else:
                ligand_path = _save_upload(_PDB_SUFFIXES, field="ligand")
                receptor = parse_pdb(
                    receptor_path, max_atoms=_MAX_DOCKING_RECEPTOR_ATOMS
                )
                ligand = parse_pdb(ligand_path, max_atoms=_MAX_DOCKING_LIGAND_ATOMS)

            if receptor.n_atoms > _MAX_DOCKING_RECEPTOR_ATOMS:
                raise ValueError("Receptor exceeds the 100000-atom HTTP budget.")
            if ligand.n_atoms > _MAX_DOCKING_LIGAND_ATOMS:
                raise ValueError("Ligand exceeds the 10000-atom HTTP budget.")

            atom_pairs = receptor.n_atoms * ligand.n_atoms
            if atom_pairs > _MAX_DOCKING_ATOM_PAIRS:
                raise ValueError(
                    "Receptor-ligand atom pairs exceed the 1000000-pair HTTP "
                    "memory budget; prepare a binding-site receptor or smaller ligand."
                )
            use_grid = request.form.get("use_grid") == "1"
            search_evaluations = (
                _DEFAULT_GRID_EVALUATIONS if use_grid else monte_carlo_evaluations
            )
            if atom_pairs * search_evaluations > _MAX_DOCKING_PAIR_EVALUATIONS:
                raise ValueError(
                    "Docking atom-pair evaluations exceed the 200000000 HTTP "
                    "work budget; reduce the structure or search settings."
                )

            box = None
            center_raw = (request.form.get("box_center") or "").strip()
            size_raw = (request.form.get("box_size") or "").strip()
            if center_raw and size_raw:
                components = [float(part) for part in center_raw.split(",")]
                if len(components) != 3:
                    raise ValueError("Box centre needs exactly three numbers.")
                box = DockingBox(
                    center=np.array(components, dtype=float),
                    half_size=np.full(3, float(size_raw)),
                )

            result = dock(
                receptor,
                ligand,
                box,
                n_runs=n_runs,
                seed=int(request.form.get("seed", 1)),
                n_steps=n_steps,
                use_grid=use_grid,
                top_n=top_n,
                rmsd_threshold=float(request.form.get("rmsd_threshold", 2.0)),
            )
            checks = (
                validate_docking(result, receptor, ligand)
                if request.form.get("validate", "1") != "0"
                else []
            )
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            for temporary in (receptor_path, ligand_path):
                if temporary:
                    _remove(temporary)

        poses = []
        for index, pose in enumerate(result.poses):
            poses.append(
                {
                    "rank": index + 1,
                    "score": pose.score,
                    "vdw": pose.breakdown.vdw,
                    "electrostatic": pose.breakdown.electrostatic,
                    "hbond": pose.breakdown.hbond,
                    "hydrophobic": pose.breakdown.hydrophobic,
                    "desolvation": pose.breakdown.desolvation,
                    "rmsd_to_best": (
                        0.0 if index == 0 else pose.rmsd_to(result.poses[0])
                    ),
                    "translation": pose.transform.translation.tolist(),
                }
            )

        plots: list[dict[str, Any]] = []
        plot_warnings = []
        try:
            _ensure_matplotlib()
            publication = request.form.get("publication") == "1"
            from bioeng_qpcr.plotting import plot_pose_ranking, plot_score_terms

            figure, _ax = plot_pose_ranking(result.poses)
            plots.append(
                _plot_payload(
                    "姿势排名与评分分解",
                    figure,
                    "docking-pose-ranking",
                    publication=publication,
                )
            )
            if result.poses:
                figure, _ax = plot_score_terms(result.poses[0].breakdown)
                plots.append(
                    _plot_payload(
                        "最佳姿势评分项",
                        figure,
                        "docking-best-pose-terms",
                        publication=publication,
                    )
                )
        except Exception:
            app.logger.exception("Docking figure generation failed")
            plot_warnings.append("对接图表未完整生成；数值结果已保留，请重试绘图并核对日志。")

        return jsonify(
            {
                "ok": True,
                "scoring": result.scoring_name,
                "plot_warnings": plot_warnings,
                "score_interpretation": SCORE_INTERPRETATION,
                "weights": list(result.weights),
                "receptor": {
                    "n_atoms": result.receptor_meta.n_atoms,
                    "n_residues": result.receptor_meta.n_residues,
                },
                "ligand": {"n_atoms": result.ligand_meta.n_atoms},
                "box": {
                    "center": result.box.center.tolist(),
                    "half_size": result.box.half_size.tolist(),
                },
                "stats": asdict(result.stats),
                "review": summarize_docking(result, checks),
                "effective_parameters": {
                    "n_runs": str(n_runs),
                    "n_steps": str(n_steps),
                    "top_n": str(top_n),
                    "seed": str(
                        result.stats.seed
                        if not use_grid
                        else request.form.get("seed", 1)
                    ),
                    "use_grid": "1" if use_grid else "0",
                    "rmsd_threshold": str(
                        float(request.form.get("rmsd_threshold", 2.0))
                    ),
                    "validate": "0"
                    if request.form.get("validate", "1") == "0"
                    else "1",
                },
                "poses": poses,
                "validation": [
                    {
                        "rule_id": check.rule_id,
                        "passed": check.passed,
                        "severity": str(
                            getattr(check.severity, "value", check.severity)
                        ),
                        "message": check.message,
                        "evidence": check.evidence,
                    }
                    for check in checks
                ],
                "plot": plots[0]["data_uri"] if plots else "",
                "plots": plots,
            }
        )

    @app.post("/api/align")
    def api_align():
        """Align a sequencing read to a reference and audit the result."""
        from bioeng_qpcr.sequence import (
            BLOSUM62,
            compare_to_reference,
            detect_variants,
            dna_matrix,
            read_sequence_file,
            trim_low_quality,
            trim_primers,
            validate_sequencing_result,
        )

        query_path: str | None = None
        reference_path: str | None = None
        try:
            query_path = _save_upload(_SEQUENCE_SUFFIXES, field="query")
            reference_path = _save_upload(_SEQUENCE_SUFFIXES, field="reference")
            records = read_sequence_file(query_path)
            references = read_sequence_file(reference_path)
            if not records or not references:
                raise ValueError("The query or reference file contains no sequences.")

            index = int(request.form.get("read_index", 0))
            if not 0 <= index < len(records):
                raise ValueError(
                    f"The query file holds {len(records)} record(s); index "
                    f"{index} is out of range."
                )
            record = records[index]
            query = record.sequence
            quality = list(record.quality) if record.quality else None
            reference = references[0].sequence

            notes: list[str] = []
            primer_result = None
            primer_5 = (request.form.get("primer_5") or "").strip() or None
            primer_3 = (request.form.get("primer_3") or "").strip() or None
            if primer_5 or primer_3:
                primer_result = trim_primers(
                    query,
                    primer_5=primer_5,
                    primer_3=primer_3,
                    max_mismatch=int(request.form.get("primer_mismatch", 2)),
                )
                query = primer_result.trimmed
                notes.append(
                    f"引物裁剪：5' 去掉 {primer_result.n_trimmed_5} 个碱基，"
                    f"3' 去掉 {primer_result.n_trimmed_3} 个"
                )
                if quality is not None:
                    quality = quality[
                        primer_result.n_trimmed_5 : len(quality)
                        - primer_result.n_trimmed_3
                    ]

            window = int(request.form.get("quality_window", 10))
            min_q = int(request.form.get("min_q", 20))
            if request.form.get("trim_quality") == "1":
                if quality is None:
                    raise ValueError("Quality trimming requires FASTQ or AB1 input.")
                trimmed = trim_low_quality(query, quality, window=window, min_q=min_q)
                query = trimmed.trimmed
                quality = quality[trimmed.kept_start : trimmed.kept_end]
                notes.append(
                    f"质量裁剪：保留第 {trimmed.kept_start}..{trimmed.kept_end} 位"
                )

            matrix = (
                BLOSUM62 if request.form.get("matrix") == "blosum62" else dna_matrix()
            )
            result = compare_to_reference(
                query,
                reference,
                matrix,
                gap_open=int(request.form.get("gap_open", 5)),
                gap_extend=int(request.form.get("gap_extend", 2)),
            )
            variants = (
                detect_variants(result, reference, query_quality=quality)
                if request.form.get("variants", "1") != "0"
                else []
            )
            report = validate_sequencing_result(
                result,
                query,
                reference,
                variants,
                quality=quality,
                coding=request.form.get("coding") == "1",
                primer_trim=primer_result,
                primer_5=primer_5,
                primer_3=primer_3,
                threshold_identity=float(request.form.get("min_identity", 0.9)),
                min_q=min_q,
            )
        except ValueError as exc:
            return _message(str(exc), 400)
        except FileNotFoundError as exc:
            return _message(f"File could not be read: {exc}", 400)
        finally:
            for temporary in (query_path, reference_path):
                if temporary:
                    _remove(temporary)

        plots: list[dict[str, Any]] = []
        plot_warnings = []
        try:
            _ensure_matplotlib()
            publication = request.form.get("publication") == "1"
            from bioeng_qpcr.plotting import (
                plot_alignment_coverage,
                plot_read_qualities,
                plot_variant_map,
            )

            figure, _ax = plot_alignment_coverage(
                len(reference),
                result.ref_start,
                result.ref_end,
                [variant.ref_position for variant in variants],
            )
            plots.append(
                _plot_payload(
                    "参考坐标覆盖与变异",
                    figure,
                    "sequence-coverage-variants",
                    publication=publication,
                )
            )
            figure, _ax = plot_variant_map(
                len(reference),
                [variant.ref_position for variant in variants],
            )
            plots.append(
                _plot_payload(
                    "变异位置图",
                    figure,
                    "sequence-variant-map",
                    publication=publication,
                )
            )
            if quality:
                figure, _ax = plot_read_qualities(
                    quality, threshold=min_q, window=window
                )
                plots.append(
                    _plot_payload(
                        "逐碱基质量分布",
                        figure,
                        "sequence-base-quality",
                        publication=publication,
                    )
                )
        except Exception:
            app.logger.exception("Sequence figure generation failed")
            plot_warnings.append("序列图表未完整生成；数值结果已保留，请重试绘图并核对日志。")

        return jsonify(
            {
                "ok": True,
                "query_id": record.id,
                "plot_warnings": plot_warnings,
                "reference_id": references[0].id,
                "notes": notes,
                "alignment": {
                    "score": result.score,
                    "identity": result.identity,
                    "similarity": result.similarity,
                    "gaps": result.gaps,
                    "gap_opens": result.gap_opens,
                    "length": result.length,
                    "ref_start": result.ref_start,
                    "ref_end": result.ref_end,
                    "query_start": result.query_start,
                    "query_end": result.query_end,
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
                    for variant in variants
                ],
                "validation": [
                    {
                        "rule_id": check.rule_id,
                        "passed": check.passed,
                        "severity": check.severity.value,
                        "message": check.message,
                    }
                    for check in report.results
                ],
                "passed": report.passed,
                "plot": plots[0]["data_uri"] if plots else "",
                "plots": plots,
            }
        )

    return app


def _preview_payload(preview) -> dict[str, Any]:
    """Serialise a :class:`~bioeng_qpcr.io.CsvPreview` for the workspace UI."""
    return {
        "path": preview.table.path,
        "encoding": preview.table.encoding,
        "delimiter": preview.table.delimiter,
        "rows_total": preview.rows_total,
        "rows_scanned": preview.rows_scanned,
        "truncated": preview.truncated,
        "size_bytes": preview.size_bytes,
        "sha256": preview.sha256,
        "schema": preview.schema_key,
        "schema_score": preview.schema_score,
        "header": list(preview.head[0]) if preview.head else [],
        "rows": [list(row) for row in preview.head[1:]],
        "columns": [asdict(column) for column in preview.columns],
        "issues": [issue.to_dict() for issue in preview.issues],
        "errors": len(preview.errors),
        "warnings": len(preview.warnings),
        "text": preview.to_text(),
    }


def _potency_payload(result, dataset, *, run_validation: bool) -> dict[str, Any]:
    """Serialise one potency result, including its audit outcomes."""
    from bioeng_qpcr.potency import run_potency_validation
    from bioeng_qpcr.web.potency_review import review_status

    payload: dict[str, Any] = {
        "compound": result.compound,
        "kind": result.kind.value,
        "model": result.best_fit.model.value,
        "message": result.message,
        "relative_value": result.relative_value,
        "relative_ci": list(result.relative_ci) if result.relative_ci else None,
        "absolute_value": result.absolute_value,
        "absolute_ci": list(result.absolute_ci) if result.absolute_ci else None,
        "control_fraction": result.control_fraction,
        "control_response": result.control_response,
        "hill_slope": result.best_fit.hill_slope,
        "bottom": result.best_fit.bottom,
        "top": result.best_fit.top,
        "r_squared": result.best_fit.r_squared,
        "n_points": result.best_fit.n_points,
        "extrapolated": result.extrapolated,
        "fold_range_covered": result.fold_range_covered,
        "confidence": result.confidence,
        "ci_method": result.ci_method,
        "warnings": list(result.warnings),
        "model_selection": {
            "ambiguous": result.comparison.ambiguous,
            "message": result.comparison.message,
            "models": [
                {
                    "model": fit.model.value,
                    "aicc": fit.aicc if math.isfinite(fit.aicc) else None,
                }
                for fit in result.comparison.fits
            ],
        },
        "validation": [],
        "plot_data": {
            "concentration": dataset.concentration.tolist(),
            "response": dataset.response.tolist(),
            "concentration_unit": dataset.concentration_unit,
            "response_unit": dataset.response_unit,
            "parameters": dict(result.best_fit.parameters),
            "direction": result.best_fit.direction.value,
        },
    }
    if run_validation:
        payload["validation"] = [
            {
                "rule_id": check.rule_id,
                "passed": check.passed,
                "severity": check.severity,
                "message": check.message,
            }
            for check in run_potency_validation(result, dataset)
        ]
    payload["review_status"] = review_status(payload)
    return payload


def _potency_summary_plot(entries: list[dict[str, Any]]) -> Any | None:
    """Draw a log-scaled potency summary from serialised results.

    The comparison is drawn from the JSON payload rather than from
    :func:`~bioeng_qpcr.plotting.plot_potency_comparison`, because the HTTP
    layer no longer holds the ``PotencyResult`` objects, and rebuilding them
    would silently lose the fitted curve behind each bar.
    """
    if not entries:
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    from bioeng_qpcr.plotting.style import PlotStyle, apply_style

    style = PlotStyle()
    figure, ax = plt.subplots(figsize=(6, max(2.4, 0.5 * len(entries) + 1.6)))
    positions = np.arange(len(entries))
    values = [entry["relative_value"] for entry in entries]
    lower: list[float] = []
    upper: list[float] = []
    for value, entry in zip(values, entries, strict=True):
        interval = entry.get("relative_ci")
        if interval:
            lower.append(max(value - interval[0], 0.0))
            upper.append(max(interval[1] - value, 0.0))
        else:
            lower.append(0.0)
            upper.append(0.0)
    ax.barh(
        positions,
        values,
        xerr=np.vstack([lower, upper]),
        height=0.6,
        error_kw={"linewidth": 1.0, "capsize": 3},
        zorder=3,
    )
    ax.set_yticks(positions, [entry["compound"] for entry in entries])
    ax.set_xscale("log")
    kind = entries[0].get("kind", "IC50")
    ax.set_xlabel(f"relative {kind} (log scale)")
    ax.set_title(f"{kind} comparison")
    ax.invert_yaxis()
    apply_style(ax, style)
    figure.tight_layout()
    return figure


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def main() -> None:
    """Run the development server (``bioeng-qpcr-web`` entry point)."""
    host = os.environ.get("BIOENG_QPCR_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("BIOENG_QPCR_WEB_PORT", "8008"))
    debug = os.environ.get("BIOENG_QPCR_WEB_DEBUG", "0") == "1"
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
