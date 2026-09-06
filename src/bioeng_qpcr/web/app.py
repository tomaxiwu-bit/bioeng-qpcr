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
import io
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_reference_targets(raw: str | None) -> list[str] | None:
    """Split a comma-separated reference-gene string into a list (or ``None``)."""
    if not raw:
        return None
    targets = [v.strip() for v in raw.split(",") if v.strip()]
    return targets or None


def _save_upload() -> str:
    """Persist the uploaded file to a temp location and return its path."""
    if "file" not in request.files:
        raise ValueError("No file was uploaded.")
    upload = request.files["file"]
    if not upload or not upload.filename:
        raise ValueError("No file was uploaded.")

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix or '<none>'}'.")

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
    return _reader_for(path)(
        path, reference_targets=reference_targets, strict=strict
    )


def _load_curve_experiment(
    path: str, reference_targets: list[str] | None = None
):
    """Load an experiment that carries raw amplification traces.

    A long-format trace CSV is preferred; files that structurally are not
    trace CSVs (and missing files) route through :func:`_load_experiment`.
    A file that *looks* like a trace CSV but is malformed raises instead of
    being silently reinterpreted.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _RDML_SUFFIXES:
        try:
            return read_amplification_csv(
                path, reference_targets=reference_targets
            )
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
            {"sample_name": e.sample_name, "reason": e.reason}
            for e in batch.exclusions
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

    @app.get("/")
    def index():
        return app.jinja_env.get_template("index.html").render()

    @app.get("/api/health")
    def health():
        from bioeng_qpcr import __version__

        return jsonify({"status": "ok", "version": __version__})

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

    return app


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