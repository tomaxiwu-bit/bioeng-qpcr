"""Review stored potency fits without repeating model selection or resampling."""

from __future__ import annotations

import math
from types import SimpleNamespace

from flask import Blueprint, Response, jsonify, request

from bioeng_qpcr.web.workbench import store
from bioeng_qpcr.workbench.service import csv_text

blueprint = Blueprint("potency_review", __name__)


def review_status(entry: dict) -> str:
    if entry.get("error"):
        return "failed"
    if (
        entry.get("extrapolated")
        or not entry.get("relative_ci")
        or entry.get("warnings")
        or entry.get("model_selection", {}).get("ambiguous")
        or any(not c["passed"] for c in entry.get("validation", []))
    ):
        return "review"
    return "passed" if entry.get("validation") else "unassessed"


def saved_result(record_id):
    record = store().get_record(record_id)
    if record["module"] != "potency":
        raise ValueError("此记录不是效价分析")
    if record["imported"]:
        raise ValueError("外部备份尚未复算，请重新分析后再审阅曲线与导出汇总")
    return record["result"]


@blueprint.errorhandler(KeyError)
def missing(exc):
    return jsonify(ok=False, error=str(exc)), 404


@blueprint.errorhandler(ValueError)
def invalid(exc):
    return jsonify(ok=False, error=str(exc)), 400


@blueprint.get("/api/lab/records/<record_id>/potency/plots")
def plots(record_id):
    from bioeng_qpcr.plotting.dose_response import (
        plot_dose_response,
        plot_dose_response_residuals,
    )
    from bioeng_qpcr.potency.fitting import CurveDirection, DoseResponseDataset
    from bioeng_qpcr.potency.models import DoseResponseModel
    from bioeng_qpcr.web.app import _plot_payload

    result = saved_result(record_id)
    compound = request.args.get("compound", "")
    entry = next((r for r in result["results"] if r["compound"] == compound), None)
    if entry is None:
        raise KeyError("化合物不存在于此分析记录")
    curve = entry.get("plot_data")
    if not curve or entry.get("error"):
        raise ValueError("此记录没有该化合物的绘图数据，请重新分析一次")
    dataset = DoseResponseDataset(
        compound=compound,
        concentration=curve["concentration"],
        response=curve["response"],
        concentration_unit=curve["concentration_unit"],
        response_unit=curve["response_unit"],
    )
    # Rehydrate the attributes consumed by the existing plotters; do not refit.
    fit = SimpleNamespace(
        model=DoseResponseModel(entry["model"]),
        parameters=curve["parameters"],
        direction=CurveDirection(curve["direction"]),
        r_squared=entry["r_squared"] if entry["r_squared"] is not None else math.nan,
        n_points=entry["n_points"],
        top=entry["top"],
        bottom=entry["bottom"],
        half_max_concentration=entry["relative_value"],
        half_max_concentration_ci=entry["relative_ci"],
    )
    rendered = []
    for plotter, label, stem in (
        (plot_dose_response, "剂量–反应曲线", "dose-response"),
        (plot_dose_response_residuals, "残差诊断", "residuals"),
    ):
        figure, _ = plotter(fit, dataset)
        rendered.append(
            _plot_payload(f"{compound} {label}", figure, stem, publication=True)
        )
    return jsonify(
        ok=True,
        compound=compound,
        plots=rendered,
        note="使用此记录已保存的拟合参数，不重新选择模型或重采样",
    )


@blueprint.get("/api/lab/records/<record_id>/potency/results.csv")
def export_csv(record_id):
    rows = []
    for entry in saved_result(record_id)["results"]:
        relative_ci, absolute_ci = entry.get("relative_ci"), entry.get("absolute_ci")
        rows.append(
            {
                "compound": entry["compound"],
                "review_status": review_status(entry),
                "kind": entry.get("kind"),
                "model": entry.get("model"),
                "concentration_unit": entry.get("plot_data", {}).get(
                    "concentration_unit", ""
                ),
                "relative_value": entry.get("relative_value"),
                "relative_ci_low": relative_ci[0] if relative_ci else None,
                "relative_ci_high": relative_ci[1] if relative_ci else None,
                "absolute_value": entry.get("absolute_value"),
                "absolute_ci_low": absolute_ci[0] if absolute_ci else None,
                "absolute_ci_high": absolute_ci[1] if absolute_ci else None,
                "ci_method": entry.get("ci_method"),
                "r_squared": entry.get("r_squared"),
                "n_points": entry.get("n_points"),
                "error": entry.get("error", ""),
                "warnings": "; ".join(entry.get("warnings", [])),
                "audit_failures": "; ".join(
                    c["rule_id"] + ": " + c["message"]
                    for c in entry.get("validation", [])
                    if not c["passed"]
                ),
            }
        )
    return Response(
        csv_text(rows),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="potency-results.csv"'},
    )
