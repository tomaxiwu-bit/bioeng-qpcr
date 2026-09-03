# ruff: noqa: E501
"""Self-contained HTML/JSON report for a qPCR readiness and quantity run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape
from typing import Any

from bioeng_qpcr.calculations import normalized_quantity_all
from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.provenance import ProvenanceManifest
from bioeng_qpcr.validation import ValidationEngine


@dataclass(frozen=True)
class ResearchReport:
    """A complete machine-readable payload and self-contained HTML view."""

    payload: dict[str, Any]
    html: str

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, ensure_ascii=False)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_research_report(
    experiment: Experiment,
    target_name: str,
    reference_targets: list[str] | None = None,
) -> ResearchReport:
    """Run readiness, normalized quantities, and provenance as one workflow."""
    readiness = ValidationEngine().run(experiment)
    analysis = normalized_quantity_all(experiment, target_name, reference_targets)
    manifest = ProvenanceManifest.from_experiment(
        experiment, parameters=analysis.parameters
    )
    manifest.exclusions = [asdict(item) for item in analysis.exclusions]

    payload = {
        "report_type": "bioeng-qpcr researcher report",
        "research_use_status": "alpha; independent review required",
        "experiment": {
            "name": experiment.name,
            "hash": experiment.content_hash(),
            "raw_input_hash": experiment.metadata.get("source_sha256"),
        },
        "readiness": readiness.to_dict(),
        "analysis": analysis.to_dict(),
        "provenance": manifest.to_dict(),
    }

    readiness_rows = [
        [result.rule_id, result.status.value, result.message]
        for result in readiness.results
    ]
    quantity_rows = [
        [
            result.sample_name,
            f"{result.normalized_quantity:.8g}",
            f"{result.log2_normalized_quantity:.4f}",
        ]
        for result in analysis.results
    ]
    exclusion_rows = [
        [item.sample_name, item.code, item.reason] for item in analysis.exclusions
    ]
    diagnostics = experiment.metadata.get("input_diagnostics") or []
    diagnostic_rows = [
        [item.get("row"), item.get("column"), item.get("code"), item.get("message")]
        for item in diagnostics
    ]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>bioeng-qpcr report — {escape(experiment.name)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2937}}
h1,h2{{color:#123c69}} .notice{{padding:1rem;background:#fff3cd;border-left:5px solid #d39e00}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #d1d5db;padding:.55rem;text-align:left}}
th{{background:#eef4f8}} code{{word-break:break-all}} .pass{{color:#166534}}
</style></head><body>
<h1>bioeng-qpcr researcher report</h1>
<p class="notice"><strong>Alpha research software.</strong> This report supports review; it is not clinical validation or full MIQE certification.</p>
<p><strong>Experiment:</strong> {escape(experiment.name)}<br><strong>Experiment hash:</strong> <code>{experiment.content_hash()}</code><br><strong>Raw input hash:</strong> <code>{escape(str(experiment.metadata.get("source_sha256") or "not available"))}</code></p>
<h2>Readiness assessment</h2>
<p>Overall: <strong>{readiness.overall_status}</strong>; rule coverage: {readiness.rule_coverage["executed"]}/{readiness.rule_coverage["available"]}.</p>
{_table(["Rule", "Status", "Finding"], readiness_rows)}
<h2>Efficiency-corrected normalized quantities</h2>
{_table(["Sample", "Normalized quantity", "log2 quantity"], quantity_rows)}
<h2>Excluded samples</h2>
{_table(["Sample", "Code", "Reason"], exclusion_rows) if exclusion_rows else "<p>None.</p>"}
<h2>Input diagnostics</h2>
{_table(["Row", "Column", "Code", "Message"], diagnostic_rows) if diagnostic_rows else "<p>None.</p>"}
<h2>Provenance</h2><pre>{escape(manifest.to_json())}</pre>
</body></html>"""
    return ResearchReport(payload=payload, html=html)
