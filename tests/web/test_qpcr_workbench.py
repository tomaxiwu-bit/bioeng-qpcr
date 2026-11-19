"""Experiment lifecycle, scientific boundaries, persistence and export contracts."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from bioeng_qpcr.io import read_amplification_csv, read_rdml, read_tidy_csv
from bioeng_qpcr.web.app import create_app
from bioeng_qpcr.workbench.acquisition import read_acquisition, validate_measurements
from bioeng_qpcr.workbench.analysis import melting_peaks, threshold_ct
from bioeng_qpcr.workbench.model import (
    AssayRecord,
    PlateAssignment,
    Project,
    SampleRecord,
    validate_design,
)
from bioeng_qpcr.workbench.service import (
    advance,
    create_project,
    csv_text,
    patch_project,
    replay,
    report_html,
)


@pytest.fixture
def client(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, WORKBENCH_DB=tmp_path / "workbench.sqlite3")
    return app.test_client()


def prepared(demo="relative-ct"):
    project = create_project("", demo)
    for step in ("samples", "design", "program"):
        advance(project, step)
    replay(project)
    advance(project, "curves")
    advance(project, "qc")
    return project


def post(client, project, route):
    response = client.post(
        f"/api/qpcr/projects/{project['id']}/{route}",
        json={"revision": project["revision"]},
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()["project"]


def test_public_relative_example_matches_independent_reference():
    project = prepared()
    project.analysis_settings["efficiency_confirmed"] = True
    advance(project, "analysis")
    kidney = project.outputs["analysis"]["rows"][1]
    assert kidney["delta_delta_cq"] == pytest.approx(-2.495)
    assert kidney["fold_change"] == pytest.approx(5.6372830205680815)
    assert project.outputs["qc"]["counts"]["NOT_ASSESSABLE"] > 0
    assert all(row["tm"] is None for row in project.outputs["curves"]["rows"])


def test_each_analysis_version_captures_its_actual_environment():
    from bioeng_qpcr import __version__

    project = prepared()
    versions = project.active_run["analysis_versions"]
    old_curves = copy.deepcopy(versions[0])
    manifest = old_curves["provenance"]
    assert manifest["package_versions"]["bioeng_qpcr"] == __version__
    assert manifest["raw_input_hash"] == project.active_run["source"]["sha256"]
    assert len(manifest["input_hash"]) == 64
    project.qc_policy["replicate_max_sd"] = 0.7
    advance(project, "qc")
    assert versions[0] == old_curves
    policy = versions[-1]["provenance"]["parameters"]["qc_policy"]
    assert policy["replicate_max_sd"] == 0.7
    assert "运行环境与输入摘要" in report_html(project)


def test_relative_requires_explicit_efficiency_assumption():
    with pytest.raises(ValueError, match="效率"):
        advance(prepared(), "analysis")


def test_absolute_public_data_has_real_curves_and_known_units():
    project = prepared("rdml-dilution")
    advance(project, "analysis")
    result = project.outputs["analysis"]
    assert result["unit"] == "cop"
    assert result["standard_curve"]["slope"] < 0
    assert result["standard_curve"]["r_squared"] > 0.98
    assert result["rows"][0]["quantity"] == pytest.approx(382.0546283308384)
    assert any(row["tm"] is not None for row in project.outputs["curves"]["rows"])
    assert len(project.active_run["measurements"]) == 26


def test_missing_upstream_cannot_be_skipped():
    with pytest.raises(ValueError, match="上游"):
        advance(create_project("", "relative-ct"), "analysis")


def test_design_edit_invalidates_results_and_retains_run_snapshot():
    project = prepared()
    snapshot = copy.deepcopy(project.runs[0])
    samples = copy.deepcopy(project.samples)
    samples[0]["name"] = "changed"
    patch_project(project, {"samples": samples})
    assert project.active_run is None
    assert project.runs[0] == snapshot
    assert project.completed == []
    assert project.outputs == {}
    assert "curves" in project.stale


def test_qc_policy_edit_preserves_raw_data_and_only_invalidates_downstream():
    project = prepared()
    original = copy.deepcopy(project.active_run)
    patch_project(
        project, {"qc_policy": {**project.qc_policy, "replicate_max_sd": 0.4}}
    )
    assert project.active_run == original
    assert "curves" in project.completed
    assert "qc" not in project.completed
    assert "curves" in project.outputs


def test_negative_control_failure_blocks_quantification_but_allows_report():
    project = prepared("rdml-dilution")
    # A deliberately failing control fixture, not a fabricated public observation.
    project.samples[0]["kind"] = "ntc"
    ntc = next(
        a
        for a in project.plate
        if next(s for s in project.samples if s["id"] == a["sample_id"])["kind"]
        == "ntc"
    )
    row = next(
        row for row in project.outputs["curves"]["rows"] if row["well"] == ntc["well"]
    )
    row["cq"] = 20.0
    advance(project, "qc")
    assert project.outputs["qc"]["blocked"]
    assert "qc" not in project.completed
    assert (
        next(n for n in project.workflow() if n["key"] == "qc")["status"] == "blocked"
    )
    with pytest.raises(ValueError):
        advance(project, "analysis")
    assert "质控阻断" in report_html(project)


def test_program_requires_matching_channels_and_acquisition_step():
    project = create_project("", "relative-ct")
    project.program["channels"] = ["HEX"]
    with pytest.raises(ValueError, match="通道"):
        validate_design(project)
    project.program["channels"] = ["SYBR/FAM"]
    for step in project.program["steps"]:
        step["acquire"] = False
    with pytest.raises(ValueError, match="采集"):
        validate_design(project)


def test_plate_aliases_do_not_bypass_duplicate_detection():
    project = create_project("", "relative-ct")
    duplicate = copy.deepcopy(project.plate[0])
    duplicate["well"] = "A1"
    project.plate.append(duplicate)
    with pytest.raises(ValueError, match="重复"):
        validate_design(project)


def test_reaction_volumes_must_sum_to_total():
    project = create_project("", "relative-ct")
    project.recipe["volume_ul"] = 1
    with pytest.raises(ValueError, match="体积"):
        validate_design(project)


def test_acquisition_rejects_unknown_wells_duplicate_measurements_and_nan():
    project = prepared()
    original = copy.deepcopy(project.active_run["measurements"][0])
    for bad in (
        {**original, "well": "H12"},
        {**original, "instrument_cq": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate_measurements(project, [bad])
    with pytest.raises(ValueError):
        validate_measurements(project, [original, original])


def test_threshold_ct_is_baseline_corrected_interpolated_crossing():
    points = [[1, 10], [2, 10], [3, 10], [4, 20], [5, 40], [6, 80]]
    ct, note = threshold_ct(
        points, {"baseline_start": 1, "baseline_end": 3, "threshold": 20}
    )
    assert ct == pytest.approx(4.5)
    assert "阈值" in note
    assert (
        threshold_ct(
            points, {"baseline_start": 1, "baseline_end": 3, "threshold": 200}
        )[0]
        is None
    )


def test_melt_sorting_and_duplicates_keep_peak_near_true_temperature():
    temperatures = np.linspace(65, 95, 121)
    fluorescence = 1000 / (1 + np.exp((temperatures - 82) * 2))
    points = list(zip(temperatures, fluorescence, strict=True))
    result = melting_peaks(list(reversed(points)) + [points[10]], 0.15)
    assert result["tm"] == pytest.approx(82, abs=0.3)
    assert len(result["peaks"]) == 1
    assert melting_peaks([], 0.15)["tm"] is None


def test_workbench_page_and_old_navigation(client):
    assert client.get("/workbench").status_code == 200
    assert b'href="/workbench"' in client.get("/").data
    assert b'href="/workbench"' in client.get("/workspace").data


def test_api_lifecycle_persistence_conflict_and_exports(client):
    response = client.post("/api/qpcr/projects", json={"demo": "relative-ct"})
    project = response.get_json()["project"]
    assert response.status_code == 201
    original_revision = project["revision"]
    for step in ("samples", "design", "program"):
        project = post(client, project, "steps/" + step)
    project = post(client, project, "replay")
    for step in ("curves", "qc"):
        project = post(client, project, "steps/" + step)
    url = f"/api/qpcr/projects/{project['id']}"
    settings = {**project["analysis_settings"], "efficiency_confirmed": True}
    project = client.patch(
        url,
        json={
            "revision": project["revision"],
            "changes": {"analysis_settings": settings},
        },
    ).get_json()["project"]
    project = post(client, project, "steps/analysis")
    project = post(client, project, "steps/report")
    persisted = client.get(url).get_json()["project"]
    assert persisted == project
    assert len(client.get("/api/qpcr/projects").get_json()["projects"]) == 1
    conflict = client.patch(
        url, json={"revision": original_revision, "changes": {"name": "stale"}}
    )
    assert conflict.status_code == 409
    for kind in (
        "project.json",
        "results.csv",
        "plate.csv",
        "protocol.json",
        "report.html",
        "acquisition.csv",
    ):
        download = client.get(url + "/export/" + kind)
        assert download.status_code == 200
        assert "attachment" in download.headers["Content-Disposition"]
    assert "5.637283" in client.get(url + "/export/results.csv").get_data(as_text=True)
    exported = json.loads(client.get(url + "/export/project.json").data)
    assert exported["runs"][0]["design_sha256"]
    fresh = create_app()
    fresh.config.update(
        TESTING=True, WORKBENCH_DB=client.application.config["WORKBENCH_DB"]
    )
    assert fresh.test_client().get(url).get_json()["project"] == persisted


def test_replay_cannot_relabel_modified_public_sample_identity(client):
    project = create_project("", "relative-ct")
    project.samples[0]["name"] = "made up"
    for step in ("samples", "design", "program"):
        advance(project, step)
    with pytest.raises(ValueError, match="重新标记"):
        replay(project)


def test_collected_run_cannot_be_overwritten():
    with pytest.raises(ValueError, match="新运行"):
        replay(prepared())


def test_real_csv_roundtrip_preserves_raw_data(client):
    project = client.post(
        "/api/qpcr/projects", json={"demo": "rdml-dilution"}
    ).get_json()["project"]
    for step in ("samples", "design", "program"):
        project = post(client, project, "steps/" + step)
    project = post(client, project, "replay")
    original = project["runs"][0]["measurements"]
    url = f"/api/qpcr/projects/{project['id']}"
    csv_data = client.get(url + "/export/acquisition.csv").data
    project = post(client, project, "new-run")
    response = client.post(
        url + "/acquire",
        data={
            "revision": str(project["revision"]),
            "file": (io.BytesIO(csv_data), "real-run.csv"),
        },
    )
    assert response.status_code == 200, response.get_json()
    updated = response.get_json()["project"]
    assert updated["runs"][-1]["measurements"] == original
    assert updated["runs"][-1]["mode"] == "file_import"
    assert updated["runs"][0]["measurements"] == original


def test_csv_formula_injection_and_html_escaping():
    assert "'=SUM" in csv_text([{"sample": "=SUM(A1:A3)"}])
    project = prepared()
    project.name = "<script>alert(1)</script>"
    assert "<script>alert(1)</script>" not in report_html(project)
    assert "&lt;script&gt;" in report_html(project)


def test_cross_origin_writes_and_result_tampering_are_rejected(client):
    response = client.post(
        "/api/qpcr/projects",
        json={"name": "test"},
        headers={"Origin": "https://evil.invalid"},
    )
    assert response.status_code == 403
    project = client.post("/api/qpcr/projects", json={"name": "test"}).get_json()[
        "project"
    ]
    response = client.patch(
        f"/api/qpcr/projects/{project['id']}",
        json={"revision": project["revision"], "changes": {"completed": ["analysis"]}},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "name,reader",
    [
        ("researcher_example.csv", read_tidy_csv),
        ("rdml_example.xml", read_rdml),
        ("amplification_curves_example.csv", read_amplification_csv),
    ],
)
def test_existing_real_file_formats_reuse_domain_trace_contract(name, reader):
    path = Path(__file__).resolve().parents[2] / "examples" / name
    experiment = reader(path)
    project = Project(
        samples=[
            asdict(SampleRecord(id=s.name, name=s.name))
            for s in experiment.samples.values()
        ],
        assays=[
            asdict(AssayRecord(id=a.name, name=a.name))
            for a in experiment.assays.values()
        ],
        plate=[
            asdict(PlateAssignment(w.position, w.sample_name, w.target_name))
            for w in experiment.wells
        ],
    )
    records = read_acquisition(path, project)
    assert len(records) == len(experiment.wells)
    if experiment.has_traces:
        assert any(row["amplification"] for row in records)


def test_analysis_versions_survive_recalculation_and_new_run():
    project = prepared()
    archived = copy.deepcopy(project.active_run["analysis_versions"])
    assert {row["step"] for row in archived} == {"curves", "qc"}
    patch_project(
        project, {"qc_policy": {**project.qc_policy, "replicate_max_sd": 0.6}}
    )
    advance(project, "qc")
    assert project.active_run["analysis_versions"][: len(archived)] == archived
    old_run = copy.deepcopy(project.active_run)
    project.invalidate("acquisition")
    project.start_run()
    assert project.runs[0] == old_run
    assert project.active_run["analysis_versions"] == []


@pytest.mark.parametrize("payload", [[], None, {"demo": "bad-key"}])
def test_malformed_create_requests_return_clear_client_error(client, payload):
    response = client.post("/api/qpcr/projects", json=payload)
    assert response.status_code == 400


@pytest.mark.parametrize(
    "key", ["expression", "standard", "melt", "multiplex", "controls"]
)
def test_experiment_templates_are_valid_designs_without_fabricated_measurements(key):
    from bioeng_qpcr.workbench.templates import EXPERIMENT_TEMPLATES

    project = create_project("", template=key)
    validate_design(project)
    assert len({a["well"] for a in project.plate}) == EXPERIMENT_TEMPLATES[key]["wells"]
    assert not project.runs and not project.outputs and not project.completed
    assert project.demo_key is None
    if key == "multiplex":
        assert len(project.plate) == 2 * len({a["well"] for a in project.plate})
        assert project.program["channels"] == ["FAM", "HEX"]


@pytest.mark.parametrize(
    "theme", ["journal", "lab", "accessible", "mono", "night", "pastel"]
)
def test_theme_exports_preserve_numerical_results(theme):
    from bioeng_qpcr.workbench.figures import figure_bytes
    from bioeng_qpcr.workbench.templates import THEMES

    project = prepared()
    project.analysis_settings["efficiency_confirmed"] = True
    advance(project, "analysis")
    advance(project, "report")
    original = copy.deepcopy(project.outputs["analysis"])
    patch_project(project, {"presentation": {**project.presentation, "theme": theme}})
    assert project.outputs["analysis"] == original
    assert "analysis" in project.completed
    if theme != "journal":
        assert "report" not in project.completed
    svg = figure_bytes(project, "relative", "svg")
    assert b"<svg" in svg
    assert THEMES[theme]["background"].encode() in svg


def test_template_catalog_and_figure_downloads(client):
    catalog = client.get("/api/qpcr/templates").get_json()
    assert len(catalog["experiments"]) == 5
    assert len(catalog["themes"]) == 6
    response = client.post("/api/qpcr/projects", json={"template": "standard"})
    assert response.status_code == 201
    project = response.get_json()["project"]
    assert project["plate"] and not project["runs"]
    for step in ("samples", "design", "program"):
        project = post(client, project, "steps/" + step)
    project = client.post(
        "/api/qpcr/projects", json={"demo": "rdml-dilution"}
    ).get_json()["project"]
    for step in ("samples", "design", "program"):
        project = post(client, project, "steps/" + step)
    project = post(client, project, "replay")
    for step in ("curves", "qc", "analysis"):
        project = post(client, project, "steps/" + step)
    root = f"/api/qpcr/projects/{project['id']}/figures/"
    for kind, fmt, signature in (
        ("standard", "svg", b"<svg"),
        ("melting", "png", b"\x89PNG"),
        ("amplification", "pdf", b"%PDF"),
    ):
        response = client.get(root + kind + "." + fmt)
        assert response.status_code == 200
        assert signature in response.data[:1000]
    assert client.get(root + "relative.svg").status_code == 400


def test_figure_export_selects_available_evidence_and_rejects_wrong_well():
    from bioeng_qpcr.workbench.figures import figure_bytes

    project = prepared("rdml-dilution")
    project.active_run["measurements"][0]["amplification"] = []
    assert b"<svg" in figure_bytes(project, "amplification", "svg")
    with pytest.raises(ValueError, match="没有采集"):
        figure_bytes(project, "amplification", "svg", well="A01", assay="not-present")


def test_backup_restore_recomputes_qc_and_preserves_original(client):
    source = prepared()
    source.analysis_settings["efficiency_confirmed"] = True
    advance(source, "analysis")
    original = client.post(
        "/api/qpcr/projects", json={"demo": "relative-ct"}
    ).get_json()["project"]
    backup = asdict(source)
    backup["outputs"]["qc"] = {"blocked": False, "tampered": True}
    response = client.post("/api/qpcr/projects/import", json=backup)
    assert response.status_code == 201, response.get_json()
    restored = response.get_json()["project"]
    assert restored["id"] not in (source.id, original["id"])
    assert restored["runs"][0]["measurements"] == source.runs[0]["measurements"]
    assert "tampered" not in restored["outputs"]["qc"]
    assert restored["analysis_settings"]["efficiency_confirmed"] is False
    assert "analysis" not in restored["completed"]
    assert "analysis" not in restored["outputs"]
    assert any(
        v.get("imported_backup") for v in restored["runs"][0]["analysis_versions"]
    )
    assert (
        client.get("/api/qpcr/projects/" + original["id"]).get_json()["project"]
        == original
    )


def test_backup_restore_rejects_corrupt_snapshot_and_bad_shapes(client):
    backup = asdict(prepared())
    backup["runs"][0]["design_sha256"] = "wrong"
    assert client.post("/api/qpcr/projects/import", json=backup).status_code == 400
    for bad in ([], {"schema_version": 99}, {"schema_version": 1, "runs": "invalid"}):
        assert client.post("/api/qpcr/projects/import", json=bad).status_code == 400
