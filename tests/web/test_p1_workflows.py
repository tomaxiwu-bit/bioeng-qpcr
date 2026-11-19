"""Scientific equivalence and review of every member of a batch."""

import copy
import io

import numpy as np
import pytest

from bioeng_qpcr.web.app import create_app
from bioeng_qpcr.workbench.model import Project
from bioeng_qpcr.workbench.service import acquire, advance, patch_project


def expression_project():
    p = Project(
        samples=[
            dict(id=s, name=s, kind="unknown", group=s) for s in ("control", "treated")
        ],
        assays=[
            dict(
                id=a,
                name=a,
                reference=a != "target",
                chemistry="dye",
                channel="SYBR/FAM",
                forward="",
                reverse="",
                probe="",
                expected_tm=None,
            )
            for a in ("target", "ref1", "ref2")
        ],
    )
    observations = {"control": [25.0, 20.0, 21.0], "treated": [23.0, 20.5, 20.8]}
    measurements = []
    for sid, values in observations.items():
        for aid, value in zip(("target", "ref1", "ref2"), values, strict=True):
            for offset in (-0.01, 0.01):
                well = f"A{len(p.plate) + 1:02d}"
                p.plate.append(dict(well=well, sample_id=sid, assay_id=aid))
                measurements.append(
                    dict(well=well, assay_id=aid, instrument_cq=value + offset)
                )
    for step in ("samples", "design", "program"):
        advance(p, step)
    acquire(
        p, measurements, {"title": "Synthetic regression fixture"}, mode="file_import"
    )
    for step in ("curves", "qc"):
        advance(p, step)
    p.analysis_settings.update(
        target_id="target",
        reference_ids=["ref1", "ref2"],
        calibrator_id="control",
        efficiency_confirmed=True,
    )
    return p


def test_equal_efficiency_multiple_references_match_known_ddct():
    p = expression_project()
    advance(p, "analysis")
    treated = p.outputs["analysis"]["rows"][1]
    assert treated["delta_delta_cq"] == pytest.approx(-2.15)
    assert treated["fold_change"] == pytest.approx(2**2.15)


def test_measured_efficiencies_match_generalized_pfaffl_and_preserve_raw_run():
    p = expression_project()
    advance(p, "analysis")
    run = copy.deepcopy(p.active_run)
    settings = {
        **p.analysis_settings,
        "relative_method": "efficiency_corrected",
        "efficiency_confirmed": False,
        "efficiencies": {
            a: {"value": e, "source": "Synthetic test characterization"}
            for a, e in (("target", 0.9), ("ref1", 0.95), ("ref2", 1.02))
        },
    }
    patch_project(p, {"analysis_settings": settings})
    assert "qc" in p.completed and "analysis" not in p.completed
    assert p.active_run == run
    advance(p, "analysis")
    treated = p.outputs["analysis"]["rows"][1]
    expected = 1.9**2 / np.sqrt(1.95 ** (-0.5) * 2.02**0.2)
    assert treated["fold_change"] == pytest.approx(expected)
    assert "delta_delta_cq" not in treated
    assert p.outputs["analysis"]["settings"]["efficiencies"]["target"]["value"] == 0.9


def test_correction_requires_evidence_for_every_selected_assay():
    p = expression_project()
    p.analysis_settings.update(relative_method="efficiency_corrected", efficiencies={})
    with pytest.raises(ValueError, match="效率"):
        advance(p, "analysis")
    p.analysis_settings["reference_ids"] = ["ref1", "ref1"]
    with pytest.raises(ValueError, match="不重复"):
        advance(p, "analysis")


def test_batch_review_plots_fourth_compound_without_refitting(tmp_path, monkeypatch):
    app = create_app()
    app.config.update(TESTING=True, WORKBENCH_DB=tmp_path / "batch.sqlite3")
    client = app.test_client()
    rows = ["compound,concentration,response"]
    for i in range(5):
        for c in np.logspace(-2, 3, 12):
            rows.append(f"C{i},{c},{5 + 95 / (1 + c / (i + 1))}")
    # Keep a failed compound in the same audit/export as the valid fits.
    rows.extend(["bad,1,20", "bad,2,30"])
    payload = client.post(
        "/api/potency",
        data={
            "file": (io.BytesIO("\n".join(rows).encode()), "batch.csv"),
            "_archive": "1",
        },
    ).get_json()
    assert payload["ok"] and len(payload["results"]) == 6
    record_id = payload["record_id"]
    from bioeng_qpcr import potency

    monkeypatch.setattr(
        potency, "fit_dose_response", lambda *a, **k: pytest.fail("Refit")
    )
    response = client.get(f"/api/lab/records/{record_id}/potency/plots?compound=C4")
    assert response.status_code == 200, response.get_json()
    assert len(response.get_json()["plots"]) == 2
    assert response.get_json()["plots"][0]["downloads"]
    csv = client.get(f"/api/lab/records/{record_id}/potency/results.csv").get_data(
        as_text=True
    )
    assert "C4" in csv and "bad,failed" in csv
    assert (
        client.get(
            f"/api/lab/records/{record_id}/potency/plots?compound=missing"
        ).status_code
        == 404
    )
    backup = client.get(f"/api/lab/records/{record_id}/export").get_json()
    restored = client.post("/api/lab/records/import", json=backup).get_json()["record"]
    assert (
        client.get(
            f"/api/lab/records/{restored['id']}/potency/plots?compound=C4"
        ).status_code
        == 400
    )
