"""History preserves exact inputs and scientific qualifications across surfaces."""

import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from bioeng_qpcr.web.app import create_app
from bioeng_qpcr.web.kinetics import finite_json

EXAMPLES = Path(__file__).resolve().parents[2] / "src/bioeng_qpcr/web/examples"
FILES = {
    "enzyme": "inhibition_example.csv",
    "ferment": "fermentation_example.csv",
    "fba": "e_coli_core.xml",
}


@pytest.fixture
def app(tmp_path):
    web = create_app()
    web.config.update(TESTING=True, WORKBENCH_DB=tmp_path / "history.sqlite3")
    return web


def analyze(client, module, **parameters):
    if module == "fba":
        pytest.importorskip("cobra")
    name = FILES[module]
    return client.post(
        f"/api/{module}",
        data={
            "file": (io.BytesIO((EXAMPLES / name).read_bytes()), name),
            "_archive": "1",
            **parameters,
        },
    )


@pytest.mark.parametrize("module", FILES)
def test_archive_restore_and_export_preserve_original_input(app, module):
    response = analyze(app.test_client(), module)
    assert response.status_code == 200, response.get_data(as_text=True)
    result = response.get_json()
    assert result["ok"] and result["rows"] and result["plots"]
    json.dumps(result, allow_nan=False)
    assert {p["format"] for p in result["plots"][0]["downloads"]} == {
        "svg",
        "pdf",
        "png",
    }
    record_id = result["record_id"]
    restarted = create_app()
    restarted.config.update(TESTING=True, WORKBENCH_DB=app.config["WORKBENCH_DB"])
    client = restarted.test_client()
    record = client.get(f"/api/lab/records/{record_id}").get_json()["record"]
    assert record["module"] == module and not record["imported"]
    assert (
        base64.b64decode(record["files"][0]["data"])
        == (EXAMPLES / FILES[module]).read_bytes()
    )
    assert (
        client.get("/api/lab/records?module=all").get_json()["records"][0]["id"]
        == record_id
    )
    exported = client.get(f"/api/lab/records/{record_id}/results.csv")
    csv_rows = list(csv.DictReader(io.StringIO(exported.data.decode("utf-8-sig"))))
    assert len(csv_rows) == len(result["rows"])
    backup = client.get(f"/api/lab/records/{record_id}/export").get_json()
    imported = client.post("/api/lab/records/import", json=backup).get_json()["record"]
    assert imported["imported"] and imported["id"] != record_id
    assert imported["result"] == record["result"]
    assert (
        client.get(f"/api/lab/records/{imported['id']}/results.csv").status_code == 400
    )
    assert client.get("/history").status_code == 200


def test_enzyme_keeps_ambiguous_mechanism_and_exact_zero_control(app):
    client = app.test_client()
    result = analyze(client, "enzyme").get_json()
    assert len(result["details"]["models"]) == 5
    assert result["metrics"][1][1] == "未决"
    assert result["details"]["best_fit"]["parameters"]["ki"] == pytest.approx(
        5.198, rel=0.001
    )
    assert any("标准误" in note for note in result["warnings"])
    result = analyze(client, "enzyme", mode="mm").get_json()
    assert len(result["rows"]) == 9
    source = (
        (EXAMPLES / FILES["enzyme"])
        .read_text(encoding="utf-8-sig")
        .replace(",0.0\n", ",0.00001\n")
    )
    response = client.post(
        "/api/enzyme",
        data={"file": (io.BytesIO(source.encode()), "rates.csv"), "mode": "mm"},
    )
    assert response.status_code == 400
    assert "[I] = 0" in response.get_json()["error"]


def test_monod_preserves_residuals_and_identifiability(app):
    payload = analyze(app.test_client(), "ferment").get_json()
    assert payload["details"]["fit"]["mu_max"] > 0
    assert len(payload["rows"]) == payload["details"]["fit"]["n_points"]
    for row in payload["rows"]:
        assert row["biomass_residual"] == pytest.approx(
            row["biomass"] - row["biomass_fit"]
        )
        assert row["substrate_residual"] == pytest.approx(
            row["substrate"] - row["substrate_fit"]
        )
    assert len(payload["details"]["identifiability"]["correlation_matrix"]) == 3


def test_fba_scenario_does_not_mutate_baseline_or_original_file(app):
    client = app.test_client()
    scenario = analyze(client, "fba", bounds="EX_o2_e,0,1000", action="fba").get_json()
    assert scenario["ok"]
    details = scenario["details"]
    assert details["baseline"]["objective_value"] == pytest.approx(0.8739215, rel=1e-5)
    assert details["scenario"]["objective_value"] == pytest.approx(0.21166295, rel=1e-5)
    assert details["scenario"]["fluxes"]["EX_o2_e"] == 0
    assert any(abs(row["difference"]) > 0.1 for row in scenario["rows"])
    again = analyze(client, "fba").get_json()
    assert (
        again["details"]["baseline"]["objective_value"]
        == details["baseline"]["objective_value"]
    )
    record = client.get(f"/api/lab/records/{scenario['record_id']}").get_json()[
        "record"
    ]
    assert record["parameters"]["bounds"] == "EX_o2_e,0,1000"


def test_infeasible_scenario_does_not_present_solver_pseudofluxes(app):
    client = app.test_client()
    payload = analyze(client, "fba", bounds="ATPM,1000,1000").get_json()
    assert payload["details"]["scenario"]["status"] == "infeasible"
    assert all(row["scenario_flux"] is None for row in payload["rows"])
    assert all(row["difference"] is None for row in payload["rows"])
    exported = client.get(f"/api/lab/records/{payload['record_id']}/results.csv")
    rows = list(csv.DictReader(io.StringIO(exported.data.decode("utf-8-sig"))))
    assert rows and all(row["scenario_status"] == "infeasible" for row in rows)
    assert all(row["scenario_flux"] == "" for row in rows)


@pytest.mark.parametrize(
    "bounds", ["MISSING,0,1", "EX_o2_e,nan,1", "EX_o2_e,2,1", "EX_o2_e,0,1;EX_o2_e,0,2"]
)
def test_invalid_scenarios_are_not_archived(app, bounds):
    client = app.test_client()
    assert analyze(client, "fba", bounds=bounds).status_code == 400
    assert not client.get("/api/lab/records?module=fba").get_json()["records"]


@pytest.mark.parametrize("action", ["blocked", "dead_ends"])
def test_network_checks_are_archivable(app, action):
    response = analyze(app.test_client(), "fba", action=action)
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["ok"] and payload["record_id"] and payload["plots"] == []
    assert payload["metrics"][0][1] > 0


@pytest.mark.parametrize("module", FILES)
def test_foreign_archive_requests_are_rejected(app, module):
    response = app.test_client().post(
        f"/api/{module}",
        data={"_archive": "1"},
        headers={"Origin": "https://unrelated.example"},
    )
    assert response.status_code == 403


def test_example_downloads_and_nonfinite_values(app):
    client = app.test_client()
    for module, name in FILES.items():
        assert client.get(f"/api/workspace/examples/{module}").get_json()["note"]
        response = client.get(f"/api/workspace/examples/{module}/download")
        assert response.data == (EXAMPLES / name).read_bytes()
        assert "attachment" in response.headers["Content-Disposition"]
    clean = finite_json(
        {
            "ci": np.nan,
            "condition": np.inf,
            "array": np.array([1.0, np.nan]),
            "flag": np.bool_(True),
        }
    )
    assert json.loads(json.dumps(clean, allow_nan=False)) == {
        "ci": None,
        "condition": None,
        "array": [1.0, None],
        "flag": True,
    }
