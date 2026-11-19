"""Persistence and failures must not lose computed results."""

import base64
import io
import json
import sqlite3

import pytest

from bioeng_qpcr.web.app import create_app
from bioeng_qpcr.workbench.store import ProjectStore


@pytest.fixture
def app(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, WORKBENCH_DB=tmp_path / "local.sqlite3")
    return app


def analyze(client, **extra):
    return client.post(
        "/api/preview",
        data={
            "file": (
                io.BytesIO(b"position,sample_name,target_name,cq\nA1,S1,G,20\n"),
                "plate.csv",
            ),
            "_archive": "1",
            **extra,
        },
    )


def test_original_file_and_result_survive_restart_and_backup(app):
    client = app.test_client()
    payload = analyze(client).get_json()
    record_id = payload["record_id"]
    restarted = create_app()
    restarted.config.update(TESTING=True, WORKBENCH_DB=app.config["WORKBENCH_DB"])
    other = restarted.test_client()
    rows = other.get("/api/lab/records?module=preview_csv").get_json()["records"]
    assert rows[0]["id"] == record_id
    record = other.get(f"/api/lab/records/{record_id}").get_json()["record"]
    assert record["result"]["rows_total"] == payload["rows_total"]
    assert base64.b64decode(record["files"][0]["data"]).endswith(b"A1,S1,G,20\n")
    exported = other.get(f"/api/lab/records/{record_id}/export")
    assert "attachment" in exported.headers["Content-Disposition"]
    backup = json.loads(exported.data)
    assert backup["provenance"] == payload["provenance"]
    assert backup["provenance"]["package_versions"]["bioeng_qpcr"]
    restored = other.post("/api/lab/records/import", json=backup)
    assert restored.status_code == 201
    imported = restored.get_json()["record"]
    assert imported["id"] != record_id and imported["imported"] is True
    assert imported["result"] == backup["result"]
    assert imported["provenance"] == backup["provenance"]
    assert (
        len(other.get("/api/lab/records?module=preview_csv").get_json()["records"]) == 2
    )


def test_failed_archive_retains_result_and_reports_error(app, monkeypatch):
    def full_disk(*args):
        raise sqlite3.OperationalError("disk is full")

    monkeypatch.setattr(ProjectStore, "save_record", full_disk)
    response = analyze(app.test_client())
    assert response.status_code == 200
    assert response.get_json()["ok"]
    assert "尚未保存" in response.get_json()["archive_error"]
    assert "record_id" not in response.get_json()


def test_archive_is_opt_in_and_rejects_cross_origin(app):
    client = app.test_client()
    assert "record_id" not in analyze(client, _archive="0").get_json()
    response = client.post(
        "/api/preview",
        data={"_archive": "1"},
        headers={"Origin": "https://unrelated.example"},
    )
    assert response.status_code == 403
    assert client.get("/api/lab/records?module=preview_csv").get_json()["records"] == []
    assert client.get("/api/lab/records?module=unknown").status_code == 400
    assert client.get("/api/lab/records/missing").status_code == 404


def test_corrupt_and_invalid_backups_do_not_overwrite(app):
    client = app.test_client()
    record_id = analyze(client).get_json()["record_id"]
    backup = client.get(f"/api/lab/records/{record_id}").get_json()["record"]
    backup["files"][0]["sha256"] = "bad"
    assert client.post("/api/lab/records/import", json=backup).status_code == 400
    assert client.post("/api/lab/records/import", json=[]).status_code == 400
    assert (
        len(client.get("/api/lab/records?module=preview_csv").get_json()["records"])
        == 1
    )


def test_export_does_not_truncate_large_results(app):
    client = app.test_client()
    record_id = analyze(client).get_json()["record_id"]
    record = client.get(f"/api/lab/records/{record_id}").get_json()["record"]
    record["result"]["large"] = "x" * 300000
    imported = client.post("/api/lab/records/import", json=record).get_json()["record"]
    exported = client.get(f"/api/lab/records/{imported['id']}/export").get_json()
    assert len(exported["result"]["large"]) == 300000


def test_oversized_upload_keeps_413_instead_of_error_during_archiving(app):
    app.config["MAX_CONTENT_LENGTH"] = 100
    response = analyze(app.test_client())
    assert response.status_code == 413


def test_legacy_backup_does_not_invent_software_version(app):
    client = app.test_client()
    record_id = analyze(client).get_json()["record_id"]
    backup = client.get(f"/api/lab/records/{record_id}").get_json()["record"]
    backup.pop("provenance")
    backup["result"].pop("provenance")
    restored = client.post("/api/lab/records/import", json=backup).get_json()["record"]
    assert restored["provenance"] is None
    assert "provenance" not in restored["result"]
    assert restored["imported"] is True
