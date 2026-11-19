"""End-to-end tests for the Flask web application.

These tests exercise the same three workflows the browser uses (validate,
analyze, curves) over the real IO / calculation / plotting layers via
Flask's built-in test client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

from flask import Flask  # noqa: E402

from bioeng_qpcr.web import create_app  # noqa: E402

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_ANALYZE_CSV = _EXAMPLES / "researcher_example.csv"
_CURVES_CSV = _EXAMPLES / "amplification_curves_example.csv"


def _upload(client, url: str, path: Path, **fields) -> dict:
    """Post a file plus form fields to *url* and return the JSON body."""
    with path.open("rb") as fh:
        data = {"file": (fh, path.name)}
        data.update(fields)
        response = client.post(
            url, data=data, content_type="multipart/form-data"
        )
    return response.get_json()


@pytest.fixture(scope="module")
def client() -> Flask.test_client_class:
    return create_app().test_client()


# ---------------------------------------------------------------------------
# Metadata & landing page
# ---------------------------------------------------------------------------


class TestIndex:
    def test_landing_page_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"bioeng-qpcr" in response.data
        assert b"MIQE" in response.data

    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "ok"
        assert "version" in body


@pytest.fixture
def local_client(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, WORKBENCH_DB=str(tmp_path / "security.sqlite3"))
    return app.test_client()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1:8008", "[::1]:8008"])
def test_loopback_hosts_and_matching_origins_are_accepted(local_client, host):
    origin = f"http://{host}"
    assert local_client.get("/api/health", base_url=origin).status_code == 200
    response = local_client.post(
        "/api/validate", base_url=origin, headers={"Origin": origin}
    )
    assert response.status_code == 400
    assert "No file" in response.get_json()["error"]


def test_untrusted_host_cannot_read_or_create_projects(local_client):
    origin = "http://untrusted.example"
    assert local_client.get("/api/qpcr/projects", base_url=origin).status_code == 400
    assert local_client.post(
        "/api/qpcr/projects", base_url=origin,
        headers={"Origin": origin}, json={"name": "untrusted"},
    ).status_code == 400


@pytest.mark.parametrize(
    "origin", ["http://untrusted.example", "https://localhost", "null"]
)
@pytest.mark.parametrize(
    "endpoint", ["/api/validate", "/api/enzyme", "/api/qpcr/projects"]
)
def test_all_analysis_writes_reject_foreign_origins(local_client, endpoint, origin):
    response = local_client.post(endpoint, headers={"Origin": origin}, json={})
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# /api/validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_tidy_csv(self, client):
        body = _upload(client, "/api/validate", _ANALYZE_CSV)
        assert body["ok"] is True
        assert body["n_failures"] >= 0
        assert "text" in body
        assert "json" in body
        # A heatmap should be generated for the valid experiment.
        assert body["heatmap"].startswith("data:image/png;base64,")

    def test_missing_file_rejected(self, client):
        response = client.post(
            "/api/validate", data={"strict": "true"}, content_type="multipart/form-data"
        )
        assert response.status_code == 400
        assert response.get_json()["ok"] is False

    def test_unsupported_extension_rejected(self, client, tmp_path):
        bogus = tmp_path / "data.zip"
        bogus.write_bytes(b"not a csv")
        with bogus.open("rb") as fh:
            response = client.post(
                "/api/validate",
                data={"file": (fh, "data.zip")},
                content_type="multipart/form-data",
            )
        assert response.status_code == 400
        assert "Unsupported" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# /api/analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_delta_delta_cq(self, client):
        body = _upload(
            client,
            "/api/analyze",
            _ANALYZE_CSV,
            method="delta_delta_cq",
            target="IL6",
            control="Control_1",
            reference_targets="GAPDH",
        )
        assert body["ok"] is True
        assert len(body["results"]) > 0
        assert body["fold_change"].startswith("data:image/png;base64,")
        assert body["heatmap"].startswith("data:image/png;base64,")

    def test_normalized_quantity(self, client):
        body = _upload(
            client,
            "/api/analyze",
            _ANALYZE_CSV,
            method="normalized_quantity",
            target="IL6",
            reference_targets="GAPDH",
        )
        assert body["ok"] is True
        assert len(body["results"]) > 0

    def test_compare_groups(self, client):
        body = _upload(
            client,
            "/api/analyze",
            _ANALYZE_CSV,
            method="compare",
            target="IL6",
            treated_group="treated",
            control_group="control",
            reference_targets="GAPDH",
        )
        assert body["ok"] is True

    def test_requires_target(self, client):
        body = _upload(
            client,
            "/api/analyze",
            _ANALYZE_CSV,
            method="delta_delta_cq",
        )
        assert body["ok"] is False
        assert "target" in body["error"]


# ---------------------------------------------------------------------------
# /api/curves
# ---------------------------------------------------------------------------


class TestCurves:
    def test_fits_amplification_curves(self, client):
        body = _upload(
            client,
            "/api/curves",
            _CURVES_CSV,
            baseline_method="constant",
        )
        assert body["ok"] is True
        assert len(body["per_well"]) == 5  # A01,A02,B01,B02,C01
        assert len(body["per_assay"]) >= 2  # IL6 + GAPDH
        assert body["plot"].startswith("data:image/png;base64,")
        assert "C01" in body["non_amplifying"]

    def test_bad_baseline_method(self, client):
        body = _upload(client, "/api/curves", _CURVES_CSV, baseline_method="bogus")
        assert body["ok"] is False

    def test_malformed_trace_csv_not_silently_downgraded(self, client, tmp_path):
        # has required trace columns but broken cycle structure -> must fail
        # loudly, not be reinterpreted as a tidy table
        bad = tmp_path / "bad_traces.csv"
        bad.write_text(
            "position,cycle,fluorescence,target_name\n"
            "A01,1,0.5,GAPDH\n"
            "A01,1,0.9,GAPDH\n",
            encoding="utf-8",
        )
        with bad.open("rb") as fh:
            response = client.post(
                "/api/curves",
                data={"file": (fh, "bad_traces.csv")},
                content_type="multipart/form-data",
            )
        body = response.get_json()
        assert body["ok"] is False
        assert "cycle" in body["error"]
