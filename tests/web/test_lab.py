"""The lab directory preserves every module and safely hands off desktop tasks."""

import pytest

from bioeng_qpcr.web.app import create_app


@pytest.fixture
def app():
    web = create_app()
    web.config["TESTING"] = True
    return web


def test_directory_matches_desktop_modules(app):
    client = app.test_client()
    catalog = client.get("/api/lab/modules").get_json()
    assert {entry["key"] for entry in catalog["modules"]} == {
        "qpcr",
        "enzyme",
        "ferment",
        "fba",
        "potency",
        "docking",
        "sequence",
        "preview",
    }
    assert catalog["desktop_available"] is False
    assert {entry["key"] for entry in catalog["primary_modules"]} == {
        "qpcr", "enzyme", "ferment", "fba", "potency",
    }
    assert {entry["key"] for entry in catalog["auxiliary_modules"]} == {
        "docking", "sequence", "preview",
    }
    page = client.get("/lab").get_data(as_text=True)
    for entry in catalog["modules"]:
        assert entry["name"] in page
        if entry["surface"] == "web":
            assert f'href="{entry["href"]}"' in page
            assert client.get(entry["href"].split("#")[0]).status_code == 200
    for path in ("/", "/workspace", "/workbench"):
        assert 'href="/lab"' in client.get(path).get_data(as_text=True)


def test_desktop_handoff_requires_local_callback(app):
    response = app.test_client().post("/api/lab/open-desktop/enzyme", json={})
    assert response.status_code == 409
    assert "Windows" in response.get_json()["error"]


def test_desktop_handoff_is_whitelisted_and_same_origin(app):
    opened = []
    app.config["DESKTOP_OPEN_MODULE"] = opened.append
    client = app.test_client()
    assert client.get("/api/lab/modules").get_json()["desktop_available"] is True
    assert (
        client.post(
            "/api/lab/open-desktop/enzyme",
            json={},
            headers={"Origin": "https://unrelated.example"},
        ).status_code
        == 403
    )
    assert client.post("/api/lab/open-desktop/enzyme", data="{}").status_code == 415
    assert client.post("/api/lab/open-desktop/qpcr", json={}).status_code == 404
    assert not opened
    for key in ("enzyme", "ferment", "fba"):
        response = client.post(
            f"/api/lab/open-desktop/{key}",
            json={},
            headers={"Origin": "http://localhost"},
        )
        assert response.status_code == 200
    assert opened == ["enzyme", "ferment", "fba"]
