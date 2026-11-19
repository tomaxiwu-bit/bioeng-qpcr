"""Tests for the workspace page and the Phase 5 web API.

The existing :mod:`tests.web.test_web` covers the qPCR workflows; this module
covers the preview, potency, docking, alignment and image endpoints, plus the
format registry that the page's documentation pane is built from.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from bioeng_qpcr.web.app import create_app

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _upload(path: str | Path, name: str | None = None):
    """Return a Flask multipart file tuple for *path*."""
    path = Path(path)
    return (io.BytesIO(path.read_bytes()), name or path.name)


def _image_bytes(width: int = 40, height: int = 30, mode: str = "L"):
    """Return an in-memory PNG."""
    import numpy as np
    from PIL import Image

    array = np.linspace(0, 255, width * height).reshape(height, width)
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode=mode).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class TestWorkspacePage:
    def test_page_is_served(self, client):
        response = client.get("/workspace")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "分析工作台" in body
        # The five requested modules must all be present in the rail.
        for marker in ("数据表预览", "图片预览", "效价分析", "分子对接", "测序比对"):
            assert marker in body

    def test_page_references_every_endpoint(self, client):
        body = client.get("/workspace").get_data(as_text=True)
        for endpoint in (
            "/api/preview",
            "/api/image",
            "/api/potency",
            "/api/dock",
            "/api/align",
            "/api/schemas",
        ):
            assert endpoint in body

    @pytest.mark.parametrize(
        ("module", "roles"),
        [
            ("preview_image", {"file"}),
            ("potency", {"file"}),
            ("docking", {"receptor", "ligand"}),
            ("sequence", {"query", "reference"}),
        ],
    )
    def test_advanced_modules_have_loadable_examples(self, client, module, roles):
        response = client.get(f"/api/workspace/examples/{module}")
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["ok"]
        assert {entry["role"] for entry in payload["files"]} == roles
        assert all(base64.b64decode(entry["data"]) for entry in payload["files"])

    def test_unknown_workspace_example_is_a_404(self, client):
        assert client.get("/api/workspace/examples/nope").status_code == 404

    def test_image_workspace_example_is_renderable(self, client):
        entry = client.get("/api/workspace/examples/preview_image").get_json()["files"][
            0
        ]
        response = client.post(
            "/api/image",
            data={
                "file": (
                    io.BytesIO(base64.b64decode(entry["data"])),
                    entry["name"],
                )
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["rendered"]
        assert payload["data_uri"].startswith("data:image/png;base64,")
        assert [plot["title"] for plot in payload["plots"]] == [
            "原始像素预览",
            "处理后预览",
        ]

    def test_upload_size_is_bounded(self, client):
        assert client.application.config["MAX_CONTENT_LENGTH"] == 64 * 1024 * 1024

    def test_existing_index_links_to_the_workspace(self, client):
        body = client.get("/").get_data(as_text=True)
        assert 'href="/workspace"' in body


class TestSchemaApi:
    def test_lists_every_registered_format(self, client):
        from bioeng_qpcr.io import SCHEMAS

        payload = client.get("/api/schemas").get_json()
        assert payload["ok"]
        assert {entry["key"] for entry in payload["schemas"]} == set(SCHEMAS)

    def test_each_entry_carries_its_documentation(self, client):
        payload = client.get("/api/schemas").get_json()
        for entry in payload["schemas"]:
            assert entry["title"]
            assert entry["summary"]
            assert entry["header"]
            assert entry["columns"]
            assert "###" in entry["markdown"]
            for column in entry["columns"]:
                assert {"name", "type", "required", "description"} <= set(column)

    def test_template_download_sets_a_filename(self, client):
        response = client.get("/api/schemas/dose_response/template")
        assert response.status_code == 200
        assert "attachment" in response.headers["Content-Disposition"]
        body = response.get_data(as_text=True)
        assert body.startswith("compound,concentration,response")

    def test_unknown_format_is_a_404(self, client):
        assert client.get("/api/schemas/nope/template").status_code == 404

    def test_example_endpoint_returns_the_rows(self, client):
        payload = client.post(
            "/api/preview/example", data={"format": "qpcr_tidy"}
        ).get_json()
        assert payload["ok"]
        assert payload["format"] == "qpcr_tidy"
        assert payload["header"][:4] == ["position", "sample_name", "target_name", "cq"]
        assert payload["rows"]
        assert all(len(row) == len(payload["header"]) for row in payload["rows"])
        assert payload["csv"].startswith("position,sample_name,target_name,cq")

    def test_workspace_clears_files_and_makes_dropzones_keyboard_accessible(
        self, client
    ):
        body = client.get("/workspace").get_data(as_text=True)
        assert "let files = {}" in body
        assert 'role: "button", tabindex: "0"' in body
        assert "state.__files" not in body
        assert "复制结果 JSON" in body
        assert 'id="steps"' in body
        assert 'id="btnNext" disabled' in body
        assert "function updateReadiness()" in body

    def test_audit_status_distinguishes_warnings_from_success(self, client):
        body = client.get("/workspace").get_data(as_text=True)
        assert "function showAuditBanner(checks, action)" in body
        assert "无阻断错误" in body
        assert 'showAuditBanner(payload.validation, "比对完成")' in body
        assert "历史结果已恢复；旧版搜索判据需要重新运行复核。" in body

    def test_header_has_no_inert_navigation_and_empty_state_is_honest(self, client):
        body = client.get("/workspace").get_data(as_text=True)
        for label in ("导入规范", "模块文档", "命令行"):
            assert f"<span>{label}</span>" not in body
        assert "请上传文件开始分析" in body

    def test_example_endpoint_requires_a_format(self, client):
        response = client.post("/api/preview/example", data={})
        assert response.status_code == 400

    def test_example_endpoint_rejects_an_unknown_format(self, client):
        response = client.post("/api/preview/example", data={"format": "nope"})
        assert response.status_code == 404


class TestPreviewApi:
    def test_previews_a_tidy_plate(self, client):
        response = client.post(
            "/api/preview",
            data={"file": _upload(EXAMPLES / "qpcr_simple_plate_example.csv")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["ok"]
        assert payload["schema"] == "qpcr_tidy"
        assert payload["rows_total"] == 64
        assert payload["errors"] == 0
        assert len(payload["columns"]) == 10
        assert payload["header"][0] == "position"
        assert payload["rows"]

    def test_reports_structural_problems_with_positions(self, client):
        body = b"position,sample_name,target_name,cq\nA1,S,G,not-a-number\n"
        response = client.post(
            "/api/preview",
            data={"file": (io.BytesIO(body), "bad.csv")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["errors"] >= 1
        issue = next(i for i in payload["issues"] if i["code"] == "INVALID_NUMBER")
        assert issue["column"] == "cq"
        assert issue["row"] == 2

    def test_format_can_be_forced(self, client):
        body = b"position,sample_name,target_name,cq\nA1,S,G,18.5\n"
        response = client.post(
            "/api/preview",
            data={"file": (io.BytesIO(body), "x.csv"), "format": "qpcr_tidy"},
            content_type="multipart/form-data",
        )
        assert response.get_json()["schema"] == "qpcr_tidy"

    def test_wrong_suffix_is_rejected(self, client):
        response = client.post(
            "/api/preview",
            data={"file": (io.BytesIO(b"x"), "image.png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.get_json()["error"]

    def test_missing_file_is_rejected(self, client):
        response = client.post("/api/preview", data={})
        assert response.status_code == 400


class TestImageApi:
    def test_reports_metadata_and_a_thumbnail(self, client):
        response = client.post(
            "/api/image",
            data={"file": (_image_bytes(), "gel.png")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["ok"]
        assert payload["info"]["format"] == "PNG"
        assert payload["info"]["width"] == 40
        assert payload["rendered"]
        assert payload["data_uri"].startswith("data:image/png;base64,")
        assert payload["palette"] == "gray"

    def test_render_can_be_skipped(self, client):
        response = client.post(
            "/api/image",
            data={"file": (_image_bytes(), "gel.png"), "render": "0"},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["info"]["format"] == "PNG"
        assert not payload["rendered"]
        assert "data_uri" not in payload

    def test_palette_and_enhancements_are_echoed(self, client):
        response = client.post(
            "/api/image",
            data={
                "file": (_image_bytes(), "gel.png"),
                "palette": "fire",
                "invert": "1",
                "gamma": "1.5",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["palette"] == "fire"
        assert any("invert" in item for item in payload["enhancements"])
        assert any("gamma" in item for item in payload["enhancements"])

    def test_unknown_palette_is_rejected(self, client):
        response = client.post(
            "/api/image",
            data={"file": (_image_bytes(), "gel.png"), "palette": "rainbow"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_format_listing_is_available(self, client):
        payload = client.get("/api/image/formats").get_json()
        assert payload["ok"]
        for expected in ("PNG", "JPEG", "TIFF", "SVG"):
            assert expected in payload["formats"]


class TestPotencyApi:
    def test_rejects_excessive_bootstrap_work_before_fitting(self, client):
        response = client.post(
            "/api/potency",
            data={"n_bootstrap": "5001"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "n_bootstrap must be between" in response.get_json()["error"]

    def test_fits_every_compound_and_runs_the_rules(self, client):
        response = client.post(
            "/api/potency",
            data={"file": _upload(EXAMPLES / "dose_response_example.csv")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert response.status_code == 200
        assert "Infinity" not in response.get_data(as_text=True)
        assert payload["ok"]
        assert [entry["compound"] for entry in payload["results"]] == [
            "CompoundA",
            "CompoundB",
        ]
        for entry in payload["results"]:
            assert entry["kind"] == "IC50"
            potency = entry["relative_value"]
            assert 2.0 < potency < 5.0 or 40.0 < potency < 50.0
            assert entry["absolute_value"] is not None
            assert len(entry["validation"]) == 11
            assert entry["model_selection"]["models"]

    def test_plot_is_returned(self, client):
        response = client.post(
            "/api/potency",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "publication": "1",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["plot"].startswith("data:image/png;base64,")
        assert len(payload["plots"]) >= 3
        assert any("残差诊断" in plot["title"] for plot in payload["plots"])
        for plot in payload["plots"]:
            assert plot["publication"]["width_in"] == pytest.approx(7.2)
            assert plot["publication"]["raster_dpi"] == 300
            assert [item["format"] for item in plot["downloads"]] == [
                "svg",
                "pdf",
                "png",
            ]
        downloads = payload["plots"][0]["downloads"]
        decoded = {
            item["format"]: base64.b64decode(item["data_uri"].split(",", 1)[1])
            for item in downloads
        }
        assert b"<svg" in decoded["svg"][:500]
        assert decoded["pdf"].startswith(b"%PDF")
        assert decoded["png"].startswith(b"\x89PNG\r\n\x1a\n")

    def test_validation_can_be_skipped(self, client):
        response = client.post(
            "/api/potency",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "validate": "0",
            },
            content_type="multipart/form-data",
        )
        assert all(not entry["validation"] for entry in response.get_json()["results"])

    def test_kind_can_be_forced(self, client):
        response = client.post(
            "/api/potency",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "kind": "EC50",
            },
            content_type="multipart/form-data",
        )
        assert all(entry["kind"] == "EC50" for entry in response.get_json()["results"])

    def test_explicit_control_response_is_used(self, client):
        response = client.post(
            "/api/potency",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "control_response": "123.5",
            },
            content_type="multipart/form-data",
        )
        assert all(
            entry["control_response"] == pytest.approx(123.5)
            for entry in response.get_json()["results"]
        )

    def test_unfittable_compound_is_reported_not_fatal(self, client):
        body = b"compound,concentration,response\nA,1,90\nA,2,80\n"
        response = client.post(
            "/api/potency",
            data={"file": (io.BytesIO(body), "short.csv")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["results"][0].get("error")

    def test_comparison_returns_a_fold_difference(self, client):
        response = client.post(
            "/api/potency/compare",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "compounds": "CompoundA,CompoundB",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["ok"]
        assert payload["fold_difference"] == pytest.approx(14.15, rel=0.1)
        assert payload["ci"][0] < payload["ratio"] < payload["ci"][1]

    def test_comparison_needs_two_names(self, client):
        response = client.post(
            "/api/potency/compare",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "compounds": "CompoundA",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_comparison_rejects_unknown_names(self, client):
        response = client.post(
            "/api/potency/compare",
            data={
                "file": _upload(EXAMPLES / "dose_response_example.csv"),
                "compounds": "CompoundA,Nope",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "Nope" in response.get_json()["error"]


class TestDockingApi:
    @pytest.mark.parametrize(
        ("field", "value"), [("n_runs", "51"), ("n_steps", "20001")]
    )
    def test_rejects_excessive_work_before_parsing_uploads(self, client, field, value):
        response = client.post(
            "/api/dock",
            data={field: value},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert f"{field} must be between" in response.get_json()["error"]

    def test_rejects_excessive_combined_docking_work(self, client):
        response = client.post(
            "/api/dock",
            data={"n_runs": "50", "n_steps": "5000"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "evaluation HTTP budget" in response.get_json()["error"]

    @pytest.mark.parametrize(
        ("limit_name", "message"),
        [
            ("_MAX_DOCKING_ATOM_PAIRS", "atom pairs exceed"),
            ("_MAX_DOCKING_PAIR_EVALUATIONS", "atom-pair evaluations exceed"),
        ],
    )
    def test_rejects_structures_that_exceed_pairwise_budgets(
        self, client, monkeypatch, limit_name, message
    ):
        import bioeng_qpcr.web.app as web_app

        monkeypatch.setattr(web_app, limit_name, 1)
        response = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "50",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert message in response.get_json()["error"]

    def test_rejects_receptor_atom_limit_while_parsing(self, client, monkeypatch):
        import bioeng_qpcr.web.app as web_app

        monkeypatch.setattr(web_app, "_MAX_DOCKING_RECEPTOR_ATOMS", 1)
        response = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "50",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "more than 1 atoms" in response.get_json()["error"]

    def test_grid_budget_uses_all_513_placements(self, client, monkeypatch):
        import bioeng_qpcr.web.app as web_app

        monkeypatch.setattr(web_app, "_MAX_DOCKING_PAIR_EVALUATIONS", 100_000)
        response = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "50",
                "use_grid": "1",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "atom-pair evaluations exceed" in response.get_json()["error"]

    def test_docks_and_returns_ranked_poses(self, client):
        response = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "150",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["ok"]
        assert payload["poses"]
        scores = [pose["score"] for pose in payload["poses"]]
        assert scores == sorted(scores)
        assert payload["poses"][0]["rmsd_to_best"] == 0.0
        assert (
            all(pose["rmsd_to_best"] > 1.9 for pose in payload["poses"][1:])
            or len(payload["poses"]) == 1
        )

    def test_breakdown_sums_to_the_total(self, client):
        response = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "120",
            },
            content_type="multipart/form-data",
        )
        top = response.get_json()["poses"][0]
        parts = (
            top["vdw"]
            + top["electrostatic"]
            + top["hbond"]
            + top["hydrophobic"]
            + top["desolvation"]
        )
        assert top["score"] == pytest.approx(parts, abs=1e-9)

    def test_audit_rules_are_reported(self, client):
        response = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "120",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["validation"]
        assert all(
            {"rule_id", "passed", "severity", "message", "evidence"} <= set(check)
            for check in payload["validation"]
        )
        assert [plot["title"] for plot in payload["plots"]] == [
            "姿势排名与评分分解",
            "最佳姿势评分项",
        ]
        assert payload["review"]["version"] == 1
        assert (
            payload["review"]["status"] != "exploratory"
            or payload["stats"]["converged"]
        )

    def test_skipping_audit_keeps_result_explicitly_unaudited(self, client):
        payload = client.post(
            "/api/dock",
            data={
                "receptor": _upload(EXAMPLES / "docking_receptor.pdb"),
                "ligand": _upload(EXAMPLES / "docking_ligand.pdb"),
                "n_runs": "1",
                "n_steps": "50",
                "validate": "0",
            },
        ).get_json()
        assert payload["ok"] and payload["validation"] == []
        assert payload["review"]["status"] == "not_audited"
        assert payload["review"]["next_steps"]

    def test_a_ligand_file_is_required_unless_splitting(self, client):
        response = client.post(
            "/api/dock",
            data={"receptor": _upload(EXAMPLES / "docking_receptor.pdb")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400


class TestAlignApi:
    def test_aligns_a_read_and_reports_rules(self, client):
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_reads.fastq"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["ok"]
        assert payload["alignment"]["identity"] == pytest.approx(1.0)
        assert payload["alignment"]["ref_end"] == 120
        assert payload["alignment"]["cigar"] == "120M"
        assert payload["validation"]
        assert payload["query_id"]

    def test_variants_are_reported_for_the_mutated_read(self, client):
        # Read 1 of the example set carries a substitution.
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_reads.fastq"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
                "read_index": "1",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["variants"], "expected the SNP-carrying read to be called"
        variant = payload["variants"][0]
        assert {"type", "ref_position", "ref_allele", "alt_allele"} <= set(variant)

    def test_quality_plot_is_returned_for_fastq(self, client):
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_reads.fastq"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["plot"].startswith("data:image/png;base64,")
        assert [plot["title"] for plot in payload["plots"]] == [
            "参考坐标覆盖与变异",
            "变异位置图",
            "逐碱基质量分布",
        ]

    def test_plain_sequence_still_returns_two_figures(self, client):
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_read_plain.txt"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
            },
            content_type="multipart/form-data",
        )
        assert [plot["title"] for plot in response.get_json()["plots"]] == [
            "参考坐标覆盖与变异",
            "变异位置图",
        ]

    @pytest.mark.parametrize("index", ["-1", "99"])
    def test_read_index_out_of_range_is_an_error(self, client, index):
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_reads.fastq"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
                "read_index": index,
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "out of range" in response.get_json()["error"]

    def test_quality_trimming_needs_quality_values(self, client):
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_read_plain.txt"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
                "trim_quality": "1",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert "FASTQ" in response.get_json()["error"]

    def test_primer_trimming_is_recorded(self, client):
        response = client.post(
            "/api/align",
            data={
                "query": _upload(EXAMPLES / "sequencing_reads.fastq"),
                "reference": _upload(EXAMPLES / "sequencing_reference.fasta"),
                "primer_5": "ATGCGTACAG",
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        assert payload["notes"]
        assert "引物裁剪" in payload["notes"][0]
