"""A figure failure must be visible without discarding calculated numbers."""

import io
from pathlib import Path

import pytest

from bioeng_qpcr.web.app import create_app

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("endpoint", "plot_name", "files", "parameters", "result_key"),
    [
        (
            "potency",
            "plot_dose_response",
            {"file": "dose_response_example.csv"},
            {},
            "results",
        ),
        (
            "dock",
            "plot_pose_ranking",
            {
                "receptor": "docking_receptor.pdb",
                "ligand": "docking_ligand.pdb",
            },
            {"n_runs": "1", "n_steps": "50"},
            "poses",
        ),
        (
            "align",
            "plot_alignment_coverage",
            {
                "query": "sequencing_reads.fastq",
                "reference": "sequencing_reference.fasta",
            },
            {},
            "alignment",
        ),
    ],
)
def test_plot_failure_is_logged_and_archived_with_a_warning(
    tmp_path,
    monkeypatch,
    caplog,
    endpoint,
    plot_name,
    files,
    parameters,
    result_key,
):
    import bioeng_qpcr.plotting as plotting

    def broken(*args, **kwargs):
        raise RuntimeError("test rendering failure")

    monkeypatch.setattr(plotting, plot_name, broken)
    app = create_app()
    app.config.update(TESTING=True, WORKBENCH_DB=tmp_path / "plots.sqlite3")
    client = app.test_client()
    response = client.post(
        f"/api/{endpoint}",
        data={
            **{
                role: (io.BytesIO((EXAMPLES / name).read_bytes()), name)
                for role, name in files.items()
            },
            **parameters,
            "_archive": "1",
        },
    )
    data = response.get_json()
    assert response.status_code == 200, data
    assert data["ok"] and data[result_key]
    assert data["plot_warnings"] and "数值结果已保留" in data["plot_warnings"][0]
    assert "test rendering failure" in caplog.text
    archived = client.get(f"/api/lab/records/{data['record_id']}").get_json()["record"]
    assert archived["result"]["plot_warnings"] == data["plot_warnings"]
