"""The CLI must preserve the same docking semantics as the browser."""

import json
from pathlib import Path

from click.testing import CliRunner

from bioeng_qpcr.cli.main import cli

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_docking_json_has_real_elements_and_explicit_unaudited_review():
    result = CliRunner().invoke(
        cli,
        [
            "dock",
            str(EXAMPLES / "docking_receptor.pdb"),
            str(EXAMPLES / "docking_ligand.pdb"),
            "--n-runs",
            "1",
            "--n-steps",
            "50",
            "--output",
            "json",
            "--no-validation",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "LIG" not in data["ligand"]["elements"]
    assert {"C", "N", "O"} <= set(data["ligand"]["elements"])
    assert data["ligand"]["residue_names"] == ["LIG"]
    assert data["review"]["status"] == "not_audited"


def test_grid_text_never_claims_global_convergence():
    result = CliRunner().invoke(
        cli,
        [
            "dock",
            str(EXAMPLES / "docking_receptor.pdb"),
            str(EXAMPLES / "docking_ligand.pdb"),
            "--grid",
            "--grid-translation-steps",
            "2",
            "--grid-rotation-samples",
            "2",
        ],
    )
    assert result.exit_code in (0, 1), result.output
    assert "网格枚举完成" in result.output
    assert "收敛: 是" not in result.output
    assert "复核建议" in result.output
