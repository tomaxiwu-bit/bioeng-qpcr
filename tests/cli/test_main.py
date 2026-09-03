"""Comprehensive tests for the bioeng-qpcr command-line interface.

Every test uses :class:`click.testing.CliRunner` together with
``isolated_filesystem()`` so that temporary CSV files and output artefacts are
created in a throwaway directory and cleaned up automatically.

Test dataset (known Cq values)::

    position,sample_name,target_name,cq
    A01,S1,GAPDH,18.5
    A02,S1,ACTB,20.3
    A03,S2,GAPDH,19.0
    A04,S2,ACTB,20.5
    A05,NTC,GAPDH,Undetermined

With ``target=ACTB``, ``control=S1``, ``reference=GAPDH``:

* S1 (control): ΔCq = 20.3 − 18.5 = 1.8 → ΔΔCq = 0 → fold change = 1.0
* S2:           ΔCq = 20.5 − 19.0 = 1.5 → ΔΔCq = −0.3 → fold change ≈ 1.231
"""

from __future__ import annotations

import csv as csv_module
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioeng_qpcr.cli.main import cli

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

#: Valid tidy CSV with two samples, two targets, one NTC well and
#: amplification efficiencies (1.0 = 100 %) for every assay.  The
#: ``efficiency`` column ensures the EFF-001 validation rule passes, and
#: ``--reference-targets GAPDH`` (passed by the tests) satisfies REF-001,
#: so all currently implemented readiness rules pass.
SAMPLE_CSV = (
    "position,sample_name,target_name,cq,efficiency\n"
    "A01,S1,GAPDH,18.5,1.0\n"
    "A02,S1,ACTB,20.3,1.0\n"
    "A03,S2,GAPDH,19.0,1.0\n"
    "A04,S2,ACTB,20.5,1.0\n"
    "A05,NTC,GAPDH,Undetermined,1.0\n"
)

#: CSV missing the required ``cq`` column -- read_tidy_csv raises ValueError.
INVALID_CSV = "position,sample_name,target_name\nA01,S1,GAPDH\nA02,S1,ACTB\n"

#: Expected fold-change values for the dataset above.
EXPECTED_S1_FOLD = 1.0
EXPECTED_S2_FOLD = 2.0**0.3  # ≈ 1.2311
RESEARCHER_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "researcher_example.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """A fresh :class:`CliRunner` for each test."""
    return CliRunner()


def _write_csv(filename: str, content: str) -> str:
    """Write *content* to *filename* in the current (isolated) directory."""
    Path(filename).write_text(content, encoding="utf-8")
    return filename


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for the ``validate`` sub-command."""

    def test_validate_exit_code_zero(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli, ["validate", csv_path, "--reference-targets", "GAPDH"]
            )
            assert result.exit_code == 0

    def test_validate_json_output(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "validate",
                    csv_path,
                    "--output",
                    "json",
                    "--reference-targets",
                    "GAPDH",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "summary" in data
            assert "overall_status" in data
            assert "compliant" in data
            assert "results" in data
            assert isinstance(data["results"], list)

    def test_validate_text_output(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "validate",
                    csv_path,
                    "--output",
                    "text",
                    "--reference-targets",
                    "GAPDH",
                ],
            )
            assert result.exit_code == 0
            text = result.stdout
            assert "PASS" in text
            assert "WARN" in text
            assert "FAIL" in text
            assert "MIQE READINESS ASSESSMENT" in text

    def test_validate_output_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            out_path = "report.txt"
            result = runner.invoke(
                cli,
                [
                    "validate",
                    csv_path,
                    "--output",
                    "text",
                    "--output-file",
                    out_path,
                    "--reference-targets",
                    "GAPDH",
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            file_content = Path(out_path).read_text(encoding="utf-8")
            assert "MIQE READINESS ASSESSMENT" in file_content

    def test_validate_output_file_json(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            out_path = "report.json"
            result = runner.invoke(
                cli,
                [
                    "validate",
                    csv_path,
                    "--output",
                    "json",
                    "--output-file",
                    out_path,
                    "--reference-targets",
                    "GAPDH",
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            assert "overall_status" in data

    def test_validate_reference_targets_marks_genes(self, runner: CliRunner) -> None:
        """--reference-targets changes the experiment content hash.

        Marking different genes as reference flips ``is_reference`` on the
        corresponding assay, which is part of the content hash.  The two
        hashes must therefore differ.  Both runs are compliant because the
        CSV includes efficiency data and each run designates a reference
        gene.
        """
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)

            result_gapdh = runner.invoke(
                cli,
                [
                    "validate",
                    csv_path,
                    "--output",
                    "json",
                    "--reference-targets",
                    "GAPDH",
                ],
            )
            result_actb = runner.invoke(
                cli,
                [
                    "validate",
                    csv_path,
                    "--output",
                    "json",
                    "--reference-targets",
                    "ACTB",
                ],
            )
            assert result_gapdh.exit_code == 0
            assert result_actb.exit_code == 0

            hash_gapdh = json.loads(result_gapdh.stdout)["experiment_hash"]
            hash_actb = json.loads(result_actb.stdout)["experiment_hash"]
            assert hash_gapdh != hash_actb

    def test_validate_invalid_csv_nonzero_exit(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("bad.csv", INVALID_CSV)
            result = runner.invoke(cli, ["validate", csv_path])
            assert result.exit_code != 0

    def test_validate_nonexistent_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["validate", "missing.csv"])
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for the ``analyze`` sub-command."""

    def test_analyze_delta_delta_cq_json(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    csv_path,
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                    "--method",
                    "delta_delta_cq",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["parameters"]["method"] == "delta_delta_cq"
            assert len(data["results"]) == 2

            by_sample = {row["sample_name"]: row for row in data["results"]}
            assert "S1" in by_sample
            assert "S2" in by_sample

            assert by_sample["S1"]["fold_change"] == pytest.approx(EXPECTED_S1_FOLD)
            assert by_sample["S2"]["fold_change"] == pytest.approx(EXPECTED_S2_FOLD)
            assert by_sample["S1"]["delta_delta_cq"] == pytest.approx(0.0)
            assert by_sample["S2"]["control_sample"] == "S1"
            assert by_sample["S2"]["reference_target"] == "GAPDH"

    def test_analyze_csv_output(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    csv_path,
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                    "--output",
                    "csv",
                    "--method",
                    "delta_delta_cq",
                ],
            )
            assert result.exit_code == 0
            # Parse the CSV output.
            reader = csv_module.DictReader(result.stdout.strip().splitlines())
            rows = list(reader)
            assert len(rows) == 2
            assert "sample_name" in reader.fieldnames
            assert "fold_change" in reader.fieldnames
            by_sample = {r["sample_name"]: r for r in rows}
            assert float(by_sample["S1"]["fold_change"]) == pytest.approx(
                EXPECTED_S1_FOLD
            )

    def test_analyze_output_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            out_path = "results.json"
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    csv_path,
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                    "--output-file",
                    out_path,
                    "--method",
                    "delta_delta_cq",
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            assert len(data["results"]) == 2

    def test_analyze_pfaffl(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    csv_path,
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                    "--method",
                    "pfaffl",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["parameters"]["method"] == "pfaffl"
            assert len(data["results"]) == 2

            by_sample = {row["sample_name"]: row for row in data["results"]}
            # With default efficiency (1.0 = 100 %) Pfaffl reduces to ΔΔCq.
            assert by_sample["S1"]["ratio"] == pytest.approx(1.0)
            assert by_sample["S2"]["ratio"] == pytest.approx(EXPECTED_S2_FOLD)

    def test_analyze_default_method_is_normalized_quantity(
        self, runner: CliRunner
    ) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    csv_path,
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["parameters"]["method"] == "normalized_quantity"
            assert "normalized_quantity" in data["results"][0]
            assert "fold_change" not in data["results"][0]

    def test_analyze_no_reference_raises_error(self, runner: CliRunner) -> None:
        """Without --reference-target and no is_reference column, analysis
        fails because no reference gene can be resolved."""
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                ["analyze", csv_path, "--target", "ACTB", "--control", "S1"],
            )
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for the ``provenance`` sub-command."""

    def test_provenance_json(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(cli, ["provenance", csv_path])
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "input_hash" in data
            assert data["input_hash"] is not None
            assert len(data["input_hash"]) == 64  # SHA-256 hex
            assert "package_versions" in data
            assert isinstance(data["package_versions"], dict)
            assert "created_at" in data
            assert "parameters" in data

    def test_provenance_parameters(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "provenance",
                    csv_path,
                    "--parameters",
                    "method=delta_delta_cq",
                    "--parameters",
                    "target=ACTB",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["parameters"]["method"] == "delta_delta_cq"
            assert data["parameters"]["target"] == "ACTB"


class TestResearchWorkflow:
    def test_compare_groups(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                [
                    "compare",
                    str(RESEARCHER_EXAMPLE),
                    "--target",
                    "IL6",
                    "--reference-targets",
                    "GAPDH,RPLP0",
                    "--treated-group",
                    "treated",
                    "--control-group",
                    "control",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["comparison"]["fold_difference"] > 1.0
            assert data["analysis_audit"]["summary"]["complete"] is True

    def test_generate_combined_report(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                [
                    "report",
                    str(RESEARCHER_EXAMPLE),
                    "--target",
                    "IL6",
                    "--reference-targets",
                    "GAPDH,RPLP0",
                    "--output-file",
                    "report.html",
                    "--json-file",
                    "report.json",
                ],
            )
            assert result.exit_code == 0
            assert "Research report saved" in result.stdout
            assert "Alpha research software" in Path("report.html").read_text(
                encoding="utf-8"
            )
            payload = json.loads(Path("report.json").read_text(encoding="utf-8"))
            assert payload["analysis"]["summary"]["exclusion_count"] == 0

    def test_provenance_output_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            out_path = "manifest.json"
            result = runner.invoke(
                cli,
                ["provenance", csv_path, "--output-file", out_path],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            assert "input_hash" in data
            assert "package_versions" in data

    def test_provenance_input_hash_matches_experiment(self, runner: CliRunner) -> None:
        """The manifest's input_hash must match the experiment content hash."""
        from bioeng_qpcr.io import read_tidy_csv

        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(cli, ["provenance", csv_path])
            assert result.exit_code == 0
            data = json.loads(result.stdout)

            experiment = read_tidy_csv(csv_path, reference_targets=None)
            assert data["input_hash"] == experiment.content_hash()


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


class TestPlot:
    """Tests for the ``plot`` sub-command."""

    def test_plot_fold_change(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            out_path = "fold_change.png"
            result = runner.invoke(
                cli,
                [
                    "plot",
                    csv_path,
                    "--type",
                    "fold_change",
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                    "--output-file",
                    out_path,
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_plot_fold_change_default_output(self, runner: CliRunner) -> None:
        """When --output-file is omitted the default ``plot.png`` is used."""
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "plot",
                    csv_path,
                    "--type",
                    "fold_change",
                    "--target",
                    "ACTB",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists("plot.png")

    def test_plot_heatmap(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            out_path = "heatmap.png"
            result = runner.invoke(
                cli,
                [
                    "plot",
                    csv_path,
                    "--type",
                    "heatmap",
                    "--output-file",
                    out_path,
                ],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_plot_fold_change_missing_target(self, runner: CliRunner) -> None:
        """fold_change without --target exits non-zero."""
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "plot",
                    csv_path,
                    "--type",
                    "fold_change",
                    "--control",
                    "S1",
                    "--reference-target",
                    "GAPDH",
                ],
            )
            assert result.exit_code != 0

    def test_plot_fold_change_missing_control(self, runner: CliRunner) -> None:
        """fold_change without --control exits non-zero."""
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli,
                [
                    "plot",
                    csv_path,
                    "--type",
                    "fold_change",
                    "--target",
                    "ACTB",
                    "--reference-target",
                    "GAPDH",
                ],
            )
            assert result.exit_code != 0

    def test_plot_missing_type_option(self, runner: CliRunner) -> None:
        """--type is required; omitting it produces a click usage error."""
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(cli, ["plot", csv_path])
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


class TestHelp:
    """``--help`` must work for the group and every sub-command."""

    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["validate", "--help"],
            ["analyze", "--help"],
            ["provenance", "--help"],
            ["plot", "--help"],
        ],
    )
    def test_help(self, runner: CliRunner, args: list[str]) -> None:
        result = runner.invoke(cli, args)
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_help_lists_subcommands(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "validate" in result.output
        assert "analyze" in result.output
        assert "provenance" in result.output
        assert "plot" in result.output

    def test_version_option(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.5.1" in result.output


# ---------------------------------------------------------------------------
# Integration: exit-code semantics
# ---------------------------------------------------------------------------


class TestExitCodes:
    """Verify the documented exit-code contract."""

    def test_compliant_returns_zero(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(
                cli, ["validate", csv_path, "--reference-targets", "GAPDH"]
            )
            assert result.exit_code == 0

    def test_usage_error_returns_two(self, runner: CliRunner) -> None:
        """Click usage errors (missing required option) exit with code 2."""
        with runner.isolated_filesystem():
            # analyze requires --target and --control.
            result = runner.invoke(cli, ["analyze", "data.csv"])
            assert result.exit_code == 2


# ---------------------------------------------------------------------------
# RDML input
# ---------------------------------------------------------------------------


class TestRdmlInput:
    """Every ingesting sub-command accepts RDML files."""

    RDML_EXAMPLE = (
        Path(__file__).resolve().parents[2] / "examples" / "rdml_example.xml"
    )

    def test_validate_rdml_xml(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "validate",
                str(self.RDML_EXAMPLE),
                "--reference-targets",
                "GAPDH,RPLP0",
            ],
        )
        assert result.exit_code == 0

    def test_analyze_rdml_matches_csv(self, runner: CliRunner) -> None:
        args = [
            "--target",
            "IL6",
            "--control",
            "Control_1",
            "--reference-target",
            "GAPDH,RPLP0",
        ]
        with runner.isolated_filesystem():
            csv_result = runner.invoke(
                cli, ["analyze", str(RESEARCHER_EXAMPLE), *args]
            )
            rdml_result = runner.invoke(
                cli, ["analyze", str(self.RDML_EXAMPLE), *args]
            )

        assert csv_result.exit_code == 0
        assert rdml_result.exit_code == 0
        csv_by_sample = {
            row["sample_name"]: row
            for row in json.loads(csv_result.stdout)["results"]
        }
        rdml_by_sample = {
            row["sample_name"]: row
            for row in json.loads(rdml_result.stdout)["results"]
        }
        assert set(csv_by_sample) == set(rdml_by_sample)
        for sample, row in csv_by_sample.items():
            assert rdml_by_sample[sample]["normalized_quantity"] == pytest.approx(
                row["normalized_quantity"]
            )

    def test_validate_rdml_zip_archive(self, runner: CliRunner) -> None:
        import zipfile

        with runner.isolated_filesystem():
            archive_path = "example.rdml"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "rdml_data.xml",
                    self.RDML_EXAMPLE.read_text(encoding="utf-8"),
                )
            result = runner.invoke(
                cli,
                [
                    "validate",
                    archive_path,
                    "--reference-targets",
                    "GAPDH,RPLP0",
                ],
            )
        assert result.exit_code == 0

    def test_non_rdml_xml_reports_error(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            xml_path = _write_csv("not_rdml.xml", "<html><body/></html>")
            result = runner.invoke(cli, ["validate", xml_path])

        assert result.exit_code == 1
        assert "rdml" in result.output.lower()


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------

#: Technical-replicate groups: one tight group of six, one clean pair.
OUTLIER_CSV = (
    "position,sample_name,target_name,cq,efficiency,is_reference,"
    "replicate_type,replicate_group\n"
    "A01,S1,GAPDH,20.01,1.0,true,technical,S1_GAPDH\n"
    "A02,S1,GAPDH,20.02,1.0,true,technical,S1_GAPDH\n"
    "A03,S1,GAPDH,20.00,1.0,true,technical,S1_GAPDH\n"
    "A04,S1,GAPDH,20.03,1.0,true,technical,S1_GAPDH\n"
    "A05,S1,GAPDH,20.02,1.0,true,technical,S1_GAPDH\n"
    "A06,S1,GAPDH,21.30,1.0,true,technical,S1_GAPDH\n"
    "B01,S2,GAPDH,19.90,1.0,true,technical,S2_GAPDH\n"
    "B02,S2,GAPDH,19.91,1.0,true,technical,S2_GAPDH\n"
)


class TestOutliers:
    """Tests for the ``outliers`` sub-command."""

    def test_outliers_text_reports_flag(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", OUTLIER_CSV)
            result = runner.invoke(cli, ["outliers", csv_path])

        assert result.exit_code == 0
        assert "A06" in result.output
        assert "Chauvenet-rejected" in result.output
        assert "Groups passing: 1" in result.output

    def test_outliers_json_output(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", OUTLIER_CSV)
            result = runner.invoke(
                cli, ["outliers", csv_path, "--output", "json"]
            )

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["summary"]["total_flagged_wells"] == 1
        assert data["summary"]["flagged_positions"] == ["A06"]
        flagged = data["groups"][0]["chauvenet_flags"][0]
        assert flagged["position"] == "A06"
        assert flagged["tail_probability"] < flagged["threshold_probability"]

    def test_outliers_exit_zero_when_clean(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", SAMPLE_CSV)
            result = runner.invoke(cli, ["outliers", csv_path])

        assert result.exit_code == 0
        assert "No Chauvenet flags" in result.output

    def test_outliers_rdml_input(self, runner: CliRunner) -> None:
        rdml_example = (
            Path(__file__).resolve().parents[2] / "examples" / "rdml_example.xml"
        )
        result = runner.invoke(cli, ["outliers", str(rdml_example)])

        assert result.exit_code == 0
        assert "group(s) assessed" in result.output

    def test_outliers_min_group_size_option(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", OUTLIER_CSV)
            result = runner.invoke(
                cli,
                [
                    "outliers",
                    csv_path,
                    "--min-group-size",
                    "3",
                    "--sd-trigger",
                    "0.0",
                ],
            )

        assert result.exit_code == 0
        assert "Review recommended" in result.output

    def test_outliers_output_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            csv_path = _write_csv("data.csv", OUTLIER_CSV)
            out_path = "outliers.json"
            result = runner.invoke(
                cli,
                ["outliers", csv_path, "--output", "json", "--output-file", out_path],
            )
            assert result.exit_code == 0
            assert os.path.exists(out_path)
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            assert data["summary"]["flagged_positions"] == ["A06"]
