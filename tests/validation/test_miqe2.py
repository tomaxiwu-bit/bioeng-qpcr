"""Tests for the MIQE 2.0 (2025) reporting checklist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioeng_qpcr.domain import InputLevel, ReplicateType
from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.io import read_tidy_csv
from bioeng_qpcr.validation import (
    MIQE2_CATEGORIES,
    MIQE2_FIELDS,
    Miqe2Report,
    miqe2_checklist,
)

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


def _minimal_experiment(rdml_version: str | None = "1.3", method=None) -> Experiment:
    wells = [
        Well(
            position="A01",
            sample_name="S1",
            target_name="GAPDH",
            cq=18.0,
            replicate_type=ReplicateType.SINGLE,
        ),
        Well(
            position="A02",
            sample_name="S1",
            target_name="IL6",
            cq=21.0,
        ),
        Well(
            position="A03",
            sample_name="NTC",
            target_name="IL6",
            cq_status="no_template",
        ),
    ]
    samples = {"S1": Sample("S1"), "NTC": Sample("NTC")}
    assays = {
        "GAPDH": Assay(name="GAPDH", is_reference=True, efficiency=0.97),
        "IL6": Assay(
            name="IL6", efficiency=0.96, primer_forward="ACGT", primer_reverse="TGCA"
        ),
    }
    metadata = {}
    if rdml_version:
        metadata["rdml_version"] = rdml_version
        metadata["source_file"] = "exp.rdml"
    return Experiment(
        name="minimal",
        wells=wells,
        samples=samples,
        assays=assays,
        input_level=InputLevel.LEVEL_C,
        method=method,
        metadata=metadata,
    )


class TestCatalog:
    def test_five_categories(self):
        assert [cat for cat, _ in MIQE2_CATEGORIES] == [
            "reagent",
            "sample",
            "rt",
            "protocol",
            "analysis",
        ]

    def test_field_ids_unique(self):
        ids = [f[0] for f in MIQE2_FIELDS]
        assert len(ids) == len(set(ids))

    def test_all_fields_reference_valid_category(self):
        valid = {cat for cat, _ in MIQE2_CATEGORIES}
        for field_id, category, *_ in MIQE2_FIELDS:
            assert category in valid, field_id

    def test_catalog_is_substantial(self):
        assert len(MIQE2_FIELDS) >= 40


class TestAssessment:
    def test_manual_checklist_empty_experiment(self):
        report = miqe2_checklist()

        assert isinstance(report, Miqe2Report)
        assert len(report) == len(MIQE2_FIELDS)
        assert report.summary["reported"] == 0
        # with no experiment and no answers nothing can be reported
        assert report.summary["missing"] + report.summary["na"] == len(MIQE2_FIELDS)

    def test_essential_fields_default_to_missing(self):
        report = miqe2_checklist(_minimal_experiment(rdml_version=None))
        essential_missing = [
            i for i in report.items if i.essential and i.status == "missing"
        ]
        # instrument, efficiency-only items, etc. require data
        assert any(i.field_id == "protocol_instrument" for i in essential_missing)
        assert all(i.essential for i in essential_missing)

    def test_rdml_source_auto_reports_raw_data(self):
        report = miqe2_checklist(_minimal_experiment())
        raw = next(i for i in report.items if i.field_id == "analysis_raw_data")
        assert raw.status == "reported"
        assert "RDML" in raw.evidence

    def test_reference_genes_auto_report_normalization(self):
        report = miqe2_checklist(_minimal_experiment())
        for field in ("analysis_normalization", "analysis_reference_validation"):
            item = next(i for i in report.items if i.field_id == field)
            assert item.status == "reported"
            assert "GAPDH" in item.evidence

    def test_primer_sequences_auto_report(self):
        report = miqe2_checklist(_minimal_experiment())
        item = next(i for i in report.items if i.field_id == "reagent_sequences")
        assert item.status == "reported"
        assert "IL6" in item.evidence

    def test_answers_override(self):
        report = miqe2_checklist(
            _minimal_experiment(),
            answers={
                "protocol_kit": True,
                "analysis_nc_frequency": False,
            },
        )
        kit = next(i for i in report.items if i.field_id == "protocol_kit")
        ncf = next(i for i in report.items if i.field_id == "analysis_nc_frequency")
        assert kit.status == "reported"
        # non-essential field answered False + no experiment evidence -> NA
        assert ncf.status == "na"

    def test_essential_answered_false_is_missing(self):
        report = miqe2_checklist(_minimal_experiment(), answers={"protocol_kit": False})
        kit = next(i for i in report.items if i.field_id == "protocol_kit")
        assert kit.status == "missing"


class TestReportOutput:
    def test_essential_completeness(self):
        report = miqe2_checklist(_minimal_experiment())
        essential_total = sum(
            1
            for i in report.items
            if i.essential and i.status != "na"
        )
        reported = sum(
            1 for i in report.items if i.essential and i.status == "reported"
        )
        assert report.essential_completeness == pytest.approx(
            reported / essential_total
        )

    def test_to_dict_roundtrip(self):
        report = miqe2_checklist(_minimal_experiment())
        payload = report.to_dict()
        assert payload["standard"] == "MIQE 2.0 (2025)"
        assert payload["experiment_name"] == "minimal"
        assert len(payload["items"]) == len(MIQE2_FIELDS)

    def test_to_json_parses(self):
        data = json.loads(miqe2_checklist().to_json())
        assert "items" in data

    def test_to_text_mentions_categories(self):
        text = miqe2_checklist(_minimal_experiment()).to_text()
        assert "Reagent Preparation" in text
        assert "essential completeness" in text.lower()

    def test_real_example_report(self):
        experiment = read_tidy_csv(
            EXAMPLES / "researcher_example.csv",
            reference_targets=["GAPDH", "RPLP0"],
        )
        report = miqe2_checklist(experiment)
        reported = {i.field_id for i in report.reported}
        assert "reagent_gene_symbols" in reported
        assert "sample_types_quantity" in reported
        # example carries efficiency and NTC wells
        assert "analysis_efficiency" in reported
        assert "analysis_nc_frequency" in reported