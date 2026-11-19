from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _validation_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_public_data.py"
    spec = importlib.util.spec_from_file_location("validate_public_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_public_dataset_must_match_pinned_hash(tmp_path):
    module = _validation_module()
    changed = tmp_path / "ct1.csv"
    changed.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._verify_pinned("ct1.csv", changed)


def test_only_explicitly_pinned_line_endings_are_accepted(tmp_path, monkeypatch):
    import hashlib

    module = _validation_module()
    lf = b"a\tb\n1\t2\n"
    crlf = lf.replace(b"\n", b"\r\n")
    monkeypatch.setitem(module.SOURCES, "sample", {
        "sha256": hashlib.sha256(lf).hexdigest(),
        "windows_crlf_sha256": hashlib.sha256(crlf).hexdigest(),
    })
    path = tmp_path / "sample.tsv"
    for raw in (lf, crlf):
        path.write_bytes(raw)
        assert module._verify_pinned("sample", path) == path
    path.write_bytes(lf.replace(b"2", b"3"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._verify_pinned("sample", path)


def test_benchmark_keeps_exclusions_and_failures_without_inflating_success(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import pandas as pd

    module = _validation_module()
    summary = pd.DataFrame(
        [
            {
                "ocnt_batch": name,
                "CYP3A4_pIC50": 6 + i,
                "activity_status": "YES",
                "rollover_status": "NO",
                "saturation_status": "YES",
                "direction": "DOWN",
                "drc_qc_status": "PASS",
                "plate_qc_status": "PASS",
                "CYP3A4_pIC50_ci_lower": 5 + i,
                "CYP3A4_pIC50_ci_upper": 7 + i,
            }
            for i, name in enumerate(("A", "B", "C", "D"))
        ]
    )
    wells = pd.DataFrame(
        [
            {
                "ocnt_batch": name,
                "concentration_M": 10 ** (-j),
                "fluorescence_norm": j,
                "outlier": False,
            }
            for name in ("A", "B", "C", "D")
            for j in range(7 if name == "D" else 8)
        ]
    )
    summary_path, wells_path = tmp_path / "summary.tsv", tmp_path / "wells.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    wells.to_csv(wells_path, sep="\t", index=False)

    def fit(dataset, **kwargs):
        if dataset.compound == "B":
            raise ValueError("unidentified plateau")
        return SimpleNamespace(
            relative_value=1e-6 if dataset.compound == "A" else 1e-8,
            best_fit=SimpleNamespace(r_squared=0.99),
            relative_ci=None,
        )

    monkeypatch.setattr(module, "fit_dose_response", fit)
    metrics, details = module._potency_validation(summary_path, wells_path)
    assert metrics["evaluated_curves"] == 3
    assert metrics["successful_fits"] == 2
    assert metrics["fit_success_rate"] == pytest.approx(2 / 3)
    assert set(details["status"]) == {"fitted", "fit_failed", "design_excluded"}
    assert (
        details.loc[details["compound"] == "B", "error"].item()
        == "unidentified plateau"
    )
