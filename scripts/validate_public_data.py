"""Reproduce the public-data validation reported in the project docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from bioeng_qpcr.calculations import delta_delta_cq
from bioeng_qpcr.domain import (
    Assay,
    Experiment,
    InputLevel,
    ReplicateType,
    Sample,
    Well,
)
from bioeng_qpcr.io import read_rdml
from bioeng_qpcr.plotting.style import apply_style, create_figure, save_figure
from bioeng_qpcr.potency import (
    CurveDirection,
    DoseResponseDataset,
    DoseResponseModel,
    fit_dose_response,
)
from bioeng_qpcr.provenance import ProvenanceManifest

SOURCES = {
    "ct1.csv": {
        "url": (
            "https://raw.githubusercontent.com/cran/pcr/"
            "6510657abfc6c92a4bb44ad3e131d10acb6ae6ec/inst/extdata/ct1.csv"
        ),
        "sha256": "d3dcd92d2991126cf5442f894a971477da75a7b07ed42659924700839ffebbf0",
    },
    "large_DNA_dilutions.rdml": {
        "url": (
            "https://raw.githubusercontent.com/RDML-consortium/rdmlpython/"
            "fce7a44e8b47b90759c646b557de568f7af77e75/experiments/"
            "untergasser/large_DNA_dilutions.rdml"
        ),
        "sha256": "2529de8272eb75c7579239d7735aa1251383118988e6fa4a7c7ef69fb97ef53d",
    },
    "inhibition.tsv": {
        "url": (
            "https://raw.githubusercontent.com/OpenADMET/Octant_CYP_blog_post/"
            "58b3d6a63d25b18bee0b68af07cf1a5b30e6cf0d/data/inhibition.tsv"
        ),
        "sha256": "19e537166a17a42dd50cc262dd6eb0a963c181830fdc52db0fba98533e01c9c6",
        "windows_crlf_sha256": (
            "c028684ad293c062f06a4f0f7fd051e4f831e12d1cc72ab20cfb14916d33968a"
        ),
    },
    "inhibition_wells.tsv": {
        "url": (
            "https://raw.githubusercontent.com/OpenADMET/Octant_CYP_blog_post/"
            "58b3d6a63d25b18bee0b68af07cf1a5b30e6cf0d/data/inhibition_wells.tsv"
        ),
        "sha256": "531a3dc8d175a9bf0be69cc8594ad310cd64e0997ed4fcd0da2067057737175c",
        "windows_crlf_sha256": (
            "c9a2390be19c2418923ec0c8fb9b655b23d5130dfea9be10ce1442d95c7cdd50"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(name: str, cache_dir: Path) -> Path:
    spec = SOURCES[name]
    path = cache_dir / name
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        spec["url"], headers={"User-Agent": "bioeng-qpcr-public-validation/1"}
    )
    pending = path.with_suffix(path.suffix + ".part")
    with (
        urllib.request.urlopen(request, timeout=90) as response,
        pending.open("wb") as target,
    ):
        while block := response.read(1024 * 1024):
            target.write(block)
    actual = _sha256(pending)
    if actual != spec["sha256"]:
        pending.unlink(missing_ok=True)
        raise ValueError(f"{name}: SHA-256 mismatch ({actual})")
    pending.replace(path)
    return path


def _verify_pinned(name: str, path: Path) -> Path:
    """Require a local dataset to match the same pinned artifact as downloads."""
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    actual = _sha256(path)
    if actual not in {
        SOURCES[name]["sha256"], SOURCES[name].get("windows_crlf_sha256"),
    }:
        raise ValueError(f"{name}: SHA-256 mismatch ({actual})")
    return path


def _public_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.pcr_root:
        ct1 = args.pcr_root / "inst" / "extdata" / "ct1.csv"
    else:
        ct1 = _fetch("ct1.csv", args.cache_dir)
    if args.rdml_root:
        rdml = (
            args.rdml_root / "experiments" / "untergasser" / "large_DNA_dilutions.rdml"
        )
    else:
        rdml = _fetch("large_DNA_dilutions.rdml", args.cache_dir)
    if args.openadmet_root:
        inhibition = args.openadmet_root / "data" / "inhibition.tsv"
        wells = args.openadmet_root / "data" / "inhibition_wells.tsv"
    else:
        inhibition = _fetch("inhibition.tsv", args.cache_dir)
        wells = _fetch("inhibition_wells.tsv", args.cache_dir)
    return {
        "ct1": _verify_pinned("ct1.csv", ct1),
        "rdml": _verify_pinned("large_DNA_dilutions.rdml", rdml),
        "inhibition": _verify_pinned("inhibition.tsv", inhibition),
        "wells": _verify_pinned("inhibition_wells.tsv", wells),
    }


def _qpcr_validation(path: Path) -> dict[str, object]:
    data = pd.read_csv(path)
    if list(data.columns) != ["c_myc", "GAPDH"] or len(data) != 12:
        raise ValueError("ct1.csv no longer matches the pinned 12-sample dataset")
    wells: list[Well] = []
    for row, values in data.iterrows():
        sample = "brain" if row < 6 else "kidney"
        for column, target in enumerate(("c_myc", "GAPDH"), start=1):
            wells.append(
                Well(
                    position=f"{row + 1:02d}:{column}",
                    sample_name=sample,
                    target_name=target,
                    cq=float(values[target]),
                    replicate_type=ReplicateType.TECHNICAL,
                    replicate_group=f"{sample}:{target}",
                )
            )
    experiment = Experiment(
        name="Livak ct1",
        wells=wells,
        samples={name: Sample(name=name) for name in ("brain", "kidney")},
        assays={
            "c_myc": Assay(name="c_myc"),
            "GAPDH": Assay(name="GAPDH", is_reference=True),
        },
        input_level=InputLevel.LEVEL_A,
    )
    result = delta_delta_cq(
        experiment,
        sample_name="kidney",
        target_name="c_myc",
        control_sample="brain",
        reference_target="GAPDH",
    )
    expected_fold = 5.6372830205680815
    return {
        "dataset": "CRAN pcr ct1 / Livak human brain-kidney example",
        "n_observations": 12,
        "delta_delta_cq": result.delta_delta_cq,
        "fold_change": result.fold_change,
        "expected_fold_change": expected_fold,
        "absolute_error": abs(result.fold_change - expected_fold),
        "passed": math.isclose(result.fold_change, expected_fold, rel_tol=1e-12),
    }


def _rdml_validation(path: Path) -> dict[str, object]:
    experiment = read_rdml(path)
    expected = {
        "rdml_version": "1.4",
        "n_wells": 294,
        "n_samples": 29,
        "n_assays": 6,
        "n_runs": 1,
        "n_traces": 294,
    }
    observed = {name: experiment.metadata[name] for name in expected}
    return {
        "dataset": "RDML Consortium Untergasser large DNA dilutions",
        "expected": expected,
        "observed": observed,
        "trace_points": int(
            sum(len(trace.cycles) for trace in experiment.traces.values())
        ),
        "passed": observed == expected,
    }


def _potency_validation(
    summary_path: Path, wells_path: Path
) -> tuple[dict, pd.DataFrame]:
    summary = pd.read_csv(summary_path, sep="\t")
    wells = pd.read_csv(wells_path, sep="\t")
    eligible = summary[
        summary["CYP3A4_pIC50"].notna()
        & summary["activity_status"].eq("YES")
        & summary["rollover_status"].eq("NO")
        & summary["saturation_status"].eq("YES")
        & summary["direction"].eq("DOWN")
        & summary["drc_qc_status"].eq("PASS")
        & summary["plate_qc_status"].eq("PASS")
    ]
    outlier = wells["outlier"].astype(str).str.lower().eq("true")
    groups = {name: frame for name, frame in wells.loc[~outlier].groupby("ocnt_batch")}
    records: list[dict[str, object]] = []
    design_excluded = 0
    for row in eligible.itertuples(index=False):
        points = groups.get(row.ocnt_batch)
        if points is None or len(points) < 8 or points["concentration_M"].nunique() < 8:
            design_excluded += 1
            records.append(
                {
                    "compound": row.ocnt_batch,
                    "status": "design_excluded",
                    "error": "Fewer than 8 observations/concentrations",
                }
            )
            continue
        dataset = DoseResponseDataset(
            compound=row.ocnt_batch,
            concentration=points["concentration_M"].to_numpy(),
            response=points["fluorescence_norm"].to_numpy(),
            concentration_unit="M",
            direction=CurveDirection.DECREASING,
        )
        try:
            result = fit_dose_response(
                dataset,
                models=[DoseResponseModel.FOUR_PL],
                allow_extrapolation=False,
            )
        except ValueError as exc:
            records.append(
                {"compound": row.ocnt_batch, "status": "fit_failed", "error": str(exc)}
            )
            continue
        estimate = -math.log10(result.relative_value)
        records.append(
            {
                "compound": row.ocnt_batch,
                "status": "fitted",
                "reference_pIC50": float(row.CYP3A4_pIC50),
                "bioeng_qpcr_pIC50": estimate,
                "difference": estimate - float(row.CYP3A4_pIC50),
                "absolute_error": abs(estimate - float(row.CYP3A4_pIC50)),
                "r_squared": result.best_fit.r_squared,
                "reference_ci_hit": bool(
                    row.CYP3A4_pIC50_ci_lower <= estimate <= row.CYP3A4_pIC50_ci_upper
                ),
                "bioeng_qpcr_ci_available": result.relative_ci is not None,
                "n_points": len(points),
            }
        )
    details = pd.DataFrame(records)
    valid = details.dropna(subset=["bioeng_qpcr_pIC50"])
    difference = valid["difference"].to_numpy(dtype=float)
    bias = float(np.mean(difference))
    difference_sd = float(np.std(difference, ddof=1))
    evaluated = len(details) - design_excluded
    metrics = {
        "dataset": "OpenADMET/Octant CYP3A4 inhibition",
        "eligible_curves": int(len(eligible)),
        "design_excluded": int(design_excluded),
        "evaluated_curves": int(evaluated),
        "successful_fits": int(len(valid)),
        "fit_success_rate": float(len(valid) / evaluated),
        "mae_pIC50": float(valid["absolute_error"].mean()),
        "rmse_pIC50": float(np.sqrt(np.mean(difference**2))),
        "median_absolute_error_pIC50": float(valid["absolute_error"].median()),
        "p90_absolute_error_pIC50": float(valid["absolute_error"].quantile(0.9)),
        "pearson_r": float(
            valid[["reference_pIC50", "bioeng_qpcr_pIC50"]].corr().iloc[0, 1]
        ),
        "spearman_rho": float(
            valid[["reference_pIC50", "bioeng_qpcr_pIC50"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "mean_bias_pIC50": bias,
        "bland_altman_lower": bias - 1.96 * difference_sd,
        "bland_altman_upper": bias + 1.96 * difference_sd,
        "median_r_squared": float(valid["r_squared"].median()),
        "reference_ci_hit_rate": float(valid["reference_ci_hit"].mean()),
        "bioeng_qpcr_ci_available_rate": float(
            valid["bioeng_qpcr_ci_available"].mean()
        ),
    }
    gates = {
        "fit_success_rate_at_least_0.99": metrics["fit_success_rate"] >= 0.99,
        "mae_pIC50_at_most_0.15": metrics["mae_pIC50"] <= 0.15,
        "pearson_r_at_least_0.90": metrics["pearson_r"] >= 0.90,
        "median_r_squared_at_least_0.95": metrics["median_r_squared"] >= 0.95,
    }
    metrics["release_gates"] = gates
    metrics["passed"] = all(gates.values())
    return metrics, details


def _plots(details: pd.DataFrame, output_dir: Path) -> list[str]:
    details = details.dropna(subset=["bioeng_qpcr_pIC50"])
    paths: list[str] = []
    reference = details["reference_pIC50"].to_numpy(dtype=float)
    estimate = details["bioeng_qpcr_pIC50"].to_numpy(dtype=float)
    difference = estimate - reference
    average = 0.5 * (estimate + reference)

    fig, ax = create_figure()
    ax.scatter(reference, estimate, s=20, alpha=0.65, color="#2b6f5e")
    limits = [
        min(reference.min(), estimate.min()),
        max(reference.max(), estimate.max()),
    ]
    ax.plot(limits, limits, "--", color="#b85c3e", label="identity")
    ax.set(
        xlabel="OpenADMET reference pIC50",
        ylabel="bioeng-qpcr pIC50",
        title="CYP3A4 potency agreement",
    )
    apply_style(ax)
    path = output_dir / "openadmet_pic50_agreement.svg"
    save_figure(fig, str(path))
    paths.append(path.name)

    fig, ax = create_figure()
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    ax.scatter(average, difference, s=20, alpha=0.65, color="#4c78a8")
    ax.axhline(bias, color="#2b6f5e", label=f"bias {bias:.3f}")
    ax.axhline(bias - 1.96 * sd, color="#b85c3e", linestyle="--")
    ax.axhline(bias + 1.96 * sd, color="#b85c3e", linestyle="--", label="95% limits")
    ax.set(
        xlabel="Mean pIC50",
        ylabel="bioeng-qpcr - reference",
        title="Bland-Altman agreement",
    )
    apply_style(ax)
    path = output_dir / "openadmet_pic50_bland_altman.svg"
    save_figure(fig, str(path))
    paths.append(path.name)

    fig, ax = create_figure()
    ax.hist(np.abs(difference), bins=25, color="#d19a2a", edgecolor="#fffdf9")
    ax.set(
        xlabel="Absolute pIC50 error",
        ylabel="Curves",
        title="Potency error distribution",
    )
    apply_style(ax)
    path = output_dir / "openadmet_pic50_error_distribution.svg"
    save_figure(fig, str(path))
    paths.append(path.name)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".validation-data"))
    parser.add_argument("--output-dir", type=Path, default=Path(".validation-results"))
    parser.add_argument("--pcr-root", type=Path)
    parser.add_argument("--rdml-root", type=Path)
    parser.add_argument("--openadmet-root", type=Path)
    args = parser.parse_args()
    paths = _public_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    potency, details = _potency_validation(paths["inhibition"], paths["wells"])
    report = {
        "provenance": ProvenanceManifest.capture_run(
            parameters={
                "model": "4PL",
                "minimum_concentrations": 8,
                "exclude_flagged_outlier_wells": True,
                "source_sha256": {key: item["sha256"] for key, item in SOURCES.items()},
            },
        ),
        "sources": SOURCES,
        "actual_input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "qpcr": _qpcr_validation(paths["ct1"]),
        "rdml": _rdml_validation(paths["rdml"]),
        "potency": potency,
    }
    report["passed"] = bool(
        report["qpcr"]["passed"] and report["rdml"]["passed"] and potency["passed"]
    )
    details.to_csv(args.output_dir / "openadmet_curve_results.csv", index=False)
    report["figures"] = _plots(details, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
