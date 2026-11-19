"""Run every documented example workflow end to end.

Usage: python scripts/verify_examples.py

Verifies that the datasets in ``examples/`` pass their own validation
layers and that the workflows described in ``docs/RESEARCHER_GUIDE.md``
and ``README.md`` still run. Requires the ``fba`` extra (COBRApy).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def verify_qpcr() -> None:
    from bioeng_qpcr.cli.main import cli

    csv_path = str(EXAMPLES / "researcher_example.csv")
    reference = "GAPDH,RPLP0"
    for argv in (
        ["validate", csv_path, "--reference-targets", reference, "--output", "text"],
        ["analyze", csv_path, "--target", "IL6", "--reference-targets", reference],
        ["compare", csv_path, "--target", "IL6", "--reference-targets", reference,
         "--treated-group", "treated", "--control-group", "control"],
    ):
        runner = cli.main(args=argv, standalone_mode=False)
        if getattr(runner, "exit_code", 0) != 0:
            raise SystemExit(f"qPCR workflow failed: {' '.join(argv)}")
    print("qPCR CLI workflows OK")


def verify_rdml() -> None:
    from bioeng_qpcr.cli.main import cli
    from bioeng_qpcr.io import read_rdml, read_tidy_csv

    rdml_path = str(EXAMPLES / "rdml_example.xml")
    for argv in (
        ["validate", rdml_path, "--reference-targets", "GAPDH,RPLP0"],
        ["analyze", rdml_path, "--target", "IL6", "--reference-targets",
         "GAPDH,RPLP0"],
    ):
        runner = cli.main(args=argv, standalone_mode=False)
        if getattr(runner, "exit_code", 0) != 0:
            raise SystemExit(f"RDML workflow failed: {' '.join(argv)}")

    csv_experiment = read_tidy_csv(
        EXAMPLES / "researcher_example.csv",
        reference_targets=["GAPDH", "RPLP0"],
    )
    rdml_experiment = read_rdml(EXAMPLES / "rdml_example.xml")
    if len(csv_experiment.wells) != len(rdml_experiment.wells):
        raise SystemExit("RDML example lost wells versus the CSV example")
    print("RDML workflows OK")


def verify_outliers() -> None:
    from bioeng_qpcr.calculations import detect_replicate_outliers
    from bioeng_qpcr.io import read_tidy_csv

    experiment = read_tidy_csv(
        EXAMPLES / "researcher_example.csv",
        reference_targets=["GAPDH", "RPLP0"],
    )
    result = detect_replicate_outliers(experiment)
    if result.n_groups == 0:
        raise SystemExit("no technical replicate groups found in example data")
    if result.total_flagged or result.review_groups:
        raise SystemExit(
            f"clean example data unexpectedly flagged: {result.summary()}"
        )
    print(f"Replicate outlier screening OK: {result.summary()}")


def verify_enzyme_kinetics() -> None:
    import numpy as np

    from bioeng_qpcr.enzyme_kinetics import (
        classify_inhibition,
        compare_inhibition_models,
        fit_michaelis_menten,
        read_kinetics_csv,
        validate_km_range,
        validate_substrate_span,
    )

    path = EXAMPLES / "enzyme_kinetics_example.csv"
    substrate, velocity, _ = read_kinetics_csv(path)
    result = fit_michaelis_menten(substrate, velocity)
    if not validate_km_range(result.km).passed:
        raise SystemExit(f"Km out of accepted range: {result.km}")
    if not validate_substrate_span(substrate, result.km).passed:
        raise SystemExit("substrate span does not bracket Km")
    print(f"Michaelis-Menten OK: Vmax={result.vmax:.3f}, Km={result.km:.3f}")

    groups: dict[float, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    inhibition_path = EXAMPLES / "inhibition_example.csv"
    with open(inhibition_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            inhibitor = float(row["inhibitor_concentration"])
            groups[inhibitor][0].append(float(row["substrate_concentration"]))
            groups[inhibitor][1].append(float(row["velocity"]))
    datasets = [
        (inhibitor, np.array(substrates), np.array(velocities))
        for inhibitor, (substrates, velocities) in sorted(groups.items())
    ]
    inhibition = classify_inhibition(datasets)
    print(f"Inhibition classification OK: {inhibition.inhibition_type}")

    comparison = compare_inhibition_models(datasets)
    best = comparison.best_fit
    print(
        f"Global inhibition model OK: {comparison.best_type.value} "
        f"(AICc={best.aicc:.2f}, w={best.akaike_weight:.3f})"
    )


def verify_reference_gene_family() -> None:
    from bioeng_qpcr.io import read_tidy_csv
    from bioeng_qpcr.reference_genes import (
        bestkeeper,
        consensus_ranking,
        delta_ct,
    )

    experiment = read_tidy_csv(
        EXAMPLES / "researcher_example.csv",
        reference_targets=["GAPDH", "RPLP0"],
    )
    candidates = ["GAPDH", "RPLP0"]
    delta = delta_ct(experiment, candidates)
    keeper = bestkeeper(experiment, candidates)
    cross = consensus_ranking(
        experiment, candidates,
        methods=("delta_ct", "bestkeeper", "genorm"),
    )
    if len(delta.ranking) != 2 or len(keeper.ranking) != 2:
        raise SystemExit("reference-gene family did not rank all candidates")
    if len(cross.ranking) != 2:
        raise SystemExit("consensus aggregation did not unify the family")
    print(f"delta-Ct OK: {delta.ranking}")
    print(f"BestKeeper OK: {keeper.ranking}")
    print(f"Consensus(cross-family) OK: {cross.ranking}")


def verify_fermentation() -> None:
    from bioeng_qpcr.fermentation_kinetics import (
        analyze_identifiability,
        fit_monod,
        monod_profile_likelihood_ci,
        read_fermentation_csv,
        validate_data_sufficiency,
        validate_mu_max_range,
    )

    time, biomass, substrate = read_fermentation_csv(
        EXAMPLES / "fermentation_example.csv"
    )
    result = fit_monod(time, biomass, substrate)
    if not validate_mu_max_range(result.mu_max).passed:
        raise SystemExit(f"mu_max out of accepted range: {result.mu_max}")
    if not validate_data_sufficiency(time, biomass, substrate).passed:
        raise SystemExit("fermentation data insufficient for fitting")
    identifiability = analyze_identifiability(result, time, biomass, substrate)
    print(f"Monod fit OK: mu_max={result.mu_max:.4f}, Ks={result.ks:.4f}")
    print(f"Identifiability OK: {identifiability}")

    profile = monod_profile_likelihood_ci(
        time, biomass, substrate,
        mu_max=result.mu_max, ks=result.ks, yxs=result.yxs,
    )
    if profile.mu_max_ci is None:
        raise SystemExit("mu_max profile-likelihood interval failed to invert")
    lo, hi = profile.mu_max_ci
    if not lo <= result.mu_max <= hi:
        raise SystemExit("mu_max outside its own profile-likelihood interval")
    print(
        f"Profile likelihood OK: mu_max 95% CI = [{lo:.4f}, {hi:.4f}], "
        f"chi2_crit={profile.chi2_critical:.2f}"
    )


def verify_fba() -> None:
    from bioeng_qpcr.metabolic_flux import (
        find_blocked_reactions,
        load_model,
        run_fba,
        validate_mass_balance,
        validate_objective_feasibility,
    )

    model = load_model("textbook")
    result = run_fba(model)
    if not validate_objective_feasibility(result).passed:
        raise SystemExit("textbook FBA solution is infeasible")
    if not validate_mass_balance(model, result).passed:
        raise SystemExit("textbook FBA solution is not mass balanced")
    blocked = find_blocked_reactions(model)
    n_blocked = len(blocked.blocked_reactions)
    print(f"FBA OK: objective={result.objective_value:.3f}, "
          f"blocked reactions={n_blocked}")


def verify_potency() -> None:
    from bioeng_qpcr.io import read_dose_response_csv
    from bioeng_qpcr.potency import fit_dose_response

    datasets = read_dose_response_csv(EXAMPLES / "dose_response_example.csv")
    results = [fit_dose_response(dataset) for dataset in datasets]
    if len(results) != 2 or any(result.best_fit.n_points != 33 for result in results):
        raise SystemExit("potency example lost compounds or replicates")
    if any(not result.best_fit.converged for result in results):
        raise SystemExit("potency example did not converge")
    expected = {"CompoundA": 3.2, "CompoundB": 45.0}
    for result in results:
        if abs(result.relative_value / expected[result.compound] - 1.0) > 0.01:
            raise SystemExit(f"{result.compound} potency moved outside 1% tolerance")
    print("Potency workflows OK: " + ", ".join(
        f"{result.compound} {result.kind.value}={result.relative_value:.3g}"
        for result in results
    ))


def verify_docking() -> None:
    from bioeng_qpcr.docking import dock, parse_pdb, validate_docking

    receptor = parse_pdb(EXAMPLES / "docking_receptor.pdb")
    ligand = parse_pdb(EXAMPLES / "docking_ligand.pdb")
    result = dock(receptor, ligand, n_runs=1, n_steps=50, seed=1)
    if not result.poses:
        raise SystemExit("docking example returned no poses")
    checks = validate_docking(result, receptor, ligand)
    if len(checks) != 10 or any(
        check.severity == "error" and not check.passed for check in checks
    ):
        raise SystemExit("docking example failed its DCK validation layer")
    print(f"Docking workflow OK: {len(result.poses)} ranked pose(s)")


def verify_sequence() -> None:
    from bioeng_qpcr.sequence import (
        compare_to_reference,
        read_sequence_file,
        validate_sequencing_result,
    )

    query = read_sequence_file(EXAMPLES / "sequencing_reads.fastq")[0]
    reference = read_sequence_file(EXAMPLES / "sequencing_reference.fasta")[0]
    result = compare_to_reference(query.sequence, reference.sequence)
    if result.identity < 0.99:
        raise SystemExit(
            f"sequence example identity regressed to {result.identity:.3f}"
        )
    report = validate_sequencing_result(result, query, reference, quality=query.quality)
    if report.n_errors:
        raise SystemExit("sequence example failed its SEQ validation layer")
    print(f"Sequence workflow OK: identity={result.identity:.3f}")


def verify_image_preview() -> None:
    from bioeng_qpcr.io import inspect_image, load_preview

    path = EXAMPLES / "image_preview_example.ppm"
    info = inspect_image(path)
    preview = load_preview(path, palette="green")
    if info.width < 1 or info.height < 1 or preview.thumbnail.size == 0:
        raise SystemExit("image preview example could not be rendered")
    print(f"Image preview workflow OK: {info.format} {info.width}x{info.height}")


def main() -> int:
    verify_qpcr()
    verify_rdml()
    verify_outliers()
    verify_enzyme_kinetics()
    verify_reference_gene_family()
    verify_fermentation()
    verify_fba()
    verify_potency()
    verify_docking()
    verify_sequence()
    verify_image_preview()
    print("All example workflows verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
