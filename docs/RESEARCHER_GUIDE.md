# Researcher workflow

`bioeng-qpcr` is alpha research software. It is designed to expose
assumptions and missing evidence; it does not certify MIQE compliance and is
not a clinical diagnostic system.

The core qPCR workflow (sections 1-5) runs on the base installation. The
enzyme-kinetics, fermentation-kinetics, and metabolic-flux modules (sections
6-9) are separate subpackages with the same validation-first design; the flux
module additionally requires the `fba` extra. Section 10 lists the other entry
points (desktop application and MCP server).

## 1. Prepare the plate export

Start from [`examples/researcher_example.csv`](../examples/researcher_example.csv).
Each row is one well. The required columns are `position`, `sample_name`,
`target_name`, and `cq`. Recommended assay-evidence columns are `efficiency`,
`r2`, `slope`, `lod_cq`, and `lloq_cq`. For normalization, mark references in
`is_reference` or pass them explicitly on the command line.

Use distinct sample names for biological replicates (`Control_1`, `Control_2`,
and so on). Mark repeated wells from the same biological sample as
`technical`, and give them a `replicate_group`. Do not encode several
biological replicates as repeated wells under one sample name; the calculator
rejects that pseudoreplication pattern.

Efficiency is a fraction: `1.0` means 100% efficiency and an amplification
factor of 2. Values must come from assay characterization; Pfaffl and the
default normalized-quantification workflow do not substitute an assumed value.

### Reading RDML files

Instruments that export RDML (`.rdml`/`.rdm` archives, or the XML they
contain) need no manual conversion:

```python
from bioeng_qpcr.io import read_rdml

experiment = read_rdml("plate.rdml")
```

The reader maps RDML onto the same validation and calculation pipeline:
targets with `<type>ref</type>` become reference genes (extra ones can be
passed as `reference_targets=`), `sample/annotation` entries with property
`group` become the sample groups used for comparisons, NTC samples keep
their no-template status, and `<adp>` amplification points become raw
traces. Technical replicates are derived automatically from repeated
sample/target pairs. Amplification efficiencies are read from
`amplificationEfficiency` and converted from the per-cycle factor
convention (1.96 = 96%) to fractions. `--lenient`-style recovery is
available as `read_rdml(path, strict=False)`.

All CLI commands accept RDML files directly, e.g.
`bioeng-qpcr analyze plate.rdml --target IL6 --reference-targets GAPDH`.
An example document mirroring
[`examples/researcher_example.csv`](../examples/researcher_example.csv) is
kept at [`examples/rdml_example.xml`](../examples/rdml_example.xml).

## 2. Validate the input and readiness evidence

```bash
bioeng-qpcr validate examples/researcher_example.csv \
  --reference-targets GAPDH,RPLP0 \
  --output json \
  --output-file readiness.json
```

Strict mode is the default. It rejects duplicate well positions, malformed or
non-finite Cq values, inconsistent status fields, conflicting sample/assay
metadata, and reference targets absent from the data. For damaged legacy data,
`--lenient` records structured diagnostics; review every diagnostic before
using the result.

Statuses mean:

- `PASS`: the supplied evidence satisfies this implemented check.
- `WARN`: review is needed; the software cannot justify a hard decision.
- `FAIL`: the supplied evidence violates an implemented threshold or policy.
- `NOT_ASSESSABLE`: required evidence was not supplied.

The rule set is a readiness subset, not the full MIQE checklist. MIQE 2.0
emphasizes assay efficiency, sensitivity, transparent raw data, and appropriate
normalization; see the [MIQE 2.0 update](https://academic.oup.com/clinchem/article/71/6/634/8119148).

### Flagging suspect technical replicates

Before averaging technical replicates, screen them for wells with a
physical cause for a bad Cq (pipetting error, a bubble, an edge-effect
well):

```bash
bioeng-qpcr outliers examples/researcher_example.csv
```

The command applies Chauvenet's criterion iteratively to every
technical-replicate group: a well is flagged when the expected number of
deviations at least that extreme falls below one half
(`n · P(|Z| ≥ z) < 0.5`), and each flag records the z-score, tail
probability, and threshold as evidence. Groups whose Cq standard
deviation exceeds `--sd-trigger` (default 0.3) without a statistical flag
are listed for manual review with their most deviant well as the
candidate. Small honest caveat: with the sample standard deviation,
groups of three or four wells are too small for Chauvenet to ever reject
statistically — such groups can only produce review recommendations.

Detection never modifies the data or the downstream analysis. Inspect the
flagged wells (ideally their raw amplification curves), then either
re-run them or document their exclusion yourself — the same command
works on RDML files.

## 3. Calculate per-sample normalized quantities

```bash
bioeng-qpcr analyze examples/researcher_example.csv \
  --target IL6 \
  --reference-targets GAPDH,RPLP0 \
  --output-file quantities.json \
  --provenance-file quantities.provenance.json
```

The default calculation uses measured efficiency and the geometric mean of
the selected reference quantities. Reference-gene stability must be justified
for the actual tissue, treatment, and assay. Multiple reference genes alone do
not establish stability. The package contains geNorm and a model-based
NormFinder implementation benchmarked against the Bioconductor NormqPCR Colon
reference values. NormFinder and the consensus method require at least three
candidate genes. Both geNorm and NormFinder require measured candidate-assay
efficiencies and apply the same efficiency-corrected log-quantity transform as
the downstream normalization. The earlier descriptive score remains available
separately as `variance_stability_score`.

The JSON always contains `results`, `exclusions`, `warnings`, `parameters`, and
`summary`. A partial analysis is rejected by default. Use `--allow-exclusions`
only after documenting why each excluded biological sample is acceptable.

## 4. Compare independent biological groups

```bash
bioeng-qpcr compare examples/researcher_example.csv \
  --target IL6 \
  --reference-targets GAPDH,RPLP0 \
  --treated-group treated \
  --control-group control \
  --output-file comparison.json
```

Technical replicates are reduced within each biological sample. Group
summaries are geometric means, and Welch's t-test is run on log2 normalized
quantities only when both groups have at least two valid biological samples.
Experimental design, power, multiplicity correction, and model suitability
remain the researcher's responsibility.

## 5. Generate an auditable handoff report

```bash
bioeng-qpcr report examples/researcher_example.csv \
  --target IL6 \
  --reference-targets GAPDH,RPLP0 \
  --output-file researcher-report.html \
  --json-file researcher-report.json
```

The report combines readiness findings, quantities, exclusions, input
diagnostics, the normalized scientific-content hash, raw-file SHA-256 hash,
package versions, analysis parameters, and validation-rule versions.

## 6. Fit enzyme-kinetics parameters

The enzyme-kinetics module fits Michaelis-Menten parameters and refuses data
its validation layer cannot defend. Input is a tidy two-column CSV
(`substrate_concentration`, `velocity`), optionally with an
`inhibitor_concentration` column for inhibition experiments.

```python
import numpy as np
from bioeng_qpcr.enzyme_kinetics import (
    fit_michaelis_menten,
    read_kinetics_csv,
    validate_km_range,
    validate_substrate_span,
)

substrate, velocity, inhibitor = read_kinetics_csv(
    "examples/enzyme_kinetics_example.csv"
)
result = fit_michaelis_menten(substrate, velocity)

print(result.vmax, result.km, result.r_squared)
print(result.km_ci, result.vmax_ci, result.warnings)

km_check = validate_km_range(result.km)          # plausible Km magnitude
span_check = validate_substrate_span(substrate, result.km)  # data brackets Km
```

On the shipped example this reports `Vmax = 99.939`, `Km = 9.401` with both
validation checks passing. Velocities must be non-negative: noisy near-zero
rates from real plate readers should be reviewed and removed before fitting
rather than silently passed through, and the fitter will reject them
explicitly. Check that the substrate span brackets the estimated Km before
trusting either parameter; `validate_substrate_span` automates that review.

`fit_lineweaver_burk` is available as a compatibility method for comparing
against legacy double-reciprocal analyses; prefer the nonlinear fit for
reporting.

## 7. Classify reversible inhibition

Inhibition classification compares Michaelis-Menten fits across several
fixed inhibitor concentrations. Build one `(inhibitor, substrate, velocity)`
tuple per concentration and classify:

```python
import csv
from collections import defaultdict

import numpy as np

from bioeng_qpcr.enzyme_kinetics import classify_inhibition

groups = defaultdict(lambda: ([], []))
with open(
    "examples/inhibition_example.csv", newline="", encoding="utf-8-sig"
) as handle:
    for row in csv.DictReader(handle):
        conc = float(row["inhibitor_concentration"])
        groups[conc][0].append(float(row["substrate_concentration"]))
        groups[conc][1].append(float(row["velocity"]))

datasets = [
    (conc, np.array(subs), np.array(vels))
    for conc, (subs, vels) in sorted(groups.items())
]
result = classify_inhibition(datasets)

print(result.inhibition_type)  # e.g. InhibitionType.COMPETITIVE
print(result.confidence, result.message)
```

The classifier reports one of `COMPETITIVE`, `NON_COMPETITIVE`,
`UNCOMPETITIVE`, `MIXED`, or `NONE` together with a coefficient-of-variation
based confidence and an explicit message. Classification is a trend-based
screening method over how apparent Km and Vmax shift with inhibitor
concentration; confirm important mechanism calls with the global fit below.

### Global model selection (AICc)

For a publishable mechanism call, globally fit the rate equation of every
candidate mechanism to all inhibitor concentrations at once and rank them by
small-sample-corrected AIC. `compare_inhibition_models` reuses the same
`datasets` list built above:

```python
from bioeng_qpcr.enzyme_kinetics import compare_inhibition_models

comparison = compare_inhibition_models(datasets)

print(comparison.best_type)  # InhibitionType.COMPETITIVE
print(comparison.message)
for m in comparison.models:  # ranked best-first by AICc
    print(m.inhibition_type.value, round(m.aicc, 2), round(m.akaike_weight, 3))
print(comparison.best_fit.parameters)
```

On the shipped example this selects competitive inhibition with
`Vmax = 98.96`, `Km = 10.14`, `Ki = 5.20` — but the mixed model lands within
ΔAICc = 0.65 of the winner, so the comparison flags the call as unresolved.
That is the honest answer when the tested inhibitor range does not yet
discriminate the two mechanisms: mixed inhibition collapses to competitive
when the ES-complex affinity is weak, so distinguishing them needs higher
inhibitor concentrations, not more replicates. As rules of thumb, treat
ΔAICc > 10 between the best and runner-up models as decisive and ΔAICc < 2
as unresolved.

Akaike weights quantify the relative support for each mechanism and sum to 1
across the candidate set. The winning `GlobalInhibitionFit` also carries
parameter standard errors, the pooled RSS, and AIC/BIC for reporting.

## 8. Fit batch-fermentation kinetics

The fermentation module fits Monod growth parameters from batch time-course
data (`time`, `biomass`, `substrate` columns) and then tells you whether the
parameters are actually identifiable:

```python
from bioeng_qpcr.fermentation_kinetics import (
    analyze_identifiability,
    fit_monod,
    read_fermentation_csv,
    validate_data_sufficiency,
    validate_mu_max_range,
)

time, biomass, substrate = read_fermentation_csv(
    "examples/fermentation_example.csv"
)
result = fit_monod(time, biomass, substrate)

print(result.mu_max, result.ks, result.yxs)
print(result.mu_max_ci, result.ks_ci, result.yxs_ci)

mu_check = validate_mu_max_range(result.mu_max)  # physical upper bound
data_check = validate_data_sufficiency(time, biomass, substrate)

identifiability = analyze_identifiability(result, time, biomass, substrate)
print(identifiability.confidence, identifiability.message)
```

On the shipped example the fit returns `mu_max = 0.4052`, `Ks = 0.3087`, but
the identifiability analysis flags the parameter set as poorly identifiable
(mu_max/ks correlation 0.99, condition number ~2.9e4) and recommends
collecting more samples near Ks. Treat that message as part of the result:
report the correlation structure and confidence interval method alongside the
point estimates, or collect the suggested data before publishing the fit.
`validate_mu_max_range` enforces a plausible physical range of
`[0.001, 5.0] h^-1` and fails the check outside it, so growth rates that
exceed known microbial maxima are surfaced rather than silently accepted.
`solve_batch_fermentation` and `batch_fermentation_ode` are available for
forward simulation once parameters are trusted.

## 9. Analyze metabolic flux

The flux module wraps COBRApy flux-balance analysis with the same
validation-first approach. Install the extra first:

```bash
python -m pip install ".[fba]"
```

```python
from bioeng_qpcr.metabolic_flux import (
    find_blocked_reactions,
    find_dead_end_metabolites,
    load_model,
    run_fba,
    validate_mass_balance,
    validate_objective_feasibility,
)

model = load_model("textbook")  # or a path to an SBML/JSON/MAT file
result = run_fba(model, objective="Biomass_Ecoli_core")

print(result.objective_value, result.status)
print(result.model_summary)

feasibility = validate_objective_feasibility(result)
mass_balance = validate_mass_balance(model, result)

blocked = find_blocked_reactions(model)
dead_ends = find_dead_end_metabolites(model)
```

`load_model("textbook")` fetches the *E. coli* core model bundled with
COBRApy and is the quickest smoke test: the shipped example reports an
objective value of `0.874` with 8 blocked reactions. The validators check
that the solution is feasible and mass balanced before you interpret fluxes;
blocked reactions and dead-end metabolites are network-curation signals, not
modeling errors. `run_fba` restores the original objective and direction
after solving, so repeated calls on the same model are safe.

## 10. Desktop application, widgets, and MCP server

- **Desktop application** — `bioeng-qpcr-gui` launches a PySide6 interface
  that covers the qPCR, enzyme-kinetics, fermentation, and flux workflows
  with progress reporting and cooperative cancellation. A Chinese-language
  manual for the packaged Windows build is kept at
  [`docs/bioeng-qpcr使用说明.pdf`](bioeng-qpcr使用说明.pdf).
- **Jupyter widgets** — `pip install ".[widgets]"` enables the ipywidgets
  front end over the same core package.
- **MCP server** — `bioeng-qpcr-mcp` exposes 15 tools (`qpcr_*`,
  `enzyme_*`, `ferment_*`, `fba_*`) for AI assistants. Each tool wraps the
  validated core API; assistant-facing errors mirror the library's
  structured validation messages.

Every workflow in sections 1-9 can also be re-run from the example datasets
with `python scripts/verify_examples.py`, which fails loudly if any
documented example stops working.

## What still needs independent validation

Before publication-critical use, compare results against instrument exports
and independently calculated reference outputs using representative assays,
inhibition/low-copy cases, and predeclared acceptance criteria. Synthetic unit
tests establish software correctness, not empirical validity. Amplification
curve efficiency methods such as LinRegPCR require separate benchmarking; the
package's exploratory trace-shape fit is not used as quantitative evidence.
See the [LinRegPCR method evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8386043/)
for relevant background.

The example datasets in [`examples/`](../examples/) are synthetic teaching
aids. They exercise every documented workflow (and two of them required the
removal of near-zero noise points that the enzyme-kinetics validation layer
would otherwise reject), but they are not measurements from real experiments.
Fit parameters recovered from them demonstrate the workflow, not empirical
accuracy.
