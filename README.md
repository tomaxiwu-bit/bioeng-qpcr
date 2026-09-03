# bioeng-qpcr

An auditable Python engine for relative qPCR analysis and machine-assessable
MIQE readiness checks.

> **Alpha software:** this package supports research and teaching workflows.
> Its readiness report is not a certification of full MIQE compliance, and
> results should be independently reviewed before clinical or high-stakes use.

## Why this project exists

The arithmetic behind qPCR analysis is compact; the difficult part is making
inputs, assumptions, exclusions, quality checks, and provenance visible. This
project separates calculation from validation so each decision can be tested
and audited.

## Features

- Efficiency-corrected normalized quantities with multiple reference genes
- Explicit opt-in ΔCq, ΔΔCq, and Pfaffl compatibility methods
- Strict CSV validation with row/column diagnostics and an auditable lenient mode
- Four-state readiness rules: `PASS`, `WARN`, `FAIL`, `NOT_ASSESSABLE`
- geNorm stability analysis and pairwise-variation reporting
- Efficiency-corrected geNorm and model-based NormFinder stability scoring
- Standard-curve efficiency fitting
- Combined HTML/JSON researcher reports, provenance manifests, CLI, and plots
- Explicit per-sample exclusions—batch analyses never silently discard data
- Responsive Monod and FBA desktop tasks with progress and cancellation requests
- 940+ automated tests with branch coverage

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install .          # core library + CLI
python -m pip install ".[plot]" # + matplotlib plots
```

Optional extras: `plot` (plots), `widgets` (Jupyter front end), `mcp` (AI
assistant server), `fba` (COBRApy metabolic flux), `gui` (PySide6 desktop
application), and `desktop-build` (PyInstaller packaging). For development:

```bash
python -m pip install -e ".[dev]"
```

## Desktop build

Build the Windows desktop bundle from an isolated environment with all
desktop features installed:

```powershell
python -m pip install -e ".[desktop-build]"
python -m PyInstaller --clean --noconfirm bioeng-qpcr.spec
dist\bioeng-qpcr\bioeng-qpcr.exe --smoke-test
```

The default PyInstaller output is a folder-based application. Distribute the
complete `dist\bioeng-qpcr` directory; the EXE is not usable without its
adjacent `_internal` directory.

Create a versioned portable ZIP with an internal file manifest and external
archive checksum after building:

```powershell
.\scripts\build_portable.ps1
```

The script runs the packaged scientific smoke test before producing the
release. Users must extract the complete archive before launching the app;
copying only `bioeng-qpcr.exe` is unsupported.

## Input CSV

The minimal tidy schema contains one measurement per row:

```csv
position,sample_name,target_name,cq,efficiency
A01,Control_1,GAPDH,19.42,0.97
A02,Control_1,IL6,24.16,1.01
A03,NTC,GAPDH,Undetermined,0.97
```

`position`, `sample_name`, `target_name`, and `cq` are required. Input is
strictly validated by default: malformed Cq values, duplicate positions,
conflicting metadata, and unknown reference targets raise structured errors
instead of being silently coerced. `strict=False` is available for recovery;
all recovered issues are retained in `experiment.metadata["input_diagnostics"]`.

Efficiency is
expressed as a fraction: `1.0` means 100% efficiency and an amplification
factor of 2 per cycle. Do not enter `100` or `2.0` in the efficiency column.
See [`examples/researcher_example.csv`](examples/researcher_example.csv) for
the full supported schema and [`docs/RESEARCHER_GUIDE.md`](docs/RESEARCHER_GUIDE.md)
for the workflow.

## Example datasets

Every module ships with a small worked dataset in [`examples/`](examples/):

| Dataset | Module | Workflow |
|---|---|---|
| `researcher_example.csv` | qPCR | full-schema treated-vs-control analysis |
| `qpcr_simple_plate_example.csv` | qPCR | minimal plate export with technical replicates |
| `enzyme_kinetics_example.csv` | enzyme kinetics | Michaelis-Menten fitting |
| `inhibition_example.csv` | enzyme kinetics | inhibition classification across four inhibitor concentrations |
| `fermentation_example.csv` | fermentation kinetics | Monod fitting and identifiability |

The *E. coli* core model for FBA is loaded directly from COBRApy
(`load_model("textbook")`), so it needs no local file.

All documented workflows are re-run against these datasets with:

```bash
python scripts/verify_examples.py
```

## Readiness assessment

```python
from bioeng_qpcr.io import read_tidy_csv
from bioeng_qpcr.validation import ValidationEngine

experiment = read_tidy_csv(
    "data.csv",
    reference_targets=["GAPDH"],
)
report = ValidationEngine().run(experiment)

print(report.to_text())
print(report.rule_coverage)
```

The engine refuses to create an empty report. Unexpected rule crashes raise a
`RuleExecutionError` rather than being mislabeled as missing scientific data.
`assessment_passed` means every executed rule passed; it does not mean that
every MIQE item was assessed.

CLI equivalent:

```bash
bioeng-qpcr validate data.csv --reference-targets GAPDH --output json
```

The command exits with code 1 for a hard rule failure or invalid input.
Warnings and unavailable evidence remain visible but do not masquerade as a
command-processing error. `assessment_passed` remains true only when every
executed rule passes.

## Relative quantification

```bash
bioeng-qpcr analyze data.csv \
  --target IL6 \
  --reference-targets GAPDH,RPLP0 \
  --provenance-file analysis.provenance.json

bioeng-qpcr compare data.csv \
  --target IL6 \
  --reference-targets GAPDH,RPLP0 \
  --treated-group treated \
  --control-group control

bioeng-qpcr report data.csv \
  --target IL6 \
  --reference-targets GAPDH,RPLP0 \
  --output-file researcher-report.html \
  --json-file researcher-report.json
```

The default method calculates per-sample normalized quantities as
`(1 + efficiency)^(-Cq)` and divides the target quantity by the geometric
mean of reference-gene quantities. Group comparisons preserve each biological
sample as an independent observation and use Welch's t-test in log2 space.

Legacy ΔΔCq must be selected explicitly. Confirm comparable target/reference
efficiencies before using it. Pfaffl requires measured assay efficiencies;
the software no longer invents a 100% default. Any incomplete sample is
reported in `exclusions`, and CLI analysis fails unless `--allow-exclusions`
is explicitly requested.

## Scientific scope and limitations

- The current readiness rules cover input integrity, assay characterization,
  amplification efficiency, reference designation/multiplicity, technical
  replicate agreement, and no-template controls. They do not cover the
  complete MIQE 2.0 checklist.
- NTC amplification has no universal Cq cutoff. Without an explicitly declared
  assay/workflow policy it produces a warning, not a fabricated hard decision.
- `normfinder` implements the model-based grouped estimator used by
  Bioconductor NormqPCR; it needs at least three candidate genes and at least
  two complete samples per group. geNorm and NormFinder require measured
  positive efficiencies for every candidate and use the same efficiency-
  corrected log-quantity scale as downstream normalization.
  `variance_stability_score` remains available as the earlier descriptive
  compatibility metric.
- `fit_logistic_trace_shape` is exploratory. It is not validated as an
  individual-amplification-curve efficiency method and is not used as MIQE
  evidence.
- Enzyme-inhibition classification remains a trend-based screening method,
  not a substitute for global nonlinear fitting and model selection.
- Monod confidence intervals use scaled residuals and a log-parameter Wald-t
  approximation. Profile-likelihood intervals remain preferable when
  parameters are weakly identifiable.
- Desktop cancellation is cooperative. It keeps the interface responsive and
  discards late results, but an active SciPy or COBRA native solver call may
  need to finish before the worker stops.
- geNorm returns no recommended gene count when its pairwise-variation
  threshold is not reached. It never silently treats targets of interest as
  reference-gene candidates.
- Synthetic unit tests establish software behavior, not scientific validity.
  Published and instrument-derived benchmark datasets remain required.

## Architecture

```text
src/bioeng_qpcr/
├── domain/                  Experiment, Sample, Assay, Well, RawTrace
├── io/                      strict tidy CSV ingestion and diagnostics
├── calculations/            normalized quantities, ΔCq, ΔΔCq, Pfaffl, efficiency
├── reference_genes/         geNorm, model-based NormFinder, consensus ranking
├── validation/              rules, engine, and readiness reports
├── statistics/              intervals and replicate-aware models
├── provenance/              hashes, versions, and parameters
├── plotting/                matplotlib-based plots
├── reporting/               combined researcher HTML/JSON report
├── enzyme_kinetics/         Michaelis-Menten fitting, inhibition classification
├── fermentation_kinetics/   Monod fitting, identifiability, ODE solving
├── metabolic_flux/          COBRApy FBA, blocked reactions, dead-end detection
├── cli/                     durable command-line interface
├── gui/                     PySide6 desktop application
├── widgets/                 ipywidgets front end
└── mcp/                     MCP server for AI assistants
```

UI and MCP adapters should remain thin layers over this independently usable
core package.

## Development

```bash
python -m pytest
python -m pytest --cov=bioeng_qpcr --cov-branch
ruff check .
```

Scientific changes should include a cited rationale, boundary tests, and—when
possible—comparison with independently produced reference outputs.

## License

MIT. See [LICENSE](LICENSE).
