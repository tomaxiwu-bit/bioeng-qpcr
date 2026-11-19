# bioeng-qpcr

本地运行、结果可追溯的 qPCR 定量分析工作台，面向科研与教学。

从 CSV / RDML 导入，到质控、相对 / 绝对定量、图表和实验报告，保留输入、
参数、警告与分析历史；另提供效价、酶动力学、发酵和 FBA 配套工具。

**0.6.8 · 研究与教学试用版** ·
[Windows 下载与版本记录](https://github.com/tomaxiwu-bit/bioeng-qpcr/releases) ·
[五分钟试用](docs/QUICKSTART_ZH.md) · [报告示例](docs/demo/qpcr-report.html) ·
[本次修复](docs/RELEASE_NOTES_v0.6.8.md)

![bioeng-qpcr 0.6.8 桌面工作台](docs/images/desktop.png)

## 先跑一次完整示例

1. 在 [Releases](https://github.com/tomaxiwu-bit/bioeng-qpcr/releases) 选择对应版本的
   `win64-portable.zip`，完整解压，打开文件夹内的 `bioeng-qpcr.exe`。便携版无需安装 Python。
2. 点击“打开综合实验工作台”，进入 qPCR，选择“脑 / 肾基因表达对比”。
3. 依次确认设计、回放公开数据、计算 Ct、检查质控，按页面说明确认示例方法假设后定量。
4. 查看 kidney / brain 的 **5.6373 倍**表达变化，下载定量 CSV、项目备份和 HTML 报告。

页面会保留示例缺少原始曲线、对照和效率验证的证据缺口。
替换真实实验数据前，请按[导入规范](docs/CSV_IMPORT_SPEC.md)准备文件，并复核实验室 SOP。
MIQE readiness 不是完整 MIQE 合规认证；仪器采集目前通过文件导入。

源码用户可以启动同一浏览器工作台：

```bash
git clone https://github.com/tomaxiwu-bit/bioeng-qpcr.git
cd bioeng-qpcr
python -m pip install ".[web,fba]"
python -m bioeng_qpcr.web
# 打开 http://127.0.0.1:8008/lab
```

公开下载与本地源码须使用相同版本；本页流程对应 0.6.8。
[安装与首次使用](docs/QUICKSTART_ZH.md) · [数据验证依据](docs/PUBLIC_DATA_VALIDATION.md) ·
[复现与构建](docs/REPRODUCIBILITY.md) · [问题反馈](https://github.com/tomaxiwu-bit/bioeng-qpcr/issues)

## 综合实验工作台

默认入口为 `/lab`，展示 qPCR、酶动力学、发酵动力学、FBA 代谢通量、效价
五个主要模块。qPCR 是其中的完整实验工作流。文件与图片预览移入辅助工具，
简化刚性对接、通用序列比对移出主导航，兼容入口、API、旧记录与备份保留。

Windows 程序点击“打开综合实验工作台”进入首页。酶动力学、发酵与 FBA 均可在浏览器分析、
自动归档、恢复和导出；原桌面页保留，并提供统一工作台入口。
`/history` 可筛选、搜索和恢复各浏览器工具的记录；qPCR 项目历史仍在 `/workbench` 中管理。
原桌面页的独立计算不会自动写入此存档，需要历史功能时使用浏览器入口。
新分析记录包含软件与科学依赖版本、输入摘要、参数和时间；旧记录不追填当前版本。
许可证与第三方组件信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
问题反馈、贡献与支持范围见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## qPCR 实验工作台

新增围绕单次实验的可视化工作流：样品与引物/探针 → 配液与 96/384 孔板位 →
温度循环与荧光通道 → 运行快照/文件采集 → Ct 与熔解峰 → 质控 →
ΔΔCt / 标准曲线定量 → CSV、JSON 与自包含 HTML 报告。

```bash
python -m pip install ".[web]"
python -m bioeng_qpcr.web
# 打开 http://127.0.0.1:8008/lab
```

在综合首页选择 qPCR，即可进入 `/workbench` 的八阶段工作流。
内置脑/肾 Ct 和 RDML 稀释实验两套公开示例，保留出处、许可和证据缺口。
当前支持真实 CSV/RDML 导入与公开数据回放；硬件控制需要厂商 SDK 接入。

模板库另外提供 5 套无测量数据的实验规划模板，以及 6 套可切换的图表与报告主题。
曲线可叠加技术重复或全部孔；相对表达支持棒棒糖、柱状和点图；标准曲线附残差图。
可下载 SVG、PDF 和 300 dpi PNG，主题切换不会修改计算结果。

项目默认保存到本机 `%LOCALAPPDATA%/bioeng-qpcr/workbench.sqlite3`（Windows），
可通过 `BIOENG_QPCR_WORKBENCH_DB` 指定路径。流程修改会使下游结果失效，
历史运行快照与分析版本保留在项目 JSON。旧版独立分析入口继续可用。

[方案评估、模块与状态设计](docs/QPCR_WORKBENCH.md) ·
[公开演示数据出处](src/bioeng_qpcr/web/examples/qpcr_workbench/README.md)

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
- RDML 1.x ingestion (`.rdml`/`.rdm` archives, gzip, or XML) into the same pipeline
- Chauvenet-criterion technical-replicate outlier screening with full statistical evidence
- Four-state readiness rules: `PASS`, `WARN`, `FAIL`, `NOT_ASSESSABLE`
- MIQE 2.0 evidence-gap planning that groups missing essential evidence into
  prioritized documentation, export, analysis, and measurement actions
- Reference-gene stability algorithm family: geNorm, model-based NormFinder,
  delta-Ct (Silver et al. 2006), and BestKeeper (Pfaffl et al. 2004), plus a
  rank-aggregated consensus over any chosen combination
- Efficiency-corrected geNorm and model-based NormFinder stability scoring
- Standard-curve efficiency fitting
- Combined HTML/JSON researcher reports, provenance manifests, CLI, and plots
- Explicit per-sample exclusions—batch analyses never silently discard data
- Responsive Monod and FBA desktop tasks with progress and cancellation requests
- Dose-response potency: 3PL/4PL/5PL/Gompertz fitting with AICc model selection,
  relative **and** absolute IC50/EC50, asymptotic or bootstrap intervals, and the
  POT-001..POT-011 audit rules
- Rigid-body molecular docking with a decomposed empirical score (van der Waals,
  electrostatics, hydrogen bonds, hydrophobic contact, desolvation), Monte Carlo
  simulated annealing or deterministic grid search, receptor-frame pose
  de-duplication, and the DCK-001..DCK-010 audit rules
- Sequencing comparison: Needleman-Wunsch, Smith-Waterman, Gotoh affine and
  glocal alignment, BLOSUM62, primer/quality trimming, variant calling,
  consensus and progressive MSA, with the SEQ-001..SEQ-008 audit rules
- CSV preview and import specifications: encoding/delimiter sniffing, typed
  column profiling, a machine-checked schema per module, generated templates,
  and [`docs/CSV_IMPORT_SPEC.md`](docs/CSV_IMPORT_SPEC.md) generated from the
  same registry the validators use
- Multi-format image preview with pseudo-colour rendering, contrast stretching
  and multi-channel composition, plus a dependency-free header sniffer
- 1500+ automated tests with branch coverage

## Installation

Requires Python 3.10 or newer.
Clone the repository and enter its directory first, as shown above.

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

## Web application

A Python-only local web workbench exposes the same validated workflows in a
browser — no JavaScript build step, no external service.  It mirrors the CLI
(`validate`, `analyze`, `curves`) so results are consistent across surfaces:

```bash
python -m pip install ".[web]"
python -m bioeng_qpcr.web        # or: bioeng-qpcr-web
# open http://127.0.0.1:8008
```

The page provides three tabs: MIQE readiness assessment (with a Cq heatmap),
relative quantification (normalized quantity / ΔΔCq / Pfaffl / group
comparison, with fold-change and heatmap plots), and per-well amplification
curve fitting (baseline correction, sigmoid Cq, per-well efficiency, and an
amplification plot).  Upload a tidy CSV, long-format amplification CSV, or
RDML file; everything runs locally and plots render as in-browser PNG data
URIs. Override the port with `BIOENG_QPCR_WEB_PORT`. The workbench accepts
`localhost` and `127.0.0.1` hosts and rejects browser writes from
other origins. It is designed for local use; changing the bind address does
not turn it into a public multi-user service.

### Workspace page

`http://127.0.0.1:8008/workspace` adds the Phase 5 modules behind a
three-column workspace (module rail, chart and results, right-hand
"分析数据 / 参数设置 / 格式说明" panel):

- **数据表预览** — preview a CSV and see every structural problem with its row
  and column, plus the format documentation and a downloadable template.
- **图片预览** — inspect any supported image and render a pseudo-coloured,
  contrast-stretched preview.
- **效价分析** — fit IC50/EC50 and read the audit rules.
- **分子对接** — dock a ligand and inspect the ranked poses.
- **测序比对** — align a read, call variants and run the SEQ rules.

The CSV-based modules offer **填充示例** (load runnable example rows) and
**下载示例** (download a template), so the expected layout is visible before
any file is uploaded.

Potency, docking, and sequence analyses can also generate a publication
bundle for every result figure: editable SVG, vector PDF, and a 300 dpi PNG
at a 7.2-inch double-column width. Prefer SVG/PDF for final typesetting and
check the target journal's required dimensions before submission.

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

Every module's input format — required and optional columns, types, units,
ranges, missing-value rules and a worked example — is specified in
[`docs/CSV_IMPORT_SPEC.md`](docs/CSV_IMPORT_SPEC.md). That file is generated
from the same schema registry the readers validate against, so it cannot
describe a format the code does not implement:

```bash
bioeng-qpcr schema --list                      # every registered format
bioeng-qpcr schema --format dose_response      # one format in full
bioeng-qpcr schema --format qpcr_tidy --write-template template.csv
bioeng-qpcr schema --write-import-spec docs/CSV_IMPORT_SPEC.md
bioeng-qpcr preview my_data.csv                # what the reader actually sees
```

### RDML input

Instrument exports in RDML (`.rdml`/`.rdm` archives or plain XML) are read
by the same pipeline (`read_rdml` from `bioeng_qpcr.io`; all CLI commands
accept them directly). Reference genes come from the RDML target type,
sample groups from annotations, and `<adp>` points become raw traces.

## Example datasets

The source repository includes a small worked dataset for every module in
[`examples/`](examples/); the Web wheel also bundles the examples used by its
one-click demo:

| Dataset | Module | Workflow |
|---|---|---|
| `researcher_example.csv` | qPCR | full-schema treated-vs-control analysis |
| `rdml_example.xml` | qPCR | the same experiment as RDML for interchange testing |
| `qpcr_simple_plate_example.csv` | qPCR | minimal plate export with technical replicates |
| `enzyme_kinetics_example.csv` | enzyme kinetics | Michaelis-Menten fitting |
| `inhibition_example.csv` | enzyme kinetics | inhibition classification across four inhibitor concentrations |
| `fermentation_example.csv` | fermentation kinetics | Monod fitting and identifiability |
| `dose_response_example.csv` | potency | two inhibitors, 11 levels × 3 replicates, long layout |
| `dose_response_matrix_example.csv` | potency | the same curves in plate-reader matrix layout |
| `docking_receptor.pdb` / `docking_ligand.pdb` | docking | receptor and ligand for a rigid-body docking run |
| `sequencing_reference.fasta` | sequence | synthetic amplicon template |
| `sequencing_reads.fastq` | sequence | six reads with known SNPs, a deletion and an insertion |
| `sequencing_read_plain.txt` | sequence | the same style of read exported as plain text |

The two dose-response examples are generated so that the shipped files recover
their true IC50 (3.2 µM and 45 µM) to within 1 %, which makes them usable as a
worked demonstration of the whole potency workflow rather than just plausible
numbers.

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
  --json-file researcher-report.json \
  --package-file researcher-package.zip
```

The ZIP is a compact RO-Crate-described handoff containing the original input,
HTML/JSON report, MIQE 2.0 checklist, provenance record and SHA-256 checksums.
Standard-curve fits also expose `result.predict_cq_interval(log10_concentration)`
for a Student-t prediction interval on a future Cq observation.

Replicated standards can be analysed directly without dropping non-detects:

```bash
bioeng-qpcr schema --format qpcr_standard_curve --write-template standards.csv
bioeng-qpcr standard-curve standards.csv --target GAPDH \
  --predict-concentration 100 --output-file calibration.json
```

The result reports efficiency with its interval, empirical LOD, LLOQ/ULOQ,
dynamic range, every concentration-level detection rate and Cq SD, plus the
thresholds used. Blank/`Undetermined` Cq rows remain in the LOD denominator.

The default method calculates per-sample normalized quantities as
`(1 + efficiency)^(-Cq)` and divides the target quantity by the geometric
mean of reference-gene quantities. Group comparisons preserve each biological
sample as an independent observation and use Welch's t-test in log2 space.

Legacy ΔΔCq must be selected explicitly. Confirm comparable target/reference
efficiencies before using it. Pfaffl requires measured assay efficiencies;
the software no longer invents a 100% default. Any incomplete sample is
reported in `exclusions`, and CLI analysis fails unless `--allow-exclusions`
is explicitly requested.

## Potency (IC50 / EC50)

```bash
bioeng-qpcr potency dose_response.csv
bioeng-qpcr potency dose_response.csv --control-response 100
bioeng-qpcr potency dose_response.csv --compare CompoundA,CompoundB --plot ic50.png
bioeng-qpcr potency plate_matrix.csv --layout matrix --ci-method bootstrap
```

Both potency values are always reported, because they answer different
questions and coincide only when the response returns to zero:

- the **relative** IC50/EC50 is the midpoint of the fitted window — what "the
  IC50" usually means in a paper, and what the inflection of a 4PL gives;
- the **absolute** IC50/EC50 is the concentration producing a chosen fraction
  (default 50 %) of the *control* response. When inhibition is incomplete, this
  is the value that actually means "50 % inhibition".

Pass the measured untreated or vehicle-control response with
`--control-response`; when it is omitted, the result is explicitly flagged as
using the mean response at the lowest tested concentration.

The fitted `log_ec50` parameter is never reported as a potency: for the 5PL and
Gompertz curves the half-maximal position is not that parameter, and the
correction is derived in closed form. Curves are fitted in `log10`
concentration, so `log_ec50` *is* a fitted parameter (with a real standard
error) and the concentration interval comes out correctly asymmetric. Model
selection ranks 3PL/4PL/5PL/Gompertz by AICc. The audit rules cover
extrapolation, the assay window relative to the control, design adequacy, Hill
slope plausibility, goodness of fit, concentration span, interval width,
plateau support, lack of fit against the nested 5PL, replicate scatter, and
monotonicity (the high-dose hook effect).

## Molecular docking

```bash
bioeng-qpcr dock receptor.pdb ligand.pdb --n-runs 8 --n-steps 4000 --write-pose top.pdb
bioeng-qpcr dock complex.pdb --split-from-receptor --box-center 12.5,-3,0 --box-size 12
```

A rigid ligand is placed by Monte Carlo simulated annealing (or a deterministic
grid scan) and scored with a decomposed empirical potential. **Lower score is
better**, and every pose reports its per-term breakdown so a ranking can be
attributed to the physics rather than trusted. Poses are de-duplicated by RMSD
*in the receptor frame* — superimposing first would collapse every rotated copy
of the ligand to zero RMSD and silently discard real binding modes. The
DCK rules check box coverage and containment, steric clashes, sampling
adequacy, ligand size, pocket occupancy, that both partners were identified,
that the top pose is favourable, rigid geometry preservation, and explicit
receptor-ligand contact.

## Sequencing comparison

```bash
bioeng-qpcr align reads.fastq reference.fasta --variants --min-identity 0.95
bioeng-qpcr align sample.ab1 reference.fasta --trim-quality
bioeng-qpcr align reads.fastq reference.fasta --primer-5 ACGTACGT --trim-quality
bioeng-qpcr align reads.fastq reference.fasta --consensus
```

Reads are mapped with a semi-global (glocal) alignment, which is what a Sanger
read against an amplicon actually is. Primers and low-quality tails are excised
before aligning, and every reported coordinate is in reference space. Variant
calling distinguishes SNPs, MNPs, insertions and deletions and flags
low-complexity context; the SEQ rules check identity, read length, reference
coverage, low-quality fraction, frameshifts, ambiguous bases, GC content and
primer excision. Applied Biosystems `.ab1`/`.abi` chromatograms are read
directly from their analysed PBAS base calls and PCON Phred qualities.

## Data preview

```bash
bioeng-qpcr preview my_data.csv --format dose_response
bioeng-qpcr preview my_data.csv --output json
```

The preview never repairs anything: ragged rows, duplicate headers, unexpected
columns, unparsable numbers and out-of-range values are all reported with their
row and column, alongside the detected encoding and delimiter and a typed
profile of every column. A file that is not valid UTF-8 is decoded as
GB18030/GB2312 with an explicit warning, because that is what Chinese Excel
exports are.

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
- Monod fitting reports log-parameter Wald-t confidence intervals, and a
  dedicated profile-likelihood routine (likelihood-ratio inversion) is
  available when parameters are weakly identifiable and the Wald
  approximation is unreliable.
- Desktop cancellation is cooperative. It keeps the interface responsive and
  discards late results, but an active SciPy or COBRA native solver call may
  need to finish before the worker stops.
- geNorm returns no recommended gene count when its pairwise-variation
  threshold is not reached. It never silently treats targets of interest as
  reference-gene candidates.
- Potency fitting covers the standard sigmoid family (3PL/4PL/5PL/Gompertz). It
  does not model time-dependence, partial inhibition kinetics, or
  non-monotone responses; a high-dose hook effect is *detected* and reported
  rather than fitted. Absolute potency requires a control response, which is
  taken from the lowest tested concentration unless supplied explicitly, so a
  dose series without a true vehicle control will report a control-relative
  value that is not what the user assumes.
- Docking uses a simplified empirical scoring function (12-6 Lennard-Jones,
  Coulomb with a distance-dependent dielectric, geometric hydrogen-bond and
  hydrophobic terms, and an optional desolvation term). It is **not** a
  validated force field and **not** a trained scoring function, the ligand is
  rigid (no induced fit), the receptor is fixed, and no explicit solvent or
  entropy is modelled. Scores are therefore a ranking heuristic for hypotheses,
  not binding free energies, and must not be compared against experimental
  affinities. Results should be re-scored and re-ranked with an established
  docking program before any experimental commitment.
- Sequence comparison assumes reads are already demultiplexed and adapter-free;
  variant calls are made from alignment columns without base-quality
  recalibration, indel realignment, or a genotyping likelihood model, so they
  are suitable for Sanger-style confirmation and amplicon QC, not for calling
  variants from deep sequencing. The multiple-sequence alignment is a pragmatic
  progressive method, not a MUSCLE/MAFFT-class aligner.
- The image preview is for *display*. Pseudo-colouring, contrast stretching and
  gamma change the pixel values shown, so intensities must always be read from
  the original file (the preview says so in its own report). The built-in
  header sniffer reports container metadata but cannot decode pixels; without
  the optional Pillow dependency, previews are unavailable and only metadata is
  reported.
- Synthetic unit tests establish software behavior, not scientific validity.
  The first retrospective public-data benchmark now covers a published qPCR
  example, 294 instrument-derived RDML traces, and 508 OpenADMET CYP3A4 curves;
  see [`docs/PUBLIC_DATA_VALIDATION.md`](docs/PUBLIC_DATA_VALIDATION.md).
  Prospective validation by an independent laboratory is still required.

## Architecture

```text
src/bioeng_qpcr/
├── domain/                  Experiment, Sample, Assay, Well, RawTrace
├── io/                      strict ingestion, CSV preview + import schemas, image preview
├── calculations/            normalized quantities, ΔCq, ΔΔCq, Pfaffl, efficiency
├── reference_genes/         geNorm, NormFinder, delta-Ct, BestKeeper, consensus ranking
├── validation/              rules, engine, and readiness reports
├── statistics/              intervals and replicate-aware models
├── provenance/              hashes, versions, and parameters
├── plotting/                matplotlib-based plots
├── reporting/               combined researcher HTML/JSON report
├── enzyme_kinetics/         Michaelis-Menten fitting, inhibition classification
├── fermentation_kinetics/   Monod fitting, profile-likelihood CI, identifiability, ODE solving
├── metabolic_flux/          COBRApy FBA, blocked reactions, dead-end detection
├── potency/                 dose-response models, IC50/EC50, POT audit rules
├── docking/                 PDB parsing, scoring, rigid-body search, DCK audit rules
├── sequence/                AB1/FASTA/FASTQ, alignment, variants, MSA, SEQ audit rules
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
python scripts/validate_public_data.py
```

Scientific changes should include a cited rationale, boundary tests, and—when
possible—comparison with independently produced reference outputs.

## License

MIT. See [LICENSE](LICENSE).
