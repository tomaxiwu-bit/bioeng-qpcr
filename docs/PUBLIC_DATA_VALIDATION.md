# Public-data validation

This benchmark is a reproducible retrospective validation against public,
experiment-derived data. It is stronger than the synthetic unit tests, but it
is not a substitute for prospective testing by an independent laboratory.

Run it from the repository root:

```bash
python scripts/validate_public_data.py
```

The script downloads four files pinned by commit and SHA-256, writes detailed
results to `.validation-results/`, and produces three independent SVG figures
(agreement, Bland-Altman bias, and error distribution). Existing local clones
can be supplied with `--pcr-root`, `--rdml-root`, and `--openadmet-root`.
OpenADMET download hashes pin upstream LF bytes. Two separately pinned CRLF
hashes accept Windows Git checkouts of those same text files; no other content
change is accepted. The summary records actual input hashes. Earlier pins used
only the CRLF form and rejected clean raw downloads; 0.6.7 corrects this.

## Datasets and attribution

1. **qPCR relative expression.** The 12-observation `ct1` human brain/kidney
   dataset distributed with the CRAN `pcr` package, derived from the Livak
   comparative Cq example. The package is GPL-3.0; the benchmark downloads the
   file at run time. The separate workbench demonstration bundles a converted
   24-measurement subset with attribution and a full GPL-3.0 license; it is not
   relicensed as MIT. Relevant publications:
   [Livak and Schmittgen (2001)](https://doi.org/10.1006/meth.2001.1262) and
   [Ahmed and Kim (2018)](https://doi.org/10.7717/peerj.4473).
2. **RDML ingestion.** The RDML Consortium/Untergasser large-DNA-dilution
   experiment contains instrument-derived raw amplification traces and is part
   of the MIT-licensed RDML-Python validation collection. Source and methods:
   [RDML-Python experiments](https://github.com/RDML-consortium/rdmlpython/tree/main/experiments)
   and the accompanying
   [qPCR validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13026326/).
3. **IC50 fitting.** OpenADMET/Octant CYP3A4 inhibition data provide 1,340
   compound summaries and 16,931 well-level measurements from 12-point
   dose-response assays. Data are CC BY 4.0. Source:
   [Zenodo DOI 10.5281/zenodo.21450795](https://doi.org/10.5281/zenodo.21450795)
   and [OpenADMET source repository](https://github.com/OpenADMET/Octant_CYP_blog_post).

## Baseline results (2026-09-20)

| Check | Result |
|---|---:|
| CRAN `ct1` kidney-vs-brain fold change | 5.637283 (exact agreement) |
| RDML imported wells / raw traces | 294 / 294 |
| OpenADMET curves meeting documented QC filter | 509 |
| Curves evaluable with at least 8 concentrations | 508 |
| Successful 4PL fits | 508 / 508 (100%) |
| pIC50 mean absolute error | 0.0931 log units |
| pIC50 root mean squared error | 0.1591 log units |
| Pearson / Spearman agreement | 0.9428 / 0.9636 |
| Median fit R-squared | 0.9795 |

The OpenADMET comparison deliberately uses the same model family (4PL), but an
independent implementation. The reference analysis and this project do not
share identical preprocessing or uncertainty models, so the reference CI hit
rate is reported diagnostically and is not used as a release gate.
Specifically, `reference_ci_hit_rate` asks whether this implementation's point
estimate lies inside the reference analysis interval. It is **not** the coverage
probability of this implementation's confidence intervals. The 100% success
rate applies only to the selected, evaluable curves, not all 1,340 compounds.
These are retrospective baseline results, not proof of independent laboratory use.

From 0.6.7, `summary.json` records the actual executing software/dependency
versions and timestamp. The detailed CSV retains excluded designs and fit
failures with status/reason, alongside successful fits. Historical summaries
are not relabelled as results of a new software version; rerun the benchmark.

The first run also exposed two real-data defects that synthetic tests had
missed: RDML `-1.0` efficiency sentinels were rejected, and weakly identified
dose-response plateaus could overflow while converting a log-scale confidence
interval. Both now have regression tests; unbounded intervals are omitted with
an explicit warning rather than crashing the analysis.

## Release gates

The [0.6.8 rerun](validation/0.6.8/summary.json) was executed on 2026-11-19.
It retains its own environment and per-curve results; the older 0.6.7 output
below remains a historical record.

The [0.6.7 rerun](validation/0.6.7/summary.json) passed the same gates with the
same numerical results (508/508 evaluable fits, pIC50 MAE 0.0930770405). The
adjacent CSV includes the one design exclusion, and figures/actual environment
are archived with it. See its `provenance.created_at` for the UTC execution time.

The executable benchmark currently requires all of the following:

- exact agreement for the documented qPCR fold-change example;
- the pinned RDML structural counts to match;
- at least 99% dose-response fit success;
- pIC50 MAE no greater than 0.15 log units;
- Pearson correlation at least 0.90;
- median curve R-squared at least 0.95.

These thresholds establish the first public-data baseline. They were not
preregistered and must not be presented as a blinded prospective validation.
