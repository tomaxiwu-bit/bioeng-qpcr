# Public example data for the qPCR workbench

These are observations from public sources, not synthesized amplification or
melting curves. Re-extract with `scripts/prepare_workbench_examples.py` and the
pinned original files named in that script. SHA-256 mismatches abort extraction.
Every example JSON also carries the original URL, license and limitations.

| File | Origin | Transformation | License |
|---|---|---|---|
| `relative-ct.json` | [CRAN pcr ct1.csv](https://github.com/cran/pcr/blob/6510657abfc6c92a4bb44ad3e131d10acb6ae6ec/inst/extdata/ct1.csv) | All 12 rows × 2 genes; six technical observations per tissue; illustrative plate positions | GPL-3.0, see `PCR-DATA-GPL-3.0.txt`; applies to these third-party example data separately from project code |
| `rdml-dilution.json` | [RDML Consortium large_DNA_dilutions.rdml](https://github.com/RDML-consortium/rdmlpython/blob/fce7a44e8b47b90759c646b557de568f7af77e75/experiments/untergasser/large_DNA_dilutions.rdml) | First run, FSTL_1_F_109, six dilution levels and two buccal samples; 26 measurements, retaining original 384-well positions, reported Cq, amplification and melting points, and primer sequences | MIT, see `RDML-MIT-LICENSE.txt` |

The RDML quantity unit `cop` is preserved literally. The workbench does not
reinterpret a source sample quantity as copies per reaction or copies per µL.
Do not infer a volume correction unless the real protocol establishes it.

Missing controls, primer sequences, raw amplification and melting traces remain
missing. The selected RDML subset contains no assigned positive/negative control
wells. Both examples therefore retain QC evidence gaps. Temperature programs
and reagent composition are explicitly marked as teaching configurations, not
as reconstructions of an executed experiment; the RDML example's 10 µL total
reaction volume comes from its source.

Replace examples with real measurements through the acquisition page.
`workbench/acquisition.py::read_acquisition` is the file adapter;
`validate_measurements` is the shared boundary for a future vendor SDK.
Never remove example attribution to present a replay as a real hardware run.
