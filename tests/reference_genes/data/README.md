# NormFinder regression fixture

`colon_normqpcr.csv` stores the linear expression matrix for 13 genes in 40
samples from the `Colon` dataset distributed by NormqPCR 1.58.0, together with
sample identifiers and normal/tumour group labels. It is third-party test data,
not newly measured data and not MIT-licensed project code.

* Package: [NormqPCR](https://www.bioconductor.org/packages/release/bioc/html/NormqPCR.html),
  authors Matthias Kohl, James Perkins, Nor Izayu Abdul Rahman; DOI
  [10.18129/B9.bioc.NormqPCR](https://doi.org/10.18129/B9.bioc.NormqPCR).
* Pinned source: `NormqPCR_1.58.0.tar.gz`, Bioconductor 3.23; URL/checksum are
  recorded in `packaging/third-party-sources.json`. Archive SHA-256:
  `f556aab9ae67d7d02deda192df6d1adf65520b0c9125797582a5b74839864a35`.
* Original member: `NormqPCR/data/Colon.RData`, SHA-256
  `d2f67f7fd62b28ed9776277ce79ec93e558eae5a78644926e02b844b5f3ddfaf`.
  Its documentation is `NormqPCR/man/Colon.Rd`.
* CSV SHA-256:
  `a0faf21aa9d5b107e9252732998014712fa71b1e8a3c565cf6688d1485fdde08`.
* Upstream package license: LGPL-3; retain the adjacent LGPL/GPL license texts.
  The upstream package cites Andersen, Jensen and Ørntoft, Cancer Research 64,
  5245–5250 (2004), as the experimental source.

The CSV rearranges the expression matrix into one sample per row. The Python
regression converts each linear value to `40 - log2(expression)` to use the Cq
convention; that additive constant does not change model stability. Expected
stability values in `test_model_normfinder.py` follow the documented NormqPCR
comparison. The original export command/R session was not recorded; source
archive, data member, current CSV and transformation are now pinned explicitly.
An R-side re-extraction comparison remains useful before publication.
