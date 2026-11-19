# Third-party notices and source access

The project's original source code is MIT licensed: see `LICENSE`. This does
not relicense dependencies or example data. The Windows bundle includes
components under BSD, MIT, Apache, PSF, MPL, LGPL and GPL terms. In particular,
the FBA solver stack includes GPL-3.0 swiglpk/GLPK; the whole portable bundle
must not be described as MIT-only. Preserve the accompanying licenses and
sources when redistributing it.

The NormFinder regression fixture has its own upstream LGPL-3 provenance and
full license texts in `tests/reference_genes/data/`; the original NormqPCR source
archive is included with the portable release. It is test data, not project-authored
experimental measurements. The original CSV export session is not available;
its current bytes and upstream data member are pinned for subsequent comparison.

## Included release materials

* `_internal/licenses/DISTRIBUTIONS.json`: build-environment distribution names,
  versions, declared licenses and paths to copied notices. It includes build/test
  tools as well as runtime dependencies; it is not a claim that every installed
  distribution executes in the application.
* `_internal/licenses/`: original wheel license/copyright files and package
  metadata, Python's full license, and notices extracted from the pinned source
  archives. Upstream commercial-license alternatives are preserved as upstream
  material; their presence does not grant this project a commercial Qt license.
* `_internal/sources/bioeng-qpcr-source.zip`: the exact application sources,
  tests, examples, lockfile and build scripts used for this local release.
* `_internal/sources/`: unmodified version-matched Qt/PySide, COBRApy, libSBML,
  swiglpk, GLPK and certifi source archives. Their origins and SHA-256 values
  are in `packaging/third-party-sources.json` and `BUILD_INFO.json`.

## Important components

| Component | Terms / source |
|---|---|
| Python | PSF license; included Python license |
| NumPy, SciPy, pandas, Matplotlib, Pillow | Their included upstream notices; bundled subcomponents have additional terms |
| PySide6 / Shiboken / Qt Core, GUI, Widgets, Network, SVG and image plugins | Open-source LGPL/GPL alternatives as specified in the included source and notices; see [Qt obligations](https://www.qt.io/development/open-source-lgpl-obligations) |
| COBRApy | LGPL-2.0-or-later OR GPL-2.0-or-later |
| libSBML | Upstream LGPL terms, with bundled third-party notices |
| swiglpk / GLPK | GPL-3.0; full corresponding upstream sources included |
| certifi | MPL-2.0; source included |
| PyInstaller bootloader | GPL with bootloader distribution exception; see copied upstream notice |

This QWidget application does not need Qt PDF rendering, the virtual keyboard
or QML/Quick. The release hook excludes their optional GUI plugins and verifies
that their Qt libraries are absent. Images are decoded by Pillow. The release
also omits the unused optional Mesa software OpenGL fallback.

Users may inspect, modify and rebuild the application, replace dynamically
loaded LGPL libraries with compatible versions, and reverse engineer as needed
to debug those modifications. No project term restricts those rights. Keep
replacement DLLs/ABI and PySide bindings compatible; alternatively rebuild from
the included application sources and matching upstream archives using the
instructions in `docs/REPRODUCIBILITY.md`. No signing or integrity enforcement
blocks modified libraries; checksum manifests are verification aids only.

## Example data

`src/bioeng_qpcr/web/examples/qpcr_workbench/README.md` and adjacent full license
files document the GPL-3.0 `pcr` Ct subset and MIT RDML subset. GPL data remain
separate from MIT project code. `KINETICS_SOURCES.md` attributes the COBRApy
tutorial model. The optional OpenADMET benchmark downloads CC BY 4.0 data from
the pinned URLs in `scripts/validate_public_data.py`; it is not bundled as user
experimental data. Synthetic planning templates are labelled and are not
measurements. Before distributing additional datasets, document their terms.

These materials document the build and upstream notices; they do not certify
every possible downstream redistribution or modified dependency combination.
