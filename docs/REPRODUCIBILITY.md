# Reproducibility and local releases

## Reproduce an analysis

Use a browser record JSON backup or qPCR project JSON. They contain raw input
hashes, parameters, measurements/results and history. From 0.6.7, new browser
analyses and each qPCR `analysis_versions` entry include a `provenance` object
with executing software/dependency versions, UTC time, Python/platform, input
hash and settings. In a frozen release `runtime.build` identifies the source
tree, Git baseline and dirty-workspace state. `version: 1` in an analysis backup
is its **schema version**, not the application version.

No version is invented for old records. Imported backups preserve their claimed
historical environment and remain unverified; rerun to create current results.
qPCR restore recalculates curves/QC and requires reconfirming quantification.
The project revision is separate from the software version. Desktop stand-alone
panels do not participate in browser automatic archiving.

`input_hash` identifies the ordered design/measurement JSON for qPCR, or the
role-to-file-hash mapping for generic browser tools. Each original upload retains
its own SHA-256. Replay raw hashes identify the upstream file, while the combined
input hash identifies the processed subset. Parameters are recorded separately.
For an unpackaged source run, `runtime.build` is null; preserve the checkout and
lockfile yourself. Neither a hash nor an imported manifest proves authenticity.

## Locked environment and verification

```bash
uv sync --locked --extra dev --extra desktop-build
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev python scripts/verify_examples.py
uv run --locked --extra dev python scripts/validate_public_data.py
```

The public benchmark downloads commit/SHA-pinned data. Cache paths can be set
with `--cache-dir`; existing clones can use the documented `--*-root` options.
Detailed CSVs retain design exclusions and fit failures; summary JSON captures
the running environment. Baseline numbers in `PUBLIC_DATA_VALIDATION.md` are
dated historical results until rerun. Confidence-interval agreement is diagnostic,
not a coverage guarantee. See that document for filtering and release thresholds.

The CI keeps a broad Python compatibility matrix using current dependencies,
and a separate Python 3.12 locked job. This distinguishes compatibility testing
from the fixed release environment. An identical lockfile does not promise
bit-for-bit binaries across operating systems or compiler versions.

## Build the Windows portable release

Run these commands in PowerShell from the repository root with Python 3.12:

```powershell
uv sync --locked --extra dev --extra desktop-build
uv run --locked --extra desktop-build python scripts/download_release_sources.py
uv run --locked --extra desktop-build python -m PyInstaller --clean --noconfirm bioeng-qpcr.spec
./scripts/build_portable.ps1 -OutputDirectory releases/0.6.8
```

Network is needed only to install dependencies and download the pinned sources.
The spec prepares notices, the exact application source ZIP and build metadata
offline; missing/modified third-party sources stop the build. Do not change
dependencies without updating their source pins. `build_portable.ps1` refuses
existing release paths and runs the frozen scientific self-check before staging.

The release contains root `LICENSE`, `THIRD_PARTY_NOTICES.md`, `BUILD_INFO.json`,
selected documentation and `_internal/licenses/` and `_internal/sources/`.
`BUILD_INFO.json` hashes each application source file and records Git baseline,
dirty state, dependencies and third-party source hashes. The source snapshot
excludes local databases, user analysis files, `.git`, credentials and caches.
A local dirty build is explicitly identified; it is not presented as a public tag.

Verify the outer `.zip.sha256` and every file against `MANIFEST.sha256` after
extraction. Keep the entire folder together. The executable is unsigned and
does not auto-update. Build checks enforce the absence of unused Qt PDF,
virtual-keyboard and QML/Quick libraries; Qt widgets, image preview and plotting
are exercised by tests and the frozen self-check.

To modify Qt/PySide/GLPK, start from the matching source archive and its upstream
build instructions. Install the rebuilt compatible library in an isolated build
environment, retain its notices/source modifications, update source pins and
rebuild with this spec. The included project source can also be run directly in
a Python environment instead of using the frozen executable.

## Research-use boundary

Public-data regression, successful tests and a reproducible release do not prove
independent laboratory adoption or clinical validity. JOSS also requires evidence
of research use and sustained public development. Real research cases, authorship,
affiliations and human verification must come from the researchers concerned.
