bioeng-qpcr portable Windows release
====================================

This application is for research use only. It is not a clinical diagnostic
device.

Installation
------------
1. Extract the entire ZIP archive to a local folder.
2. Keep bioeng-qpcr.exe and the adjacent _internal folder together.
3. Run bioeng-qpcr.exe from the extracted folder.
4. The first panel is the comprehensive lab workbench. Click its button to open
   the module directory in your default browser; keep the desktop app running while
   using it. The server listens only on 127.0.0.1.

Workbench v0.6.7
---------------
The /lab home lists five main research modules: qPCR, potency, enzyme kinetics,
fermentation and FBA. Generic previews are auxiliary tools; simplified docking
and sequence alignment have legacy links. Old records and APIs remain supported.
Enzyme kinetics, fermentation,
and FBA now also run in the browser with shared history, backup and CSV export.
Their original Windows panels remain available as independent analysis tools.
The qPCR workbench includes eight experiment stages, two public-data demonstrations, five
planning templates, and six chart/report themes are included. Templates
provide editable designs; they do not contain fabricated measurement data.
SVG/PDF and 300 dpi PNG exports are available after analysis.

Browser analysis tools automatically archive successful results, parameters,
and original files in the local database. Use the history bar to restore or
export a record. Imported backup results are marked unverified until rerun.
qPCR project JSON backups restore into a new project; current curves and QC
are recalculated and quantification must be confirmed again.
Use /history to search all browser analysis records and restore them. The original
enzyme, fermentation and FBA desktop pages have a browser-workbench button;
independent desktop calculations do not write to this archive.
JSON backup imports are limited to 64 MiB. Back up the SQLite database for
larger histories. Unrun drafts remain in browser memory only.

Projects are saved to %LOCALAPPDATA%\bioeng-qpcr\workbench.sqlite3 by default,
outside this portable program folder. Moving the program does not migrate
that database. Keep a backup or export project JSON before moving computers.
The BIOENG_QPCR_WORKBENCH_DB environment variable can specify another location.

Acquisition currently supports CSV/RDML imports and public-data replay.
Live control of an instrument requires its vendor SDK.

Docking review
--------------
Search diagnostics, structural checks and recommended next steps are shown
together. A score gap never proves convergence. Grid completion does not
establish global convergence. Old records remain unchanged and require a new
run for the updated review. Even passed checks do not establish binding affinity.

Do not copy bioeng-qpcr.exe by itself. The executable depends on libraries in
_internal, including Qt, SciPy, Matplotlib, and Pillow. Moving only the EXE can
cause errors such as "No module named 'PIL'".

Integrity
---------
MANIFEST.sha256 contains the SHA-256 checksum of every packaged file. The
release ZIP has a separate .sha256 file beside it.

Version records and licenses
---------------------------
New browser analyses and qPCR analysis-history entries record the actual software,
scientific dependencies, input hashes and parameters. Old records do not acquire
today's software version. Imported backups remain historical claims until rerun.
BUILD_INFO.json identifies the build environment, source hashes, Git baseline and
any uncommitted changes. It is not a public release tag.

Project source is MIT licensed; the bundle is not MIT-only. See LICENSE and
THIRD_PARTY_NOTICES.md. Original notices are in _internal/licenses and matching
copyleft-component sources plus the exact application source ZIP in
_internal/sources. Keep these materials with redistributed copies. LGPL library
replacement and rebuilding are permitted; see docs/REPRODUCIBILITY.md.

If a chart cannot be generated, the browser shows a warning and keeps numerical
results and that warning in the archive. Read the warning before using the report.

This portable release is not code-signed and does not provide automatic
updates or an uninstaller.


P1 analysis additions
---------------------
qPCR supports multiple references and measured efficiency correction.
Enzyme kinetics includes background global inhibition model comparison.
Potency review supports search, filtering, pagination, full CSV and stored-fit
plots for any compound without refitting. Older records need one fresh run
to store plotting parameters; imported backups remain unverified until rerun.

P2 workbench additions
---------------------
Browser enzyme fitting includes MM and five-model global inhibition comparison.
Browser Monod analysis includes confidence intervals, identifiability and residuals.
FBA compares the original model with up to 20 explicit reaction-bound changes;
the source model file is never edited. The public E. coli core model is bundled
with source attribution. Larger models remain supported by the desktop tools.
