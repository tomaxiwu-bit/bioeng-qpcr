# Contributing and support

Questions, reproducible bug reports and focused pull requests are welcome at
[GitHub Issues](https://github.com/tomaxiwu-bit/bioeng-qpcr/issues) and
[Pull requests](https://github.com/tomaxiwu-bit/bioeng-qpcr/pulls).
Support is best effort; no response-time or instrument-support guarantee is offered.
Do not upload identifiable participant information, private laboratory data or credentials.

## Reproduce a problem

Include the software version, OS, entry point (browser/desktop/CLI), exact input
format, parameters, expected result and observed result. Attach a minimal public
or synthetic example and the JSON backup when appropriate. For a numerical
disagreement, supply the reference calculation, units, tolerance and treatment
of missing values/replicates. Include `BUILD_INFO.json` for a portable-build issue.
Use the environment recorded in the result, not the current application's version,
when describing a historical analysis.

## Development

Python 3.10–3.13 is supported. Python 3.12 is used for the Windows release.

```bash
uv sync --locked --extra dev
uv run --locked --extra dev ruff check .
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev python scripts/verify_examples.py
```

Headless Qt tests use `QT_QPA_PLATFORM=offscreen`. GUI and FBA dependencies remain
optional for library users. Windows source-release commands are documented in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Keep scientific calculations in the shared library. Update callers, import/export
contracts and documentation together; test a numerical change against an
independent expected result and add a small regression for a bug. Preserve raw
data and historical records. Document changed assumptions, units and failure
conditions. Do not use successful execution alone as evidence of scientific validity.

New datasets need a public source URL, pinned revision/checksum, reuse license,
extraction steps and limitations. Contributions to project code use the MIT
license; third-party data and components keep their own terms. Disclose AI-assisted
contributions and the human verification performed; do not fabricate author or
research-use evidence. The repository's CI tests supported Python versions,
installed wheels, public datasets and the frozen Windows application.
