"""Readers for commercial qPCR instrument export files.

Three mainstream vendor exports are ingested into the same
:class:`~bioeng_qpcr.domain.entities.Experiment` model the RDML and tidy-CSV
readers produce, so validation, quantification and the amplification-curve
pipeline run unchanged:

* ``ThermoQuantStudio`` -- the ``Results`` worksheet of an ``.xlsx`` export
  (metadata block followed by a columnar results table with one row per well
  per target).  Cq, sample/target names, task classification (NTC vs unknown)
  and omitted-well flags are captured.
* ``BioRadCFX`` -- the result worksheet (``Well/Fluor/Target/Content/Sample/
  Cq/SQ`` columns) plus a key/value ``Run Information`` worksheet.  Wells that
  Bio-Rad reports as negative controls are marked ``ntc``, and the
  ``SQ`` (Starting Quantity) column maps to ``Well.n0``.
* ``LightCycler480`` -- the raw fluorescence text export (one row per well
  per cycle, ``Sample position/Cycle number/Fluorescence data``), which yields
  raw amplification traces that feed the amplification-curve pipeline.

The dispatcher :func:`read_vendor_export` sniffs the source from its
structure, not its extension, so an ``.xlsx`` can be either QuantStudio or
CFX and a text export is recognised as LightCycler raw fluorescence.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bioeng_qpcr.domain import ReplicateType
from bioeng_qpcr.domain.entities import Assay, Experiment, RawTrace, Sample, Well
from bioeng_qpcr.io.diagnostics import CSVValidationIssue

__all__ = ["read_vendor_export"]

_POSITION_RE = re.compile(r"^([A-Ha-h])(\d+)$")

#: RDML-style NA strings appearing in vendor result cells.
_NA_TEXT = {"undetermined", "na", "n/a", "undefined", "no cq", "no data"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text.casefold() != "nan" else ""


def _normalize_position(value: str) -> str:
    """Normalise a plate address (``A1`` -> ``A01``); pass through otherwise."""
    match = _POSITION_RE.match(value)
    if not match:
        return value
    return f"{match.group(1).upper()}{int(match.group(2)):02d}"


def _grid(sheet: Any) -> list[list[Any]]:
    """All cell values of an openpyxl worksheet as a row-major list."""
    return [[cell.value for cell in row] for row in sheet.iter_rows()]


def _grep_header_row(grid: list[list[Any]], needle: str) -> int | None:
    """First row whose cells contain an exact match for *needle*."""
    for index, row in enumerate(grid):
        if any(_clean(cell).casefold() == needle for cell in row):
            return index
    return None


def _header_map(row: list[Any]) -> dict[str, int]:
    """Map lower-cased column headers to their column index."""
    return {
        _clean(cell).casefold(): col for col, cell in enumerate(row) if _clean(cell)
    }


def _to_float(value: Any) -> float | None:
    text = _clean(value)
    if not text or text.casefold() in _NA_TEXT:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if not np.isfinite(number) else number


def _make_issues() -> list[CSVValidationIssue]:
    return []


# ---------------------------------------------------------------------------
# Thermo Fisher QuantStudio
# ---------------------------------------------------------------------------


def _read_quantstudio(
    workbook: Any, *, reference_targets: list[str]
) -> Experiment:
    issues = _make_issues()
    if "Results" not in workbook.sheetnames:
        raise ValueError("Not a QuantStudio export: no 'Results' worksheet found")

    grid = _grid(workbook["Results"])
    header_row = _grep_header_row(grid, "well")
    if header_row is None:
        raise ValueError("QuantStudio 'Results' worksheet has no 'Well' header")
    header = _header_map(grid[header_row])
    for required in ("well", "sample name", "target name"):
        if required not in header:
            raise ValueError(
                f"QuantStudio 'Results' missing required column {required!r}"
            )

    rows: list[tuple[str, str, str, float | None, str]] = []
    n_skipped = 0
    for row in grid[header_row + 1 :]:
        if not any(_clean(cell) for cell in row):
            continue
        if "omit" in header and _clean(row[header["omit"]]).casefold() in {
            "1",
            "true",
        }:
            n_skipped += 1
            continue
        position = _normalize_position(
            _clean(
                row[header["well position"]]
                if "well position" in header
                else row[header["well"]]
            )
        )
        sample_name = _clean(row[header["sample name"]])
        target_name = _clean(row[header["target name"]])
        if not sample_name or not target_name:
            n_skipped += 1
            continue

        task = _clean(row[header["task"]] if "task" in header else "").casefold()
        cq = _to_float(row[header["ct"]] if "ct" in header else None)
        cq_status = (
            "no_template"
            if task == "ntc"
            else ("undetermined" if cq is None else "determined")
        )
        rows.append((position, sample_name, target_name, cq, cq_status))

    if not rows:
        raise ValueError("QuantStudio export contains no usable result rows")

    # Multiplex wells put several targets under one position; the domain model
    # requires unique positions, so suffix repeats like the CFX/RDML readers.
    position_counts = Counter(r[0] for r in rows)
    wells = [
        Well(
            position=(
                position
                if position_counts[position] == 1
                else f"{position}:{target_name}"
            ),
            sample_name=sample_name,
            target_name=target_name,
            cq=cq,
            replicate_type=ReplicateType.SINGLE,
            cq_status=cq_status,
        )
        for position, sample_name, target_name, cq, cq_status in rows
    ]

    reference = set(reference_targets or [])
    return Experiment(
        name="quantstudio",
        wells=wells,
        samples={
            name: Sample(name=name) for name in sorted({w.sample_name for w in wells})
        },
        assays={
            target: Assay(name=target, is_reference=target in reference)
            for target in sorted({w.target_name for w in wells})
        },
        metadata={
            "vendor": "ThermoQuantStudio",
            "input_diagnostics": [i.to_dict() for i in issues],
            "n_wells": len(wells),
            "skipped_rows": n_skipped,
        },
    )


# ---------------------------------------------------------------------------
# Bio-Rad CFX Maestro
# ---------------------------------------------------------------------------


def _read_cfx(workbook: Any, *, reference_targets: list[str]) -> Experiment:
    issues = _make_issues()

    result_sheet = _find_cfx_result_sheet(workbook)
    if result_sheet is None:
        raise ValueError("Not a CFX export: no result worksheet (Well/Cq) found")

    grid = _grid(workbook[result_sheet])
    header_row = _grep_header_row(grid, "well")
    if header_row is None:
        raise ValueError("CFX result worksheet has no 'Well' header")
    columns = _header_map(grid[header_row])

    run_info: dict[str, str] = {}
    for sheet_name in workbook.sheetnames:
        if sheet_name == result_sheet:
            continue
        for row in _grid(workbook[sheet_name])[:60]:
            if not row:
                continue
            key = _clean(row[0])
            value = _clean(row[1] if len(row) > 1 else None)
            if key:
                run_info[key] = value

    rows: list[tuple[str, str, str, float | None, str, float | None, str]] = []
    n_skipped = 0
    for row in grid[header_row + 1 :]:
        if not any(_clean(cell) for cell in row):
            continue
        sample_name = _clean(row[columns["sample"]] if "sample" in columns else "")
        target_name = _clean(row[columns["target"]] if "target" in columns else "")
        if not sample_name or not target_name:
            n_skipped += 1
            continue
        position = _normalize_position(_clean(row[columns["well"]]))
        content = _clean(
            row[columns["content"]] if "content" in columns else ""
        ).casefold()
        cq = _to_float(row[columns["cq"]] if "cq" in columns else None)
        sq = _to_float(row[columns["sq"]] if "sq" in columns else None)
        fluor = _clean(row[columns["fluor"]] if "fluor" in columns else "")
        is_ntc = content in {"neg ctrl", "neg", "ntc", "no template", "no_template"}
        cq_status = (
            "no_template"
            if is_ntc
            else ("undetermined" if cq is None else "determined")
        )
        rows.append((position, sample_name, target_name, cq, cq_status, sq, fluor))

    if not rows:
        raise ValueError("CFX export contains no usable result rows")

    # Multiplex wells put several targets under one position; the domain
    # model requires unique positions, so suffix repeats like the RDML reader.
    position_counts = Counter(r[0] for r in rows)
    fully_positions = {
        pos: (count == 1) for pos, count in position_counts.items()
    }
    wells = [
        Well(
            position=(
                position
                if fully_positions[position]
                else f"{position}:{target_name}"
            ),
            sample_name=sample_name,
            target_name=target_name,
            cq=cq,
            cq_status=cq_status,
            n0=sq,
        )
        for position, sample_name, target_name, cq, cq_status, sq, _fluor in rows
    ]

    reference = set(reference_targets or [])
    dye_by_target: dict[str, str] = {}
    for _pos, _sample, target, _cq, _cq_status, _sq, fluor in rows:
        if fluor and target not in dye_by_target:
            dye_by_target[target] = fluor
    return Experiment(
        name="cfx",
        wells=wells,
        samples={
            name: Sample(name=name) for name in sorted({w.sample_name for w in wells})
        },
        assays={
            target: Assay(
                name=target,
                is_reference=target in reference,
                dye_chemistry=dye_by_target.get(target),
            )
            for target in sorted({w.target_name for w in wells})
        },
        metadata={
            "vendor": "BioRadCFX",
            "run_information": run_info,
            "input_diagnostics": [i.to_dict() for i in issues],
            "n_wells": len(wells),
            "skipped_rows": n_skipped,
        },
    )


def _find_cfx_result_sheet(workbook: Any) -> str | None:
    """Locate the CFX result worksheet by its header content."""
    for name in workbook.sheetnames:
        for row in _grid(workbook[name])[:5]:
            cells = {_clean(cell).casefold() for cell in row}
            has_cq_target = "cq" in cells and ("fluor" in cells or "target" in cells)
            if "well" in cells and has_cq_target:
                return name
    return None


# ---------------------------------------------------------------------------
# Roche LightCycler 480 raw fluorescence text
# ---------------------------------------------------------------------------


def _is_lc480_text(first_line: str) -> bool:
    lowered = first_line.casefold()
    return (
        "sample position" in lowered
        and "cycle number" in lowered
        and "fluorescence data" in lowered
    )


def _read_lc480_dataframe(path: Path) -> pd.DataFrame:
    """Read LC480 tab-export, tolerating non-UTF-8 encodings."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"{path.name}: could not decode as UTF-8 or latin-1")


def _read_lc480(path: Path) -> Experiment:
    df = _read_lc480_dataframe(path)
    df.columns = [str(c).strip() for c in df.columns]
    lowered = {c.casefold(): c for c in df.columns}
    pos_col = lowered.get("sample position")
    cycle_col = lowered.get("cycle number")
    fluor_cols = [
        c for k, c in lowered.items() if "fluorescence data" in k
    ]
    if pos_col is None or cycle_col is None:
        raise ValueError(
            "Not a LightCycler 480 raw-fluorescence export: missing "
            "'Sample position' or 'Cycle number' columns"
        )
    if not fluor_cols:
        raise ValueError(
            "Not a LightCycler 480 raw-fluorescence export: no "
            "'Fluorescence data' column found"
        )
    if len(fluor_cols) > 1:
        raise ValueError(
            "multi-channel LightCycler 480 text export detected with "
            f"{len(fluor_cols)} fluorescence columns; only single-dye "
            "exports are supported - split channels or export as RDML"
        )
    fluor_col = fluor_cols[0]

    name_col = lowered.get("sample name")
    wells: dict[str, str] = {}
    traces: dict[str, RawTrace] = {}
    for position, group in df.groupby(pos_col, sort=True):
        pos = _normalize_position(_clean(position))
        cyc = np.asarray(group[cycle_col], dtype=float)
        fluor = np.asarray(group[fluor_col], dtype=float)
        if not np.all(np.isfinite(cyc)) or not np.all(np.isfinite(fluor)):
            continue
        try:
            trace = RawTrace(position=pos, cycles=cyc, fluorescence=fluor)
        except ValueError:
            continue
        traces[pos] = trace
        sample_name = (_clean(group[name_col].iloc[0]) if name_col else "") or (
            f"well_{pos}"
        )
        wells[pos] = sample_name

    if not traces:
        raise ValueError(
            "LightCycler 480 export contains no usable amplification traces"
        )

    return Experiment(
        name=path.stem,
        wells=[
            Well(
                position=pos,
                sample_name=sample_name,
                target_name="unknown",
                cq_status="undetermined",
            )
            for pos, sample_name in sorted(wells.items())
        ],
        samples={
            name: Sample(name=name) for name in sorted(set(wells.values()))
        },
        assays={"unknown": Assay(name="unknown")},
        traces=traces,
        metadata={
            "vendor": "LightCycler480",
            "input_diagnostics": [],
            "n_wells": len(traces),
            "wells_with_trace": len(traces),
        },
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def read_vendor_export(
    path: str | Path,
    *,
    reference_targets: list[str] | None = None,
) -> Experiment:
    """Read a commercial instrument export into an :class:`Experiment`.

    The format is detected structurally:

    * ``.xlsx`` containing a ``Results`` worksheet -> QuantStudio.
    * ``.xlsx`` containing a result sheet (``Well``/``Cq``) -> Bio-Rad CFX.
    * Tab-separated raw-fluorescence text -> Roche LightCycler 480.

    Parameters
        path:
            Path to the vendor export file.
        reference_targets:
            Target names to mark as reference assays (QuantStudio / CFX).

    Returns
        :class:`~bioeng_qpcr.domain.entities.Experiment` with wells, samples,
        assays and (for LC480) raw amplification traces populated.

    Raises
        ValueError:
            When the file does not match a supported vendor export.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xls":
        raise ValueError(
            f"{path.name}: legacy binary .xls is not supported; re-export from "
            "the instrument software as .xlsx"
        )
    if suffix == ".xlsx":
        try:
            import openpyxl  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - guarded by dependency
            raise ValueError(
                "reading .xlsx vendor exports requires the 'openpyxl' package"
            ) from exc
        workbook = openpyxl.load_workbook(path, data_only=True)
        try:
            if "Results" in workbook.sheetnames:
                return _read_quantstudio(
                    workbook, reference_targets=reference_targets or []
                )
            if _find_cfx_result_sheet(workbook) is not None:
                return _read_cfx(workbook, reference_targets=reference_targets or [])
            raise ValueError(
                f"{path.name}: unrecognised Excel vendor export (expected a "
                "QuantStudio 'Results' sheet or a CFX result sheet)"
            )
        finally:
            workbook.close()

    with open(path, encoding="utf-8", errors="replace") as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line.strip()
                break
    if _is_lc480_text(first_line):
        return _read_lc480(path)
    raise ValueError(
        f"{path.name}: unrecognised vendor export; expected a QuantStudio .xlsx, "
        "a CFX .xlsx, or a LightCycler 480 raw-fluorescence text file"
    )