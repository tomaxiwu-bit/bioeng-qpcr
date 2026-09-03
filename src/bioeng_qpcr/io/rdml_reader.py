"""Strict, auditable reader for the RDML qPCR interchange format.

RDML (Real-time PCR Data Markup Language) is the RDML consortium's open
XML format for exchanging complete qPCR experiments, including raw
amplification curves.  This reader ingests RDML versions 1.x into the
same :class:`~bioeng_qpcr.domain.Experiment` model the tidy-CSV reader
produces, so validation, quantification, and statistics work unchanged.

Container handling
------------------
Per the consortium's file-format notes, an RDML archive (extension
``.rdml`` or ``.rdm``) is a ZIP container whose primary member is
``rdml_data.xml``.  Uncompressed XML and gzip-compressed XML are also
accepted; the container is sniffed from magic bytes, not the extension.

Schema conventions implemented
------------------------------
* Measurements chain: ``experiment -> run -> react -> data``, where a
  ``react`` is one physical reaction (well) referencing a sample, and each
  ``data`` element references one target.
* ``data/cq`` uses ``-1.0`` as the "not available" sentinel; a missing or
  empty element means the same.  An absent Cq yields an ``undetermined``
  well.
* ``data/excl`` present means the instrument software excluded the entry
  from evaluation; the well is skipped and the exclusion recorded.
* ``sample/type`` ``ntc`` marks no-template controls.
* ``target/type`` ``ref`` marks reference genes.
* ``target/amplificationEfficiency`` is the per-cycle amplification
  factor (1.95 = 95 % efficiency) and is converted to the library's
  fractional convention (0.95).
* ``run/pcrFormat`` converts the row-first ``react`` id into a plate
  address such as ``A01`` when the format allows.
* ``sample/annotation`` with property ``group`` maps to the sample group,
  enabling treated-vs-control comparisons.

Conventions beyond the schema
-----------------------------
A ``react`` holding several ``data`` elements (multiplex measurement)
yields one well per target.  Positions beyond the first are suffixed with
the target id (``"A01:GAPDH"``) because the domain model requires unique
well positions; this is recorded as a diagnostic.

Technical replicates are derived from RDML's sample semantics: wells
sharing a sample id measure the same template solution, so wells of a
sample/target pair measured more than once are marked ``technical`` and
receive a ``"{sample}_{target}"`` replicate group.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from bioeng_qpcr.domain import (
    Assay,
    Experiment,
    InputLevel,
    RawTrace,
    ReplicateType,
    Sample,
    Well,
)
from bioeng_qpcr.io.diagnostics import CSVValidationError, CSVValidationIssue

__all__ = ["read_rdml"]

#: Issue codes that abort strict-mode ingestion.  Warning-level codes
#: (excluded data entries, multiplex position suffixes) are recorded but
#: never raise: they describe valid RDML, not corrupt input.
_ERROR_CODES: frozenset[str] = frozenset(
    {
        "DUPLICATE_REACT",
        "INVALID_CQ",
        "INVALID_EFFICIENCY",
        "INVALID_SAMPLE_TYPE",
        "INVALID_TARGET_TYPE",
        "INVALID_TRACE_POINT",
        "MISSING_SAMPLE_REF",
        "MISSING_TARGET_REF",
        "CONFLICTING_SAMPLE_GROUP",
        "UNKNOWN_REFERENCE_TARGET",
    }
)

#: Legal ``sample/type`` values in RDML 1.x.
_SAMPLE_TYPES: frozenset[str] = frozenset(
    {"unkn", "ntc", "nac", "std", "ntp", "nrt", "pos", "opt"}
)

#: Plausible per-cycle amplification factors (1.0 = no amplification,
#: 2.0 = perfect doubling).  Values outside this range cannot be valid.
_EFFICIENCY_FACTOR_RANGE = (1.0, 3.0)

_NTC_SAMPLE_TYPE = "ntc"
_REFERENCE_TARGET_TYPE = "ref"


# ---------------------------------------------------------------------------
# XML helpers (namespace-agnostic)
# ---------------------------------------------------------------------------


def _local_name(tag: str) -> str:
    """Strip any XML namespace from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    """First direct child whose local element name is *name*."""
    for child in element:
        if _local_name(child.tag) == name:
            return child
    return None


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    """All direct children whose local element name is *name*."""
    return [child for child in element if _local_name(child.tag) == name]


def _text(element: ET.Element | None, name: str) -> str | None:
    """Stripped text content of child *name*, or ``None`` if absent/empty."""
    if element is None:
        return None
    child = _child(element, name)
    if child is None:
        return None
    text = (child.text or "").strip()
    return text or None


def _ref_id(element: ET.Element | None) -> str:
    """Value of the ``id`` attribute of a reference-style element."""
    if element is None:
        return ""
    return (element.get("id") or "").strip()


def _add_issue(
    issues: list[CSVValidationIssue],
    *,
    code: str,
    message: str,
    location: str | None = None,
    value: Any = None,
) -> None:
    issues.append(
        CSVValidationIssue(
            code=code,
            message=message,
            column=location,
            value=None if value is None else str(value),
        )
    )


# ---------------------------------------------------------------------------
# Container unwrapping and plate geometry
# ---------------------------------------------------------------------------


def _unwrap_container(raw: bytes) -> tuple[bytes, str]:
    """Return ``(xml_bytes, container_format)`` for an RDML file payload.

    Supports ZIP archives (the canonical ``.rdml``/``.rdm`` container),
    gzip-compressed XML, and plain XML, detected by magic bytes.
    """
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            xml_members = [n for n in names if n.lower().endswith(".xml")]
            preferred = [
                n
                for n in xml_members
                if n.rsplit("/", 1)[-1].lower() == "rdml_data.xml"
            ]
            if not preferred and not xml_members:
                raise ValueError(
                    "RDML ZIP archive contains no XML member; expected "
                    "rdml_data.xml"
                )
            member = preferred[0] if preferred else xml_members[0]
            return archive.read(member), "zip"
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw), "gzip"
    return raw, "xml"


def _row_letter(row: int) -> str:
    """Spreadsheet-style row label: 0 -> A, 25 -> Z, 26 -> AA, ..."""
    letters = ""
    index = row + 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _plate_position(
    react_id: int, rows: int, columns: int, row_label: str
) -> str | None:
    """Convert a row-first ``react`` id into a plate address.

    Returns ``None`` when the plate geometry is unknown (free format) or
    the id falls outside the plate.
    """
    if rows <= 0 or columns <= 0 or react_id < 1 or react_id > rows * columns:
        return None
    row, col = divmod(react_id - 1, columns)
    width = max(2, len(str(columns)))
    if row_label.upper() == "ABC":
        return f"{_row_letter(row)}{col + 1:0{width}d}"
    return f"R{row + 1}C{col + 1:0{width}d}"


def _parse_pcr_format(run_el: ET.Element) -> tuple[int, int, str]:
    fmt_el = _child(run_el, "pcrFormat")
    if fmt_el is None:
        return -1, -1, ""
    try:
        rows = int(float(_text(fmt_el, "rows") or "-1"))
        columns = int(float(_text(fmt_el, "columns") or "-1"))
    except ValueError:
        return -1, -1, ""
    return rows, columns, (_text(fmt_el, "rowLabel") or "").strip()


# ---------------------------------------------------------------------------
# Master element parsing
# ---------------------------------------------------------------------------


def _parse_samples(
    root: ET.Element, issues: list[CSVValidationIssue]
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(sample_types, sample_groups)`` from ``<sample>`` masters."""
    sample_types: dict[str, str] = {}
    sample_groups: dict[str, str] = {}
    for sample_el in _children(root, "sample"):
        sample_id = (sample_el.get("id") or "").strip()
        if not sample_id:
            continue
        type_text = (_text(sample_el, "type") or "unkn").strip()
        if type_text not in _SAMPLE_TYPES:
            _add_issue(
                issues,
                code="INVALID_SAMPLE_TYPE",
                message=f"unknown sample type; expected one of {sorted(_SAMPLE_TYPES)}",
                location=f"sample '{sample_id}'",
                value=type_text,
            )
            type_text = "unkn"
        sample_types[sample_id] = type_text

        for annotation_el in _children(sample_el, "annotation"):
            property_name = (_text(annotation_el, "property") or "").strip()
            property_value = (_text(annotation_el, "value") or "").strip()
            if property_name.lower() != "group" or not property_value:
                continue
            existing = sample_groups.get(sample_id)
            if existing is not None and existing != property_value:
                _add_issue(
                    issues,
                    code="CONFLICTING_SAMPLE_GROUP",
                    message=f"sample previously assigned group {existing!r}",
                    location=f"sample '{sample_id}'",
                    value=property_value,
                )
            elif existing is None:
                sample_groups[sample_id] = property_value
    return sample_types, sample_groups


def _parse_targets(
    root: ET.Element, issues: list[CSVValidationIssue]
) -> dict[str, dict[str, Any]]:
    """Parse ``<target>`` masters into raw assay metadata."""
    targets: dict[str, dict[str, Any]] = {}
    for target_el in _children(root, "target"):
        target_id = (target_el.get("id") or "").strip()
        if not target_id:
            continue
        type_text = (_text(target_el, "type") or "toi").strip()
        if type_text not in {"ref", "toi"}:
            _add_issue(
                issues,
                code="INVALID_TARGET_TYPE",
                message="unknown target type; expected 'ref' or 'toi'",
                location=f"target '{target_id}'",
                value=type_text,
            )
            type_text = "toi"

        efficiency: float | None = None
        efficiency_text = _text(target_el, "amplificationEfficiency")
        if efficiency_text is not None:
            try:
                factor = float(efficiency_text)
            except ValueError:
                factor = None
            low, high = _EFFICIENCY_FACTOR_RANGE
            if factor is None or not math.isfinite(factor) or not low <= factor <= high:
                _add_issue(
                    issues,
                    code="INVALID_EFFICIENCY",
                    message=(
                        "amplificationEfficiency must be a per-cycle factor "
                        f"between {low} and {high}"
                    ),
                    location=f"target '{target_id}'",
                    value=efficiency_text,
                )
            else:
                efficiency = factor - 1.0

        sequences_el = _child(target_el, "sequences")
        primer_forward = _oligo_sequence(sequences_el, "forwardPrimer")
        primer_reverse = _oligo_sequence(sequences_el, "reversePrimer")
        probe = _oligo_sequence(sequences_el, "probe1")

        targets[target_id] = {
            "is_reference": type_text == _REFERENCE_TARGET_TYPE,
            "efficiency": efficiency,
            "primer_forward": primer_forward,
            "primer_reverse": primer_reverse,
            "probe": probe,
        }
    return targets


def _oligo_sequence(sequences_el: ET.Element | None, name: str) -> str | None:
    """Sequence text of one oligo (primer/probe) inside ``<sequences>``."""
    if sequences_el is None:
        return None
    return _text(_child(sequences_el, name), "sequence")


# ---------------------------------------------------------------------------
# Measurement parsing
# ---------------------------------------------------------------------------


def _parse_cq(
    data_el: ET.Element,
    issues: list[CSVValidationIssue],
    *,
    location: str,
) -> float | None:
    cq_el = _child(data_el, "cq")
    if cq_el is None:
        return None
    text = (cq_el.text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        _add_issue(
            issues,
            code="INVALID_CQ",
            message="cq is not a number",
            location=location,
            value=text,
        )
        return None
    if not math.isfinite(value):
        _add_issue(
            issues,
            code="INVALID_CQ",
            message="cq must be a finite number",
            location=location,
            value=text,
        )
        return None
    if value == -1.0:  # RDML "not available" sentinel
        return None
    if value < 0:
        _add_issue(
            issues,
            code="INVALID_CQ",
            message="only the -1.0 sentinel may be negative",
            location=location,
            value=text,
        )
        return None
    return value


def _parse_adp(
    data_el: ET.Element,
    position: str,
    issues: list[CSVValidationIssue],
) -> RawTrace | None:
    """Assemble a raw amplification trace from ``<adp>`` data points."""
    cycles: list[float] = []
    fluorescence: list[float] = []
    for adp_el in _children(data_el, "adp"):
        cycle_text = _text(adp_el, "cyc")
        fluor_text = _text(adp_el, "fluor")
        if cycle_text is None or fluor_text is None:
            _add_issue(
                issues,
                code="INVALID_TRACE_POINT",
                message="adp point needs both cyc and fluor",
                location=f"well {position}",
            )
            continue
        try:
            cycle = float(cycle_text)
            fluor = float(fluor_text)
        except ValueError:
            _add_issue(
                issues,
                code="INVALID_TRACE_POINT",
                message="adp point values must be numbers",
                location=f"well {position}",
                value=f"{cycle_text};{fluor_text}",
            )
            continue
        if not (math.isfinite(cycle) and math.isfinite(fluor)):
            _add_issue(
                issues,
                code="INVALID_TRACE_POINT",
                message="adp point values must be finite",
                location=f"well {position}",
                value=f"{cycle_text};{fluor_text}",
            )
            continue
        cycles.append(cycle)
        fluorescence.append(fluor)
    if not cycles:
        return None
    return RawTrace(
        position=position,
        cycles=np.asarray(cycles, dtype=float),
        fluorescence=np.asarray(fluorescence, dtype=float),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_rdml(
    path: str | Path,
    experiment_name: str | None = None,
    reference_targets: list[str] | None = None,
    *,
    strict: bool = True,
) -> Experiment:
    """Read an RDML file (ZIP, gzip, or plain XML) into an Experiment.

    The first ``<experiment>`` in the file is read; additional experiments
    are ignored and recorded in the metadata.  Runs are merged, with well
    positions prefixed by the run id when the file holds more than one
    run.

    Parameters
        path:
            Path to an ``.rdml``/``.rdm`` archive or XML file.
        experiment_name:
            Override for the experiment name (defaults to the first
            experiment's id, then the file stem).
        reference_targets:
            Extra reference-gene target ids; targets with RDML
            ``<type>ref</type>`` are always references.
        strict:
            Raise :class:`CSVValidationError` on error-level issues such
            as unresolved references or invalid values.  Warning-level
            diagnostics (excluded entries, multiplex position suffixes)
            never raise.

    Returns
        :class:`~bioeng_qpcr.domain.Experiment` at input level C, carrying
        wells, samples, assays, raw traces, and ingestion diagnostics.

    Raises
        ValueError:
            If the file is not a readable RDML document.
        CSVValidationError:
            In strict mode, when error-level issues are found.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    try:
        xml_bytes, container = _unwrap_container(raw_bytes)
    except (ValueError, zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"{path.name}: cannot read RDML container ({exc})") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"{path.name} does not contain valid XML ({exc})") from exc
    if _local_name(root.tag).lower() != "rdml":
        raise ValueError(
            f"{path.name}: expected an <rdml> root element, "
            f"found <{_local_name(root.tag)}>"
        )

    issues: list[CSVValidationIssue] = []
    sample_types, sample_groups = _parse_samples(root, issues)
    target_infos = _parse_targets(root, issues)

    experiments = _children(root, "experiment")
    if not experiments:
        raise CSVValidationError(
            [
                CSVValidationIssue(
                    code="EMPTY_FILE",
                    message="RDML document contains no experiment",
                )
            ]
        )
    experiment_el = experiments[0]
    name = (
        experiment_name
        if experiment_name is not None
        else (experiment_el.get("id") or "").strip() or path.stem
    )

    ref_set = {
        str(target).strip()
        for target in reference_targets or []
        if str(target).strip()
    }
    ref_set.update(
        target_id
        for target_id, info in target_infos.items()
        if info["is_reference"]
    )
    for target in sorted(ref_set - set(target_infos)):
        _add_issue(
            issues,
            code="UNKNOWN_REFERENCE_TARGET",
            message="reference target is not defined in the RDML file",
            location="reference_targets",
            value=target,
        )

    runs = _children(experiment_el, "run")
    if not runs:
        _add_issue(
            issues,
            code="EMPTY_FILE",
            message="experiment contains no run with measurements",
        )
        raise CSVValidationError(issues)

    wells: list[Well] = []
    traces: dict[str, RawTrace] = {}
    n_excluded = 0
    n_multiplex = 0

    for run_el in runs:
        run_id = (run_el.get("id") or "").strip() or "run"
        rows, columns, row_label = _parse_pcr_format(run_el)
        run_prefix = f"{run_id}:" if len(runs) > 1 else ""
        seen_react_keys: set[str] = set()

        for react_el in _children(run_el, "react"):
            react_key = (react_el.get("id") or "").strip()
            try:
                react_no = int(react_key)
            except ValueError:
                react_no = None
            seen_key = str(react_no) if react_no is not None else react_key
            if not seen_key or seen_key in seen_react_keys:
                _add_issue(
                    issues,
                    code="DUPLICATE_REACT",
                    message="react id is empty or occurs twice in this run",
                    location=f"run '{run_id}'",
                    value=react_key,
                )
                continue
            seen_react_keys.add(seen_key)

            sample_ref = _ref_id(_child(react_el, "sample"))
            if not sample_ref or sample_ref not in sample_types:
                _add_issue(
                    issues,
                    code="MISSING_SAMPLE_REF",
                    message="react references an undefined or missing sample",
                    location=f"run '{run_id}', react {react_key}",
                    value=sample_ref,
                )
                continue
            is_ntc = sample_types[sample_ref] == _NTC_SAMPLE_TYPE

            base_position = (
                _plate_position(react_no, rows, columns, row_label)
                if react_no is not None
                else None
            ) or react_key

            data_elements = _children(react_el, "data")
            data_index = 0
            for data_el in data_elements:
                if _child(data_el, "excl") is not None:
                    n_excluded += 1
                    _add_issue(
                        issues,
                        code="EXCLUDED_DATA",
                        message=(
                            "data entry flagged with <excl> by the instrument "
                            "software was skipped"
                        ),
                        location=f"run '{run_id}', react {react_key}",
                    )
                    continue

                target_ref = _ref_id(_child(data_el, "tar"))
                if not target_ref or target_ref not in target_infos:
                    _add_issue(
                        issues,
                        code="MISSING_TARGET_REF",
                        message="data references an undefined or missing target",
                        location=f"run '{run_id}', react {react_key}",
                        value=target_ref,
                    )
                    continue

                if data_index > 0:
                    n_multiplex += 1
                    position = f"{run_prefix}{base_position}:{target_ref}"
                    _add_issue(
                        issues,
                        code="MULTIPLEX_POSITION_SUFFIX",
                        message=(
                            "multiplex react: well position suffixed with the "
                            "target id to keep positions unique"
                        ),
                        location=f"run '{run_id}', react {react_key}",
                        value=position,
                    )
                else:
                    position = f"{run_prefix}{base_position}"

                location = f"run '{run_id}', well {position}"
                cq = _parse_cq(data_el, issues, location=location)
                cq_status = (
                    "no_template"
                    if is_ntc
                    else ("determined" if cq is not None else "undetermined")
                )
                wells.append(
                    Well(
                        position=position,
                        sample_name=sample_ref,
                        target_name=target_ref,
                        cq=cq,
                        replicate_type=ReplicateType.SINGLE,
                        replicate_group=None,
                        cq_status=cq_status,
                    )
                )
                data_index += 1

                trace = _parse_adp(data_el, position, issues)
                if trace is not None:
                    traces[position] = trace

    if strict and any(issue.code in _ERROR_CODES for issue in issues):
        raise CSVValidationError(issues)
    if not wells:
        raise CSVValidationError(
            issues
            or [
                CSVValidationIssue(
                    code="NO_VALID_WELLS", message="no usable wells remain"
                )
            ]
        )

    # RDML semantics: identical sample ids measure the same template
    # solution, so repeated sample/target wells are technical replicates.
    pair_counts = Counter((well.sample_name, well.target_name) for well in wells)
    resolved_wells = [
        (
            replace(
                well,
                replicate_type=ReplicateType.TECHNICAL,
                replicate_group=f"{well.sample_name}_{well.target_name}",
            )
            if pair_counts[(well.sample_name, well.target_name)] > 1
            else well
        )
        for well in wells
    ]

    referenced_samples = {well.sample_name for well in resolved_wells}
    samples = {
        sample_id: Sample(
            name=sample_id,
            group=sample_groups.get(sample_id),
            replicate_type=(
                ReplicateType.TECHNICAL
                if any(
                    pair_counts[(sample_id, target)] > 1
                    for target in {well.target_name for well in resolved_wells
                                   if well.sample_name == sample_id}
                )
                else ReplicateType.SINGLE
            ),
        )
        for sample_id in sorted(referenced_samples)
    }

    referenced_targets = {well.target_name for well in resolved_wells}
    assays = {
        target_id: Assay(
            name=target_id,
            efficiency=target_infos[target_id]["efficiency"],
            is_reference=target_id in ref_set,
            primer_forward=target_infos[target_id]["primer_forward"],
            primer_reverse=target_infos[target_id]["primer_reverse"],
            probe=target_infos[target_id]["probe"],
        )
        for target_id in sorted(referenced_targets)
    }

    return Experiment(
        name=name,
        wells=resolved_wells,
        samples=samples,
        assays=assays,
        traces=traces,
        input_level=InputLevel.LEVEL_C,
        method=None,
        metadata={
            "source_file": str(path),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "parse_mode": "strict" if strict else "lenient",
            "rdml_version": root.get("version") or "unknown",
            "container_format": container,
            "input_diagnostics": [issue.to_dict() for issue in issues],
            "n_wells": len(resolved_wells),
            "n_samples": len(samples),
            "n_assays": len(assays),
            "n_runs": len(runs),
            "ignored_experiments": [
                (el.get("id") or "").strip() for el in experiments[1:]
            ],
            "excluded_data_entries": n_excluded,
            "multiplex_data_entries": n_multiplex,
            "n_traces": len(traces),
        },
    )
