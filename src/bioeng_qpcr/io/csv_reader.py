"""Strict, auditable reader for the Level A tidy qPCR CSV format."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from bioeng_qpcr.domain import (
    Assay,
    Experiment,
    InputLevel,
    ReplicateType,
    Sample,
    Well,
)
from bioeng_qpcr.io.diagnostics import CSVValidationError, CSVValidationIssue

__all__ = ["read_tidy_csv"]

REQUIRED_COLUMNS: tuple[str, ...] = ("position", "sample_name", "target_name", "cq")
NTC_NAMES: frozenset[str] = frozenset({"NTC", "NO_TEMPLATE", "BLANK"})
_UNDETERMINED_TOKENS: frozenset[str] = frozenset(
    {"", "undetermined", "nan", "na", "n/a", "none", "null"}
)
_TRUE_TOKENS: frozenset[str] = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS: frozenset[str] = frozenset({"false", "0", "no", "n", "f"})
_CQ_STATUSES: frozenset[str] = frozenset({"determined", "undetermined", "no_template"})


def _is_missing(value: Any) -> bool:
    """Return ``True`` for empty or missing scalar values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _parse_cq(raw: Any) -> float | None:
    """Compatibility helper that parses Cq without diagnostics.

    Researcher-facing ingestion calls the strict diagnostic parser below.
    """
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    text = str(raw).strip()
    if text.lower() in _UNDETERMINED_TOKENS:
        return None
    try:
        value = float(text)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _parse_float(raw: Any) -> float | None:
    """Compatibility helper that parses finite optional numeric values."""
    if _is_missing(raw):
        return None
    try:
        value = float(str(raw).strip())
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _parse_replicate_type(raw: Any) -> ReplicateType:
    """Compatibility helper; strict ingestion reports invalid values."""
    if _is_missing(raw):
        return ReplicateType.SINGLE
    try:
        return ReplicateType(str(raw).strip().lower())
    except ValueError:
        return ReplicateType.SINGLE


def _parse_bool(raw: Any) -> bool:
    """Compatibility helper; strict ingestion reports unknown tokens."""
    if _is_missing(raw):
        return False
    return str(raw).strip().lower() in _TRUE_TOKENS


def _is_ntc(sample_name: str) -> bool:
    return sample_name.upper() in NTC_NAMES


def _infer_cq_status(cq: float | None, is_ntc: bool) -> str:
    if is_ntc:
        return "no_template"
    if cq is None:
        return "undetermined"
    return "determined"


def _add_issue(
    issues: list[CSVValidationIssue],
    *,
    code: str,
    message: str,
    row: int | None = None,
    column: str | None = None,
    value: Any = None,
) -> None:
    issues.append(
        CSVValidationIssue(
            code=code,
            message=message,
            row=row,
            column=column,
            value=None if value is None else str(value),
        )
    )


def _diagnostic_float(
    raw: Any,
    *,
    row: int,
    column: str,
    issues: list[CSVValidationIssue],
    allow_missing: bool = True,
    non_negative: bool = False,
) -> float | None:
    if _is_missing(raw):
        if not allow_missing:
            _add_issue(
                issues,
                code="MISSING_VALUE",
                message="a numeric value is required",
                row=row,
                column=column,
            )
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        _add_issue(
            issues,
            code="INVALID_NUMBER",
            message="expected a finite number",
            row=row,
            column=column,
            value=raw,
        )
        return None
    if not math.isfinite(value):
        _add_issue(
            issues,
            code="NONFINITE_NUMBER",
            message="NaN and infinite values are not valid measurements",
            row=row,
            column=column,
            value=raw,
        )
        return None
    if non_negative and value < 0:
        _add_issue(
            issues,
            code="NEGATIVE_VALUE",
            message="value must be non-negative",
            row=row,
            column=column,
            value=raw,
        )
        return None
    return value


def _diagnostic_cq(
    raw: Any, *, row: int, issues: list[CSVValidationIssue]
) -> float | None:
    if _is_missing(raw) or str(raw).strip().lower() in _UNDETERMINED_TOKENS:
        return None
    return _diagnostic_float(
        raw,
        row=row,
        column="cq",
        issues=issues,
        allow_missing=True,
        non_negative=True,
    )


def read_tidy_csv(
    path: str | Path,
    experiment_name: str | None = None,
    reference_targets: list[str] | None = None,
    *,
    strict: bool = True,
) -> Experiment:
    """Read a Level A tidy CSV with row-level validation diagnostics.

    In strict mode (the default), malformed identifiers, numeric values,
    categorical values, duplicate well positions, and conflicting sample or
    assay metadata raise :class:`CSVValidationError`. In lenient mode,
    recoverable values are coerced and every issue is retained in
    ``experiment.metadata["input_diagnostics"]``; structurally unusable rows
    and duplicate wells are excluded.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(column).strip().lower() for column in df.columns]

    issues: list[CSVValidationIssue] = []
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        for column in missing_columns:
            _add_issue(
                issues,
                code="MISSING_COLUMN",
                message="required column is absent",
                row=1,
                column=column,
            )
        raise CSVValidationError(issues)
    if df.empty:
        raise CSVValidationError(
            [CSVValidationIssue(code="EMPTY_FILE", message="CSV contains no data rows")]
        )

    name = experiment_name if experiment_name is not None else path.stem
    ref_set = {
        str(target).strip() for target in reference_targets or [] if str(target).strip()
    }

    has = {
        column: column in df.columns
        for column in (
            "replicate_type",
            "replicate_group",
            "cq_status",
            "sample_group",
            "efficiency",
            "r2",
            "slope",
            "is_reference",
            "primer_forward",
            "primer_reverse",
            "probe",
            "lod_cq",
            "lloq_cq",
        )
    }

    wells: list[Well] = []
    sample_info: dict[str, dict[str, Any]] = {}
    assay_info: dict[str, dict[str, Any]] = {}
    positions: set[str] = set()

    for frame_index, row_data in df.iterrows():
        csv_row = int(frame_index) + 2
        position = str(row_data["position"]).strip()
        sample_name = str(row_data["sample_name"]).strip()
        target_name = str(row_data["target_name"]).strip()

        missing_identifier = False
        for column, value in (
            ("position", position),
            ("sample_name", sample_name),
            ("target_name", target_name),
        ):
            if not value:
                missing_identifier = True
                _add_issue(
                    issues,
                    code="EMPTY_IDENTIFIER",
                    message="identifier must not be empty",
                    row=csv_row,
                    column=column,
                )
        if missing_identifier:
            continue
        if position in positions:
            _add_issue(
                issues,
                code="DUPLICATE_WELL",
                message="well position occurs more than once",
                row=csv_row,
                column="position",
                value=position,
            )
            continue
        positions.add(position)

        cq = _diagnostic_cq(row_data["cq"], row=csv_row, issues=issues)
        ntc = _is_ntc(sample_name)

        if has["cq_status"] and not _is_missing(row_data["cq_status"]):
            cq_status = str(row_data["cq_status"]).strip().lower()
            if cq_status not in _CQ_STATUSES:
                _add_issue(
                    issues,
                    code="INVALID_CQ_STATUS",
                    message=f"expected one of {sorted(_CQ_STATUSES)}",
                    row=csv_row,
                    column="cq_status",
                    value=row_data["cq_status"],
                )
                cq_status = _infer_cq_status(cq, ntc)
        else:
            cq_status = _infer_cq_status(cq, ntc)
        if cq is None and cq_status == "determined":
            _add_issue(
                issues,
                code="INCONSISTENT_CQ_STATUS",
                message="a determined well must contain a numeric Cq",
                row=csv_row,
                column="cq_status",
                value=cq_status,
            )
            cq_status = "undetermined"
        elif cq is not None and cq_status == "undetermined":
            _add_issue(
                issues,
                code="INCONSISTENT_CQ_STATUS",
                message="an undetermined well must not contain a numeric Cq",
                row=csv_row,
                column="cq_status",
                value=cq_status,
            )
            cq_status = _infer_cq_status(cq, ntc)

        replicate_type = ReplicateType.SINGLE
        if has["replicate_type"] and not _is_missing(row_data["replicate_type"]):
            raw_type = str(row_data["replicate_type"]).strip().lower()
            try:
                replicate_type = ReplicateType(raw_type)
            except ValueError:
                _add_issue(
                    issues,
                    code="INVALID_REPLICATE_TYPE",
                    message="expected technical, biological, or single",
                    row=csv_row,
                    column="replicate_type",
                    value=row_data["replicate_type"],
                )

        replicate_group = None
        if has["replicate_group"] and not _is_missing(row_data["replicate_group"]):
            replicate_group = str(row_data["replicate_group"]).strip()

        wells.append(
            Well(
                position=position,
                sample_name=sample_name,
                target_name=target_name,
                cq=cq,
                replicate_type=replicate_type,
                replicate_group=replicate_group,
                cq_status=cq_status,
            )
        )

        group = (
            str(row_data["sample_group"]).strip()
            if has["sample_group"] and not _is_missing(row_data["sample_group"])
            else None
        )
        existing_sample = sample_info.get(sample_name)
        if existing_sample is None:
            sample_info[sample_name] = {
                "group": group,
                "replicate_type": replicate_type,
            }
        else:
            if group and existing_sample["group"] and group != existing_sample["group"]:
                _add_issue(
                    issues,
                    code="CONFLICTING_SAMPLE_GROUP",
                    message=(
                        "sample was previously assigned to "
                        f"{existing_sample['group']!r}"
                    ),
                    row=csv_row,
                    column="sample_group",
                    value=group,
                )
            elif group and existing_sample["group"] is None:
                existing_sample["group"] = group
            previous_type = existing_sample["replicate_type"]
            if replicate_type is not previous_type:
                _add_issue(
                    issues,
                    code="CONFLICTING_REPLICATE_TYPE",
                    message=(
                        f"sample previously used replicate type {previous_type.value!r}"
                    ),
                    row=csv_row,
                    column="replicate_type",
                    value=replicate_type.value,
                )

        efficiency = (
            _diagnostic_float(
                row_data["efficiency"],
                row=csv_row,
                column="efficiency",
                issues=issues,
                non_negative=True,
            )
            if has["efficiency"]
            else None
        )
        r2 = (
            _diagnostic_float(row_data["r2"], row=csv_row, column="r2", issues=issues)
            if has["r2"]
            else None
        )
        if r2 is not None and not 0.0 <= r2 <= 1.0:
            _add_issue(
                issues,
                code="INVALID_RANGE",
                message="r2 must be between 0 and 1",
                row=csv_row,
                column="r2",
                value=r2,
            )
            r2 = None
        slope = (
            _diagnostic_float(
                row_data["slope"], row=csv_row, column="slope", issues=issues
            )
            if has["slope"]
            else None
        )
        lod_cq = (
            _diagnostic_float(
                row_data["lod_cq"],
                row=csv_row,
                column="lod_cq",
                issues=issues,
                non_negative=True,
            )
            if has["lod_cq"]
            else None
        )
        lloq_cq = (
            _diagnostic_float(
                row_data["lloq_cq"],
                row=csv_row,
                column="lloq_cq",
                issues=issues,
                non_negative=True,
            )
            if has["lloq_cq"]
            else None
        )
        if lod_cq is not None and lloq_cq is not None and lloq_cq > lod_cq:
            _add_issue(
                issues,
                code="INVALID_RANGE",
                message="lloq_cq must not exceed lod_cq",
                row=csv_row,
                column="lloq_cq",
                value=lloq_cq,
            )
            lloq_cq = None

        is_reference = target_name in ref_set
        if has["is_reference"] and not _is_missing(row_data["is_reference"]):
            bool_token = str(row_data["is_reference"]).strip().lower()
            if bool_token in _TRUE_TOKENS:
                is_reference = True
            elif bool_token not in _FALSE_TOKENS:
                _add_issue(
                    issues,
                    code="INVALID_BOOLEAN",
                    message="expected true/false, yes/no, or 1/0",
                    row=csv_row,
                    column="is_reference",
                    value=row_data["is_reference"],
                )

        assay_row = {
            "efficiency": efficiency,
            "r2": r2,
            "slope": slope,
            "lod_cq": lod_cq,
            "lloq_cq": lloq_cq,
            "is_reference": is_reference,
            "primer_forward": str(row_data["primer_forward"]).strip()
            if has["primer_forward"] and not _is_missing(row_data["primer_forward"])
            else None,
            "primer_reverse": str(row_data["primer_reverse"]).strip()
            if has["primer_reverse"] and not _is_missing(row_data["primer_reverse"])
            else None,
            "probe": str(row_data["probe"]).strip()
            if has["probe"] and not _is_missing(row_data["probe"])
            else None,
        }
        existing_assay = assay_info.get(target_name)
        if existing_assay is None:
            assay_info[target_name] = assay_row
        else:
            for field_name, new_value in assay_row.items():
                old_value = existing_assay[field_name]
                if new_value is None:
                    continue
                if old_value is None:
                    existing_assay[field_name] = new_value
                    continue
                values_conflict = (
                    not math.isclose(old_value, new_value, rel_tol=1e-9, abs_tol=1e-12)
                    if isinstance(old_value, float) and isinstance(new_value, float)
                    else old_value != new_value
                )
                if values_conflict:
                    _add_issue(
                        issues,
                        code="CONFLICTING_ASSAY_METADATA",
                        message=f"target previously used {field_name}={old_value!r}",
                        row=csv_row,
                        column=field_name,
                        value=new_value,
                    )

    for target in sorted(ref_set - set(assay_info)):
        _add_issue(
            issues,
            code="UNKNOWN_REFERENCE_TARGET",
            message="reference target is not present in the CSV",
            column="reference_targets",
            value=target,
        )

    if strict and issues:
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

    samples = {
        sample_name: Sample(
            name=sample_name,
            group=data["group"],
            replicate_type=data["replicate_type"],
        )
        for sample_name, data in sample_info.items()
    }
    assays = {
        target_name: Assay(name=target_name, **data)
        for target_name, data in assay_info.items()
    }

    return Experiment(
        name=name,
        wells=wells,
        samples=samples,
        assays=assays,
        traces={},
        input_level=InputLevel.LEVEL_A,
        method=None,
        metadata={
            "source_file": str(path),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "parse_mode": "strict" if strict else "lenient",
            "input_diagnostics": [issue.to_dict() for issue in issues],
            "n_wells": len(wells),
            "n_samples": len(samples),
            "n_assays": len(assays),
        },
    )
