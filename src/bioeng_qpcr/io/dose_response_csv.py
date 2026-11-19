"""Reader that turns a dose-response CSV into :class:`DoseResponseDataset`s.

The format this module accepts is declared in
:mod:`bioeng_qpcr.io.csv_preview` (schemas ``dose_response`` and
``dose_response_matrix``), which is also where the authoritative prose
specification lives.  Two layouts are supported because both appear in
practice:

*Long (one row per observation)*
    Columns ``compound``, ``concentration``, ``response`` and an optional
    ``replicate``.  This is what you get from a plate-reader export that has
    already been unpivoted.

*Matrix (first column = concentration, one column per compound)*
    The classic 96-well export: a concentration column followed by one column
    per compound.  Written by almost every plate reader, and not worth making
    the user pivot by hand.

Optional normalisation to "per cent of control" is offered because it is a
near-universal convention in inhibition assays, but it is *opt-in*: doing it
silently would change the meaning of the ``response`` column and break the
absolute-potency calculation, which needs the original scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioeng_qpcr.io.csv_preview import (
    CsvTable,
    normalise_header,
    read_csv_table,
)
from bioeng_qpcr.potency.fitting import CurveDirection, DoseResponseDataset

__all__ = [
    "NotDoseResponseFile",
    "read_dose_response_csv",
]

#: Header spellings accepted for the concentration column.
_CONCENTRATION_NAMES = frozenset(
    {"concentration", "conc", "dose", "concentration_um", "concentration_nm", "x"}
)

#: Header spellings accepted for the response column.
_RESPONSE_NAMES = frozenset({"response", "signal", "value", "y", "inhibition"})

#: Header spellings accepted for the compound column.
_COMPOUND_NAMES = frozenset({"compound", "drug", "treatment", "series", "name"})

#: Header spellings accepted for the replicate column.
_REPLICATE_NAMES = frozenset({"replicate", "rep", "replicate_id", "well"})


class NotDoseResponseFile(ValueError):
    """Raised when a file is structurally not a dose-response table.

    Distinct from a plain :class:`ValueError`, which signals a file that *is*
    a dose-response table but contains an unusable value.
    """


def _to_dataset(
    compound: str,
    concentrations: list[float],
    responses: list[float],
    replicates: list[str],
    *,
    normalize: str,
    direction: CurveDirection,
) -> DoseResponseDataset:
    """Assemble one :class:`DoseResponseDataset`, applying normalisation."""
    concentration = np.asarray(concentrations, dtype=float)
    response = np.asarray(responses, dtype=float)

    if normalize == "percent_of_control":
        control_mask = np.isclose(concentration, float(np.min(concentration)))
        control = float(np.mean(response[control_mask])) if control_mask.any() else 0.0
        if control == 0.0:
            raise ValueError(
                f"Cannot normalise compound {compound!r} to per cent of "
                "control: the mean response at the lowest concentration is 0."
            )
        response = response / control * 100.0

    return DoseResponseDataset(
        compound=compound,
        concentration=concentration,
        response=response,
        direction=direction,
        replicate_group=tuple(replicates) if replicates else None,
    )


def _read_long(
    table: CsvTable,
    *,
    normalize: str,
    default_compound: str,
    direction: CurveDirection,
) -> list[DoseResponseDataset]:
    """Parse the long (tidy) dose-response layout."""
    headers = {normalise_header(name): index for index, name in enumerate(table.header)}
    concentration_index = next(
        (headers[name] for name in headers if name in _CONCENTRATION_NAMES), None
    )
    response_index = next(
        (headers[name] for name in headers if name in _RESPONSE_NAMES), None
    )
    if concentration_index is None or response_index is None:
        raise NotDoseResponseFile(
            "long-format dose-response tables need a concentration column "
            f"(one of {sorted(_CONCENTRATION_NAMES)}) and a response column "
            f"(one of {sorted(_RESPONSE_NAMES)})."
        )
    compound_index = next(
        (headers[name] for name in headers if name in _COMPOUND_NAMES), None
    )
    replicate_index = next(
        (headers[name] for name in headers if name in _REPLICATE_NAMES), None
    )

    grouped: dict[str, tuple[list[float], list[float], list[str]]] = {}
    for row_index, row in enumerate(table.rows):
        raw_concentration = row[concentration_index].strip()
        raw_response = row[response_index].strip()
        if not raw_concentration or not raw_response:
            continue
        try:
            concentration = float(raw_concentration)
            response = float(raw_response)
        except ValueError as exc:
            raise ValueError(
                f"row {row_index + 2}: expected numbers in the concentration "
                f"and response columns, got {raw_concentration!r} and "
                f"{raw_response!r}."
            ) from exc
        compound = (
            row[compound_index].strip()
            if compound_index is not None and row[compound_index].strip()
            else default_compound
        )
        replicate = (
            row[replicate_index].strip()
            if replicate_index is not None
            else f"r{row_index + 1}"
        )
        bucket = grouped.setdefault(compound, ([], [], []))
        bucket[0].append(concentration)
        bucket[1].append(response)
        bucket[2].append(replicate)

    if not grouped:
        raise NotDoseResponseFile("the file contains no usable data rows.")

    return [
        _to_dataset(
            compound,
            concentrations,
            responses,
            replicates,
            normalize=normalize,
            direction=direction,
        )
        for compound, (concentrations, responses, replicates) in grouped.items()
    ]


def _read_matrix(
    table: CsvTable,
    *,
    normalize: str,
    direction: CurveDirection,
) -> list[DoseResponseDataset]:
    """Parse the wide (matrix) dose-response layout."""
    headers = [normalise_header(name) for name in table.header]
    concentration_index = next(
        (index for index, name in enumerate(headers) if name in _CONCENTRATION_NAMES),
        None,
    )
    if concentration_index is None:
        raise NotDoseResponseFile(
            "matrix-format dose-response tables need the concentration "
            f"column first (one of {sorted(_CONCENTRATION_NAMES)})."
        )

    series_indices = [
        index
        for index in range(len(table.header))
        if index != concentration_index and not _is_metadata_column(headers[index])
    ]
    if not series_indices:
        raise NotDoseResponseFile(
            "matrix-format dose-response tables need at least one compound "
            "column after the concentration column."
        )

    grouped: dict[str, tuple[list[float], list[float], list[str]]] = {}
    for row_index, row in enumerate(table.rows):
        raw_concentration = row[concentration_index].strip()
        if not raw_concentration:
            continue
        try:
            concentration = float(raw_concentration)
        except ValueError as exc:
            raise ValueError(
                f"row {row_index + 2}: expected a number in the concentration "
                f"column, got {raw_concentration!r}."
            ) from exc
        for index in series_indices:
            raw = row[index].strip()
            if not raw:
                continue  # a blank cell is a missing observation, not a zero
            try:
                response = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"row {row_index + 2}, column '{table.header[index]}': "
                    f"expected a number, got {raw!r}."
                ) from exc
            compound = table.header[index].strip()
            bucket = grouped.setdefault(compound, ([], [], []))
            bucket[0].append(concentration)
            bucket[1].append(response)
            bucket[2].append(f"r{row_index + 1}")

    if not grouped:
        raise NotDoseResponseFile("the file contains no usable data values.")

    return [
        _to_dataset(
            compound,
            concentrations,
            responses,
            replicates,
            normalize=normalize,
            direction=direction,
        )
        for compound, (concentrations, responses, replicates) in grouped.items()
    ]


def _is_metadata_column(header: str) -> bool:
    """Return whether a matrix column is metadata rather than a series."""
    return header in _REPLICATE_NAMES or header in _COMPOUND_NAMES


def read_dose_response_csv(
    path: str | Path,
    *,
    layout: str = "auto",
    normalize: str = "none",
    compound_name: str | None = None,
    direction: CurveDirection = CurveDirection.AUTO,
    encoding: str | None = None,
    delimiter: str | None = None,
) -> list[DoseResponseDataset]:
    """Read a dose-response CSV into one dataset per compound.

    Parameters
        path:
            File to read.
        layout:
            ``"auto"`` (default), ``"long"`` or ``"matrix"``.  Auto-detection
            prefers the long layout, because a file that has both a
            ``compound`` column and several response columns is ambiguous and
            the long layout carries strictly more information.
        normalize:
            ``"none"`` (default) keeps the response as measured;
            ``"percent_of_control"`` divides every response by the mean
            response at the lowest concentration of its compound and
            multiplies by 100.
        compound_name:
            Name to use for a long-format file that has no ``compound``
            column.  Defaults to the file stem.
        direction:
            Expected curve direction; ``AUTO`` infers it per compound.
        encoding, delimiter:
            Overrides for the sniffed values.

    Returns
        One :class:`~bioeng_qpcr.potency.DoseResponseDataset` per compound,
        in file order.

    Raises
        FileNotFoundError:
            If *path* does not exist.
        NotDoseResponseFile:
            If the file matches neither layout.
        ValueError:
            If *layout* or *normalize* is unknown, or a value cannot be parsed.
    """
    if layout not in ("auto", "long", "matrix"):
        raise ValueError(
            f"layout must be 'auto', 'long' or 'matrix', got {layout!r}."
        )
    if normalize not in ("none", "percent_of_control"):
        raise ValueError(
            "normalize must be 'none' or 'percent_of_control', got "
            f"{normalize!r}."
        )

    path = Path(path)
    table = read_csv_table(path, encoding=encoding, delimiter=delimiter)
    headers = {normalise_header(name) for name in table.header}

    resolved = layout
    if resolved == "auto":
        has_concentration = bool(headers & _CONCENTRATION_NAMES)
        has_response = bool(headers & _RESPONSE_NAMES)
        if has_concentration and has_response:
            # An explicit response column is the strongest evidence: the
            # matrix layout never has one, so a file with both a
            # concentration and a response column is long-format even when it
            # also carries extra columns.
            resolved = "long"
        elif has_concentration and len(table.header) > 1:
            resolved = "matrix"
        else:
            raise NotDoseResponseFile(
                f"'{path.name}' is not a dose-response table: expected a "
                "concentration column plus either a response column (long "
                f"layout) or one column per compound (matrix layout). Found "
                f"columns: {table.header}."
            )

    if resolved == "long":
        return _read_long(
            table,
            normalize=normalize,
            default_compound=compound_name or path.stem,
            direction=direction,
        )
    return _read_matrix(table, normalize=normalize, direction=direction)
