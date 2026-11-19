"""CSV preview and machine-checked import specifications.

Every analysis module in this project consumes a CSV with a specific shape,
and the single most common support question for a tool like this is *"what
should my file look like?"*.  This module answers that in two ways that share
one source of truth:

1. :func:`preview_csv` reads a file **defensively** and returns a
   :class:`CsvPreview`: the detected encoding and delimiter, the header, a
   typed profile of every column, the first rows as they will be interpreted,
   and a list of structural problems.  It never raises on a malformed file --
   the whole point is to show the user what the parser actually sees.
2. :data:`SCHEMAS` is a registry of :class:`CsvSchema` objects, one per
   supported import format.  A schema is both executable (it validates rows
   and produces location-aware diagnostics) and self-documenting
   (:meth:`CsvSchema.to_markdown` renders the tables that
   ``docs/CSV_IMPORT_SPEC.md`` is generated from).

Why the reader does not use :mod:`pandas`
-----------------------------------------
``pandas.read_csv`` silently coerces, drops, or reinterprets exactly the
things a user needs to see when their file is wrong: a ragged row becomes
``NaN``, a duplicate header becomes ``col.1``, a latin-1 byte sequence becomes
mojibake.  A preview whose job is to explain a file must not repair it, so the
rows are parsed with the standard :mod:`csv` module and every anomaly is
reported instead of hidden.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioeng_qpcr.io.diagnostics import CSVValidationIssue

__all__ = [
    "SCHEMAS",
    "ColumnProfile",
    "ColumnSpec",
    "CsvPreview",
    "CsvSchema",
    "CsvTable",
    "detect_schema",
    "import_spec_markdown",
    "list_schemas",
    "preview_csv",
    "read_csv_table",
    "schema_for",
    "sniff_delimiter",
    "sniff_encoding",
    "validate_rows_against_schema",
    "write_template_csv",
]

#: Encodings tried in order when nothing else is known.  ``utf-8`` comes
#: first and ``utf-8-sig`` is *not* listed: Python's ``utf-8`` codec happily
#: decodes a BOM-prefixed file (yielding a leading ``\ufeff``), so listing
#: ``utf-8-sig`` first would make every plain UTF-8 file report itself as
#: "utf-8-sig".  The BOM is detected separately instead.  ``gb18030`` covers
#: GBK/GB2312 files produced by Chinese versions of Excel, and ``latin-1`` is
#: the final byte-preserving fallback that can never fail.
_ENCODING_CANDIDATES: tuple[str, ...] = (
    "utf-8",
    "gb18030",
    "utf-16",
    "latin-1",
)

#: UTF-8 byte order mark.
_UTF8_BOM = b"\xef\xbb\xbf"

#: Delimiters considered when sniffing.
_DELIMITER_CANDIDATES: tuple[str, ...] = (",", "\t", ";", "|")

#: Unit tokens recognised at the end of a column name, in a trailing
#: ``_unit``, ``(unit)`` or ``[unit]`` position.
_UNIT_TOKENS: tuple[str, ...] = (
    "uM",
    "um",
    "mM",
    "nM",
    "pM",
    "fM",
    "M",
    "%",
    "mg/mL",
    "ug/mL",
    "ng/mL",
    "mg/L",
    "g/L",
    "min",
    "h",
    "s",
    "rpm",
    "C",
    "K",
    "mL",
    "uL",
    "L",
    "mg",
    "ug",
    "ng",
    "g",
    "Da",
    "kDa",
    "M-1s-1",
    "U/mg",
    "RFU",
)

_UNIT_PATTERN = re.compile(r"[\(\[]\s*([^\)\]]{1,12})\s*[\)\]]\s*$")

#: Header-name normalisation: lower-case, collapse separators to ``_``.
_SEPARATOR_PATTERN = re.compile(r"[\s\-\.]+")
_INVALID_HEADER_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+")


# ---------------------------------------------------------------------------
# Schema model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Declaration of one column in an import format.

    Attributes
        name:
            Canonical column name as the reader expects it (matched
            case-insensitively after normalisation).
        type:
            One of ``"number"``, ``"integer"``, ``"text"``, ``"boolean"``,
            ``"datetime"``.
        required:
            Whether the column must be present.
        unit:
            Physical unit the values are expected in, for display.
        description:
            What the column means.
        aliases:
            Alternative accepted spellings of the header.
        minimum, maximum:
            Inclusive numeric bounds.
        choices:
            Permitted values for a categorical column.
        allow_missing:
            Whether the cell may be blank (blank means "not determined" for
            the columns where that is meaningful).
        missing_tokens:
            Extra tokens treated as blank beyond the default set.
        example:
            A representative value, used in the generated template.
    """

    name: str
    type: str = "number"
    required: bool = True
    unit: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    allow_missing: bool = False
    missing_tokens: tuple[str, ...] = ()
    example: str = ""

    def accepted_names(self) -> tuple[str, ...]:
        """Return every accepted spelling of this column, canonical first."""
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class CsvSchema:
    """An import format: its columns, its rules, and its documentation.

    Attributes
        key:
            Registry key (also the identifier used by the CLI and the UI).
        title:
            Human-readable name.
        summary:
            One-paragraph explanation of when to use this format.
        columns:
            The declared columns, in the order they should appear.
        notes:
            Extra rules that cannot be expressed per-column (e.g. "values
            must be sorted by cycle").
        example_rows:
            A minimal valid example, used verbatim by
            :func:`write_template_csv`.
        delimiter:
            Canonical delimiter.  The reader still sniffs, but the generated
            template and the documentation use this one.
        encoding:
            Canonical encoding for written files.
        layout:
            ``"tidy"`` for one observation per row, ``"matrix"`` for a wide
            layout whose extra columns are data series.
        open_series_columns:
            When ``True`` (matrix layout) any number of additional numeric
            columns may follow the declared ones; each is treated as one
            data series and its header is the series name.
        min_rows:
            Minimum number of data rows for the file to be usable.
        header_example:
            Optional explicit example header, when it differs from the
            column names (matrix layouts).
    """

    key: str
    title: str
    summary: str
    columns: tuple[ColumnSpec, ...]
    notes: tuple[str, ...] = ()
    example_rows: tuple[tuple[str, ...], ...] = ()
    delimiter: str = ","
    encoding: str = "utf-8"
    layout: str = "tidy"
    open_series_columns: bool = False
    min_rows: int = 1
    header_example: tuple[str, ...] = ()

    @property
    def required_columns(self) -> tuple[ColumnSpec, ...]:
        """Return the columns that must be present."""
        return tuple(column for column in self.columns if column.required)

    def header(self) -> tuple[str, ...]:
        """Return the header row this schema expects."""
        if self.header_example:
            return self.header_example
        return tuple(column.name for column in self.columns)

    def find(self, header: str) -> ColumnSpec | None:
        """Return the spec matching *header*, or ``None``.

        Matching is case-insensitive and ignores separator style, so
        ``Sample Name``, ``sample-name`` and ``sample_name`` all resolve to
        the same canonical column.
        """
        normalised = normalise_header(header)
        for column in self.columns:
            if any(
                normalise_header(name) == normalised
                for name in column.accepted_names()
            ):
                return column
        return None

    def find_in(self, header: list[str], column: ColumnSpec) -> bool:
        """Return whether *column* appears anywhere in *header*."""
        return any(self.find(name) is column for name in header)

    def present_columns(self, header: list[str]) -> tuple[ColumnSpec, ...]:
        """Return the declared columns present in *header*, in file order."""
        found: list[ColumnSpec] = []
        for name in header:
            spec = self.find(name)
            if spec is not None and spec not in found:
                found.append(spec)
        return tuple(found)

    def example_width_error(self) -> str | None:
        """Return why the example rows are malformed, or ``None`` if they fit.

        Example rows are written verbatim by :func:`write_template_csv` and
        rendered by the web UI, so a row that does not match the header width
        produces a template that no reader will accept.  That failure is easy
        to introduce and invisible unless it is checked, so it is checked.
        """
        width = len(self.header())
        for index, row in enumerate(self.example_rows):
            if len(row) != width:
                return (
                    f"example row {index + 1} has {len(row)} field(s) but the "
                    f"header declares {width}"
                )
        return None

    def to_markdown(self) -> str:
        """Render this format as a Markdown section.

        The output is the authoritative description of the import format and
        is what ``docs/CSV_IMPORT_SPEC.md`` is generated from, so the code and
        the documentation cannot drift apart.
        """
        lines: list[str] = [f"### `{self.key}` — {self.title}", ""]
        lines.append(self.summary)
        lines.append("")
        lines.append(
            f"* 布局：`{self.layout}`　|　分隔符：`{self.delimiter!r}`　|　"
            f"编码：`{self.encoding}`　|　最少数据行：{self.min_rows}"
        )
        lines.append("")
        lines.append("| 列名 | 类型 | 必填 | 单位 | 允许范围 / 取值 | 说明 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for column in self.columns:
            constraints: list[str] = []
            if column.minimum is not None and column.maximum is not None:
                constraints.append(f"{column.minimum:g} ~ {column.maximum:g}")
            elif column.minimum is not None:
                constraints.append(f"≥ {column.minimum:g}")
            elif column.maximum is not None:
                constraints.append(f"≤ {column.maximum:g}")
            if column.choices:
                constraints.append(" / ".join(f"`{c}`" for c in column.choices))
            if column.type == "number" and not constraints:
                constraints.append("任意有限实数")
            if column.allow_missing:
                constraints.append("可留空")
            if column.aliases:
                constraints.append(
                    "别名：" + "、".join(f"`{a}`" for a in column.aliases)
                )
            lines.append(
                "| `{name}` | {type} | {required} | {unit} | {constraints} | "
                "{description} |".format(
                    name=column.name,
                    type=column.type,
                    required="是" if column.required else "否",
                    unit=column.unit or "—",
                    constraints="<br>".join(constraints) or "—",
                    description=column.description,
                )
            )
        if self.open_series_columns:
            lines.append("")
            lines.append(
                "* 声明列之后可继续追加**任意数量**的数值列；每个追加列的"
                "表头即为该数据系列（化合物）的名称。"
            )
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"* {note}")
        if self.example_rows:
            lines.append("")
            lines.append("示例：")
            lines.append("")
            lines.append("```csv")
            lines.append(self.delimiter.join(self.header()))
            for row in self.example_rows:
                lines.append(self.delimiter.join(row))
            lines.append("```")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Header / value helpers
# ---------------------------------------------------------------------------


def normalise_header(header: str) -> str:
    """Normalise a header for tolerant comparison.

    Lower-cases, trims, strips surrounding quotes and BOM characters, and
    collapses spaces/dashes/dots to single underscores.

    Parameters
        header:
            Raw header text.

    Returns
        The normalised key.
    """
    text = str(header).strip().strip('"').strip("'").lstrip("\ufeff")
    text = _SEPARATOR_PATTERN.sub("_", text)
    return text.strip("_").lower()


def _to_float(text: str) -> float | None:
    """Parse *text* as a finite float, or return ``None``."""
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "f"})
_BLANK_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "nd"})

_DATE_PATTERN = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
)


def _is_blank(text: str, extra: tuple[str, ...] = ()) -> bool:
    """Return whether *text* represents a missing value."""
    token = str(text).strip().lower()
    return token in _BLANK_TOKENS or token in {e.lower() for e in extra}


def _is_integral(text: str) -> bool:
    """Return whether *text* is a finite number with no fractional part.

    ``1``, ``1.0`` and ``1e3`` all qualify; ``1.5`` does not.  Spreadsheets
    routinely write whole numbers as ``1.0``, so requiring the literal
    integer syntax would reject correct files.
    """
    value = _to_float(text)
    return value is not None and float(value).is_integer()


def _infer_type(values: list[str]) -> str:
    """Infer the narrowest type that describes every non-blank value."""
    seen = [value for value in values if not _is_blank(value)]
    if not seen:
        return "empty"
    if all(_is_integral(value) for value in seen):
        return "integer"
    if all(_to_float(value) is not None for value in seen):
        return "number"
    lowered = {value.strip().lower() for value in seen}
    if lowered <= (_TRUE_TOKENS | _FALSE_TOKENS):
        return "boolean"
    if all(_DATE_PATTERN.match(value.strip()) for value in seen):
        return "datetime"
    return "text"


def _detect_unit(header: str) -> str | None:
    """Extract a unit from a header such as ``Concentration (uM)``."""
    match = _UNIT_PATTERN.search(str(header))
    if match:
        return match.group(1).strip()
    parts = normalise_header(header).split("_")
    if len(parts) > 1 and parts[-1] in {token.lower() for token in _UNIT_TOKENS}:
        return parts[-1]
    for token in _UNIT_TOKENS:
        if str(header).strip().endswith(f"_{token}"):
            return token
    return None


# ---------------------------------------------------------------------------
# Raw reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvTable:
    """A defensively parsed CSV file.

    Attributes
        path:
            Source path, or ``"<string>"`` for in-memory input.
        encoding:
            Encoding that successfully decoded the bytes.
        delimiter:
            Delimiter used to split the rows.
        header:
            The header cells (empty when the file has no usable header).
        rows:
            Data rows, each padded/truncated to ``len(header)``.
        raw_row_count:
            Number of physical data rows before padding.
        issues:
            Structural problems found while reading.
    """

    path: str
    encoding: str
    delimiter: str
    header: list[str]
    rows: list[list[str]]
    raw_row_count: int
    issues: list[CSVValidationIssue] = field(default_factory=list)


def sniff_encoding(
    raw: bytes, *, candidates: tuple[str, ...] | None = None
) -> tuple[str, bool]:
    """Determine the encoding of *raw* bytes.

    UTF-8 (with or without BOM) is tried first, then GB18030 (the superset of
    GBK/GB2312 used by Chinese Excel), then UTF-16, and finally latin-1 which
    cannot fail.  Decoding is verified by round-tripping, so a file that
    merely *decodes* without error under a wrong codec is still rejected.

    Parameters
        raw:
            Raw file bytes.
        candidates:
            Override for the candidate list (mainly for tests).

    Returns
        ``(encoding, fell_back)`` where *fell_back* is ``True`` when the file
        did not decode as UTF-8 and a legacy codec was used, which the caller
        reports as a warning because it usually means the file came out of a
        spreadsheet configured for a non-UTF-8 locale.
    """
    if raw.startswith(_UTF8_BOM):
        return "utf-8-sig", False
    order = candidates if candidates is not None else _ENCODING_CANDIDATES
    for index, encoding in enumerate(order):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if "\x00" in text and encoding not in ("utf-16", "utf-32"):
            continue
        return encoding, index > 0
    return "latin-1", True


def sniff_delimiter(text: str) -> str:
    """Guess the delimiter of *text*.

    Uses :class:`csv.Sniffer` when it succeeds and falls back to counting
    candidate delimiters outside quoted regions, choosing the one with the
    most consistent per-line count.  Consistency matters more than raw
    frequency: a comma-separated file whose free-text column contains
    semicolons must not be mistaken for a semicolon-separated file.

    Parameters
        text:
            Decoded file content.

    Returns
        The most likely delimiter.
    """
    sample = "\n".join(text.splitlines()[:25])
    try:
        sniffer = csv.Sniffer()
        return sniffer.sniff(
            sample, delimiters="".join(_DELIMITER_CANDIDATES)
        ).delimiter
    except csv.Error:
        pass

    best, best_score = ",", -1.0
    for delimiter in _DELIMITER_CANDIDATES:
        try:
            rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
        except csv.Error:
            continue
        counts = [len(row) for row in rows if row]
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        # Reward columns and consistency, penalise raggedness.
        score = mode * (counts.count(mode) / len(counts))
        if score > best_score:
            best, best_score = delimiter, score
    return best


def read_csv_table(
    path: str | Path,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    max_rows: int | None = None,
) -> CsvTable:
    """Read a CSV file without repairing anything.

    Parameters
        path:
            File to read.
        encoding:
            Force an encoding instead of sniffing.
        delimiter:
            Force a delimiter instead of sniffing.
        max_rows:
            Stop after this many data rows (used by the preview).

    Returns
        A :class:`CsvTable`.  Structural problems are reported in
        ``table.issues`` rather than raised.

    Raises
        FileNotFoundError:
            If *path* does not exist.
        ValueError:
            If the file is empty.
    """
    path = Path(path)
    raw = path.read_bytes()
    issues: list[CSVValidationIssue] = []
    if not raw.strip():
        raise ValueError(f"CSV file '{path}' is empty.")

    if encoding is None:
        encoding, fell_back = sniff_encoding(raw)
        if fell_back:
            issues.append(
                CSVValidationIssue(
                    code="ENCODING_FALLBACK",
                    message=(
                        "the file is not valid UTF-8 and was decoded as "
                        f"{encoding!r}; re-export it as UTF-8 to avoid "
                        "mangled characters"
                    ),
                    severity="warning",
                )
            )
    text = raw.decode(encoding, errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]

    if delimiter is None:
        delimiter = sniff_delimiter(text)

    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        raw_rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"CSV file '{path}' could not be parsed: {exc}") from exc

    raw_rows = [row for row in raw_rows if any(cell.strip() for cell in row)]
    if not raw_rows:
        raise ValueError(f"CSV file '{path}' contains no data.")

    header = [str(cell) for cell in raw_rows[0]]
    width = len(header)
    data_rows: list[list[str]] = []
    for index, row in enumerate(raw_rows[1 : (max_rows + 1) if max_rows else None]):
        csv_row = index + 2
        if len(row) != width:
            if len(row) < width:
                issues.append(
                    CSVValidationIssue(
                        code="SHORT_ROW",
                        message=(
                            f"row has {len(row)} field(s) but the header "
                            f"declares {width}; missing cells were treated "
                            "as blank"
                        ),
                        row=csv_row,
                        severity="warning",
                    )
                )
                row = [*row, *([""] * (width - len(row)))]
            else:
                issues.append(
                    CSVValidationIssue(
                        code="LONG_ROW",
                        message=(
                            f"row has {len(row)} field(s) but the header "
                            f"declares {width}; extra cells were discarded"
                        ),
                        row=csv_row,
                        value=str(row[width:]),
                        severity="warning",
                    )
                )
                row = row[:width]
        data_rows.append([str(cell) for cell in row])

    for name in header:
        if not str(name).strip():
            issues.append(
                CSVValidationIssue(
                    code="EMPTY_HEADER",
                    message="the header row contains an unnamed column",
                    severity="error",
                )
            )
            break
    normalised = [normalise_header(name) for name in header]
    duplicates = sorted(
        {name for name in normalised if name and normalised.count(name) > 1}
    )
    for name in duplicates:
        issues.append(
            CSVValidationIssue(
                code="DUPLICATE_HEADER",
                message=f"the header declares '{name}' more than once",
                column=name,
                severity="error",
            )
        )

    return CsvTable(
        path=str(path),
        encoding=encoding,
        delimiter=delimiter,
        header=header,
        rows=data_rows,
        raw_row_count=max(len(raw_rows) - 1, 0),
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    """What the sample of a column actually contains.

    Attributes
        name:
            Header as written in the file.
        index:
            Zero-based column position.
        inferred_type:
            ``"integer"``, ``"number"``, ``"boolean"``, ``"datetime"``,
            ``"text"`` or ``"empty"``.
        n_missing, missing_fraction:
            Blank/NA counts and their fraction of the scanned rows.
        n_unique:
            Number of distinct non-blank values.
        examples:
            Up to five representative values.
        minimum, maximum, mean:
            Numeric summary, when the column is numeric.
        unit:
            Unit parsed from the header, when present.
        matched_spec:
            Name of the schema column this header matched, when a schema was
            supplied.
    """

    name: str
    index: int
    inferred_type: str
    n_missing: int
    missing_fraction: float
    n_unique: int
    examples: tuple[str, ...]
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    unit: str | None = None
    matched_spec: str | None = None


@dataclass(frozen=True)
class CsvPreview:
    """A complete, non-destructive description of a CSV file.

    Attributes
        table:
            The defensively parsed table the preview was built from.
        columns:
            Per-column profiles.
        head:
            The first rows exactly as they will be interpreted, with the
            header as the first element.
        rows_scanned:
            Number of data rows actually profiled.
        rows_total:
            Number of data rows present in the file.
        truncated:
            ``True`` when only a prefix of the file was profiled.
        size_bytes, sha256:
            File size and content hash, for provenance.
        issues:
            Structural problems (from reading) plus schema violations.
        schema_key:
            Registry key of the schema used for validation, if any.
        schema_score:
            How well the header matched that schema, in ``[0, 1]``.
    """

    table: CsvTable
    columns: tuple[ColumnProfile, ...]
    head: tuple[tuple[str, ...], ...]
    rows_scanned: int
    rows_total: int
    truncated: bool
    size_bytes: int
    sha256: str
    issues: tuple[CSVValidationIssue, ...]
    schema_key: str | None
    schema_score: float

    @property
    def errors(self) -> list[CSVValidationIssue]:
        """Issues that must be fixed before the file can be analysed."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[CSVValidationIssue]:
        """Issues that do not block analysis but change the interpretation."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_text(self, *, max_table_rows: int = 12) -> str:
        """Render a human-readable preview report.

        Parameters
            max_table_rows:
                How many data rows to include in the rendered table.

        Returns
            A plain-text report suitable for a terminal or a text panel.
        """
        lines: list[str] = []
        lines.append("CSV 预览")
        lines.append("=" * 52)
        lines.append(f"文件:     {self.table.path}")
        lines.append(f"编码:     {self.table.encoding}")
        lines.append(f"分隔符:   {self.table.delimiter!r}")
        lines.append(f"大小:     {self.size_bytes:,} bytes")
        lines.append(f"SHA256:   {self.sha256[:16]}…")
        lines.append(
            f"数据行:   {self.rows_total}"
            + (f"（已扫描 {self.rows_scanned} 行）" if self.truncated else "")
        )
        lines.append(f"列数:     {len(self.columns)}")
        if self.schema_key:
            lines.append(
                f"匹配格式: {self.schema_key} "
                f"（匹配度 {self.schema_score:.0%}）"
            )
        else:
            lines.append("匹配格式: 未匹配到内置格式（可按通用 CSV 处理）")

        lines.append("")
        lines.append("数据内容")
        lines.append("-" * 52)
        for row_index, row in enumerate(self.head[: max_table_rows + 1]):
            marker = "H" if row_index == 0 else str(row_index)
            lines.append(
                f"{marker:>3} | " + " | ".join(cell[:18] for cell in row)
            )
        if len(self.head) > max_table_rows + 1:
            remaining = len(self.head) - max_table_rows - 1
            lines.append(f"    …  余下 {remaining} 行未显示")

        lines.append("")
        lines.append("列概况")
        lines.append("-" * 52)
        for column in self.columns:
            summary = f"  {column.name!r} ({column.index})  {column.inferred_type}"
            if column.unit:
                summary += f"  unit={column.unit}"
            if column.n_missing:
                summary += f"  空值 {column.n_missing} ({column.missing_fraction:.0%})"
            if column.minimum is not None:
                summary += f"  范围 [{column.minimum:g}, {column.maximum:g}]"
            if column.matched_spec:
                summary += f"  → {column.matched_spec}"
            lines.append(summary)

        if self.issues:
            lines.append("")
            lines.append(f"问题（{len(self.issues)}）")
            lines.append("-" * 52)
            for issue in self.issues[:40]:
                prefix = {"error": "[错误]", "warning": "[警告]", "info": "[提示]"}
                lines.append(f"{prefix.get(issue.severity, '[?]')} {issue.format()}")
            if len(self.issues) > 40:
                lines.append(f"  … 余下 {len(self.issues) - 40} 条未显示")
        else:
            lines.append("")
            lines.append("未发现结构性问题。")
        return "\n".join(lines)


def preview_csv(
    path: str | Path,
    *,
    schema: CsvSchema | str | None = None,
    preview_rows: int = 200,
    display_rows: int = 20,
    encoding: str | None = None,
    delimiter: str | None = None,
) -> CsvPreview:
    """Describe a CSV file and check it against an import specification.

    Parameters
        path:
            File to preview.
        schema:
            A :class:`CsvSchema`, a registry key, or ``None`` to auto-detect
            the best-matching registered format.
        preview_rows:
            How many data rows to scan when profiling columns.
        display_rows:
            How many rows to include in ``head``.
        encoding, delimiter:
            Overrides for the sniffed values.

    Returns
        A :class:`CsvPreview`.  Structural problems never raise; they are
        reported in ``preview.issues`` with a severity.

    Raises
        FileNotFoundError:
            If *path* does not exist.
        ValueError:
            If the file is empty or unparseable at the byte level.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    table = read_csv_table(
        path,
        encoding=encoding,
        delimiter=delimiter,
        max_rows=preview_rows,
    )

    resolved: CsvSchema | None = None
    score = 0.0
    if isinstance(schema, CsvSchema):
        resolved = schema
        score = _schema_match_score(schema, table.header)
    elif isinstance(schema, str):
        resolved = schema_for(schema)
        score = _schema_match_score(resolved, table.header)
    else:
        key, score = detect_schema(table.header)
        resolved = SCHEMAS[key] if key else None

    issues = list(table.issues)
    if resolved is not None:
        issues.extend(validate_rows_against_schema(table, resolved))
    if table.raw_row_count > len(table.rows):
        issues.append(
            CSVValidationIssue(
                code="PREVIEW_TRUNCATED",
                message=(
                    f"only the first {len(table.rows)} of "
                    f"{table.raw_row_count} data rows were profiled"
                ),
                severity="info",
            )
        )

    columns = tuple(
        _profile_column(index, table, resolved)
        for index in range(len(table.header))
    )
    head: list[tuple[str, ...]] = [tuple(table.header)]
    head.extend(tuple(row) for row in table.rows[:display_rows])

    return CsvPreview(
        table=table,
        columns=columns,
        head=tuple(head),
        rows_scanned=len(table.rows),
        rows_total=table.raw_row_count,
        truncated=table.raw_row_count > len(table.rows),
        size_bytes=size,
        sha256=digest,
        issues=tuple(issues),
        schema_key=resolved.key if resolved else None,
        schema_score=score,
    )


def _profile_column(
    index: int, table: CsvTable, schema: CsvSchema | None
) -> ColumnProfile:
    """Build the :class:`ColumnProfile` for one column of *table*."""
    name = table.header[index]
    values = [row[index] for row in table.rows]
    inferred = _infer_type(values)
    missing = sum(1 for value in values if _is_blank(value))
    non_blank = [value.strip() for value in values if not _is_blank(value)]
    seen: list[str] = []
    for value in non_blank:
        if value not in seen:
            seen.append(value)
        if len(seen) >= 5:
            break

    minimum = maximum = mean = None
    numbers = [_to_float(value) for value in non_blank]
    numeric = [value for value in numbers if value is not None]
    if numeric and len(numeric) == len(non_blank):
        minimum = min(numeric)
        maximum = max(numeric)
        mean = sum(numeric) / len(numeric)

    matched = schema.find(name) if schema is not None else None
    return ColumnProfile(
        name=name,
        index=index,
        inferred_type=inferred,
        n_missing=missing,
        missing_fraction=(missing / len(values)) if values else 0.0,
        n_unique=len(set(non_blank)),
        examples=tuple(seen),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        unit=_detect_unit(name),
        matched_spec=matched.name if matched else None,
    )


def _schema_match_score(schema: CsvSchema, header: list[str]) -> float:
    """Fraction of the schema's required columns present in *header*."""
    required = schema.required_columns
    if not required:
        return 0.0
    present = sum(1 for column in required if schema.find_in(header, column))
    return present / len(required)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_rows_against_schema(
    table: CsvTable, schema: CsvSchema
) -> list[CSVValidationIssue]:
    """Check *table* against *schema* and return every violation.

    Parameters
        table:
            Parsed table.
        schema:
            Format the table is supposed to follow.

    Returns
        Location-aware issues: missing/unknown/duplicate columns, unparsable
        values, out-of-range numbers, invalid categories, and blank cells in
        columns that may not be blank.
    """
    issues: list[CSVValidationIssue] = []

    for column in schema.required_columns:
        if not schema.find_in(table.header, column):
            issues.append(
                CSVValidationIssue(
                    code="MISSING_COLUMN",
                    message=(
                        "required column is absent; accepted spellings: "
                        + ", ".join(repr(name) for name in column.accepted_names())
                    ),
                    column=column.name,
                )
            )

    matched_specs: list[tuple[int, ColumnSpec]] = []
    for index, header in enumerate(table.header):
        spec = schema.find(header)
        if spec is None:
            if not schema.open_series_columns:
                issues.append(
                    CSVValidationIssue(
                        code="UNKNOWN_COLUMN",
                        message=(
                            "column is not part of this format and will be "
                            "ignored; remove it to avoid ambiguity"
                        ),
                        column=header,
                        severity="warning",
                    )
                )
        else:
            matched_specs.append((index, spec))

    if len(table.rows) < schema.min_rows:
        issues.append(
            CSVValidationIssue(
                code="TOO_FEW_ROWS",
                message=(
                    f"format needs at least {schema.min_rows} data row(s), "
                    f"found {len(table.rows)}"
                ),
            )
        )

    for index, spec in matched_specs:
        for row_index, row in enumerate(table.rows):
            csv_row = row_index + 2
            raw = row[index] if index < len(row) else ""
            if _is_blank(raw, spec.missing_tokens):
                if spec.required and not spec.allow_missing:
                    issues.append(
                        CSVValidationIssue(
                            code="MISSING_VALUE",
                            message="a value is required in this column",
                            column=spec.name,
                            row=csv_row,
                        )
                    )
                continue

            text = raw.strip()
            if spec.type in ("number", "integer"):
                value = _to_float(text)
                if value is None:
                    issues.append(
                        CSVValidationIssue(
                            code="INVALID_NUMBER",
                            message=f"expected a finite {spec.type}",
                            column=spec.name,
                            row=csv_row,
                            value=raw,
                        )
                    )
                    continue
                if spec.type == "integer" and not float(value).is_integer():
                    issues.append(
                        CSVValidationIssue(
                            code="INVALID_INTEGER",
                            message="expected a whole number",
                            column=spec.name,
                            row=csv_row,
                            value=raw,
                        )
                    )
                    continue
                if spec.minimum is not None and value < spec.minimum:
                    issues.append(
                        CSVValidationIssue(
                            code="OUT_OF_RANGE",
                            message=f"value must be >= {spec.minimum:g}",
                            column=spec.name,
                            row=csv_row,
                            value=raw,
                        )
                    )
                if spec.maximum is not None and value > spec.maximum:
                    issues.append(
                        CSVValidationIssue(
                            code="OUT_OF_RANGE",
                            message=f"value must be <= {spec.maximum:g}",
                            column=spec.name,
                            row=csv_row,
                            value=raw,
                        )
                    )
            elif spec.type == "boolean":
                if text.lower() not in (_TRUE_TOKENS | _FALSE_TOKENS):
                    issues.append(
                        CSVValidationIssue(
                            code="INVALID_BOOLEAN",
                            message="expected true/false, yes/no, or 1/0",
                            column=spec.name,
                            row=csv_row,
                            value=raw,
                        )
                    )
            elif spec.type == "datetime":
                if not _DATE_PATTERN.match(text):
                    issues.append(
                        CSVValidationIssue(
                            code="INVALID_DATE",
                            message="expected an ISO-like date (YYYY-MM-DD)",
                            column=spec.name,
                            row=csv_row,
                            value=raw,
                        )
                    )
            if spec.choices and text not in spec.choices:
                issues.append(
                    CSVValidationIssue(
                        code="INVALID_CHOICE",
                        message="expected one of: " + ", ".join(spec.choices),
                        column=spec.name,
                        row=csv_row,
                        value=raw,
                    )
                )

    duplicate_rows = _duplicate_key_rows(table, schema)
    if duplicate_rows:
        issues.append(
            CSVValidationIssue(
                code="DUPLICATE_ROW",
                message=(
                    f"{duplicate_rows} row(s) repeat an identical key; the "
                    "first occurrence wins and the rest are ignored"
                ),
                severity="warning",
            )
        )
    return issues


def _duplicate_key_rows(table: CsvTable, schema: CsvSchema) -> int:
    """Count rows whose key columns repeat an earlier row's values."""
    key_specs = [
        column
        for column in schema.columns
        if column.name in {"position", "cycle", "sample_name", "target_name"}
    ]
    if not key_specs:
        return 0
    indices = [
        table.header.index(header)
        for header in table.header
        if schema.find(header) in key_specs
    ]
    if not indices:
        return 0
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in table.rows:
        key = tuple(row[index].strip() for index in indices if index < len(row))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _spec(name: str, column_type: str = "number", **kwargs: Any) -> ColumnSpec:
    """Shorthand constructor used by the registry below."""
    return ColumnSpec(name, type=column_type, **kwargs)


#: The tidy Level A qPCR plate format read by
#: :func:`bioeng_qpcr.io.read_tidy_csv`.
_QPCR_TIDY = CsvSchema(
    key="qpcr_tidy",
    title="qPCR 整理型（Level A）数据表",
    summary=(
        "一行一口（one row per well）的 qPCR 结果表，是 MIQE 验证、相对定量与"
        "分组比较的唯一必需输入。列名不区分大小写，空格/连字符/下划线等价。"
    ),
    columns=(
        _spec(
            "position",
            "text",
            description="孔位编号，全表唯一（如 A1、A2、P1-1）",
        ),
        _spec(
            "sample_name",
            "text",
            description=(
                "样本名；NTC / NO_TEMPLATE / BLANK 会被识别为无模板对照"
            ),
        ),
        _spec("target_name", "text", description="靶标（基因）名"),
        _spec(
            "cq",
            "number",
            minimum=0.0,
            allow_missing=True,
            missing_tokens=("undetermined",),
            description="定量循环值；未检出可留空或写 Undetermined",
        ),
        _spec(
            "replicate_type",
            "text",
            required=False,
            choices=("technical", "biological", "single"),
            allow_missing=True,
            description="重复类型",
        ),
        _spec(
            "replicate_group",
            "text",
            required=False,
            allow_missing=True,
            description="重复组标签",
        ),
        _spec(
            "cq_status",
            "text",
            required=False,
            choices=("determined", "undetermined", "no_template"),
            allow_missing=True,
            description="Cq 状态；留空时按 cq 与样本名自动推断",
        ),
        _spec(
            "sample_group",
            "text",
            required=False,
            allow_missing=True,
            description="分组名，用于分组比较",
        ),
        _spec(
            "efficiency",
            "number",
            required=False,
            allow_missing=True,
            description="扩增效率（1.0 = 100%）",
        ),
        _spec(
            "r2",
            "number",
            required=False,
            allow_missing=True,
            minimum=0.0,
            maximum=1.0,
            description="标准曲线 R²",
        ),
        _spec(
            "slope",
            "number",
            required=False,
            allow_missing=True,
            description="标准曲线斜率",
        ),
        _spec(
            "is_reference",
            "boolean",
            required=False,
            allow_missing=True,
            description="是否内参基因；留空时按 reference_targets 判断",
        ),
        _spec(
            "primer_forward",
            "text",
            required=False,
            allow_missing=True,
            description="正向引物序列",
        ),
        _spec(
            "primer_reverse",
            "text",
            required=False,
            allow_missing=True,
            description="反向引物序列",
        ),
        _spec(
            "probe",
            "text",
            required=False,
            allow_missing=True,
            description="探针序列",
        ),
        _spec(
            "lod_cq",
            "number",
            required=False,
            allow_missing=True,
            minimum=0.0,
            description="检出限 Cq",
        ),
        _spec(
            "lloq_cq",
            "number",
            required=False,
            allow_missing=True,
            minimum=0.0,
            description="定量限 Cq",
        ),
    ),
    notes=(
        "`position` 必须全表唯一；重复孔位会被拒绝。",
        "未检出（Undetermined）应留空或写 `Undetermined`，不要写 0，"
        "否则会被当成真实的极低 Cq。",
        "`lloq_cq` 必须小于等于 `lod_cq`。",
        "同一 `target_name` 的效率/R²/引物等元数据必须一致，"
        "冲突会报 `CONFLICTING_ASSAY_METADATA`。",
    ),
    example_rows=(
        (
            "A1", "Control_1", "GAPDH", "18.42", "technical", "Control_1_rep1",
            "determined", "control", "0.98", "0.999", "-3.36", "true",
            "GACCAGGTTCTGCTTGTC", "CTTGCTCTCAGTATCCTCG", "CGAGAAGCTGTGCTACGTC",
            "37.0", "35.0",
        ),
        (
            "A3", "Control_1", "IL6", "25.13", "technical", "Control_1_rep1",
            "determined", "control", "", "", "", "false",
            "AGACAGCCACTCACCTCTTC", "GTTCCTGAGAGACTTGGCTG", "", "", "",
        ),
        (
            "B1", "Treated_1", "GAPDH", "18.66", "technical", "Treated_1_rep1",
            "determined", "treated", "0.98", "0.999", "-3.36", "true",
            "GACCAGGTTCTGCTTGTC", "CTTGCTCTCAGTATCCTCG", "CGAGAAGCTGTGCTACGTC",
            "37.0", "35.0",
        ),
        (
            "B3", "Treated_1", "IL6", "22.87", "technical", "Treated_1_rep1",
            "determined", "treated", "", "", "", "false",
            "AGACAGCCACTCACCTCTTC", "GTTCCTGAGAGACTTGGCTG", "", "", "",
        ),
        (
            "C1", "NTC", "IL6", "", "single", "", "no_template", "", "", "", "",
            "false", "AGACAGCCACTCACCTCTTC", "GTTCCTGAGAGACTTGGCTG", "", "", "",
        ),
    ),
)

#: Replicated qPCR standards for efficiency, LOD and quantifiable range.
_QPCR_STANDARD_CURVE = CsvSchema(
    key="qpcr_standard_curve",
    title="qPCR 标准曲线与检出限数据",
    summary=(
        "每行一个标准品重复，用于拟合扩增效率、Cq 预测区间，并从重复检测率与"
        "Cq 精密度评估经验 LOD、LLOQ、ULOQ 和动态范围。"
    ),
    columns=(
        _spec(
            "concentration",
            "number",
            minimum=1e-300,
            description="标准品浓度或拷贝数；全表必须使用同一单位且严格大于 0",
        ),
        _spec(
            "cq",
            "number",
            minimum=0.0,
            allow_missing=True,
            missing_tokens=("undetermined",),
            description="测得的 Cq；未检出必须留空或写 Undetermined",
        ),
        _spec(
            "replicate",
            "text",
            required=False,
            allow_missing=True,
            description="重复编号，仅用于追溯",
        ),
    ),
    notes=(
        "每个浓度水平建议至少 3 个重复；未检出也必须保留为一行，否则检测率会被高估。",
        "LOD 判据默认检测率 ≥95%；LLOQ 还要求重复 Cq 标准差 ≤0.5，"
        "两项阈值均可按实验室 SOP 调整并随结果记录。",
        "至少提供 3 个不同且有有限 Cq 的浓度水平，才能拟合标准曲线。",
    ),
    example_rows=(
        ("1", "", "1"),
        ("1", "Undetermined", "2"),
        ("1", "35.2", "3"),
        ("10", "31.8", "1"),
        ("10", "31.7", "2"),
        ("10", "31.9", "3"),
        ("100", "28.5", "1"),
        ("100", "28.4", "2"),
        ("100", "28.6", "3"),
        ("1000", "25.1", "1"),
        ("1000", "25.2", "2"),
        ("1000", "25.0", "3"),
    ),
    min_rows=3,
)

#: Long-format amplification traces read by
#: :func:`bioeng_qpcr.io.read_amplification_csv`.
_AMPLIFICATION_TRACES = CsvSchema(
    key="amplification_traces",
    title="扩增曲线原始荧光数据（long 格式）",
    summary=(
        "每个循环一行的原始荧光数据，用于基线校正、逐孔 sigmoid 拟合与效率估计。"
        "没有该表也能做相对定量，但只有它能做曲线级质控。"
    ),
    columns=(
        _spec(
            "position",
            "text",
            description="孔位编号；同一孔位的循环必须连续且不重复",
        ),
        _spec(
            "cycle",
            "integer",
            minimum=1.0,
            description="循环数，从 1 开始",
        ),
        _spec(
            "fluorescence",
            "number",
            description="该循环的荧光读数（原始值，勿做归一化）",
        ),
        _spec(
            "target_name",
            "text",
            required=False,
            allow_missing=True,
            description="靶标名，可省略",
        ),
        _spec(
            "sample_name",
            "text",
            required=False,
            allow_missing=True,
            description="样本名，可省略",
        ),
    ),
    notes=(
        "`position` + `cycle` 组合必须唯一。",
        "不要提交已经做过基线校正或归一化的荧光值；软件会自己做并记录方法。",
    ),
    example_rows=(
        ("A1", "1", "1023.5", "GAPDH", "Control_1"),
        ("A1", "2", "1011.2", "GAPDH", "Control_1"),
        ("A1", "3", "1044.8", "GAPDH", "Control_1"),
    ),
    min_rows=3,
)

#: Enzyme kinetics (Michaelis-Menten and inhibition) as read by
#: :func:`bioeng_qpcr.enzyme_kinetics.read_kinetics_csv`.
_ENZYME_KINETICS = CsvSchema(
    key="enzyme_kinetics",
    title="酶动力学数据表",
    summary=(
        "`[S]` 与初速度的配对表。带 `inhibitor_concentration` 列时同一文件可同时"
        "用于抑制类型判别与全局模型选择。"
    ),
    columns=(
        _spec(
            "substrate_concentration",
            "number",
            minimum=0.0,
            unit="同 Km 单位",
            description="底物浓度 [S]",
        ),
        _spec("velocity", "number", minimum=0.0, description="初速度 v（单位/时间）"),
        _spec(
            "inhibitor_concentration",
            "number",
            required=False,
            minimum=0.0,
            description="抑制剂浓度 [I]；抑制剂判别必需，且需至少两个不同取值",
        ),
    ),
    notes=(
        "至少需要 5 个不同 [S] 点，且应同时覆盖 Km 以下与以上（ENZ-002）。",
        "`inhibitor_concentration` 至少需要 2 个不同取值，其中至少一个大于 0。",
    ),
    example_rows=(
        ("0.1", "1.12", "0"),
        ("0.5", "4.31", "0"),
        ("2.0", "8.94", "0"),
        ("10.0", "12.05", "0"),
        ("0.5", "2.11", "5"),
        ("2.0", "5.32", "5"),
    ),
    min_rows=5,
)

#: Batch fermentation kinetics as read by
#: :func:`bioeng_qpcr.fermentation_kinetics.read_fermentation_csv`.
_FERMENTATION_KINETICS = CsvSchema(
    key="fermentation_kinetics",
    title="发酵动力学时间序列",
    summary="时间、生物量与底物浓度的三元组，用于 Monod 参数拟合与可辨识性分析。",
    columns=(
        _spec(
            "time",
            "number",
            minimum=0.0,
            unit="h",
            description="培养时间",
        ),
        _spec(
            "biomass",
            "number",
            minimum=0.0,
            unit="g/L",
            description="生物量浓度 X",
        ),
        _spec(
            "substrate",
            "number",
            minimum=0.0,
            unit="g/L",
            description="限制性底物浓度 S",
        ),
    ),
    notes=(
        "时间必须严格递增且不得重复。",
        "至少 6 个时间点，否则参数不可辨识。",
    ),
    example_rows=(
        ("0", "0.10", "10.00"),
        ("2", "0.21", "9.42"),
        ("4", "0.48", "8.15"),
        ("6", "1.02", "6.40"),
        ("8", "1.94", "4.12"),
        ("10", "3.05", "2.10"),
    ),
    min_rows=6,
)

#: Long-format dose-response table for IC50/EC50 determination.
_DOSE_RESPONSE_LONG = CsvSchema(
    key="dose_response",
    title="剂量-反应数据表（long 格式）",
    summary=(
        "一个化合物一组长格式数据：每一行是一对（浓度, 响应）。含 `compound` 列时"
        "一个文件可放多个化合物，用于批量 IC50/EC50 与相对效价比较。"
    ),
    columns=(
        _spec(
            "compound",
            "text",
            description="化合物名称；同一化合物的行构成一条曲线",
        ),
        _spec(
            "concentration",
            "number",
            minimum=1e-15,
            description="浓度，必须严格大于 0（曲线在 log10 空间拟合）",
        ),
        _spec(
            "response",
            "number",
            description="响应值，任意单位（吸光、荧光、活率百分比等）",
        ),
        _spec(
            "replicate",
            "text",
            required=False,
            allow_missing=True,
            description="重复标签；提供后可做孔内精密度检查（POT-010）",
        ),
    ),
    notes=(
        "浓度必须 > 0：log10(0) 未定义；未处理/溶剂对照响应请通过界面字段"
        "或 CLI `--control-response` 传入。",
        "建议至少 5 个不同浓度水平、跨度不少于 2 个数量级（POT-003 / POT-006）。",
        "响应方向由数据决定：随浓度上升为 EC50，随浓度下降为 IC50；可显式指定。",
        "如需百分比化，请自行先归一化；软件会按原始响应拟合并同时报告相对与绝对效价。",
    ),
    example_rows=(
        ("CompoundA", "0.01", "98.2", "1"),
        ("CompoundA", "0.1", "96.4", "1"),
        ("CompoundA", "1", "88.1", "1"),
        ("CompoundA", "10", "52.3", "1"),
        ("CompoundA", "100", "14.7", "1"),
        ("CompoundA", "1000", "5.2", "1"),
        ("CompoundB", "0.01", "99.0", "1"),
        ("CompoundB", "0.1", "97.8", "1"),
        ("CompoundB", "1", "95.2", "1"),
        ("CompoundB", "10", "81.6", "1"),
        ("CompoundB", "100", "40.4", "1"),
        ("CompoundB", "1000", "8.9", "1"),
    ),
    min_rows=5,
)

#: Wide dose-response matrix: one concentration column plus one column per
#: compound.  This is the layout most people export from a plate reader.
_DOSE_RESPONSE_MATRIX = CsvSchema(
    key="dose_response_matrix",
    title="剂量-反应矩阵表（wide 格式）",
    summary=(
        "第一列为浓度，其余每一列是一个化合物（表头即化合物名）。这是酶标仪导出"
        "的常见形状，软件会自动转成长格式再拟合。"
    ),
    layout="matrix",
    open_series_columns=True,
    columns=(
        _spec(
            "concentration",
            "number",
            minimum=1e-15,
            aliases=("conc", "dose", "concentration_uM"),
            description="浓度；该列之后的每一列都是响应值",
        ),
    ),
    notes=(
        "与 `dose_response` 二选一即可；自动识别时若存在 `compound` 列，"
        "则优先按 long 格式解析。",
        "矩阵中的空格视为缺失，不会当作 0 参与拟合。",
        "所有化合物列共用同一浓度序列。",
    ),
    header_example=("concentration", "CompoundA", "CompoundB"),
    example_rows=(
        ("0.01", "98.2", "99.0"),
        ("0.1", "96.4", "97.8"),
        ("1", "88.1", "95.2"),
        ("10", "52.3", "81.6"),
        ("100", "14.7", "40.4"),
        ("1000", "5.2", "8.9"),
    ),
    min_rows=5,
)

#: Registry of every import format the software understands.
SCHEMAS: dict[str, CsvSchema] = {
    schema.key: schema
    for schema in (
        _QPCR_TIDY,
        _QPCR_STANDARD_CURVE,
        _AMPLIFICATION_TRACES,
        _ENZYME_KINETICS,
        _FERMENTATION_KINETICS,
        _DOSE_RESPONSE_LONG,
        _DOSE_RESPONSE_MATRIX,
    )
}


def schema_for(key: str) -> CsvSchema:
    """Return the schema registered under *key*.

    Raises
        KeyError:
            If no schema uses that key.
    """
    try:
        return SCHEMAS[key]
    except KeyError:
        raise KeyError(
            f"Unknown CSV schema {key!r}; available: {sorted(SCHEMAS)}"
        ) from None


def list_schemas() -> list[CsvSchema]:
    """Return every registered schema, ordered by key."""
    return [SCHEMAS[key] for key in sorted(SCHEMAS)]


def detect_schema(header: list[str]) -> tuple[str | None, float]:
    """Find the registered format that best matches *header*.

    The matrix dose-response format is deliberately only considered when the
    tidy one does not match, because both accept a bare ``concentration``
    column and the tidy format carries more information.

    Parameters
        header:
            Header cells of the file under inspection.

    Returns
        ``(key, score)``.  ``key`` is ``None`` when no format matches at least
        half of its required columns.
    """
    best_key: str | None = None
    best_score = 0.0
    # ``dose_response_matrix`` is excluded here: it has only one required
    # column, so it would match almost any file with a concentration column.
    # It is considered only after the tidy formats have been ruled out.
    tidy_candidates = (
        "qpcr_tidy",
        "qpcr_standard_curve",
        "dose_response",
        "enzyme_kinetics",
        "fermentation_kinetics",
        "amplification_traces",
    )
    for key in tidy_candidates:
        schema = SCHEMAS[key]
        score = _schema_match_score(schema, header)
        extra = _extra_column_penalty(schema, header)
        adjusted = score - extra
        if adjusted > best_score:
            best_key, best_score = key, adjusted
    if best_key is None or best_score < 0.5:
        matrix = SCHEMAS["dose_response_matrix"]
        if matrix.find_in(header, matrix.columns[0]) and len(header) >= 2:
            return "dose_response_matrix", 1.0
        return None, max(best_score, 0.0)
    return best_key, best_score


def _extra_column_penalty(schema: CsvSchema, header: list[str]) -> float:
    """Small penalty for unexpected columns, to break near-ties sensibly."""
    if schema.open_series_columns:
        return 0.0
    unknown = sum(1 for name in header if schema.find(name) is None)
    return 0.05 * unknown


# ---------------------------------------------------------------------------
# Documentation and templates
# ---------------------------------------------------------------------------


_SPEC_PREAMBLE = """# CSV 导入格式规范

本文档由代码生成，是 `bioeng-qpcr` 唯一权威的导入格式说明。每个分析模块都只
接受下列表格格式之一；**列名不区分大小写**，空格、连字符、点号与下划线在匹配时
等价（`Sample Name`、`sample-name`、`sample_name` 视为同一列）。

## 通用规则

1. **编码**：优先 UTF-8（可带 BOM）。若文件不是合法 UTF-8，软件会退回
   GB18030/GB2312 解码并给出警告；请尽量从 Excel 另存为「CSV UTF-8」。
2. **分隔符**：支持逗号、制表符、分号、竖线，自动探测。同一个文件内必须统一。
3. **表头**：必须有表头行，且表头不能为空、不能重复。
4. **缺失值**：可留空，或写 `NA` / `N/A` / `NaN` / `null` / `none` / `-` / `ND`。
   **不要用 0 表示缺失**——0 会被当作真实测量值参与拟合。
5. **数值**：必须能解析为有限实数。不接受 `1,234`（千分位）、`12%`、`>100`
   等带修饰的写法；请在导出前清洗。
6. **行完整性**：每行列数应与表头一致，否则会报 `SHORT_ROW` / `LONG_ROW`。
7. **重复行**：同一主键（如孔位、循环数）重复时以首行为准并给出警告。

## 各模块格式

"""

_SPEC_FOOTER = """## 在软件中自查

命令行：

```bash
bioeng-qpcr schema --list                  # 列出全部格式
bioeng-qpcr schema --format dose_response  # 查看某个格式的完整说明
bioeng-qpcr preview my_data.csv            # 预览任意 CSV 并自动匹配格式
bioeng-qpcr schema --format qpcr_tidy --write-template my_template.csv
```

图形界面与网页界面中，「数据预览」页会显示与 `preview` 相同的内容，并可直接
下载对应格式的模板文件。
"""


def import_spec_markdown() -> str:
    """Render the complete import specification as Markdown.

    This is the generator for ``docs/CSV_IMPORT_SPEC.md``: the documentation
    is produced from the same registry the validators use, so it cannot
    describe a format the code does not implement.
    """
    parts = [_SPEC_PREAMBLE]
    for schema in list_schemas():
        parts.append(schema.to_markdown())
    parts.append(_SPEC_FOOTER)
    return "\n".join(parts)


def write_template_csv(
    path: str | Path,
    schema: CsvSchema | str,
    *,
    n_rows: int = 0,
    overwrite: bool = True,
) -> Path:
    """Write a template CSV for *schema*.

    Parameters
        path:
            Destination file.
        schema:
            A :class:`CsvSchema` or a registry key.
        n_rows:
            When greater than zero, repeat the schema's example rows until
            this many data rows have been written (cycling through them).
            When zero, write exactly the schema's example rows.
        overwrite:
            Whether an existing file may be replaced.

    Returns
        The path written.

    Raises
        KeyError:
            If *schema* is an unknown key.
        FileExistsError:
            If the file exists and *overwrite* is ``False``.
        ValueError:
            If the schema has no example rows, or its example rows do not match
            its own header width (a schema-authoring bug that would otherwise
            ship a malformed template).
    """
    resolved = schema if isinstance(schema, CsvSchema) else schema_for(schema)
    if not resolved.example_rows:
        raise ValueError(
            f"Schema {resolved.key!r} has no example rows to build a template from."
        )
    width_error = resolved.example_width_error()
    if width_error is not None:
        raise ValueError(
            f"Schema {resolved.key!r} cannot produce a valid template: "
            f"{width_error}."
        )
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")

    rows = list(resolved.example_rows)
    if n_rows > 0:
        rows = [rows[index % len(rows)] for index in range(n_rows)]

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=resolved.delimiter, lineterminator="\n")
    writer.writerow(resolved.header())
    for row in rows:
        writer.writerow(list(row))
    path.write_text(buffer.getvalue(), encoding=resolved.encoding)
    return path
