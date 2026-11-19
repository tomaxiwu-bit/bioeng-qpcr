"""Sequence input/output and basic sequence utilities.

This module implements dependency-free readers and writers for the sequence
formats used in Sanger / ABI sequencing workflows, together with lightweight
sequence cleaning and statistics helpers.

Formats supported
-----------------
* **FASTA** -- ``>id description`` header followed by one or more lines of
  sequence.  Multi-record files, multi-line sequences, ``;`` comment lines,
  Windows/Unix line endings, mixed case and embedded whitespace are all
  handled.  See :func:`read_fasta`.
* **FASTQ** -- four-line ``@``/sequence/``+``/quality blocks (multi-line
  sequences are tolerated).  Quality is decoded with the Phred+33 convention
  into a list of integer Phred scores; a length mismatch between sequence and
  quality raises :class:`ValueError`.  See :func:`read_fastq`.
* **Plain sequence text** -- ABI chromatogram exports often drop FASTA
  headers and emit raw sequence broken across lines.  :func:`read_plain_sequence`
  joins every non-empty, non-comment line (``>``/``#``/``;`` prefixed lines are
  ignored) into one :class:`SeqRecord` per paragraph.

:func:`read_sequence_file` auto-detects the format from the first meaningful
line and dispatches to the correct parser, raising a clear error when the
file does not match any known format.

Sequences are normalised by :func:`clean_sequence` (whitespace, digits and
``*`` removed, upper-cased) which preserves IUPAC degenerate bases for DNA/RNA
and the 20 standard amino acids for protein.  :func:`identify_sequence_type`
and :func:`sequence_statistics` provide elementary characterisation.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path

# IUPAC nucleotide + protein alphabets used for type inference and cleaning.
_NUCLEOTIDE_LETTERS: frozenset[str] = frozenset("ACGTURYSWKMBDHVN")
_PROTEIN_LETTERS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWYXBZ")
# Characters stripped from raw sequence input (whitespace, digits, stop '*').
_STRIP_PATTERN = re.compile(r"[\s\d\*]")

_DEFAULT_FASTA_WIDTH: int = 60
_DEFAULT_QUALITY_OFFSET: int = 33
_ABIF_ENTRY_SIZE: int = 28


@unique
class SequenceType(str, Enum):
    """Broad biopolymer class inferred from a sequence's residue alphabet."""

    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SeqRecord:
    """An immutable holder for one biological sequence and its metadata.

    Attributes
        id:
            Primary identifier (the FASTA/FASTQ header token before the first
            space, or a synthesized id for plain-text input).
        description:
            Free-text description taken from the remainder of the header
            (empty string when absent).
        sequence:
            The cleaned, upper-cased sequence string.
        quality:
            Per-base Phred quality scores (Phred+33 decoded) for FASTQ
            records, or ``None`` when the source format carries no quality.
        annotations:
            Arbitrary string key/value metadata (e.g. instrument, run id).
    """

    id: str
    sequence: str
    description: str = ""
    quality: list[int] | None = None
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceStatistics:
    """Summary statistics for a single sequence.

    Attributes
        length:
            Number of residues (after cleaning).
        gc_fraction:
            Fraction of G+C among defined nucleotide bases; ``0.0`` for
            protein or unknown sequences.
        composition:
            Count of each residue symbol observed, upper-cased.
        n_fraction:
            Fraction of ambiguous ``N`` (or ``X`` for protein) residues.
        sequence_type:
            Inferred :class:`SequenceType`.
    """

    length: int
    gc_fraction: float
    composition: dict[str, int]
    n_fraction: float
    sequence_type: SequenceType


def clean_sequence(sequence: str) -> str:
    """Normalise a raw sequence string for downstream analysis.

    Removes whitespace, digits and ``*`` stop symbols, and upper-cases the
    remaining residues.  IUPAC degenerate bases (``N``, ``R``, ``Y`` ...) and
    protein residues are preserved.

    Parameters
        sequence:
            Raw sequence text as emitted by a sequencer or file.

    Returns
        The cleaned, upper-cased sequence string.
    """
    return _STRIP_PATTERN.sub("", sequence).upper()


def identify_sequence_type(sequence: str) -> SequenceType:
    """Infer whether *sequence* is DNA, RNA, protein or unknown.

    The decision is purely alphabet-based: a residue set contained in the
    nucleotide alphabet is reported as DNA (or RNA when ``U`` appears without
    ``T``); a residue set contained in the protein alphabet but not the
    nucleotide alphabet is reported as protein; otherwise the type is
    :attr:`SequenceType.UNKNOWN`.

    Parameters
        sequence:
            Cleaned (or raw) sequence string.

    Returns
        The inferred :class:`SequenceType`.
    """
    letters = set(clean_sequence(sequence))
    if not letters:
        return SequenceType.UNKNOWN
    if letters <= _NUCLEOTIDE_LETTERS:
        if "U" in letters and "T" not in letters:
            return SequenceType.RNA
        return SequenceType.DNA
    if letters <= _PROTEIN_LETTERS:
        return SequenceType.PROTEIN
    return SequenceType.UNKNOWN


def sequence_statistics(sequence: str) -> SequenceStatistics:
    """Compute length, GC content, composition and ambiguity statistics.

    Parameters
        sequence:
            Cleaned (or raw) sequence string.

    Returns
        A :class:`SequenceStatistics` describing the sequence.
    """
    cleaned = clean_sequence(sequence)
    composition: dict[str, int] = {}
    for residue in cleaned:
        composition[residue] = composition.get(residue, 0) + 1

    seq_type = identify_sequence_type(cleaned)
    gc_fraction = 0.0
    if seq_type in (SequenceType.DNA, SequenceType.RNA):
        defined = sum(composition.get(base, 0) for base in "ACGTU")
        gc = composition.get("G", 0) + composition.get("C", 0)
        gc_fraction = (gc / defined) if defined else 0.0

    ambiguous = composition.get("N", 0) + composition.get("X", 0)
    n_fraction = (ambiguous / len(cleaned)) if cleaned else 0.0

    return SequenceStatistics(
        length=len(cleaned),
        gc_fraction=gc_fraction,
        composition=composition,
        n_fraction=n_fraction,
        sequence_type=seq_type,
    )


def _split_header(header: str) -> tuple[str, str]:
    """Split a FASTA/FASTQ header into ``(id, description)``.

    The id is everything before the first whitespace; the rest (stripped) is
    the description.  A bare ``>`` header yields an empty id and description.
    """
    header = header.strip()
    if not header:
        return "", ""
    parts = header.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()


def _make_record(
    header: str, seq_parts: list[str], quality: list[int] | None = None
) -> SeqRecord:
    """Build a :class:`SeqRecord` from a header and accumulated sequence lines."""
    record_id, description = _split_header(header)
    sequence = clean_sequence("".join(seq_parts))
    return SeqRecord(
        id=record_id,
        description=description,
        sequence=sequence,
        quality=quality,
    )


def read_fasta(path: str | Path) -> list[SeqRecord]:
    """Parse one or more FASTA records from *path*.

    Parameters
        path:
            Path to a FASTA file (``>``-delimited records, multi-line
            sequences allowed).

    Returns
        A list of :class:`SeqRecord` objects, one per record.

    Raises
        ValueError:
            If the file is empty, is not FASTA, or contains a record without
            a header line.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    records: list[SeqRecord] = []
    header: str | None = None
    seq_parts: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(_make_record(header, seq_parts))
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)

    if header is not None:
        records.append(_make_record(header, seq_parts))
    elif not records:
        if not any(line.startswith(">") for line in lines if line.strip()):
            raise ValueError(
                f"{path} does not look like FASTA: no '>' header found."
            )
    if not records:
        raise ValueError(f"{path} is empty or contains no FASTA records.")
    return records


def read_fastq(path: str | Path) -> list[SeqRecord]:
    """Parse FASTQ records (Phred+33) from *path*.

    Parameters
        path:
            Path to a FASTQ file (``@id`` / sequence / ``+`` / quality blocks).
            Multi-line sequences and multi-line quality are tolerated.

    Returns
        A list of :class:`SeqRecord` objects with decoded integer Phred
        quality scores.

    Raises
        ValueError:
            If a record is malformed, the ``+`` separator is missing, or the
            sequence and quality strings differ in length.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [line.rstrip("\n") for line in text.splitlines()]
    if not any(line.startswith("@") for line in lines):
        raise ValueError(f"{path} does not look like FASTQ: no '@' header found.")

    records: list[SeqRecord] = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith("@"):
            i += 1
            continue
        header = lines[i][1:].strip()
        j = i + 1
        seq_parts: list[str] = []
        while j < n and not lines[j].startswith("+"):
            seq_parts.append(lines[j])
            j += 1
        if j >= n:
            raise ValueError(
                f"Incomplete FASTQ record {header!r}: missing '+' separator."
            )
        seq = "".join(seq_parts)
        k = j + 1
        qual_parts: list[str] = []
        while k < n and len("".join(qual_parts)) < len(seq):
            qual_parts.append(lines[k])
            k += 1
        qual = "".join(qual_parts)
        if len(seq) != len(qual):
            raise ValueError(
                f"FASTQ record {header!r}: sequence length {len(seq)} != "
                f"quality length {len(qual)}."
            )
        quality = [_decode_phred(char) for char in qual]
        records.append(_make_record(header, [seq], quality=quality))
        i = k

    if not records:
        raise ValueError(f"{path} contains no complete FASTQ records.")
    return records


def _decode_phred(char: str) -> int:
    """Decode a single Phred+33 quality character to an integer score."""
    value = ord(char) - _DEFAULT_QUALITY_OFFSET
    if value < 0 or value > 93:
        raise ValueError(
            f"Phred+33 quality character {char!r} decodes to {value}, "
            "outside the valid [0, 93] range."
        )
    return value


def _quality_to_string(quality: list[int]) -> str:
    """Encode integer Phred scores back to a Phred+33 quality string."""
    chars = []
    for score in quality:
        if score < 0 or score > 93:
            raise ValueError(
                f"Phred score {score} outside the valid [0, 93] range."
            )
        chars.append(chr(score + _DEFAULT_QUALITY_OFFSET))
    return "".join(chars)


def read_plain_sequence(path: str | Path) -> list[SeqRecord]:
    """Parse ABI-style plain sequence text from *path*.

    Each non-empty line is treated as a continuation of the sequence.  Lines
    beginning with ``>``, ``#`` or ``;`` are ignored as comments.  The whole
    file is returned as a single :class:`SeqRecord` whose id is the file stem.

    Parameters
        path:
            Path to a plain-text sequence file.

    Returns
        A one-element list containing the concatenated :class:`SeqRecord`.

    Raises
        ValueError:
            If the file yields no sequence residues.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    seq_parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[0] in (">", "#", ";"):
            continue
        seq_parts.append(line)
    sequence = clean_sequence("".join(seq_parts))
    if not sequence:
        raise ValueError(f"{path} contains no sequence residues.")
    record = SeqRecord(
        id=Path(path).stem,
        description="plain sequence text",
        sequence=sequence,
    )
    return [record]


def read_ab1(path: str | Path) -> list[SeqRecord]:
    """Read the analysed base calls and Phred qualities from an ABIF file.

    Applied Biosystems ``.ab1``/``.abi`` chromatograms store their analysed
    sequence in ``PBAS2`` and qualities in ``PCON2`` (with ``*1`` as the
    legacy fallback). Only these standard ABIF directory tags are needed by
    the existing alignment and audit pipeline.
    """
    data = Path(path).read_bytes()
    if len(data) < 6 + _ABIF_ENTRY_SIZE or data[:4] != b"ABIF":
        raise ValueError(f"{path} does not look like an ABIF chromatogram.")

    version = struct.unpack_from(">H", data, 4)[0]
    root = _read_abif_entry(data, 6)
    count = root[4]
    directory_offset = root[6]
    if count <= 0 or count > 100_000:
        raise ValueError(f"{path} has an invalid ABIF directory size: {count}.")
    if directory_offset + count * _ABIF_ENTRY_SIZE > len(data):
        raise ValueError(f"{path} has a truncated ABIF directory.")

    entries: dict[tuple[bytes, int], tuple[int, bytes]] = {}
    for index in range(count):
        offset = directory_offset + index * _ABIF_ENTRY_SIZE
        entry = _read_abif_entry(data, offset)
        tag, number, element_type, _size, _count, data_size, data_offset, _handle = (
            entry
        )
        if data_size <= 4:
            value = data[offset + 20 : offset + 20 + data_size]
        else:
            end = data_offset + data_size
            if data_offset < 0 or end > len(data):
                label = tag.decode("ascii", "replace")
                raise ValueError(
                    f"{path} has a truncated ABIF value for {label}{number}."
                )
            value = data[data_offset:end]
        entries[(tag, number)] = (element_type, value)

    base_key = (b"PBAS", 2) if (b"PBAS", 2) in entries else (b"PBAS", 1)
    if base_key not in entries:
        raise ValueError(f"{path} contains no analysed PBAS base calls.")
    try:
        sequence = clean_sequence(
            entries[base_key][1].decode("ascii").replace("\x00", "")
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} contains non-ASCII PBAS base calls.") from exc
    if not sequence:
        raise ValueError(f"{path} contains an empty PBAS base call sequence.")

    quality_key = (b"PCON", base_key[1])
    quality = list(entries[quality_key][1]) if quality_key in entries else None
    if quality is not None and len(quality) != len(sequence):
        raise ValueError(
            f"{path} has {len(sequence)} base calls but {len(quality)} quality values."
        )
    return [
        SeqRecord(
            id=Path(path).stem,
            description="ABIF chromatogram",
            sequence=sequence,
            quality=quality,
            annotations={
                "source_format": "ab1",
                "abif_version": f"{version / 100:.2f}",
                "basecall_tag": f"PBAS{base_key[1]}",
            },
        )
    ]


def _read_abif_entry(
    data: bytes, offset: int
) -> tuple[bytes, int, int, int, int, int, int, int]:
    """Unpack one 28-byte big-endian ABIF directory entry."""
    if offset < 0 or offset + _ABIF_ENTRY_SIZE > len(data):
        raise ValueError("Truncated ABIF directory entry.")
    return struct.unpack_from(">4sIHHIIII", data, offset)


def detect_format(path: str | Path) -> str:
    """Return ``"ab1"``, ``"fasta"``, ``"fastq"`` or ``"plain"`` for *path*.

    Parameters
        path:
            Path to a sequence file.

    Returns
        ``"ab1"``, ``"fasta"``, ``"fastq"`` or ``"plain"``.

    Raises
        ValueError:
            If the file is empty or matches no known format.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"ABIF"):
        return "ab1"
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is neither ABIF nor UTF-8 sequence text.") from exc
    first = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped:
            first = stripped
            break
    if not first:
        raise ValueError(f"{path} is empty.")
    if first.startswith(">"):
        return "fasta"
    if first.startswith("@"):
        has_plus = any(
            line.strip().startswith("+") for line in text.splitlines()
        )
        if has_plus:
            return "fastq"
        return "plain"
    return "plain"


def read_sequence_file(path: str | Path) -> list[SeqRecord]:
    """Auto-detect the format of *path* and parse it accordingly.

    Parameters
        path:
            Path to an AB1/ABI, FASTA, FASTQ or plain-text sequence file.

    Returns
        A list of parsed :class:`SeqRecord` objects.

    Raises
        ValueError:
            If the format cannot be determined or the file is invalid.
    """
    fmt = detect_format(path)
    if fmt == "ab1":
        return read_ab1(path)
    if fmt == "fasta":
        return read_fasta(path)
    if fmt == "fastq":
        return read_fastq(path)
    return read_plain_sequence(path)


def write_fasta(
    records: list[SeqRecord] | SeqRecord,
    path: str | Path,
    *,
    width: int = _DEFAULT_FASTA_WIDTH,
) -> None:
    """Write one or more :class:`SeqRecord` objects to *path* as FASTA.

    Parameters
        records:
            A single record or a list of records to serialise.
        path:
            Destination file path.
        width:
            Maximum number of residues per wrapped sequence line.

    Raises
        ValueError:
            If *width* is not positive.
    """
    if width <= 0:
        raise ValueError(f"FASTA line width must be positive, got {width}.")
    if isinstance(records, SeqRecord):
        records = [records]

    parts: list[str] = []
    for record in records:
        header = record.id
        if record.description:
            header = f"{record.id} {record.description}"
        parts.append(f">{header}")
        sequence = record.sequence
        for start in range(0, len(sequence), width):
            parts.append(sequence[start : start + width])
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_fastq(
    records: list[SeqRecord] | SeqRecord,
    path: str | Path,
) -> None:
    """Write one or more :class:`SeqRecord` objects to *path* as FASTQ.

    Records without quality are rejected because FASTQ requires it.

    Parameters
        records:
            A single record or a list of records to serialise.
        path:
            Destination file path.

    Raises
        ValueError:
            If any record lacks quality scores or its quality length does not
            match its sequence length.
    """
    if isinstance(records, SeqRecord):
        records = [records]

    parts: list[str] = []
    for record in records:
        if record.quality is None:
            raise ValueError(
                f"Record {record.id!r} has no quality scores; cannot write FASTQ."
            )
        if len(record.quality) != len(record.sequence):
            raise ValueError(
                f"Record {record.id!r}: sequence/quality length mismatch."
            )
        header = record.id
        if record.description:
            header = f"{record.id} {record.description}"
        parts.append(f"@{header}")
        parts.append(record.sequence)
        parts.append("+")
        parts.append(_quality_to_string(record.quality))
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")
