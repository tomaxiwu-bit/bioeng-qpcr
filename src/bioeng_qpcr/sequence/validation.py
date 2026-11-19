"""Auditable quality-validation rules for sequencing comparisons.

This module implements a small, self-contained validation framework for the
sequencing-comparison workflow.  Each rule carries a stable identifier of the
form ``SEQ-00x``, a boolean ``passed`` flag, a human-readable ``message`` and a
:class:`Severity` level, mirroring the auditability philosophy of the rest of
bioeng-qpcr without depending on the qPCR MIQE rule engine.

Rules are organised by concern:

============  ===========  ==================================================
ID            Severity     Check
============  ===========  ==================================================
SEQ-001       ERROR/WARN   Alignment identity meets a caller-supplied threshold.
SEQ-002       ERROR        Read length is not shorter than a minimum.
SEQ-003       WARN         Query covers a sufficient fraction of the reference.
SEQ-004       WARN         Low-quality base fraction is below a threshold.
SEQ-005       ERROR/WARN   No frameshift (indel length a multiple of 3) for
                           coding sequences.
SEQ-006       WARN         Fraction of ambiguous ``N`` bases is acceptable.
SEQ-007       WARN         GC content is within an expected range.
SEQ-008       WARN         Primers were fully excised (none remain at the ends).
============  ===========  ==================================================

:func:`validate_sequencing_result` runs the battery against a comparison
outcome and returns every :class:`SequenceValidationResult`; individual rules
are also exposed so tests can assert failures in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique

from bioeng_qpcr.sequence.align import AlignmentResult
from bioeng_qpcr.sequence.compare import PrimerTrim, Variant
from bioeng_qpcr.sequence.fasta import SeqRecord, identify_sequence_type


@unique
class Severity(str, Enum):
    """Severity level of a validation rule outcome."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class SequenceValidationResult:
    """Outcome of a single sequencing-validation rule.

    Attributes
        rule_id:
            Stable rule identifier (e.g. ``"SEQ-001"``).
        passed:
            Whether the sequence under test satisfies the rule.
        message:
            Human-readable explanation of the outcome.
        severity:
            :class:`Severity` of the rule (only meaningful when it fails).
        evidence:
            Structured supporting values (thresholds, observed counts, ...).
    """

    rule_id: str
    passed: bool
    message: str
    severity: Severity
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate outcome of a validation battery.

    Attributes
        results:
            All individual :class:`SequenceValidationResult` objects.
        passed:
            ``True`` only when every rule passed.
        n_errors, n_warnings:
            Counts of failing ERROR / WARNING rules.
    """

    results: list[SequenceValidationResult]
    passed: bool
    n_errors: int
    n_warnings: int


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def check_identity(
    alignment: AlignmentResult,
    *,
    threshold: float = 0.9,
    error_below: float | None = None,
) -> SequenceValidationResult:
    """SEQ-001: alignment identity must reach a caller-supplied threshold.

    Parameters
        alignment:
            The alignment to assess.
        threshold:
            Minimum acceptable identity (default 0.9).
        error_below:
            When the identity falls below this stricter value the rule is an
            ERROR; otherwise a WARNING.  Defaults to *threshold* (ERROR).

    Returns
        A :class:`SequenceValidationResult`.
    """
    identity = alignment.identity
    hard = threshold if error_below is None else error_below
    passed = identity >= threshold
    severity = Severity.ERROR if (not passed and identity < hard) else Severity.WARNING
    return SequenceValidationResult(
        rule_id="SEQ-001",
        passed=passed,
        message=(
            f"Alignment identity {identity:.1%} "
            f"{'meets' if passed else 'below'} threshold {threshold:.1%}."
        ),
        severity=severity,
        evidence={"identity": identity, "threshold": threshold},
    )


def check_read_length(
    query: str | SeqRecord,
    *,
    min_length: int = 50,
) -> SequenceValidationResult:
    """SEQ-002: the read must not be shorter than *min_length*.

    Parameters
        query:
            Read sequence or record.
        min_length:
            Minimum acceptable length (default 50).

    Returns
        A :class:`SequenceValidationResult`.
    """
    length = len(query.sequence) if isinstance(query, SeqRecord) else len(query)
    passed = length >= min_length
    return SequenceValidationResult(
        rule_id="SEQ-002",
        passed=passed,
        message=(
            f"Read length {length} "
            f"{'meets' if passed else 'below'} minimum {min_length}."
        ),
        severity=Severity.ERROR,
        evidence={"length": length, "min_length": min_length},
    )


def check_coverage(
    alignment: AlignmentResult,
    reference: str | SeqRecord,
    *,
    min_fraction: float = 0.8,
) -> SequenceValidationResult:
    """SEQ-003: the read should cover a sufficient fraction of the reference.

    Parameters
        alignment:
            The glocal alignment of read to reference.
        reference:
            The reference sequence.
        min_fraction:
            Minimum fraction of the reference length that must be covered.

    Returns
        A :class:`SequenceValidationResult`.
    """
    if isinstance(reference, SeqRecord):
        ref_len = len(reference.sequence)
    else:
        ref_len = len(reference)
    covered = alignment.ref_end - alignment.ref_start
    fraction = (covered / ref_len) if ref_len else 0.0
    passed = fraction >= min_fraction
    return SequenceValidationResult(
        rule_id="SEQ-003",
        passed=passed,
        message=(
            f"Read covers {fraction:.1%} of the reference "
            f"({covered}/{ref_len}); minimum {min_fraction:.1%}."
        ),
        severity=Severity.WARNING,
        evidence={
            "fraction": fraction,
            "covered": covered,
            "reference_length": ref_len,
        },
    )


def check_low_quality_fraction(
    query: str | SeqRecord,
    quality: list[int] | None = None,
    *,
    max_fraction: float = 0.2,
    min_q: int = 20,
) -> SequenceValidationResult:
    """SEQ-004: the fraction of low-quality bases must stay below a threshold.

    Parameters
        query:
            Read sequence or record.
        quality:
            Per-base Phred scores; if ``None`` and *query* is a record with
            quality, those are used.
        max_fraction:
            Maximum tolerated fraction of bases with ``Q < min_q``.
        min_q:
            Quality threshold defining a "low-quality" base.

    Returns
        A :class:`SequenceValidationResult`.
    """
    if quality is None and isinstance(query, SeqRecord):
        quality = query.quality
    if quality is None:
        return SequenceValidationResult(
            rule_id="SEQ-004",
            passed=True,
            message="No quality data available; low-quality check skipped.",
            severity=Severity.INFO,
            evidence={"max_fraction": max_fraction},
        )
    low = sum(1 for q in quality if q < min_q)
    fraction = (low / len(quality)) if quality else 0.0
    passed = fraction <= max_fraction
    return SequenceValidationResult(
        rule_id="SEQ-004",
        passed=passed,
        message=(
            f"Low-quality base fraction {fraction:.1%} "
            f"{'within' if passed else 'exceeds'} limit {max_fraction:.1%}."
        ),
        severity=Severity.WARNING,
        evidence={"fraction": fraction, "low_bases": low, "min_q": min_q},
    )


def check_frameshift(
    variants: list[Variant],
    *,
    coding: bool = True,
) -> SequenceValidationResult:
    """SEQ-005: indels should be a multiple of 3 for coding sequences.

    Parameters
        variants:
            Detected variants (insertions / deletions).
        coding:
            If ``False`` the rule is informational only (frameshifts are
            irrelevant for non-coding sequences).

    Returns
        A :class:`SequenceValidationResult`.
    """
    if not coding:
        return SequenceValidationResult(
            rule_id="SEQ-005",
            passed=True,
            message="Non-coding sequence; frameshift check not applicable.",
            severity=Severity.INFO,
            evidence={},
        )
    frameshifts = [
        v for v in variants
        if v.type.value in ("insertion", "deletion") and v.length % 3 != 0
    ]
    passed = not frameshifts
    severity = Severity.ERROR if frameshifts else Severity.INFO
    if passed:
        message = "No frameshifting indels detected."
    else:
        positions = ", ".join(str(v.ref_position) for v in frameshifts)
        message = (
            f"{len(frameshifts)} frameshifting indel(s) at reference "
            f"position(s) {positions} (length not a multiple of 3)."
        )
    return SequenceValidationResult(
        rule_id="SEQ-005",
        passed=passed,
        message=message,
        severity=severity,
        evidence={"frameshift_positions": [v.ref_position for v in frameshifts]},
    )


def check_ambiguous_bases(
    query: str | SeqRecord,
    *,
    max_fraction: float = 0.05,
) -> SequenceValidationResult:
    """SEQ-006: the ambiguous ``N`` fraction must stay below a threshold.

    Parameters
        query:
            Read sequence or record.
        max_fraction:
            Maximum tolerated ``N`` (or ``X`` for protein) fraction.

    Returns
        A :class:`SequenceValidationResult`.
    """
    seq = query.sequence if isinstance(query, SeqRecord) else query
    n_count = sum(1 for c in seq.upper() if c in ("N", "X"))
    fraction = (n_count / len(seq)) if seq else 0.0
    passed = fraction <= max_fraction
    return SequenceValidationResult(
        rule_id="SEQ-006",
        passed=passed,
        message=(
            f"Ambiguous base fraction {fraction:.1%} "
            f"{'within' if passed else 'exceeds'} limit {max_fraction:.1%}."
        ),
        severity=Severity.WARNING,
        evidence={"fraction": fraction, "n_bases": n_count},
    )


def check_gc_content(
    query: str | SeqRecord,
    *,
    min_gc: float = 0.2,
    max_gc: float = 0.8,
) -> SequenceValidationResult:
    """SEQ-007: GC content should lie within an expected range.

    Parameters
        query:
            Read sequence or record.
        min_gc, max_gc:
            Acceptable GC fraction bounds.

    Returns
        A :class:`SequenceValidationResult`.
    """
    seq = query.sequence if isinstance(query, SeqRecord) else query
    gc = _gc_fraction(seq)
    passed = min_gc <= gc <= max_gc
    return SequenceValidationResult(
        rule_id="SEQ-007",
        passed=passed,
        message=(
            f"GC content {gc:.1%} is "
            f"{'within' if passed else 'outside'} expected range "
            f"[{min_gc:.1%}, {max_gc:.1%}]."
        ),
        severity=Severity.WARNING,
        evidence={"gc": gc, "min_gc": min_gc, "max_gc": max_gc},
    )


def check_primers_excised(
    trim: PrimerTrim,
    primer_5: str | None = None,
    primer_3: str | None = None,
) -> SequenceValidationResult:
    """SEQ-008: primers should have been fully excised (none left at ends).

    Parameters
        trim:
            Outcome of :func:`bioeng_qpcr.sequence.compare.trim_primers`.
        primer_5, primer_3:
            The primers that were searched for.

    Returns
        A :class:`SequenceValidationResult`.
    """
    failures: list[str] = []
    if primer_5 is not None and not trim.found_5:
        failures.append("5'")
    if primer_3 is not None and not trim.found_3:
        failures.append("3'")
    passed = not failures
    message = (
        "All supplied primers were excised."
        if passed
        else f"Primer(s) not fully excised at end(s): {', '.join(failures)}."
    )
    return SequenceValidationResult(
        rule_id="SEQ-008",
        passed=passed,
        message=message,
        severity=Severity.WARNING,
        evidence={"unexcised_ends": failures},
    )


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def validate_sequencing_result(
    alignment: AlignmentResult,
    query: str | SeqRecord,
    reference: str | SeqRecord,
    variants: list[Variant] | None = None,
    *,
    quality: list[int] | None = None,
    coding: bool = False,
    primer_trim: PrimerTrim | None = None,
    primer_5: str | None = None,
    primer_3: str | None = None,
    threshold_identity: float = 0.9,
    min_length: int = 50,
    min_coverage_fraction: float = 0.8,
    max_low_quality_fraction: float = 0.2,
    max_ambiguous_fraction: float = 0.05,
    min_gc: float = 0.2,
    max_gc: float = 0.8,
    min_q: int = 20,
) -> ValidationReport:
    """Run the full SEQ-00x validation battery on a comparison outcome.

    Parameters
        alignment:
            The read/reference alignment.
        query, reference:
            The read and reference sequences.
        variants:
            Detected variants (used by the frameshift rule).
        quality:
            Optional per-base Phred scores for the low-quality rule.
        coding:
            Whether the sequence is coding (enables the frameshift rule).
        primer_trim, primer_5, primer_3:
            Primer-trim outcome and primers for the excision rule.
        threshold_identity, min_length, min_coverage_fraction, ...:
            Tunable thresholds for the individual rules.

    Returns
        A :class:`ValidationReport` aggregating all rule outcomes.
    """
    results: list[SequenceValidationResult] = [
        check_identity(alignment, threshold=threshold_identity),
        check_read_length(query, min_length=min_length),
        check_coverage(alignment, reference, min_fraction=min_coverage_fraction),
        check_low_quality_fraction(
            query, quality, max_fraction=max_low_quality_fraction, min_q=min_q
        ),
        check_ambiguous_bases(query, max_fraction=max_ambiguous_fraction),
        check_gc_content(query, min_gc=min_gc, max_gc=max_gc),
    ]
    if variants is not None:
        results.append(check_frameshift(variants, coding=coding))
    if primer_trim is not None:
        results.append(
            check_primers_excised(primer_trim, primer_5=primer_5, primer_3=primer_3)
        )

    n_errors = sum(
        1 for r in results
        if not r.passed and r.severity is Severity.ERROR
    )
    n_warnings = sum(
        1 for r in results
        if not r.passed and r.severity is Severity.WARNING
    )
    return ValidationReport(
        results=results,
        passed=n_errors == 0,
        n_errors=n_errors,
        n_warnings=n_warnings,
    )


#: Stable catalogue of the built-in SEQ-00x rule functions.
SEQ_RULES: list[str] = [
    "SEQ-001",
    "SEQ-002",
    "SEQ-003",
    "SEQ-004",
    "SEQ-005",
    "SEQ-006",
    "SEQ-007",
    "SEQ-008",
]


def _gc_fraction(sequence: str) -> float:
    """Return the GC fraction of a nucleotide sequence (0.0 for protein)."""
    cleaned = sequence.upper()
    if not cleaned:
        return 0.0
    seq_type = identify_sequence_type(cleaned)
    if seq_type in ("dna", "rna"):
        defined = sum(cleaned.count(b) for b in "ACGTU")
        if not defined:
            return 0.0
        return (cleaned.count("G") + cleaned.count("C")) / defined
    return 0.0
