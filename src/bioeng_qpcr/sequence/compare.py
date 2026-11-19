"""Sequencing-result comparison and downstream analysis.

This module turns raw Sanger / ABI reads into interpretable variation calls.
It builds on :mod:`bioeng_qpcr.sequence.fasta` (I/O) and
:mod:`bioeng_qpcr.sequence.align` (pairwise alignment) and adds the workflow
specific to sequencing:

* :func:`compare_to_reference` -- map a read onto a reference (glocal / semi-
  global alignment, because a read is usually only a sub-region of the
  template).
* :func:`trim_primers` -- locate and excise 5'/3' primers allowing a few
  mismatches (for Sanger primers and vector sequences).
* :func:`trim_low_quality` -- sliding-window quality trimming of FASTQ reads.
* :func:`detect_variants` -- walk an alignment column-by-column and emit typed
  :class:`Variant` calls (SNP / insertion / deletion / MNP) with optional
  per-position depth and quality.
* :func:`build_consensus` -- pile up reads over a reference into a consensus
  sequence with simple-majority or threshold voting and IUPAC ambiguity codes.
* :func:`multiple_sequence_alignment` -- a pragmatic progressive MSA (longest
  sequence as anchor, remaining reads added in k-mer-similarity order, gaps
  propagated to all rows) producing equal-length rows.
* :func:`summarize_alignment` -- per-column conservation, Shannon entropy,
  overall and pairwise identity, and variable-site detection.
* :func:`summarize_sequencing_quality` -- read-length, quality, GC and
  low-quality-read statistics across a read set.

All algorithms are implemented in pure Python/numpy with no external
dependencies beyond numpy and scipy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum, unique

import numpy as np

from bioeng_qpcr.sequence.align import (
    AlignmentResult,
    SubstitutionMatrix,
    align_glocal,
    dna_matrix,
    kmer_containment,
)
from bioeng_qpcr.sequence.fasta import (
    SeqRecord,
    SequenceType,
    clean_sequence,
    identify_sequence_type,
)


def _as_str(value: str | SeqRecord) -> str:
    """Return the sequence string of *value* (accepting str or SeqRecord)."""
    if isinstance(value, SeqRecord):
        return value.sequence
    return clean_sequence(value)


_DNA_COMPLEMENT = str.maketrans("ACGTRYMKBDHVN", "TGCAYRKMVHDBN")
_RNA_COMPLEMENT = str.maketrans("ACGURYMKBDHVN", "UGCAYRKMVHDBN")


def reverse_complement(sequence: str | SeqRecord) -> str:
    """Return the reverse complement of a DNA or RNA sequence."""
    cleaned = _as_str(sequence)
    sequence_type = identify_sequence_type(cleaned)
    if sequence_type is SequenceType.DNA:
        return cleaned.translate(_DNA_COMPLEMENT)[::-1]
    if sequence_type is SequenceType.RNA:
        return cleaned.translate(_RNA_COMPLEMENT)[::-1]
    raise ValueError("Reverse complement is only defined for DNA or RNA.")


# ---------------------------------------------------------------------------
# Variant calling
# ---------------------------------------------------------------------------


@unique
class VariantType(str, Enum):
    """Type of a called sequence variant."""

    SNP = "snp"
    INSERTION = "insertion"
    DELETION = "deletion"
    MNP = "mnp"


@dataclass(frozen=True)
class Variant:
    """A single called variant between a read and its reference.

    Attributes
        type:
            One of :class:`VariantType`.
        ref_position:
            0-based reference coordinate of the variant.  For an insertion it
            is the reference base *preceding* the inserted residues; for a
            deletion it is the first deleted reference base.
        ref_allele:
            Reference allele (``""`` for insertions).
        alt_allele:
            Sequenced allele (``""`` for deletions).
        length:
            Number of affected reference bases (1 for SNP).
        depth:
            Coverage depth at the variant position when a pileup coverage map
            is supplied, else ``None``.
        quality:
            Minimum Phred quality of the read bases supporting the variant,
            or ``None`` when unavailable (e.g. a pure deletion).
        in_low_complexity:
            Whether the variant lies in a homopolymer / low-complexity region.
    """

    type: VariantType
    ref_position: int
    ref_allele: str
    alt_allele: str
    length: int
    depth: int | None = None
    quality: int | None = None
    in_low_complexity: bool = False


def compare_to_reference(
    query: str | SeqRecord,
    reference: str | SeqRecord,
    matrix: SubstitutionMatrix | None = None,
    *,
    gap_open: int = 5,
    gap_extend: int = 2,
    auto_orient: bool = True,
) -> AlignmentResult:
    """Align a sequencing read to a reference (semi-global glocal alignment).

    Parameters
        query:
            The read (usually shorter than the reference).
        reference:
            The template sequence.
        matrix:
            Substitution matrix (defaults to DNA match/mismatch).
        gap_open, gap_extend:
            Affine gap penalties.
        auto_orient:
            For nucleotide reads, also align the reverse complement and keep
            it when its score is higher.

    Returns
        The glocal :class:`AlignmentResult`.
    """
    if matrix is None:
        matrix = dna_matrix()
    query_str = _as_str(query)
    reference_str = _as_str(reference)
    forward = align_glocal(
        query_str, reference_str, matrix, gap_open=gap_open,
        gap_extend=gap_extend,
    )
    if not auto_orient or identify_sequence_type(query_str) not in {
        SequenceType.DNA,
        SequenceType.RNA,
    }:
        return forward
    reverse = align_glocal(
        reverse_complement(query_str),
        reference_str,
        matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return (
        replace(reverse, query_orientation="reverse_complement")
        if reverse.score > forward.score
        else forward
    )


def _hamming(window: str, pattern: str, max_mismatch: int) -> int:
    """Count mismatches between *pattern* and *window* (N treated as match)."""
    mismatches = 0
    for a, b in zip(window, pattern, strict=True):
        if a == "N" or b == "N":
            continue
        if a != b:
            mismatches += 1
            if mismatches > max_mismatch:
                return mismatches
    return mismatches


def _locate_primer(
    seq: str, primer: str, max_mismatch: int, end: str
) -> tuple[int, int] | None:
    """Find the ``(start, end)`` span of *primer* near a sequence end.

    Returns ``None`` when no placement within *max_mismatch* is found.
    """
    plen = len(primer)
    if plen == 0 or plen > len(seq):
        return None
    if end == "5":
        starts = range(0, min(max_mismatch + 1, len(seq) - plen + 1))
    else:
        lo = max(0, len(seq) - plen - max_mismatch)
        hi = len(seq) - plen + 1
        starts = range(lo, hi)
    best: tuple[int, int] | None = None
    best_mis = max_mismatch + 1
    for start in starts:
        end_pos = start + plen
        if end_pos > len(seq):
            continue
        mismatches = _hamming(seq[start:end_pos], primer, max_mismatch)
        if mismatches < best_mis:
            best_mis = mismatches
            best = (start, end_pos)
    return best


@dataclass(frozen=True)
class PrimerTrim:
    """Outcome of primer trimming.

    Attributes
        trimmed:
            Sequence with both primers excised.
        n_trimmed_5, n_trimmed_3:
            Number of residues removed from the 5' and 3' ends.
        found_5, found_3:
            Whether each primer was located and removed.
        kept_start, kept_end:
            0-based half-open interval of *trimmed* within the input.
    """

    trimmed: str
    n_trimmed_5: int
    n_trimmed_3: int
    found_5: bool
    found_3: bool
    kept_start: int
    kept_end: int


def trim_primers(
    sequence: str | SeqRecord,
    primer_5: str | None = None,
    primer_3: str | None = None,
    *,
    max_mismatch: int = 2,
) -> PrimerTrim:
    """Locate and excise 5'/3' primers (allowing up to *max_mismatch*).

    Parameters
        sequence:
            Read sequence (str or :class:`SeqRecord`).
        primer_5, primer_3:
            Forward / reverse primer sequences; either may be ``None``.
        max_mismatch:
            Maximum allowed mismatches per primer.

    Returns
        A :class:`PrimerTrim` with the trimmed sequence and counts.

    Raises
        ValueError:
            If both primers are ``None`` or *max_mismatch* is negative.
    """
    if primer_5 is None and primer_3 is None:
        raise ValueError("At least one primer must be supplied.")
    if max_mismatch < 0:
        raise ValueError(f"max_mismatch must be >= 0, got {max_mismatch}.")
    seq = _as_str(sequence)
    start = 0
    end = len(seq)
    found_5 = False
    found_3 = False

    if primer_5 is not None:
        loc = _locate_primer(seq, clean_sequence(primer_5), max_mismatch, "5")
        if loc is not None:
            start = loc[1]
            found_5 = True
    if primer_3 is not None:
        loc = _locate_primer(seq, clean_sequence(primer_3), max_mismatch, "3")
        if loc is not None:
            end = loc[0]
            found_3 = True

    return PrimerTrim(
        trimmed=seq[start:end],
        n_trimmed_5=start,
        n_trimmed_3=len(seq) - end,
        found_5=found_5,
        found_3=found_3,
        kept_start=start,
        kept_end=end,
    )


@dataclass(frozen=True)
class QualityTrim:
    """Outcome of low-quality trimming.

    Attributes
        trimmed:
            Retained high-quality subsequence.
        kept_start, kept_end:
            0-based half-open interval of *trimmed* within the input.
        n_trimmed_5, n_trimmed_3:
            Residues removed from each end.
        low_quality_fraction:
            Fraction of the input sequence that fell outside the kept region.
    """

    trimmed: str
    kept_start: int
    kept_end: int
    n_trimmed_5: int
    n_trimmed_3: int
    low_quality_fraction: float


def trim_low_quality(
    sequence: str | SeqRecord,
    quality: list[int] | None = None,
    *,
    window: int = 10,
    min_q: int = 20,
) -> QualityTrim:
    """Trim low-quality residues from a read via a sliding window.

    A residue is kept only if it participates in at least one window of size
    *window* whose mean Phred quality is ``>= min_q``.  The longest contiguous
    run of kept residues is returned, so both ends and interior low-quality
    stretches are removed.

    Parameters
        sequence:
            Read sequence (str or :class:`SeqRecord`).
        quality:
            Per-base Phred scores; if ``None`` and *sequence* is a
            :class:`SeqRecord` with quality, those are used.
        window:
            Sliding-window size.
        min_q:
            Minimum mean window quality to retain a residue.

    Returns
        A :class:`QualityTrim`.

    Raises
        ValueError:
            If *quality* is missing or has the wrong length, or *window* < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    seq = _as_str(sequence)
    if quality is None and isinstance(sequence, SeqRecord):
        quality = sequence.quality
    if quality is None:
        raise ValueError("Phred quality scores are required for low-quality trimming.")
    if len(quality) != len(seq):
        raise ValueError(
            f"quality length {len(quality)} != sequence length {len(seq)}."
        )

    length = len(seq)
    kept = [False] * length
    if length >= window:
        q = np.asarray(quality, dtype=float)
        cumsum = np.concatenate([[0.0], np.cumsum(q)])
        for p in range(length):
            w_start = max(0, p - window + 1)
            w_end = min(length, p + window)
            span = w_end - w_start
            mean = (cumsum[w_end] - cumsum[w_start]) / span
            if mean >= min_q:
                kept[p] = True
    else:
        kept = [q >= min_q for q in quality]

    # Longest contiguous run of kept residues.
    best_run = (0, 0)
    best_len = 0
    run_start = None
    for p, is_kept in enumerate(kept):
        if is_kept:
            if run_start is None:
                run_start = p
        else:
            if run_start is not None:
                run_len = p - run_start
                if run_len > best_len:
                    best_len = run_len
                    best_run = (run_start, p)
                run_start = None
    if run_start is not None:
        run_len = length - run_start
        if run_len > best_len:
            best_run = (run_start, length)

    start, end = best_run
    n_low = length - (end - start)
    return QualityTrim(
        trimmed=seq[start:end],
        kept_start=start,
        kept_end=end,
        n_trimmed_5=start,
        n_trimmed_3=length - end,
        low_quality_fraction=(n_low / length) if length else 0.0,
    )


def _has_homopolymer(context: str, window: int) -> bool:
    """Return whether *context* contains a run of *window* identical residues."""
    run = 1
    for i in range(1, len(context)):
        if context[i] == context[i - 1] and context[i] != "-":
            run += 1
            if run >= window:
                return True
        else:
            run = 1
    return False


def detect_variants(
    alignment: AlignmentResult,
    reference: str | SeqRecord | None = None,
    *,
    query_quality: list[int] | None = None,
    coverage: dict[int, int] | None = None,
    low_complexity_window: int = 5,
) -> list[Variant]:
    """Call variants by walking an alignment column-by-column.

    Parameters
        alignment:
            An :class:`AlignmentResult` (typically from :func:`compare_to_reference`).
        reference:
            Original reference string; used for low-complexity context.  When
            omitted, ``in_low_complexity`` is reported as ``False``.
        query_quality:
            Per-base Phred scores of the original (unaligned) query, used to
            report the minimum supporting quality of each variant.
        coverage:
            Optional mapping from reference position to read depth; reported
            as the variant's ``depth``.
        low_complexity_window:
            Homopolymer run length that flags a variant as low complexity.

    Returns
        A list of :class:`Variant` objects in reference-coordinate order.
    """
    aq = alignment.aligned_query
    ar = alignment.aligned_reference
    length = len(aq)
    ref_str = _as_str(reference) if reference is not None else None
    if (
        query_quality is not None
        and alignment.query_orientation == "reverse_complement"
    ):
        query_quality = list(reversed(query_quality))

    qidx = alignment.query_start
    ridx = alignment.ref_start
    variants: list[Variant] = []
    i = 0
    while i < length:
        cq = aq[i]
        cr = ar[i]
        if cq == "-" and cr != "-":
            j = i
            refb = ""
            while j < length and aq[j] == "-" and ar[j] != "-":
                refb += ar[j]
                j += 1
            variants.append(_make_variant(
                VariantType.DELETION, ridx, refb, "", qidx, query_quality,
                coverage, ref_str, low_complexity_window,
            ))
            ridx += len(refb)
            i = j
        elif cr == "-" and cq != "-":
            j = i
            altb = ""
            while j < length and ar[j] == "-" and aq[j] != "-":
                altb += aq[j]
                j += 1
            variants.append(_make_variant(
                VariantType.INSERTION, ridx, "", altb, qidx, query_quality,
                coverage, ref_str, low_complexity_window,
            ))
            qidx += len(altb)
            i = j
        elif cq == cr:
            qidx += 1
            ridx += 1
            i += 1
        else:
            j = i
            refb = ""
            altb = ""
            while (
                j < length
                and aq[j] != "-"
                and ar[j] != "-"
                and aq[j] != ar[j]
            ):
                refb += ar[j]
                altb += aq[j]
                j += 1
            vtype = VariantType.MNP if len(refb) > 1 else VariantType.SNP
            variants.append(_make_variant(
                vtype, ridx, refb, altb, qidx, query_quality,
                coverage, ref_str, low_complexity_window,
            ))
            qidx += len(refb)
            ridx += len(refb)
            i = j
    return variants


def _make_variant(
    vtype: VariantType,
    ref_position: int,
    ref_allele: str,
    alt_allele: str,
    q_start: int,
    query_quality: list[int] | None,
    coverage: dict[int, int] | None,
    ref_str: str | None,
    window: int,
    depth: int | None = None,
) -> Variant:
    """Assemble a :class:`Variant`, computing quality/depth/low-complexity."""
    quality: int | None = None
    if query_quality is not None and alt_allele:
        seg = query_quality[q_start : q_start + len(alt_allele)]
        if seg:
            quality = int(min(seg))
    depth_val = None
    if coverage is not None:
        depth_val = coverage.get(ref_position)
    in_low = False
    if ref_str is not None:
        ctx = ref_str[
            max(0, ref_position - window) : ref_position + len(ref_allele) + window
        ]
        in_low = _has_homopolymer(ctx, window)
    return Variant(
        type=vtype,
        ref_position=ref_position,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        length=max(len(ref_allele), len(alt_allele), 1),
        depth=depth_val,
        quality=quality,
        in_low_complexity=in_low,
    )


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


_IUPAC_CODE: dict[frozenset[str], str] = {
    frozenset("A"): "A",
    frozenset("C"): "C",
    frozenset("G"): "G",
    frozenset("T"): "T",
    frozenset("AG"): "R",
    frozenset("CT"): "Y",
    frozenset("GC"): "S",
    frozenset("AT"): "W",
    frozenset("GT"): "K",
    frozenset("AC"): "M",
    frozenset("CGT"): "B",
    frozenset("AGT"): "D",
    frozenset("ACT"): "H",
    frozenset("ACG"): "V",
    frozenset("ACGT"): "N",
}


def _iupac_code(bases: set[str]) -> str:
    """Return the IUPAC degenerate code for a set of DNA bases."""
    code = _IUPAC_CODE.get(frozenset(bases))
    return code if code is not None else "N"


@dataclass(frozen=True)
class ConsensusResult:
    """Consensus sequence built by piling reads onto a reference.

    Attributes
        consensus:
            The consensus sequence (IUPAC degenerate at ambiguous sites).
        depth:
            Per-reference-position read depth (including the reference prior).
        method:
            Voting method used (``"simple_majority"`` or ``"threshold"``).
        threshold:
            Fraction threshold used by the ``threshold`` method.
        n_ambiguous:
            Number of reference positions that resolved to an ambiguous code.
        reference:
            The reference sequence the reads were piled onto.
    """

    consensus: str
    depth: list[int]
    method: str
    threshold: float
    n_ambiguous: int
    reference: str


def build_consensus(
    reads: list[str | SeqRecord],
    reference: str | SeqRecord,
    *,
    method: str = "simple_majority",
    threshold: float = 0.7,
    matrix: SubstitutionMatrix | None = None,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> ConsensusResult:
    """Build a consensus sequence from reads piled onto a reference.

    Each read is glocally aligned to the reference and its residues are piled
    onto the corresponding reference position.  The reference itself counts
    as one prior observation, so uncovered positions default to the reference.
    Voting is either simple majority (plurality) or a fraction threshold;
    below-threshold positions emit an IUPAC degenerate code (or ``N`` for
    protein).

    Parameters
        reads:
            Aligned or raw read sequences.
        reference:
            Template sequence.
        method:
            ``"simple_majority"`` or ``"threshold"``.
        threshold:
            Minimum fraction for ``"threshold"`` voting.
        matrix, gap_open, gap_extend:
            Alignment parameters for the per-read glocal mapping.

    Returns
        A :class:`ConsensusResult`.

    Raises
        ValueError:
            If *method* is unknown or *reads* is empty.
    """
    if method not in ("simple_majority", "threshold"):
        raise ValueError(
            f"Unknown consensus method {method!r}; expected "
            "'simple_majority' or 'threshold'."
        )
    if not reads:
        raise ValueError("At least one read is required to build a consensus.")
    if matrix is None:
        matrix = dna_matrix()

    ref = _as_str(reference)
    n = len(ref)
    piles: list[Counter] = [Counter() for _ in range(n)]
    for read in reads:
        aln = align_glocal(_as_str(read), ref, matrix, gap_open=gap_open,
                           gap_extend=gap_extend)
        ridx = aln.ref_start
        for qc, rc in zip(aln.aligned_query, aln.aligned_reference, strict=True):
            if rc != "-":
                if qc != "-":
                    piles[ridx][qc] += 1
                ridx += 1
    # Reference acts as a prior observation at every position.
    for idx, base in enumerate(ref):
        piles[idx][base] += 1

    consensus_chars: list[str] = []
    depth: list[int] = []
    n_ambiguous = 0
    is_dna = identify_sequence_type(ref) in ("dna", "rna")
    for idx in range(n):
        counter = piles[idx]
        total = sum(counter.values())
        best_base, best_count = counter.most_common(1)[0]
        if method == "threshold" and best_count / total < threshold:
            bases = set(counter.keys())
            code = _iupac_code(bases) if is_dna else "X"
            consensus_chars.append(code)
            n_ambiguous += 1
        else:
            consensus_chars.append(best_base)
        depth.append(total)
    return ConsensusResult(
        consensus="".join(consensus_chars),
        depth=depth,
        method=method,
        threshold=threshold,
        n_ambiguous=n_ambiguous,
        reference=ref,
    )


# ---------------------------------------------------------------------------
# Multiple sequence alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSAResult:
    """Outcome of a multiple sequence alignment.

    Attributes
        rows:
            Equal-length aligned sequences (with ``-`` gaps), in the caller's
            input order, so ``rows[i]`` is the alignment of ``sequences[i]``.
        n_sequences:
            Number of input sequences.
        length:
            Alignment width (length of every row).
        guide_order:
            Original 0-based indices in the construction order (anchor first),
            kept for provenance.
    """

    rows: list[str]
    n_sequences: int
    length: int
    guide_order: list[int] = field(default_factory=list)


def multiple_sequence_alignment(
    sequences: list[str | SeqRecord],
    matrix: SubstitutionMatrix | None = None,
    *,
    gap_open: int = 5,
    gap_extend: int = 2,
    k: int = 4,
) -> MSAResult:
    """Progressive multiple sequence alignment (anchor + greedy addition).

    The longest sequence is chosen as the anchor.  Remaining sequences are
    ordered by k-mer containment to the anchor (most similar added first) and
    each is glocally aligned to the current gapped profile; gaps introduced in
    the profile are propagated to every existing row so all rows stay
    equal-length.  This is a pragmatic progressive MSA, not a profile-HMM or
    tree-based method such as MUSCLE.

    Parameters
        sequences:
            Two or more sequences (str or :class:`SeqRecord`).
        matrix, gap_open, gap_extend:
            Alignment parameters.
        k:
            K-mer size for the similarity ordering.

    Returns
        An :class:`MSAResult` with equal-length aligned rows.
    """
    if matrix is None:
        matrix = dna_matrix()
    seqs = [_as_str(s) for s in sequences]
    if len(seqs) < 2:
        return MSAResult(
            rows=list(seqs),
            n_sequences=len(seqs),
            length=len(seqs[0]) if seqs else 0,
            guide_order=list(range(len(seqs))),
        )

    anchor_index = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    anchor_seq = seqs[anchor_index]
    others = [i for i in range(len(seqs)) if i != anchor_index]
    others.sort(
        key=lambda i: kmer_containment(seqs[i], anchor_seq, k), reverse=True
    )

    guide_order = [anchor_index, *others]
    profile_rows: list[list[str]] = [list(anchor_seq)]
    row_ids: list[int] = [anchor_index]
    anchor = list(anchor_seq)
    for idx in others:
        anchor_row = "".join(anchor)
        aln = align_glocal(
            seqs[idx], anchor_row, matrix, gap_open=gap_open,
            gap_extend=gap_extend,
        )
        # A glocal alignment reports only the mapped region of the reference,
        # so the anchor's free flanks (before ``ref_start`` and after
        # ``ref_end``) are absent from ``aligned_reference``.  Using that row
        # as the new profile would silently delete those anchor residues from
        # every row, so the flanks are restored here and the incoming
        # sequence receives matching gaps instead.
        leading = anchor_row[: aln.ref_start]
        trailing = anchor_row[aln.ref_end :]
        reference_row = leading + aln.aligned_reference + trailing
        if reference_row.replace("-", "") != anchor_row:
            raise ValueError(
                "Internal MSA error: the propagated reference row is not a "
                "gapped version of the profile it came from. This indicates "
                "an unresolved glocal-alignment inconsistency."
            )
        query_row = "-" * len(leading) + aln.aligned_query + "-" * len(trailing)

        new_rows: list[list[str]] = []
        for row in profile_rows:
            new_row: list[str] = []
            ri = 0
            for char in reference_row:
                if char == "-":
                    new_row.append("-")
                else:
                    new_row.append(row[ri])
                    ri += 1
            new_rows.append(new_row)
        profile_rows = new_rows
        profile_rows.append(list(query_row))
        row_ids.append(idx)
        anchor = list(reference_row)

    if len({len(row) for row in profile_rows}) != 1:
        raise ValueError(
            "Internal MSA error: aligned rows have inconsistent lengths "
            f"{sorted({len(row) for row in profile_rows})}."
        )
    # Rows are returned in the caller's input order so that
    # ``zip(sequences, result.rows)`` pairs each sequence with its own row.
    # The anchor-first construction order is preserved in ``guide_order``.
    rows = [""] * len(seqs)
    for position, idx in enumerate(row_ids):
        rows[idx] = "".join(profile_rows[position])
    return MSAResult(
        rows=rows,
        n_sequences=len(seqs),
        length=len(rows[0]) if rows else 0,
        guide_order=guide_order,
    )


# ---------------------------------------------------------------------------
# Alignment / quality summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentSummary:
    """Conservation and identity statistics for a multiple alignment.

    Attributes
        n_sequences, length:
            Number of rows and alignment width.
        conservation_per_column:
            Per-column fraction of the most common residue (gaps excluded).
        entropy_per_column:
            Per-column Shannon entropy (bits) over non-gap residues.
        overall_identity:
            Mean pairwise identity (gaps excluded) across all row pairs.
        variable_sites:
            Column indices whose non-gap residues are not all identical.
        pairwise_identity:
            Symmetric matrix of pairwise identities (row-major).
        most_common_per_column:
            Majority non-gap residue per column (``"-"`` if all gapped).
    """

    n_sequences: int
    length: int
    conservation_per_column: list[float]
    entropy_per_column: list[float]
    overall_identity: float
    variable_sites: list[int]
    pairwise_identity: list[list[float]]
    most_common_per_column: list[str]


def summarize_alignment(rows: list[str]) -> AlignmentSummary:
    """Compute conservation, entropy, identity and variable-site statistics.

    Parameters
        rows:
            Equal-length aligned sequences (with ``-`` gaps).

    Returns
        An :class:`AlignmentSummary`.

    Raises
        ValueError:
            If fewer than two rows are supplied or rows differ in length.
    """
    if len(rows) < 2:
        raise ValueError("Need at least two aligned rows to summarize.")
    n_seq = len(rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All aligned rows must have the same length.")

    conservation: list[float] = []
    entropy: list[float] = []
    most_common: list[str] = []
    variable_sites: list[int] = []
    for col in range(width):
        counts = Counter(row[col] for row in rows if row[col] != "-")
        total = sum(counts.values())
        if total == 0:
            conservation.append(0.0)
            entropy.append(0.0)
            most_common.append("-")
            continue
        top_base, top_count = counts.most_common(1)[0]
        conservation.append(top_count / total)
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * np.log2(p)
        entropy.append(ent)
        most_common.append(top_base)
        if top_count != total:
            variable_sites.append(col)

    pairwise: list[list[float]] = [[0.0] * n_seq for _ in range(n_seq)]
    id_sum = 0.0
    id_pairs = 0
    for a in range(n_seq):
        for b in range(a + 1, n_seq):
            matches = 0
            compared = 0
            for col in range(width):
                ca = rows[a][col]
                cb = rows[b][col]
                if ca == "-" or cb == "-":
                    continue
                compared += 1
                if ca == cb:
                    matches += 1
            ident = (matches / compared) if compared else 0.0
            pairwise[a][b] = ident
            pairwise[b][a] = ident
            id_sum += ident
            id_pairs += 1
    overall = (id_sum / id_pairs) if id_pairs else 0.0

    return AlignmentSummary(
        n_sequences=n_seq,
        length=width,
        conservation_per_column=conservation,
        entropy_per_column=entropy,
        overall_identity=overall,
        variable_sites=variable_sites,
        pairwise_identity=pairwise,
        most_common_per_column=most_common,
    )


@dataclass(frozen=True)
class SequencingQualitySummary:
    """Aggregate quality statistics for a set of sequencing reads.

    Attributes
        n_reads:
            Number of reads summarised.
        read_lengths:
            Per-read sequence lengths.
        mean_length, length_min, length_max:
            Length statistics.
        mean_quality_per_position:
            Mean Phred quality at each position (length = longest read).
        overall_mean_quality:
            Mean Phred quality across all read bases.
        gc_fraction_mean:
            Mean GC fraction across reads.
        low_quality_read_fraction:
            Fraction of reads whose mean quality is below *min_q*.
        min_q:
            Quality threshold used to flag low-quality reads.
    """

    n_reads: int
    read_lengths: list[int]
    mean_length: float
    length_min: int
    length_max: int
    mean_quality_per_position: list[float]
    overall_mean_quality: float
    gc_fraction_mean: float
    low_quality_read_fraction: float
    min_q: int


def summarize_sequencing_quality(
    records: list[SeqRecord],
    *,
    min_q: int = 20,
) -> SequencingQualitySummary:
    """Summarise read lengths, qualities, GC and low-quality read fraction.

    Parameters
        records:
            Read :class:`SeqRecord` objects (those without quality are skipped
            for quality statistics).
        min_q:
            Mean-quality threshold below which a read is "low quality".

    Returns
        A :class:`SequencingQualitySummary`.

    Raises
        ValueError:
            If no records are supplied.
    """
    if not records:
        raise ValueError("Cannot summarise an empty read set.")
    lengths = [len(r.sequence) for r in records]
    mean_len = float(np.mean(lengths))
    gc_values = [_gc_fraction(r.sequence) for r in records]
    gc_mean = float(np.mean(gc_values)) if gc_values else 0.0

    max_len = max(lengths) if lengths else 0
    per_position = np.full((len(records), max_len), np.nan)
    all_quals: list[int] = []
    low_reads = 0
    for ridx, r in enumerate(records):
        if r.quality is not None:
            for pos, q in enumerate(r.quality):
                per_position[ridx, pos] = q
                all_quals.append(q)
            if r.quality:
                if float(np.mean(r.quality)) < min_q:
                    low_reads += 1
    mean_q_pos = list(np.nanmean(per_position, axis=0))
    overall_q = float(np.mean(all_quals)) if all_quals else 0.0
    low_frac = (low_reads / len(records)) if records else 0.0

    return SequencingQualitySummary(
        n_reads=len(records),
        read_lengths=lengths,
        mean_length=mean_len,
        length_min=min(lengths),
        length_max=max(lengths),
        mean_quality_per_position=mean_q_pos,
        overall_mean_quality=overall_q,
        gc_fraction_mean=gc_mean,
        low_quality_read_fraction=low_frac,
        min_q=min_q,
    )


def _gc_fraction(sequence: str) -> float:
    """Return the GC fraction of a nucleotide sequence (0.0 for protein)."""
    cleaned = clean_sequence(sequence)
    if not cleaned:
        return 0.0
    defined = sum(cleaned.count(b) for b in "ACGTU")
    if not defined:
        return 0.0
    gc = cleaned.count("G") + cleaned.count("C")
    return gc / defined
