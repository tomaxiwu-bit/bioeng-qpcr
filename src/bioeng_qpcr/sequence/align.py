"""Pairwise and k-mer sequence alignment algorithms.

This module provides self-contained, dependency-free implementations of the
core sequence-alignment algorithms used by the sequencing-comparison
workflow.  All dynamic-programming (DP) routines use :mod:`numpy` matrices
for speed and raise a clear :class:`ValueError` when the problem size would
exhaust memory.

Algorithms
----------
* **Needleman-Wunsch** (:func:`needleman_wunsch`) -- global alignment with a
  *linear* gap penalty (every gap residue costs the same ``gap`` amount).
* **Smith-Waterman** (:func:`smith_waterman`) -- local alignment with a
  linear gap penalty; only the highest-scoring diagonal fragment is returned.
* **Gotoh / affine gap** (:func:`needleman_wunsch_affine`,
  :func:`smith_waterman_affine`) -- global and local alignment where a gap of
  length *k* costs ``gap_open + (k - 1) * gap_extend``.  This matches the
  EMBOSS ``needle``/``water`` ``--gapopen``/``--gapextend`` semantics: a
  single-residue gap costs exactly ``gap_open`` (not ``gap_open + gap_extend``
  and not ``2 * gap_open``).
* **Glocal / semi-global** (:func:`align_glocal`) -- global alignment with
  *free* end gaps in the reference, used to map a short read (which is only a
  sub-region of the reference) onto its template.

Scoring is supplied through a :class:`SubstitutionMatrix`.  DNA is scored with
a simple match/mismatch matrix (see :func:`dna_matrix`); protein uses the
built-in :data:`BLOSUM62` table (the full, untruncated NCBI matrix).  Arbitrary
matrices can be built with :func:`custom_matrix`.

Finally, :func:`kmer_similarity` offers an O(n) heuristic (k-mer Jaccard /
containment) for rapidly clustering or screening hundreds of reads without a
full DP pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, unique

import numpy as np

# Raising this above (len_q * len_r) cells triggers a memory guard instead of
# allocating an unbounded DP table.  25e6 int cells ~= 200 MB for three matrices.
_MAX_CELLS: float = 25_000_000.0

# The Gotoh recurrence keeps four double-precision matrices plus two byte
# pointer matrices (~34 bytes per cell), so its budget is tighter than the
# single-matrix linear recurrences: 8e6 cells is roughly 270 MB.
_MAX_CELLS_AFFINE: float = 8_000_000.0

# ---------------------------------------------------------------------------
# Substitution matrices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstitutionMatrix:
    """A symmetric substitution scoring matrix.

    The matrix is stored as a callable ``score(a, b)``.  For a match/mismatch
    matrix the callable compares residues directly; for a full table (BLOSUM62)
    it performs a symmetric dictionary lookup with a fallback default.

    Attributes
        name:
            Human-readable name (e.g. ``"BLOSUM62"``, ``"dna"``).
        _score:
            The underlying scoring callable.
    """

    name: str
    _score: Callable[[str, str], float]

    def score(self, a: str, b: str) -> float:
        """Return the substitution score between residues *a* and *b*."""
        return self._score(a, b)


def dna_matrix(match: int = 1, mismatch: int = -1) -> SubstitutionMatrix:
    """Build a DNA/RNA match-mismatch substitution matrix.

    Parameters
        match:
            Score awarded when two aligned residues are identical.
        mismatch:
            Score (usually negative) when they differ.

    Returns
        A :class:`SubstitutionMatrix` scoring equal residues as *match*.
    """
    return SubstitutionMatrix(
        name="dna",
        _score=lambda a, b: float(match if a == b else mismatch),
    )


def custom_matrix(
    table: dict[tuple[str, str], int],
    *,
    default: int = -1,
    name: str = "custom",
) -> SubstitutionMatrix:
    """Build a substitution matrix from an explicit symmetric lookup table.

    Parameters
        table:
            Mapping from ``(a, b)`` residue pair to score.  Lookups are
            symmetric, so only one orientation needs to be present.
        default:
            Score used when a pair is absent from *table*.
        name:
            Name attached to the resulting matrix.

    Returns
        A :class:`SubstitutionMatrix` wrapping *table*.
    """
    lookup = {(a, b): float(v) for (a, b), v in table.items()}

    def _score(a: str, b: str) -> float:
        direct = lookup.get((a, b))
        if direct is not None:
            return direct
        swapped = lookup.get((b, a))
        if swapped is not None:
            return swapped
        return float(default)

    return SubstitutionMatrix(name=name, _score=_score)


# Full NCBI BLOSUM62 matrix (24 x 24, including ambiguous B/Z/X and '*').
_BLOSUM62_SYMBOLS = "ARNDCQEGHILKMFPSTWYVBZX*"
_BLOSUM62_ROWS: list[list[int]] = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4],  # noqa: E501
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4],  # noqa: E501
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4],  # noqa: E501
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4],  # noqa: E501
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],  # noqa: E501
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4],  # noqa: E501
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],  # noqa: E501
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4],  # noqa: E501
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4],  # noqa: E501
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, -1, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4],  # noqa: E501
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4],  # noqa: E501
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4],  # noqa: E501
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4],  # noqa: E501
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, -1, 0, -3, 0, 6, -3, -2, -2, 1, 3, -1, -3, -3, -1, -4],  # noqa: E501
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -3, 7, -1, -1, -2, -1, -2, -2, -1, -1, -4],  # noqa: E501
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -1, 0, 0, 0, -4],  # noqa: E501
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4],  # noqa: E501
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -2, -3, -2, 11, 2, -3, -4, -3, -2, -4],  # noqa: E501
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -1, -2, -2, 2, 7, -1, -3, -2, -1, -4],  # noqa: E501
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -1, 0, -3, -1, 4, -3, -2, -1, -4],  # noqa: E501
    [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4],  # noqa: E501
    [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],  # noqa: E501
    [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -2, -1, -1, -1, -1, -1, -4],  # noqa: E501
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1],  # noqa: E501
]

#: Frozen, symmetric lookup table for :data:`BLOSUM62`.
BLOSUM62_TABLE: dict[tuple[str, str], int] = {}
for _i, _sym_a in enumerate(_BLOSUM62_SYMBOLS):
    for _j, _sym_b in enumerate(_BLOSUM62_SYMBOLS):
        BLOSUM62_TABLE[(_sym_a, _sym_b)] = _BLOSUM62_ROWS[_i][_j]

#: The complete NCBI BLOSUM62 substitution matrix.
BLOSUM62 = SubstitutionMatrix(name="BLOSUM62", _score=lambda a, b: float(
    BLOSUM62_TABLE.get((a, b), BLOSUM62_TABLE.get((b, a), -1))
))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@unique
class AlignmentMode(str, Enum):
    """Alignment mode selecting boundary/gap treatment."""

    GLOBAL = "global"
    LOCAL = "local"
    GLOCAL = "glocal"


@dataclass(frozen=True)
class AlignmentResult:
    """The outcome of a pairwise alignment.

    Attributes
        aligned_query:
            Query sequence with ``-`` gap characters, same length as
            *aligned_reference*.
        aligned_reference:
            Reference sequence with ``-`` gap characters.
        score:
            Raw alignment score under the supplied matrix and gap scheme.
        identity:
            Fraction of aligned (non-gap) columns that are exact matches.
        similarity:
            Fraction of aligned (non-gap) columns with a positive substitution
            score (equals identity for simple DNA match/mismatch matrices).
        gaps:
            Total number of gap characters in the alignment.
        length:
            Number of aligned columns (length of the gapped strings).
        query_start, query_end:
            0-based half-open interval of the query covered by the alignment.
        ref_start, ref_end:
            0-based half-open interval of the reference covered.
        cigar:
            Compact CIGAR string (``M``=aligned, ``I``=query insertion,
            ``D``=reference insertion/deletion).
        matches, mismatches:
            Counts of aligned match / mismatch columns.
        gap_opens, gap_extends:
            Number of gap runs and total extended (beyond the first) residues.
        mode:
            The :class:`AlignmentMode` that produced this result.
        matrix_name:
            Name of the substitution matrix used.
        query_orientation:
            ``"forward"`` or ``"reverse_complement"`` for workflows that
            test both nucleotide read orientations.
    """

    aligned_query: str
    aligned_reference: str
    score: float
    identity: float
    similarity: float
    gaps: int
    length: int
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    cigar: str
    matches: int
    mismatches: int
    gap_opens: int
    gap_extends: int
    mode: AlignmentMode
    matrix_name: str
    query_orientation: str = "forward"


def _finalize(
    aq: list[str],
    ar: list[str],
    score: float,
    mode: AlignmentMode,
    matrix: SubstitutionMatrix,
    query_start: int,
    query_end: int,
    ref_start: int,
    ref_end: int,
) -> AlignmentResult:
    """Compute derived statistics and build an :class:`AlignmentResult`."""
    aligned_query = "".join(reversed(aq))
    aligned_reference = "".join(reversed(ar))
    matches = 0
    mismatches = 0
    similar = 0
    gaps = 0
    gap_opens = 0
    gap_extends = 0
    in_gap = False
    cigar_ops: list[tuple[str, int]] = []

    for qc, rc in zip(aligned_query, aligned_reference, strict=True):
        if qc == "-" or rc == "-":
            gaps += 1
            if not in_gap:
                gap_opens += 1
                in_gap = True
            else:
                gap_extends += 1
            op = "I" if qc != "-" else "D"
        else:
            in_gap = False
            if qc == rc:
                matches += 1
            else:
                mismatches += 1
            if matrix.score(qc, rc) > 0:
                similar += 1
            op = "M"
        if cigar_ops and cigar_ops[-1][0] == op:
            cigar_ops[-1] = (op, cigar_ops[-1][1] + 1)
        else:
            cigar_ops.append((op, 1))

    cigar = "".join(f"{count}{op}" for op, count in cigar_ops)
    aligned_columns = matches + mismatches
    identity = (matches / aligned_columns) if aligned_columns else 0.0
    similarity = (similar / aligned_columns) if aligned_columns else 0.0

    return AlignmentResult(
        aligned_query=aligned_query,
        aligned_reference=aligned_reference,
        score=score,
        identity=identity,
        similarity=similarity,
        gaps=gaps,
        length=len(aligned_query),
        query_start=query_start,
        query_end=query_end,
        ref_start=ref_start,
        ref_end=ref_end,
        cigar=cigar,
        matches=matches,
        mismatches=mismatches,
        gap_opens=gap_opens,
        gap_extends=gap_extends,
        mode=mode,
        matrix_name=matrix.name,
    )


def _memory_guard(m: int, n: int) -> None:
    """Raise :class:`ValueError` when the DP table would exceed the budget."""
    if m * n > _MAX_CELLS:
        raise ValueError(
            f"Alignment of lengths {m} x {n} = {m * n} cells exceeds the "
            f"memory budget ({int(_MAX_CELLS)} cells). Trim the sequences or "
            "use k-mer pre-screening before full dynamic programming."
        )


# ---------------------------------------------------------------------------
# Linear gap (Needleman-Wunsch / Smith-Waterman)
# ---------------------------------------------------------------------------


def needleman_wunsch(
    query: str,
    reference: str,
    *,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> AlignmentResult:
    """Global alignment with a linear (constant) gap penalty.

    Parameters
        query:
            Query sequence (upper-case residues recommended).
        reference:
            Reference sequence.
        match:
            Score for an aligned match.
        mismatch:
            Score for an aligned mismatch.
        gap:
            Penalty (usually negative) applied per gap residue.

    Returns
        A global :class:`AlignmentResult`.
    """
    matrix = dna_matrix(match=match, mismatch=mismatch)
    return _needleman_wunsch_linear(query, reference, matrix, gap)


def smith_waterman(
    query: str,
    reference: str,
    *,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> AlignmentResult:
    """Local alignment with a linear gap penalty (Smith-Waterman).

    Only the highest-scoring ungapped-terminating diagonal fragment is
    returned; surrounding mismatches are discarded.

    Parameters
        query, reference:
            Sequences to align locally.
        match, mismatch, gap:
            Scoring parameters (see :func:`needleman_wunsch`).

    Returns
        A local :class:`AlignmentResult`.
    """
    matrix = dna_matrix(match=match, mismatch=mismatch)
    return _smith_waterman_linear(query, reference, matrix, gap)


def _needleman_wunsch_linear(
    query: str,
    reference: str,
    matrix: SubstitutionMatrix,
    gap: int,
) -> AlignmentResult:
    """Core linear-gap global alignment (numpy DP + traceback)."""
    q = list(query)
    r = list(reference)
    m, n = len(q), len(r)
    _memory_guard(m, n)

    H = np.zeros((m + 1, n + 1), dtype=np.int64)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    H[0, 1:] = np.arange(1, n + 1) * gap
    H[1:, 0] = np.arange(1, m + 1) * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = int(H[i - 1, j - 1]) + int(matrix.score(q[i - 1], r[j - 1]))
            up = int(H[i - 1, j]) + gap
            left = int(H[i, j - 1]) + gap
            best = diag
            choice = 0
            if up > best:
                best = up
                choice = 1
            if left > best:
                best = left
                choice = 2
            H[i, j] = best
            trace[i, j] = choice

    aq: list[str] = []
    ar: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i, j] == 0:
            aq.append(q[i - 1])
            ar.append(r[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i, j] == 1):
            # "up" consumed a query residue only, so that residue is aligned
            # against a gap in the reference.
            aq.append(q[i - 1])
            ar.append("-")
            i -= 1
        else:
            # "left" consumed a reference residue only: a gap in the query.
            aq.append("-")
            ar.append(r[j - 1])
            j -= 1
    return _finalize(aq, ar, float(H[m, n]), AlignmentMode.GLOBAL, matrix, 0, m, 0, n)


def _smith_waterman_linear(
    query: str,
    reference: str,
    matrix: SubstitutionMatrix,
    gap: int,
) -> AlignmentResult:
    """Core linear-gap local alignment (numpy DP + traceback)."""
    q = list(query)
    r = list(reference)
    m, n = len(q), len(r)
    _memory_guard(m, n)

    H = np.zeros((m + 1, n + 1), dtype=np.int64)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)
    best = 0
    bi, bj = 0, 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = int(H[i - 1, j - 1]) + int(matrix.score(q[i - 1], r[j - 1]))
            up = int(H[i - 1, j]) + gap
            left = int(H[i, j - 1]) + gap
            best_local = max(0, diag, up, left)
            choice = 0
            if best_local == diag and diag > 0:
                choice = 0
            elif best_local == up and up > 0:
                choice = 1
            elif best_local == left and left > 0:
                choice = 2
            H[i, j] = best_local
            trace[i, j] = choice
            if best_local > best:
                best = best_local
                bi, bj = i, j

    aq: list[str] = []
    ar: list[str] = []
    i, j = bi, bj
    while i > 0 and j > 0 and H[i, j] > 0:
        if trace[i, j] == 0:
            aq.append(q[i - 1])
            ar.append(r[j - 1])
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            # "up": the query residue is aligned against a reference gap.
            aq.append(q[i - 1])
            ar.append("-")
            i -= 1
        else:
            # "left": the reference residue is aligned against a query gap.
            aq.append("-")
            ar.append(r[j - 1])
            j -= 1
    q_end, r_end = bi, bj
    return _finalize(
        aq, ar, float(best), AlignmentMode.LOCAL, matrix, i, q_end, j, r_end
    )


# ---------------------------------------------------------------------------
# Affine gap (Gotoh)
# ---------------------------------------------------------------------------


def needleman_wunsch_affine(
    query: str,
    reference: str,
    matrix: SubstitutionMatrix | None = None,
    *,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> AlignmentResult:
    """Global alignment with affine (open + extend) gap penalties -- Gotoh.

    Parameters
        query, reference:
            Sequences to align.
        matrix:
            Substitution matrix (defaults to the DNA match/mismatch matrix).
        gap_open:
            Penalty to open a gap (charged once, including the first residue).
        gap_extend:
            Penalty per additional gap residue.  A gap of length *k* therefore
            costs ``gap_open + (k - 1) * gap_extend``.

    Returns
        A global affine :class:`AlignmentResult`.
    """
    if matrix is None:
        matrix = dna_matrix()
    return _gotoh(query, reference, matrix, gap_open, gap_extend, AlignmentMode.GLOBAL)


def smith_waterman_affine(
    query: str,
    reference: str,
    matrix: SubstitutionMatrix | None = None,
    *,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> AlignmentResult:
    """Local alignment with affine gap penalties -- Gotoh / Smith-Waterman.

    Parameters
        query, reference:
            Sequences to align locally.
        matrix:
            Substitution matrix (defaults to DNA).
        gap_open, gap_extend:
            Affine gap parameters (see :func:`needleman_wunsch_affine`).

    Returns
        A local affine :class:`AlignmentResult`.
    """
    if matrix is None:
        matrix = dna_matrix()
    return _gotoh(query, reference, matrix, gap_open, gap_extend, AlignmentMode.LOCAL)


def align_glocal(
    query: str,
    reference: str,
    matrix: SubstitutionMatrix | None = None,
    *,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> AlignmentResult:
    """Semi-global (glocal) alignment with free reference end gaps.

    Leading and trailing regions of the reference that are not covered by the
    query are left unpenalised, which makes this the natural choice for
    mapping a short read (a sub-region of the reference) onto its template.
    End gaps in the query are still penalised.

    Parameters
        query:
            The (typically shorter) read sequence.
        reference:
            The (typically longer) template sequence.
        matrix:
            Substitution matrix (defaults to DNA).
        gap_open, gap_extend:
            Affine gap parameters.

    Returns
        A glocal :class:`AlignmentResult` whose ``ref_start``/``ref_end``
        bracket the mapped region.
    """
    if matrix is None:
        matrix = dna_matrix()
    return _gotoh(query, reference, matrix, gap_open, gap_extend, AlignmentMode.GLOCAL)


def _gotoh(
    query: str,
    reference: str,
    matrix: SubstitutionMatrix,
    gap_open: int,
    gap_extend: int,
    mode: AlignmentMode,
) -> AlignmentResult:
    """Affine-gap dynamic programming core for global / local / glocal.

    Implements the Gotoh three-matrix recurrence.  Boundary and stopping
    conditions switch on *mode*:

    * GLOBAL -- both ends anchored, free leading reference gaps disabled.
    * LOCAL -- every cell clipped at zero; traceback from the global maximum
      stops when the score returns to zero.
    * GLOCAL -- first reference row is zero (free leading reference gap), the
      final score is the best over the last row (free trailing reference gap),
      and traceback stops when the query is exhausted.

    Matrix conventions (stated explicitly because "Ix"/"Iy" are used both
    ways in the literature):

    * ``M[i, j]`` -- ``query[i-1]`` is paired with ``reference[j-1]``.
    * ``E[i, j]`` -- vertical move: ``query[i-1]`` is paired with a **gap in
      the reference**.
    * ``F[i, j]`` -- horizontal move: ``reference[j-1]`` is paired with a
      **gap in the query**.

    The traceback carries the *matrix state* (M / E / F) rather than only the
    arg-max of the combined ``H`` matrix.  Reading the arg-max back at every
    step is a classic Gotoh implementation bug: it silently swaps the two
    gap types and can emit an "alignment" whose rows are not gapped versions
    of the input sequences at all.
    """
    q = list(query)
    r = list(reference)
    m, n = len(q), len(r)
    if m * n > _MAX_CELLS_AFFINE:
        raise ValueError(
            f"Affine alignment of lengths {m} x {n} = {m * n} cells exceeds "
            f"the memory budget ({int(_MAX_CELLS_AFFINE)} cells); the Gotoh "
            "recurrence keeps four tables of doubles. Trim the sequences or "
            "use k-mer pre-screening first."
        )

    INF = float("inf")
    M = np.full((m + 1, n + 1), -INF, dtype=np.float64)
    E = np.full((m + 1, n + 1), -INF, dtype=np.float64)
    F = np.full((m + 1, n + 1), -INF, dtype=np.float64)
    H = np.full((m + 1, n + 1), -INF, dtype=np.float64)
    # Which matrix achieved the optimum at each cell (0 = M, 1 = E, 2 = F).
    choice = np.zeros((m + 1, n + 1), dtype=np.int8)
    # For E and F, which precursor matrix the cell was extended from.
    from_e = np.zeros((m + 1, n + 1), dtype=np.int8)  # 1 = extended from E
    from_f = np.zeros((m + 1, n + 1), dtype=np.int8)  # 2 = extended from F

    if mode is AlignmentMode.GLOCAL:
        # The query must be consumed end to end; the reference may be
        # truncated at both ends, so the whole first row starts free.
        H[0, :] = 0.0
        M[0, :] = 0.0
        F[0, :] = 0.0
    elif mode is AlignmentMode.LOCAL:
        # Local alignments must be free to start anywhere, so the whole
        # first row and first column are zero.  Charging gap costs along
        # them (as in the global case) would discard the first matched
        # residue of every local alignment.
        H[0, :] = 0.0
        M[0, :] = 0.0
        H[:, 0] = 0.0
        M[:, 0] = 0.0
    else:
        H[0, 0] = 0.0
        M[0, 0] = 0.0
        for i in range(1, m + 1):
            cost = -(gap_open + (i - 1) * gap_extend)
            M[i, 0] = cost
            E[i, 0] = cost
            H[i, 0] = cost
        for j in range(1, n + 1):
            cost = -(gap_open + (j - 1) * gap_extend)
            M[0, j] = cost
            F[0, j] = cost
            H[0, j] = cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = float(matrix.score(q[i - 1], r[j - 1]))
            diag = H[i - 1, j - 1] + s
            if mode is AlignmentMode.LOCAL:
                diag = max(0.0, diag)
            M[i, j] = diag

            open_e = M[i - 1, j] - gap_open
            extend_e = E[i - 1, j] - gap_extend
            if extend_e > open_e:
                E[i, j] = extend_e
                from_e[i, j] = 1
            else:
                E[i, j] = open_e
                from_e[i, j] = 0

            open_f = M[i, j - 1] - gap_open
            extend_f = F[i, j - 1] - gap_extend
            if extend_f > open_f:
                F[i, j] = extend_f
                from_f[i, j] = 2
            else:
                F[i, j] = open_f
                from_f[i, j] = 0

            best = M[i, j]
            ch = 0
            if E[i, j] > best:
                best = E[i, j]
                ch = 1
            if F[i, j] > best:
                best = F[i, j]
                ch = 2
            if mode is AlignmentMode.LOCAL and best < 0.0:
                best = 0.0
            H[i, j] = best
            choice[i, j] = ch

    if mode is AlignmentMode.GLOCAL:
        last_row = H[m, :]
        j_end = int(np.argmax(last_row))
        i_end = m
        score = float(last_row[j_end])
    elif mode is AlignmentMode.LOCAL:
        i_end, j_end = divmod(int(np.argmax(H)), n + 1)
        score = float(H[i_end, j_end])
    else:
        i_end, j_end = m, n
        score = float(H[m, n])

    aq: list[str] = []
    ar: list[str] = []
    i, j = i_end, j_end
    state = int(choice[i_end, j_end])
    while i > 0 or j > 0:
        if mode is AlignmentMode.GLOCAL and i == 0:
            break  # free leading reference gap: stop without emitting it
        if mode is AlignmentMode.LOCAL and H[i, j] <= 0.0:
            # A zero cell marks the start of the local alignment; residues
            # before it are deliberately excluded from the result.
            break
        if i == 0:
            # Only reference residues remain (global mode): query gaps.
            aq.append("-")
            ar.append(r[j - 1])
            j -= 1
            continue
        if j == 0:
            # Only query residues remain (global mode): reference gaps.
            aq.append(q[i - 1])
            ar.append("-")
            i -= 1
            continue

        if state == 0:  # M: pair query[i-1] with reference[j-1]
            aq.append(q[i - 1])
            ar.append(r[j - 1])
            i -= 1
            j -= 1
            state = int(choice[i, j])
        elif state == 1:  # E: query[i-1] against a reference gap
            aq.append(q[i - 1])
            ar.append("-")
            state = 0 if from_e[i, j] == 0 else 1
            i -= 1
        else:  # F: reference[j-1] against a query gap
            aq.append("-")
            ar.append(r[j - 1])
            state = 0 if from_f[i, j] == 0 else 2
            j -= 1

    if mode is AlignmentMode.GLOCAL:
        q_start, r_start = 0, j
    elif mode is AlignmentMode.LOCAL:
        q_start, r_start = i, j
    else:
        q_start, r_start = 0, 0
    return _finalize(
        aq, ar, score, mode, matrix, q_start, i_end, r_start, j_end
    )


# ---------------------------------------------------------------------------
# k-mer based fast similarity
# ---------------------------------------------------------------------------


def kmer_set(sequence: str, k: int = 4) -> set[str]:
    """Return the set of k-mers (substrings of length *k*) in *sequence*.

    Parameters
        sequence:
            Input sequence (cleaned upstream by the caller if desired).
        k:
            K-mer size (>= 1).

    Returns
        The set of distinct k-mers.

    Raises
        ValueError:
            If *k* is not positive or the sequence is shorter than *k*.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    seq = sequence.upper()
    if len(seq) < k:
        raise ValueError(
            f"Sequence length {len(seq)} < k={k}; no k-mers available."
        )
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def kmer_jaccard(seq_a: str, seq_b: str, k: int = 4) -> float:
    """Jaccard similarity of the k-mer sets of two sequences.

    Defined as ``|A ∩ B| / |A ∪ B|``; equals 1.0 for identical sequences and
    approaches 0.0 for unrelated sequences.

    Parameters
        seq_a, seq_b:
            Input sequences.
        k:
            K-mer size.

    Returns
        The Jaccard coefficient in ``[0, 1]``.
    """
    a = kmer_set(seq_a, k)
    b = kmer_set(seq_b, k)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def kmer_containment(seq_a: str, seq_b: str, k: int = 4) -> float:
    """Containment of *seq_a*'s k-mers within *seq_b*'s k-mers.

    Defined as ``|A ∩ B| / |A|``; equals 1.0 when every k-mer of *seq_a*
    appears in *seq_b* (e.g. *seq_a* is a substring or close subsequence).

    Parameters
        seq_a, seq_b:
            Input sequences.
        k:
            K-mer size.

    Returns
        The containment fraction in ``[0, 1]``.
    """
    a = kmer_set(seq_a, k)
    b = kmer_set(seq_b, k)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def kmer_similarity(
    seq_a: str,
    seq_b: str,
    k: int = 4,
    *,
    method: str = "jaccard",
) -> float:
    """Fast heuristic similarity between two sequences via shared k-mers.

    Parameters
        seq_a, seq_b:
            Input sequences.
        k:
            K-mer size (typically 4, 6 or 8).
        method:
            ``"jaccard"`` for set overlap, ``"containment"`` for one-directional
            containment of *seq_a* within *seq_b*.

    Returns
        The requested similarity in ``[0, 1]``.

    Raises
        ValueError:
            If *method* is not one of the supported names.
    """
    if method == "jaccard":
        return kmer_jaccard(seq_a, seq_b, k)
    if method == "containment":
        return kmer_containment(seq_a, seq_b, k)
    raise ValueError(
        f"Unknown k-mer method {method!r}; expected 'jaccard' or 'containment'."
    )
