"""Shared fixtures and brute-force references for sequence tests.

Everything here is generated synthetically (no network, no large files) so the
suite runs in well under 30 seconds.

The template is deliberately **non-repetitive**.  On a periodic template (for
example ``"ACGT" * 105``, used by an earlier revision of this fixture) the
placement of an insertion or a deletion is genuinely ambiguous: a gap can slide
by one period, or past an identical neighbouring base, without changing the
alignment score at all.  A test that pinned a variant to a specific position on
such a template would be asserting the tie-breaking order of the dynamic
programming rather than a biological fact, and would have broken as soon as an
equally-correct implementation made a different arbitrary choice.

The template below is therefore pseudo-random, and every engineered variant is
placed at a position that has been checked to be the *unique* optimum (see
:func:`_unique_deletion_start` and :func:`_unique_insertion`).  The resulting
positions are exported as module constants so tests assert against them instead
of hard-coding numbers that depend on the generator seed.
"""

from __future__ import annotations

from functools import cache

from bioeng_qpcr.sequence.fasta import SeqRecord


def _synthetic_reference(length: int = 420, seed: int = 20260917) -> str:
    """Return a deterministic, non-repetitive DNA template.

    Uses a plain linear congruential generator so the template is identical
    across Python and NumPy versions, which a seeded NumPy generator cannot
    guarantee.

    Parameters
        length:
            Number of bases to generate.
        seed:
            Generator seed.

    Returns
        The upper-case template sequence.
    """
    bases = "ACGT"
    state = seed
    out: list[str] = []
    for _ in range(length):
        state = (1103515245 * state + 12345) % (2**31)
        out.append(bases[(state >> 16) & 3])
    return "".join(out)


#: The synthetic amplicon used as the reference template.
REFERENCE: str = _synthetic_reference()


def make_reference_sequence() -> str:
    """Return the synthetic reference template as a string."""
    return REFERENCE


def make_reference_record() -> SeqRecord:
    """Return the synthetic reference as a :class:`SeqRecord`."""
    return SeqRecord(
        id="ref_gene_001", description="synthetic amplicon", sequence=REFERENCE
    )


# ---------------------------------------------------------------------------
# Uniquely-placeable variant positions
# ---------------------------------------------------------------------------


def unique_deletion_start(
    template: str, low: int, high: int, size: int
) -> int:
    """Return a deletion start whose placement cannot slide.

    Deleting ``template[p : p + size]`` yields the same optimal score at
    ``p - 1`` whenever ``template[p-1] == template[p-1+size]``, and at
    ``p + 1`` whenever ``template[p] == template[p+size]``.  Both conditions
    are excluded so the reported position is a property of the data rather
    than of the optimiser's tie-breaking.

    Parameters
        template:
            Template sequence the variant is engineered into.
        low, high:
            Half-open window to search for a safe position.
        size:
            Number of deleted bases.

    Returns
        The first safe deletion start.

    Raises
        AssertionError:
            If no safe position exists in the window.
    """
    for start in range(low, high):
        if start < 1 or start + size >= len(template):
            continue
        shifts_left = template[start - 1] == template[start - 1 + size]
        shifts_right = template[start] == template[start + size]
        if not shifts_left and not shifts_right:
            return start
    raise AssertionError(
        f"no unambiguously placeable deletion of {size} bases in [{low}, {high})"
    )


def unique_insertion(template: str, low: int, high: int) -> tuple[int, str]:
    """Return an ``(index, base)`` insertion that cannot slide.

    Inserting base ``X`` at index ``p`` is indistinguishable from inserting it
    at ``p - 1`` when ``X == template[p-1]`` and at ``p + 1`` when
    ``X == template[p]``.  Both cases are excluded.

    Parameters
        template:
            Template sequence the variant is engineered into.
        low, high:
            Half-open window to search for a safe position.

    Returns
        ``(index, base)`` where *base* differs from both flanking bases.

    Raises
        AssertionError:
            If no safe insertion exists in the window.
    """
    for index in range(low, high):
        if index < 1 or index >= len(template):
            continue
        for base in "ACGT":
            if base != template[index - 1] and base != template[index]:
                return index, base
    raise AssertionError(
        f"no unambiguously placeable insertion in [{low}, {high})"
    )


def different_base(template: str, index: int) -> str:
    """Return a base different from ``template[index]``."""
    for base in "ACGT":
        if base != template[index]:
            return base
    raise AssertionError("unreachable: no alternative base exists")


#: Read 02: single substitution at this reference position.
READ_02_SNP_POS: int = 50

#: Read 03: two substitutions at these reference positions.
READ_03_SNP_POS_A: int = 90
READ_03_SNP_POS_B: int = 95

#: Read 04: an in-frame deletion of :data:`READ_04_DEL_SIZE` bases at this
#: reference position.
READ_04_DEL_START: int = unique_deletion_start(REFERENCE, 126, 145, 3)
READ_04_DEL_SIZE: int = 3

#: Read 05: a one-base insertion of :data:`READ_05_INS_BASE` at this reference
#: position.
READ_05_INS_POS, READ_05_INS_BASE = unique_insertion(REFERENCE, 156, 172)


def make_read_set() -> list[SeqRecord]:
    """Return a small read set covering the reference with known variants.

    ``read_01`` is clean; ``read_02`` carries a substitution at
    :data:`READ_02_SNP_POS`; ``read_03`` carries two substitutions at
    :data:`READ_03_SNP_POS_A` and :data:`READ_03_SNP_POS_B`; ``read_04``
    carries a :data:`READ_04_DEL_SIZE`-base deletion at
    :data:`READ_04_DEL_START`; ``read_05`` carries a one-base insertion at
    :data:`READ_05_INS_POS`; ``read_06`` is a clean tail read.

    Every variant position is checked to be uniquely determined by the
    template, so the expected variant calls are exact rather than
    tie-break-dependent.
    """
    reads: list[SeqRecord] = []
    reads.append(SeqRecord(id="read_01_clean", sequence=REFERENCE[0:120]))

    window = list(REFERENCE[40:180])
    window[READ_02_SNP_POS - 40] = different_base(REFERENCE, READ_02_SNP_POS)
    reads.append(SeqRecord(id="read_02_snp", sequence="".join(window)))

    window = list(REFERENCE[70:210])
    window[READ_03_SNP_POS_A - 70] = different_base(REFERENCE, READ_03_SNP_POS_A)
    window[READ_03_SNP_POS_B - 70] = different_base(REFERENCE, READ_03_SNP_POS_B)
    reads.append(SeqRecord(id="read_03_snps", sequence="".join(window)))

    start = READ_04_DEL_START
    stop = start + READ_04_DEL_SIZE
    reads.append(
        SeqRecord(
            id="read_04_del3",
            sequence=REFERENCE[100:start] + REFERENCE[stop:260],
        )
    )

    inserted = list(REFERENCE[140:300])
    inserted.insert(READ_05_INS_POS - 140, READ_05_INS_BASE)
    reads.append(SeqRecord(id="read_05_ins1", sequence="".join(inserted)))

    reads.append(SeqRecord(id="read_06_clean", sequence=REFERENCE[180:300]))
    return reads


@cache
def _brute_nw(
    i: int, j: int, a: str, b: str, match: int, mismatch: int, gap: int
) -> int:
    """Brute-force global alignment score (reference for NW tests)."""
    if i == 0 and j == 0:
        return 0
    best = -10**9
    if i > 0 and j > 0:
        sc = match if a[i - 1] == b[j - 1] else mismatch
        best = max(best, _brute_nw(i - 1, j - 1, a, b, match, mismatch, gap) + sc)
    if i > 0:
        best = max(best, _brute_nw(i - 1, j, a, b, match, mismatch, gap) + gap)
    if j > 0:
        best = max(best, _brute_nw(i, j - 1, a, b, match, mismatch, gap) + gap)
    return best


def brute_force_nw(
    a: str, b: str, *, match: int = 1, mismatch: int = -1, gap: int = -1
) -> int:
    """Public wrapper around :func:`_brute_nw` for tiny sequences."""
    return _brute_nw(len(a), len(b), a, b, match, mismatch, gap)
