"""Tests for pairwise and k-mer alignment algorithms."""

from __future__ import annotations

import pytest

from bioeng_qpcr.sequence.align import (
    BLOSUM62,
    AlignmentMode,
    SubstitutionMatrix,
    align_glocal,
    custom_matrix,
    dna_matrix,
    kmer_containment,
    kmer_jaccard,
    kmer_set,
    kmer_similarity,
    needleman_wunsch,
    needleman_wunsch_affine,
    smith_waterman,
    smith_waterman_affine,
)
from tests.sequence.conftest import brute_force_nw

_NW_PAIRS = [
    ("ACGT", "ACGT"),
    ("ACGT", "A-GT"),
    ("GATTACA", "GCATGCU"),
    ("ACGT", "TGCA"),
    ("ATGCGT", "ATCGT"),
]


class TestNeedlemanWunsch:
    def test_identical_sequences_score_and_identity(self):
        res = needleman_wunsch("ACGTACGT", "ACGTACGT", match=2, gap=-1)
        assert res.score == pytest.approx(16)
        assert res.identity == pytest.approx(1.0)
        assert res.gaps == 0
        assert "-" not in res.aligned_query

    def test_matches_brute_force(self):
        for a, b in _NW_PAIRS:
            for match, mismatch, gap in [(1, -1, -1), (2, -1, -2), (1, -2, -3)]:
                got = needleman_wunsch(a, b, match=match, mismatch=mismatch, gap=gap)
                expected = brute_force_nw(a, b, match=match, mismatch=mismatch, gap=gap)
                assert got.score == pytest.approx(expected), (a, b, match, gap)

    def test_known_textbook_example(self):
        # 'GATTACA' vs 'GCATGCU' with match=+1, mismatch=-1, gap=-1.
        res = needleman_wunsch("GATTACA", "GCATGCU", match=1, mismatch=-1, gap=-1)
        expected = brute_force_nw("GATTACA", "GCATGCU")
        assert res.score == pytest.approx(expected)


class TestSmithWaterman:
    def test_local_finds_embedded_segment(self):
        res = smith_waterman("NNNACGTACGTNNN", "ACGTACGT", match=1, gap=-1)
        assert res.aligned_query == "ACGTACGT"
        assert res.score == pytest.approx(8)
        assert res.identity == pytest.approx(1.0)
        assert res.mode is AlignmentMode.LOCAL

    def test_local_excludes_flanking_noise(self):
        res = smith_waterman("XXXXACGTXXXX", "ACGT", match=1, gap=-1)
        # Only the high-scoring diagonal fragment is reported.
        assert "X" not in res.aligned_query
        assert res.aligned_query == "ACGT"


class TestGotohAffine:
    def test_no_gap_matches_linear_nw(self):
        for a, b in [("ACGT", "ACGT"), ("ACGT", "ACGA"), ("ATCG", "ATGC")]:
            affine = needleman_wunsch_affine(
                a, b, dna_matrix(1, -1), gap_open=5, gap_extend=2
            )
            # The linear API takes the per-residue gap *score*, so the
            # equivalent of gap_open=5 is gap=-5.
            linear = needleman_wunsch(a, b, match=1, mismatch=-1, gap=-5)
            assert affine.score == pytest.approx(linear.score)

    def test_single_gap_costs_gap_open_only(self):
        # One extra residue in the reference => a single gap in the query.
        res = needleman_wunsch_affine(
            "ACGT", "ACGTT", dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        assert res.score == pytest.approx(4 - 5)  # 4 matches - gap_open
        assert res.gap_opens == 1
        assert res.gap_extends == 0

    def test_three_base_gap_is_open_plus_two_extends(self):
        res = needleman_wunsch_affine(
            "ACGT", "ACGTTTT", dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        # 4 matches - (gap_open + 2 * gap_extend) = 4 - 9 = -5.
        assert res.score == pytest.approx(4 - (5 + 2 * 2))
        assert res.gap_opens == 1

    def test_local_affine_matches_linear_sw_without_gaps(self):
        a, b = "NNNACGTACGTNNN", "ACGTACGT"
        affine = smith_waterman_affine(
            a, b, dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        linear = smith_waterman(a, b, match=1, gap=-1)
        assert affine.score == pytest.approx(linear.score)


class TestGlocal:
    def test_substring_maps_without_end_gaps(self):
        reference = "ACGTACGTACGT"
        query = "GTACGT"  # reference[2:8]
        res = align_glocal(
            query, reference, dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        assert res.identity == pytest.approx(1.0)
        assert res.aligned_query == "GTACGT"  # no query-end gaps
        assert res.ref_start == 2
        assert res.ref_end == 8
        assert res.mode is AlignmentMode.GLOCAL

    def test_glocal_reports_mapped_region_only(self):
        reference = "AAAA" + "CCCC" + "GGGG"
        query = "CCCC"
        res = align_glocal(
            query, reference, dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        assert res.ref_start == 4
        assert res.ref_end == 8
        assert res.identity == pytest.approx(1.0)


class TestSubstitutionMatrices:
    def test_blosum62_known_values(self):
        assert BLOSUM62.score("W", "W") == 11
        assert BLOSUM62.score("C", "C") == 9
        assert BLOSUM62.score("A", "A") == 4
        assert BLOSUM62.score("A", "W") == -3
        assert BLOSUM62.score("G", "G") == 6
        # Symmetry.
        assert BLOSUM62.score("W", "A") == BLOSUM62.score("A", "W")

    def test_blosum62_does_not_penalise_identical_residues(self):
        for base in "ACDEFGHIKLMNPQRSTVWY":
            assert BLOSUM62.score(base, base) >= 4

    def test_custom_matrix(self):
        table = {("A", "A"): 2, ("A", "T"): -3}
        matrix = custom_matrix(table, default=-1)
        assert matrix.score("A", "A") == 2
        assert matrix.score("A", "T") == -3
        assert matrix.score("T", "A") == -3  # symmetric lookup
        assert matrix.score("G", "C") == -1  # default


class TestKmerSimilarity:
    def test_identical_sequences_jaccard_one(self):
        assert kmer_jaccard("ACGT" * 10, "ACGT" * 10, k=4) == pytest.approx(1.0)

    def test_disjoint_sequences_near_zero(self):
        a = "AC" * 20
        b = "GT" * 20
        assert kmer_jaccard(a, b, k=4) == pytest.approx(0.0)
        assert kmer_similarity(a, b, k=4, method="jaccard") == pytest.approx(0.0)

    def test_substring_containment(self):
        # A non-repetitive template is required: with a perfectly periodic
        # sequence such as "ACGT"*10 every k-mer of the prefix also occurs
        # in the suffix, so the Jaccard coefficient legitimately equals 1.
        full = "ACGTTGCAAGTCCTAGGCATCGATCGTTACGGATCCAAGCTTAGCGATCGATCGTAGCATGCA"
        sub = full[:30]
        assert kmer_containment(sub, full, k=4) == pytest.approx(1.0)
        assert kmer_jaccard(sub, full, k=4) < 1.0
        assert kmer_jaccard(sub, full, k=4) > 0.0

    def test_kmer_set_size(self):
        assert len(kmer_set("ACGT", 2)) == 3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown k-mer method"):
            kmer_similarity("ACGT", "ACGT", method="bogus")


class TestMemoryGuard:
    def test_large_alignment_raises(self):
        with pytest.raises(ValueError, match="memory budget"):
            needleman_wunsch("A" * 6000, "A" * 6000)


class TestResultContainer:
    def test_cigar_and_counts(self):
        res = needleman_wunsch("ACGT", "ACGT", match=1, gap=-1)
        assert res.cigar == "4M"
        assert res.matches == 4
        assert res.mismatches == 0

    def test_dna_matrix_helper(self):
        m = dna_matrix(2, -1)
        assert isinstance(m, SubstitutionMatrix)
        assert m.score("A", "A") == 2
        assert m.score("A", "T") == -1
