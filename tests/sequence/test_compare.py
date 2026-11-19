"""Tests for sequencing-comparison workflows (compare.py)."""

from __future__ import annotations

import pytest

from bioeng_qpcr.sequence.align import (
    align_glocal,
    dna_matrix,
    needleman_wunsch,
    needleman_wunsch_affine,
    smith_waterman,
    smith_waterman_affine,
)
from bioeng_qpcr.sequence.compare import (
    AlignmentSummary,
    ConsensusResult,
    MSAResult,
    PrimerTrim,
    QualityTrim,
    VariantType,
    build_consensus,
    compare_to_reference,
    detect_variants,
    multiple_sequence_alignment,
    summarize_alignment,
    summarize_sequencing_quality,
    trim_low_quality,
    trim_primers,
)
from bioeng_qpcr.sequence.fasta import SeqRecord
from tests.sequence.conftest import (
    READ_02_SNP_POS,
    READ_04_DEL_SIZE,
    READ_04_DEL_START,
    READ_05_INS_BASE,
    READ_05_INS_POS,
    different_base,
    make_read_set,
    make_reference_sequence,
    unique_deletion_start,
    unique_insertion,
)


class TestCompareToReference:
    def test_reverse_complement_read_is_oriented_automatically(self):
        reference = "AACCGTTAAGTC"
        reverse_read = "GACTTAACGGTT"
        result = compare_to_reference(reverse_read, reference)
        assert result.identity == pytest.approx(1.0)
        assert result.query_orientation == "reverse_complement"

    def test_reverse_complement_reverses_quality_for_variant_calls(self):
        reference = "AACCGTTAAGTC"
        reverse_read = "GACTTAACAGTT"  # one mismatch after orientation
        result = compare_to_reference(reverse_read, reference)
        variants = detect_variants(
            result, reference, query_quality=list(range(30, 42))
        )
        assert result.query_orientation == "reverse_complement"
        assert len(variants) == 1
        assert variants[0].quality == 38

    def test_substring_maps_with_high_identity(self):
        reference = make_reference_sequence()
        reads = make_read_set()
        res = compare_to_reference(reads[0], reference)  # read_01_clean
        assert res.identity == pytest.approx(1.0)
        assert res.ref_start == 0
        assert res.ref_end == 120

    def test_returns_glocal_result(self):
        res = compare_to_reference("GTACGT", "ACGTACGTACGT")
        assert res.mode.value == "glocal"


class TestTrimPrimers:
    def test_exact_primers_trimmed(self):
        query = "ACGTACGTACTGCA"
        trim = trim_primers(query, primer_5="ACGT", primer_3="TGCA")
        assert isinstance(trim, PrimerTrim)
        assert trim.trimmed == "ACGTAC"
        assert trim.n_trimmed_5 == 4
        assert trim.n_trimmed_3 == 4
        assert trim.found_5 and trim.found_3

    def test_mismatched_primers_still_found(self):
        # 5' primer "ACGA" has one mismatch vs the read prefix "ACGT".
        query = "ACGTACGTACTGCA"
        trim = trim_primers(query, primer_5="ACGA", primer_3="TGCA", max_mismatch=2)
        assert trim.found_5
        assert trim.trimmed == "ACGTAC"

    def test_no_primers_raises(self):
        with pytest.raises(ValueError, match="At least one primer"):
            trim_primers("ACGT", primer_5=None, primer_3=None)


class TestTrimLowQuality:
    def test_high_quality_read_passes_through(self):
        seq = "A" * 40
        quality = [40] * 40
        trim = trim_low_quality(seq, quality, window=10, min_q=20)
        assert isinstance(trim, QualityTrim)
        assert trim.trimmed == seq
        assert trim.n_trimmed_5 == 0
        assert trim.n_trimmed_3 == 0

    def test_low_quality_tail_is_trimmed(self):
        seq = "A" * 40
        quality = [40] * 15 + [2] * 25
        trim = trim_low_quality(seq, quality, window=10, min_q=20)
        assert trim.n_trimmed_3 > 0
        assert len(trim.trimmed) < len(seq)
        # Quality of the retained region is improved versus the input.
        mean_in = sum(quality) / len(quality)
        mean_out = sum(trim.trimmed and [quality[i] for i in range(
            trim.kept_start, trim.kept_end)]) / (trim.kept_end - trim.kept_start)
        assert mean_out > mean_in

    def test_fully_low_read_becomes_empty(self):
        seq = "A" * 20
        quality = [2] * 20
        trim = trim_low_quality(seq, quality, window=10, min_q=20)
        assert trim.trimmed == ""

    def test_requires_quality(self):
        with pytest.raises(ValueError, match="Phred quality"):
            trim_low_quality("ACGT", None)


class TestDetectVariants:
    """Variant calling against a non-repetitive template.

    Each engineered variant is placed at a position that is the *unique*
    optimum for its alignment (see ``tests.sequence.conftest``), so the
    assertions below test the caller rather than the tie-breaking order of
    the dynamic program.
    """

    #: A short, non-repetitive template to engineer variants into.
    TEMPLATE = make_reference_sequence()[8:38]

    def _reference(self) -> str:
        return self.TEMPLATE

    def test_single_snp_called(self):
        reference = self._reference()
        position = 7
        alt = different_base(reference, position)
        query = reference[:position] + alt + reference[position + 1 :]
        aln = compare_to_reference(query, reference)
        variants = detect_variants(aln, reference)
        snps = [v for v in variants if v.type is VariantType.SNP]
        assert len(snps) == 1
        v = snps[0]
        assert v.ref_position == position
        assert v.ref_allele == reference[position]
        assert v.alt_allele == alt

    def test_snp_reports_quality_and_depth(self):
        reference = self._reference()
        position = 7
        query = (
            reference[:position]
            + different_base(reference, position)
            + reference[position + 1 :]
        )
        quality = [40] * len(query)
        coverage = {i: 3 for i in range(len(reference))}
        aln = compare_to_reference(query, reference)
        variants = detect_variants(
            aln, reference, query_quality=quality, coverage=coverage
        )
        assert len(variants) == 1
        assert variants[0].quality == 40
        assert variants[0].depth == 3

    def test_deletion_called(self):
        reference = self._reference()
        start = unique_deletion_start(reference, 10, 20, 3)
        query = reference[:start] + reference[start + 3 :]
        aln = compare_to_reference(query, reference)
        variants = detect_variants(aln, reference)
        dels = [v for v in variants if v.type is VariantType.DELETION]
        assert len(dels) == 1
        assert dels[0].ref_position == start
        assert dels[0].ref_allele == reference[start : start + 3]
        assert dels[0].alt_allele == ""
        assert dels[0].length == 3

    def test_insertion_called(self):
        reference = self._reference()
        position, base = unique_insertion(reference, 12, 22)
        query = reference[:position] + base + reference[position:]
        aln = compare_to_reference(query, reference)
        variants = detect_variants(aln, reference)
        ins = [v for v in variants if v.type is VariantType.INSERTION]
        assert len(ins) == 1
        assert ins[0].ref_position == position
        assert ins[0].alt_allele == base

    def test_mnp_called(self):
        reference = self._reference()
        position = 4
        replacements = "".join(
            different_base(reference, index)
            for index in (position, position + 1)
        )
        query = reference[:position] + replacements + reference[position + 2 :]
        aln = compare_to_reference(query, reference)
        variants = detect_variants(aln, reference)
        mnps = [v for v in variants if v.type is VariantType.MNP]
        assert len(mnps) == 1
        assert mnps[0].ref_allele == reference[position : position + 2]
        assert mnps[0].alt_allele == replacements
        assert mnps[0].length == 2

    def test_variants_on_real_read_set(self):
        reference = make_reference_sequence()
        reads = make_read_set()

        aln = compare_to_reference(reads[1], reference)
        variants = detect_variants(aln, reference)
        assert any(
            v.type is VariantType.SNP and v.ref_position == READ_02_SNP_POS
            for v in variants
        )

        aln = compare_to_reference(reads[3], reference)
        variants = detect_variants(aln, reference)
        assert any(
            v.type is VariantType.DELETION
            and v.ref_position == READ_04_DEL_START
            and v.length == READ_04_DEL_SIZE
            for v in variants
        )

        aln = compare_to_reference(reads[4], reference)
        variants = detect_variants(aln, reference)
        assert any(
            v.type is VariantType.INSERTION
            and v.ref_position == READ_05_INS_POS
            and v.alt_allele == READ_05_INS_BASE
            for v in variants
        )

    def test_clean_read_produces_no_variants(self):
        reference = make_reference_sequence()
        reads = make_read_set()
        aln = compare_to_reference(reads[0], reference)
        assert detect_variants(aln, reference) == []


class TestAlignmentInvariants:
    """Regression tests for the alignment rows themselves.

    The dynamic-programming score can be correct while the traceback emits
    rows that are not gapped versions of the inputs (the two gap types get
    swapped, or a leading residue is dropped).  These tests pin the
    invariant that every public aligner must satisfy.
    """

    PAIRS = [
        ("ACGT", "ACGTT"),
        ("ACGTACGT", "ACGT"),
        ("GATTACA", "GCATGCU"),
        ("ATGCGTAC", "ATGCGTA"),
        ("ACGTACGTAA", "ACGTAA"),
        ("TTTTGGGG", "GGGG"),
    ]

    def test_global_rows_reconstruct_the_inputs(self):
        for query, reference in self.PAIRS:
            linear = needleman_wunsch(query, reference, match=2, mismatch=-1, gap=-2)
            affine = needleman_wunsch_affine(
                query, reference, dna_matrix(2, -1), gap_open=5, gap_extend=2
            )
            for result, label in ((linear, "linear"), (affine, "affine")):
                assert result.aligned_query.replace("-", "") == query, (
                    label,
                    query,
                    reference,
                )
                assert result.aligned_reference.replace("-", "") == reference, (
                    label,
                    query,
                    reference,
                )

    def test_local_rows_are_substrings_of_the_inputs(self):
        query, reference = "NNNACGTACGTNNN", "ACGTACGT"
        linear = smith_waterman(query, reference, match=1, gap=-1)
        affine = smith_waterman_affine(
            query, reference, dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        for result in (linear, affine):
            assert result.aligned_query.replace("-", "") in query
            assert result.aligned_reference.replace("-", "") in reference

    def test_glocal_rows_reconstruct_the_covered_region(self):
        reference = make_reference_sequence()
        query = reference[40:180]
        result = align_glocal(
            query, reference, dna_matrix(1, -1), gap_open=5, gap_extend=2
        )
        assert result.aligned_query.replace("-", "") == query
        assert (
            result.aligned_reference.replace("-", "")
            == reference[result.ref_start : result.ref_end]
        )

    def test_multiple_sequence_alignment_preserves_every_residue(self):
        sequences = ["ACGTACGT", "ACGTACGTT", "ACGTTACGT", "ACGTACGT"]
        msa = multiple_sequence_alignment(sequences)
        assert len({len(row) for row in msa.rows}) == 1
        for row, original in zip(msa.rows, sequences, strict=True):
            assert row.replace("-", "") == original


class TestBuildConsensus:
    def test_simple_majority_keeps_reference(self):
        reference = "ACGTACGT"
        reads = ["ACGTACGT", "ACGTATGT"]  # one read carries a SNP at pos 5
        consensus = build_consensus(reads, reference, method="simple_majority")
        assert isinstance(consensus, ConsensusResult)
        assert consensus.consensus == reference  # C wins 2:1
        assert consensus.n_ambiguous == 0

    def test_threshold_below_limit_emits_degenerate_code(self):
        reference = "ACGTACGT"
        reads = ["ACGTACGT", "ACGTATGT"]
        # Position 5 sees C (reference + one read) against T (one read):
        # the leading base holds 2 of 3 observations = 0.667 < 0.7, so the
        # site is ambiguous and resolves to the C/T degenerate code "Y".
        consensus = build_consensus(
            reads, reference, method="threshold", threshold=0.7
        )
        assert consensus.consensus[5] == "Y"
        assert consensus.n_ambiguous == 1

    def test_four_way_disagreement_becomes_N(self):
        reference = "ACGT"
        # Position 1 sees C (reference + one read) plus G, T and A, so no
        # two-base code can describe the site and it resolves to "N".
        reads = ["AGGT", "ACTT", "ATGT", "AATT"]
        consensus = build_consensus(
            reads, reference, method="threshold", threshold=0.7
        )
        assert consensus.consensus[1] == "N"
        # Position 2 is a pure G/T split -> "K".
        assert consensus.consensus[2] == "K"
        assert consensus.n_ambiguous == 2

    def test_iupac_degenerate_code(self):
        reference = "ACGTACGT"
        reads = ["ACGTACGT", "ACGTTCGT"]  # A/T disagreement at pos 4
        consensus = build_consensus(
            reads, reference, method="threshold", threshold=0.7
        )
        assert consensus.consensus[4] == "W"  # A or T

    def test_empty_reads_raises(self):
        with pytest.raises(ValueError, match="At least one read"):
            build_consensus([], "ACGT")


class TestMultipleSequenceAlignment:
    def test_rows_equal_length_and_conserved_columns(self):
        seqs = [
            "ACGTACGTACGTACGT",
            "ACGTACGTTCGTACGT",
            "ACGTACGTTCGTACCT",
            "ACGTACGTACGTACGT",
        ]
        msa = multiple_sequence_alignment(seqs)
        assert isinstance(msa, MSAResult)
        assert msa.n_sequences == 4
        assert len({len(r) for r in msa.rows}) == 1  # all equal length
        summary = summarize_alignment(msa.rows)
        assert summary.length == len(msa.rows[0])
        assert summary.variable_sites  # SNPs introduced variation

    def test_insertion_produces_gaps_and_equal_rows(self):
        seqs = ["ACGTACGT", "ACGTACGTT"]  # second has a trailing insertion
        msa = multiple_sequence_alignment(seqs)
        assert len({len(r) for r in msa.rows}) == 1
        assert any("-" in r for r in msa.rows)
        # The longer sequence is the anchor, so the *shorter* one carries the
        # gap and the anchor keeps all nine of its residues.
        assert msa.rows[0] == "ACGTACGT-"
        assert msa.rows[1] == "ACGTACGTT"

    def test_single_sequence_returns_itself(self):
        msa = multiple_sequence_alignment(["ACGTACGT"])
        assert msa.rows == ["ACGTACGT"]


class TestAlignmentSummary:
    def test_identity_and_entropy(self):
        rows = ["ACGTACGT", "ACGTACGT", "ACGTTCGT"]
        summary = summarize_alignment(rows)
        assert isinstance(summary, AlignmentSummary)
        assert summary.overall_identity > 0.8
        assert summary.pairwise_identity[0][1] == pytest.approx(1.0)
        # Variable site at the SNP column.
        assert summary.variable_sites
        # Conservation is 1.0 at a fully conserved column.
        assert max(summary.conservation_per_column) == pytest.approx(1.0)

    def test_requires_equal_length_rows(self):
        with pytest.raises(ValueError, match="same length"):
            summarize_alignment(["ACGT", "ACG"])


class TestSequencingQualitySummary:
    def test_length_and_low_quality_fraction(self):
        records = [
            SeqRecord(id="a", sequence="A" * 20, quality=[40] * 20),
            SeqRecord(id="b", sequence="A" * 20, quality=[10] * 20),
        ]
        summary = summarize_sequencing_quality(records, min_q=20)
        assert summary.n_reads == 2
        assert summary.mean_length == pytest.approx(20)
        assert summary.low_quality_read_fraction == pytest.approx(0.5)
        assert summary.read_lengths == [20, 20]

    def test_empty_read_set_raises(self):
        with pytest.raises(ValueError, match="empty read set"):
            summarize_sequencing_quality([])
