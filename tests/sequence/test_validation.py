"""Tests for the SEQ-00x sequencing-validation rule battery."""

from __future__ import annotations

from bioeng_qpcr.sequence.compare import (
    PrimerTrim,
    Variant,
    VariantType,
    compare_to_reference,
    detect_variants,
)
from bioeng_qpcr.sequence.fasta import SeqRecord
from bioeng_qpcr.sequence.validation import (
    SEQ_RULES,
    SequenceValidationResult,
    Severity,
    ValidationReport,
    check_ambiguous_bases,
    check_coverage,
    check_frameshift,
    check_gc_content,
    check_identity,
    check_low_quality_fraction,
    check_primers_excised,
    check_read_length,
    validate_sequencing_result,
)
from tests.sequence.conftest import make_read_set, make_reference_sequence


class TestIdentityRule:
    def test_low_identity_fails(self):
        reference = "ACGTACGTACGT"
        query = "TTTTTTTTTTTT"
        aln = compare_to_reference(query, reference)
        result = check_identity(aln, threshold=0.9)
        assert isinstance(result, SequenceValidationResult)
        assert not result.passed
        assert result.severity is Severity.ERROR

    def test_high_identity_passes(self):
        reference = "ACGTACGTACGT"
        aln = compare_to_reference(reference, reference)
        assert check_identity(aln, threshold=0.9).passed


class TestReadLengthRule:
    def test_short_read_fails(self):
        assert not check_read_length("A" * 10, min_length=50).passed

    def test_long_read_passes(self):
        assert check_read_length("A" * 60, min_length=50).passed


class TestCoverageRule:
    def test_partial_coverage_warns(self):
        reference = make_reference_sequence()
        short_read = "ACGT" * 5  # 20 bp against a 430 bp reference
        aln = compare_to_reference(short_read, reference)
        result = check_coverage(aln, reference, min_fraction=0.8)
        assert not result.passed
        assert result.severity is Severity.WARNING

    def test_full_coverage_passes(self):
        reference = "ACGTACGT"
        aln = compare_to_reference(reference, reference)
        assert check_coverage(aln, reference, min_fraction=0.8).passed


class TestLowQualityRule:
    def test_many_low_bases_fails(self):
        record = SeqRecord(id="r", sequence="A" * 20, quality=[10] * 10 + [40] * 10)
        result = check_low_quality_fraction(record, max_fraction=0.2, min_q=20)
        assert not result.passed

    def test_clean_read_passes(self):
        record = SeqRecord(id="r", sequence="A" * 20, quality=[40] * 20)
        assert check_low_quality_fraction(record, max_fraction=0.2, min_q=20).passed

    def test_missing_quality_is_info(self):
        record = SeqRecord(id="r", sequence="A" * 20)
        assert check_low_quality_fraction(record).passed
        assert check_low_quality_fraction(record).severity is Severity.INFO


class TestFrameshiftRule:
    def test_single_base_indel_fails_for_coding(self):
        variant = Variant(
            type=VariantType.INSERTION,
            ref_position=10,
            ref_allele="",
            alt_allele="A",
            length=1,
        )
        result = check_frameshift([variant], coding=True)
        assert not result.passed
        assert result.severity is Severity.ERROR

    def test_in_frame_deletion_passes(self):
        variant = Variant(
            type=VariantType.DELETION,
            ref_position=10,
            ref_allele="ACG",
            alt_allele="",
            length=3,
        )
        assert check_frameshift([variant], coding=True).passed

    def test_non_coding_is_info(self):
        variant = Variant(
            type=VariantType.INSERTION,
            ref_position=10,
            ref_allele="",
            alt_allele="A",
            length=1,
        )
        assert check_frameshift([variant], coding=False).passed

    def test_real_frameshift_read_detected(self):
        reference = make_reference_sequence()
        reads = make_read_set()
        aln = compare_to_reference(reads[4], reference)  # 1-bp insertion
        variants = detect_variants(aln, reference)
        result = check_frameshift(variants, coding=True)
        assert not result.passed


class TestAmbiguousBasesRule:
    def test_high_n_fraction_fails(self):
        record = SeqRecord(id="r", sequence="N" * 10 + "A" * 10)
        assert not check_ambiguous_bases(record, max_fraction=0.05).passed

    def test_clean_sequence_passes(self):
        assert check_ambiguous_bases("A" * 20).passed


class TestGcContentRule:
    def test_extreme_gc_fails(self):
        assert not check_gc_content("A" * 20, min_gc=0.2, max_gc=0.8).passed

    def test_moderate_gc_passes(self):
        assert check_gc_content("ACGTACGTAC", min_gc=0.2, max_gc=0.8).passed


class TestPrimerExcisionRule:
    def test_unexcised_primer_fails(self):
        trim = PrimerTrim(
            trimmed="ACGT",
            n_trimmed_5=0,
            n_trimmed_3=0,
            found_5=False,
            found_3=True,
            kept_start=0,
            kept_end=4,
        )
        result = check_primers_excised(trim, primer_5="ACGT")
        assert not result.passed
        assert result.severity is Severity.WARNING

    def test_excised_primers_pass(self):
        trim = PrimerTrim(
            trimmed="ACGT",
            n_trimmed_5=4,
            n_trimmed_3=0,
            found_5=True,
            found_3=True,
            kept_start=4,
            kept_end=4,
        )
        assert check_primers_excised(trim, primer_5="ACGT").passed


class TestValidationBattery:
    def test_clean_read_passes_battery(self):
        reference = make_reference_sequence()
        reads = make_read_set()
        aln = compare_to_reference(reads[0], reference)  # clean substring
        report = validate_sequencing_result(
            aln, reads[0], reference, threshold_identity=0.9
        )
        assert isinstance(report, ValidationReport)
        assert report.passed  # only ERROR-level failures block the report
        assert report.n_errors == 0

    def test_bad_read_fails_battery(self):
        reference = make_reference_sequence()
        query = "T" * 10  # short and unrelated
        aln = compare_to_reference(query, reference)
        report = validate_sequencing_result(aln, query, reference)
        assert not report.passed
        assert report.n_errors >= 1

    def test_rule_catalogue_is_complete(self):
        assert "SEQ-001" in SEQ_RULES
        assert len(SEQ_RULES) == 8
