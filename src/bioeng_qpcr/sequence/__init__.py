"""Sequence analysis: reading, alignment and sequencing comparison.

This package adds self-contained, dependency-free sequencing-comparison
capabilities to bioeng-qpcr:

* :mod:`bioeng_qpcr.sequence.fasta` -- FASTA/FASTQ/plain I/O, format
  auto-detection, sequence cleaning and statistics.
* :mod:`bioeng_qpcr.sequence.align` -- Needleman-Wunsch, Smith-Waterman, Gotoh
  affine (global/local) and glocal alignment plus BLOSUM62 and k-mer
  similarity.
* :mod:`bioeng_qpcr.sequence.compare` -- read/reference comparison, primer and
  quality trimming, variant detection, consensus, MSA and summary statistics.
* :mod:`bioeng_qpcr.sequence.validation` -- the SEQ-00x audit rule battery.
"""

from __future__ import annotations

from bioeng_qpcr.sequence.align import (
    BLOSUM62,
    BLOSUM62_TABLE,
    AlignmentMode,
    AlignmentResult,
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
from bioeng_qpcr.sequence.compare import (
    AlignmentSummary,
    ConsensusResult,
    MSAResult,
    PrimerTrim,
    QualityTrim,
    Variant,
    VariantType,
    build_consensus,
    compare_to_reference,
    detect_variants,
    multiple_sequence_alignment,
    reverse_complement,
    summarize_alignment,
    summarize_sequencing_quality,
    trim_low_quality,
    trim_primers,
)
from bioeng_qpcr.sequence.fasta import (
    SeqRecord,
    SequenceStatistics,
    SequenceType,
    clean_sequence,
    detect_format,
    identify_sequence_type,
    read_ab1,
    read_fasta,
    read_fastq,
    read_plain_sequence,
    read_sequence_file,
    sequence_statistics,
    write_fasta,
    write_fastq,
)
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

__all__ = [
    # align
    "AlignmentMode",
    "AlignmentResult",
    "BLOSUM62",
    "BLOSUM62_TABLE",
    "SubstitutionMatrix",
    "align_glocal",
    "custom_matrix",
    "dna_matrix",
    "kmer_containment",
    "kmer_jaccard",
    "kmer_set",
    "kmer_similarity",
    "needleman_wunsch",
    "needleman_wunsch_affine",
    "smith_waterman",
    "smith_waterman_affine",
    # compare
    "AlignmentSummary",
    "ConsensusResult",
    "MSAResult",
    "PrimerTrim",
    "QualityTrim",
    "Variant",
    "VariantType",
    "build_consensus",
    "compare_to_reference",
    "detect_variants",
    "multiple_sequence_alignment",
    "reverse_complement",
    "summarize_alignment",
    "summarize_sequencing_quality",
    "trim_low_quality",
    "trim_primers",
    # fasta
    "SeqRecord",
    "SequenceStatistics",
    "SequenceType",
    "clean_sequence",
    "detect_format",
    "identify_sequence_type",
    "read_ab1",
    "read_fasta",
    "read_fastq",
    "read_plain_sequence",
    "read_sequence_file",
    "sequence_statistics",
    "write_fasta",
    "write_fastq",
    # validation
    "SEQ_RULES",
    "SequenceValidationResult",
    "Severity",
    "ValidationReport",
    "check_ambiguous_bases",
    "check_coverage",
    "check_frameshift",
    "check_gc_content",
    "check_identity",
    "check_low_quality_fraction",
    "check_primers_excised",
    "check_read_length",
    "validate_sequencing_result",
]
