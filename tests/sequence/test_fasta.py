"""Tests for FASTA/FASTQ/plain sequence I/O and statistics."""

from __future__ import annotations

import dataclasses
import struct

import pytest

from bioeng_qpcr.sequence.fasta import (
    SeqRecord,
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


class TestReadFasta:
    def test_multi_record_and_multiline(self, tmp_path):
        path = tmp_path / "multi.fasta"
        path.write_text(
            ">seq1 description one\nACGT\nACGT\n>seq2\nTTTT\n",
            encoding="utf-8",
        )
        records = read_fasta(path)
        assert len(records) == 2
        assert records[0].id == "seq1"
        assert records[0].description == "description one"
        assert records[0].sequence == "ACGTACGT"
        assert records[1].sequence == "TTTT"

    def test_lowercase_and_whitespace_cleaned(self, tmp_path):
        path = tmp_path / "case.fasta"
        path.write_text(">s\na c g t\n", encoding="utf-8")
        record = read_fasta(path)[0]
        assert record.sequence == "ACGT"

    def test_comment_lines_ignored(self, tmp_path):
        path = tmp_path / "comment.fasta"
        path.write_text("; header comment\n>s\nACGT\n", encoding="utf-8")
        records = read_fasta(path)
        assert len(records) == 1
        assert records[0].sequence == "ACGT"

    def test_windows_line_endings(self, tmp_path):
        path = tmp_path / "win.fasta"
        path.write_bytes(b">s\r\nACGT\r\nTTTT\r\n")
        records = read_fasta(path)
        assert records[0].sequence == "ACGTTTTT"

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.fasta"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no '>' header"):
            read_fasta(path)


class TestReadFastq:
    def test_quality_decoded_phred33(self, tmp_path):
        # 'I' = 40, '!' = 0
        path = tmp_path / "r.fastq"
        path.write_text("@r\nAC\n+\nI!\n", encoding="utf-8")
        record = read_fastq(path)[0]
        assert record.sequence == "AC"
        assert record.quality == [40, 0]

    def test_quality_length_mismatch_raises(self, tmp_path):
        path = tmp_path / "bad.fastq"
        path.write_text("@r\nACGT\n+\nII\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sequence length"):
            read_fastq(path)

    def test_multiline_sequence_accepted(self, tmp_path):
        path = tmp_path / "ml.fastq"
        path.write_text("@r\nAC\nGT\n+\nII\nII\n", encoding="utf-8")
        record = read_fastq(path)[0]
        assert record.sequence == "ACGT"
        assert record.quality == [40, 40, 40, 40]


class TestRoundTrip:
    def test_fasta_roundtrip(self, tmp_path):
        records = [
            SeqRecord(id="a", description="d", sequence="ACGTACGT"),
            SeqRecord(id="b", description="", sequence="TTTTGGGG"),
        ]
        path = tmp_path / "rt.fasta"
        write_fasta(records, path, width=4)
        back = read_fasta(path)
        assert [r.sequence for r in back] == ["ACGTACGT", "TTTTGGGG"]
        assert back[0].description == "d"

    def test_fastq_roundtrip(self, tmp_path):
        records = [
            SeqRecord(id="a", sequence="ACGT", quality=[35, 36, 37, 38]),
        ]
        path = tmp_path / "rt.fastq"
        write_fastq(records, path)
        back = read_fastq(path)
        assert back[0].sequence == "ACGT"
        assert back[0].quality == [35, 36, 37, 38]

    def test_write_fastq_rejects_missing_quality(self, tmp_path):
        record = SeqRecord(id="a", sequence="ACGT")
        with pytest.raises(ValueError, match="no quality"):
            write_fastq([record], tmp_path / "x.fastq")


class TestPlainSequence:
    def test_plain_ignores_comments(self, tmp_path):
        path = tmp_path / "plain.txt"
        path.write_text(
            "; ABI export\n# sample\n>ignored\nACGT\nTTTT\n", encoding="utf-8"
        )
        records = read_plain_sequence(path)
        assert len(records) == 1
        assert records[0].sequence == "ACGTTTTT"


class TestFormatDetection:
    def test_detect_fasta_fastq_plain(self, tmp_path):
        fa = tmp_path / "a.fasta"
        fa.write_text(">s\nACGT\n", encoding="utf-8")
        fq = tmp_path / "b.fastq"
        fq.write_text("@s\nACGT\n+\nIIII\n", encoding="utf-8")
        pl = tmp_path / "c.txt"
        pl.write_text("ACGT\nTTTT\n", encoding="utf-8")
        assert detect_format(fa) == "fasta"
        assert detect_format(fq) == "fastq"
        assert detect_format(pl) == "plain"

    def test_read_sequence_file_dispatches(self, tmp_path):
        path = tmp_path / "a.fasta"
        path.write_text(">s\nACGT\n", encoding="utf-8")
        records = read_sequence_file(path)
        assert records[0].sequence == "ACGT"

    def test_ab1_base_calls_and_quality_dispatch(self, tmp_path):
        sequence = b"ACGTN"
        quality = bytes([40, 39, 10, 30, 20])
        directory_offset = 34
        payload_offset = directory_offset + 2 * 28
        pack = struct.pack
        root = pack(
            ">4sIHHIIII", b"tdir", 1, 1023, 28, 2, 56, directory_offset, 0
        )
        pbas = pack(
            ">4sIHHIIII", b"PBAS", 2, 2, 1, 5, 5, payload_offset, 0
        )
        pcon = pack(
            ">4sIHHIIII", b"PCON", 2, 1, 1, 5, 5, payload_offset + 5, 0
        )
        path = tmp_path / "sample.ab1"
        path.write_bytes(
            b"ABIF" + pack(">H", 101) + root + pbas + pcon + sequence + quality
        )

        assert detect_format(path) == "ab1"
        assert read_ab1(path)[0].quality == [40, 39, 10, 30, 20]
        record = read_sequence_file(path)[0]
        assert record.sequence == "ACGTN"
        assert record.annotations["abif_version"] == "1.01"


class TestCleaningAndStats:
    def test_clean_sequence(self):
        assert clean_sequence("a 1c*gT 9") == "ACGT"

    def test_identify_sequence_type(self):
        assert identify_sequence_type("ACGTACGT") is SequenceType.DNA
        assert identify_sequence_type("ACGUACGU") is SequenceType.RNA
        assert identify_sequence_type("MKTAYIA") is SequenceType.PROTEIN
        assert identify_sequence_type("") is SequenceType.UNKNOWN

    def test_sequence_statistics(self):
        stats = sequence_statistics("ACGTACGTNN")
        assert stats.length == 10
        assert stats.gc_fraction == pytest.approx(0.5)
        assert stats.n_fraction == pytest.approx(0.2)
        assert stats.composition["A"] == 2

    def test_seqrecord_is_frozen(self):
        record = SeqRecord(id="a", description="", sequence="ACGT")
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.sequence = "TTTT"
