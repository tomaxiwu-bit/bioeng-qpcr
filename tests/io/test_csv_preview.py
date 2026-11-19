"""Tests for the CSV preview engine and the import-format registry."""

from __future__ import annotations

import pytest

from bioeng_qpcr.io.csv_preview import (
    SCHEMAS,
    ColumnSpec,
    CsvSchema,
    detect_schema,
    import_spec_markdown,
    list_schemas,
    normalise_header,
    preview_csv,
    read_csv_table,
    schema_for,
    sniff_delimiter,
    sniff_encoding,
    validate_rows_against_schema,
    write_template_csv,
)


def _write(tmp_path, name: str, text: str, encoding: str = "utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


class TestSniffing:
    def test_utf8_is_detected_without_fallback(self):
        encoding, fell_back = sniff_encoding(b"compound,response\nA,1\n")
        assert encoding == "utf-8"
        assert not fell_back

    def test_utf8_bom_is_detected(self):
        encoding, _ = sniff_encoding("a,b\n1,2\n".encode("utf-8-sig"))
        assert encoding == "utf-8-sig"

    def test_gb18030_falls_back_and_flags_it(self):
        # A Chinese-Excel export: valid GB18030, invalid UTF-8.
        encoding, fell_back = sniff_encoding("样本,响应\n对照,1\n".encode("gb18030"))
        assert encoding == "gb18030"
        assert fell_back

    def test_comma_delimiter(self):
        assert sniff_delimiter("a,b,c\n1,2,3\n4,5,6\n") == ","

    def test_tab_delimiter(self):
        assert sniff_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"

    def test_semicolon_delimiter(self):
        assert sniff_delimiter("a;b;c\n1;2;3\n") == ";"

    def test_free_text_commas_do_not_beat_the_real_delimiter(self):
        # Semicolon-separated file whose last column contains commas.
        text = "a;b;note\n1;2;x, y, z\n3;4;p, q, r\n5;6;s, t, u\n"
        assert sniff_delimiter(text) == ";"


class TestHeaderNormalisation:
    @pytest.mark.parametrize(
        "raw",
        [
            "sample_name",
            "Sample Name",
            "sample-name",
            "  Sample.Name  ",
            "\ufeffsample_name",
        ],
    )
    def test_spellings_collapse_to_one_key(self, raw):
        assert normalise_header(raw) == "sample_name"


class TestReadTable:
    def test_basic_read(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x,y\n1,2\n3,4\n")
        table = read_csv_table(path)
        assert table.header == ["x", "y"]
        assert table.rows == [["1", "2"], ["3", "4"]]
        assert table.raw_row_count == 2
        assert table.issues == []

    def test_short_row_is_padded_and_reported(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x,y,z\n1,2\n")
        table = read_csv_table(path)
        assert table.rows == [["1", "2", ""]]
        assert any(issue.code == "SHORT_ROW" for issue in table.issues)

    def test_long_row_is_truncated_and_reported(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x,y\n1,2,3\n")
        table = read_csv_table(path)
        assert table.rows == [["1", "2"]]
        assert any(issue.code == "LONG_ROW" for issue in table.issues)

    def test_duplicate_headers_are_reported(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x,x\n1,2\n")
        table = read_csv_table(path)
        assert any(issue.code == "DUPLICATE_HEADER" for issue in table.issues)

    def test_empty_header_is_reported(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x,\n1,2\n")
        table = read_csv_table(path)
        assert any(issue.code == "EMPTY_HEADER" for issue in table.issues)

    def test_non_utf8_produces_a_warning_not_a_failure(self, tmp_path):
        path = _write(tmp_path, "a.csv", "样本,响应\n对照,1\n", encoding="gb18030")
        table = read_csv_table(path)
        assert any(issue.code == "ENCODING_FALLBACK" for issue in table.issues)
        assert table.header == ["样本", "响应"]

    def test_empty_file_raises(self, tmp_path):
        path = _write(tmp_path, "a.csv", "   \n")
        with pytest.raises(ValueError, match="empty"):
            read_csv_table(path)

    def test_blank_lines_are_skipped(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x,y\n1,2\n\n3,4\n")
        table = read_csv_table(path)
        assert len(table.rows) == 2

    def test_max_rows_limits_the_scan(self, tmp_path):
        path = _write(tmp_path, "a.csv", "x\n1\n2\n3\n4\n5\n")
        table = read_csv_table(path, max_rows=2)
        assert len(table.rows) == 2
        assert table.raw_row_count == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_csv_table(tmp_path / "nope.csv")


class TestPreview:
    def test_profiles_column_types(self, tmp_path):
        path = _write(
            tmp_path,
            "typed.csv",
            "name,count,value,ratio,flag\n"
            "a,1,1.5,0.9,true\n"
            "b,2,2.5,0.8,false\n"
            "c,3,3.5,0.7,true\n",
        )
        preview = preview_csv(path)
        types = {column.name: column.inferred_type for column in preview.columns}
        assert types == {
            "name": "text",
            "count": "integer",
            "value": "number",
            "ratio": "number",
            "flag": "boolean",
        }

    def test_integral_floats_count_as_integers(self, tmp_path):
        # Spreadsheets write whole numbers as 1.0; that must not be "number".
        path = _write(tmp_path, "cycle.csv", "cycle\n1.0\n2.0\n3.0\n")
        preview = preview_csv(path)
        assert preview.columns[0].inferred_type == "integer"

    def test_missing_tokens_are_counted(self, tmp_path):
        # Blank *lines* are skipped entirely (they carry no row), so only the
        # explicit NA tokens count as missing values here.
        path = _write(tmp_path, "m.csv", "x\n1\n\nNA\nnull\n4\n")
        preview = preview_csv(path)
        column = preview.columns[0]
        assert column.n_missing == 2
        assert column.missing_fraction == pytest.approx(0.5)

    def test_empty_cells_in_a_multi_column_file_count_as_missing(self, tmp_path):
        path = _write(tmp_path, "m2.csv", "x,y\n1,\n,2\n3,4\n")
        preview = preview_csv(path)
        missing = {column.name: column.n_missing for column in preview.columns}
        assert missing == {"x": 1, "y": 1}

    def test_unit_is_extracted_from_the_header(self, tmp_path):
        path = _write(tmp_path, "u.csv", "concentration (uM)\n1\n2\n")
        preview = preview_csv(path)
        assert preview.columns[0].unit == "uM"

    def test_numeric_summary_is_computed(self, tmp_path):
        path = _write(tmp_path, "n.csv", "x\n1\n2\n3\n4\n")
        preview = preview_csv(path)
        column = preview.columns[0]
        assert column.minimum == 1.0
        assert column.maximum == 4.0
        assert column.mean == pytest.approx(2.5)
        assert column.n_unique == 4

    def test_head_holds_the_interpreted_rows(self, tmp_path):
        path = _write(tmp_path, "h.csv", "x,y\n1,2\n3,4\n")
        preview = preview_csv(path, display_rows=1)
        assert preview.head[0] == ("x", "y")
        assert preview.head[1] == ("1", "2")
        assert len(preview.head) == 2

    def test_provenance_fields(self, tmp_path):
        path = _write(tmp_path, "p.csv", "x\n1\n")
        preview = preview_csv(path)
        assert preview.size_bytes == path.stat().st_size
        assert len(preview.sha256) == 64

    def test_truncation_is_recorded(self, tmp_path):
        rows = "x\n" + "".join(f"{index}\n" for index in range(50))
        path = _write(tmp_path, "t.csv", rows)
        preview = preview_csv(path, preview_rows=10)
        assert preview.truncated
        assert preview.rows_scanned == 10
        assert preview.rows_total == 50
        assert any(issue.code == "PREVIEW_TRUNCATED" for issue in preview.issues)

    def test_text_report_is_readable(self, tmp_path):
        path = _write(tmp_path, "r.csv", "x\n1\n2\n")
        report = preview_csv(path).to_text()
        assert "CSV 预览" in report
        assert "列概况" in report


class TestSchemaDetection:
    def test_qpcr_tidy_is_detected(self, tmp_path):
        path = _write(
            tmp_path,
            "q.csv",
            "position,sample_name,target_name,cq\nA1,S1,GAPDH,18.5\n",
        )
        preview = preview_csv(path)
        assert preview.schema_key == "qpcr_tidy"
        assert preview.schema_score == 1.0

    def test_long_dose_response_is_detected(self, tmp_path):
        path = _write(
            tmp_path,
            "d.csv",
            "compound,concentration,response\nA,1,90\nA,10,50\nA,100,10\n",
        )
        assert preview_csv(path).schema_key == "dose_response"

    def test_matrix_dose_response_is_detected(self, tmp_path):
        path = _write(
            tmp_path, "m.csv", "concentration,CompoundA,CompoundB\n1,90,88\n10,50,60\n"
        )
        assert preview_csv(path).schema_key == "dose_response_matrix"

    def test_unknown_header_matches_nothing(self):
        key, score = detect_schema(["foo", "bar", "baz"])
        assert key is None
        assert score < 0.5

    def test_explicit_schema_overrides_detection(self, tmp_path):
        path = _write(
            tmp_path, "q.csv", "position,sample_name,target_name,cq\nA1,S,G,1\n"
        )
        preview = preview_csv(path, schema="qpcr_tidy")
        assert preview.schema_key == "qpcr_tidy"


class TestSchemaValidation:
    def test_missing_required_column_is_an_error(self, tmp_path):
        path = _write(tmp_path, "bad.csv", "position,sample_name\nA1,S\n")
        preview = preview_csv(path, schema="qpcr_tidy")
        codes = {issue.code for issue in preview.issues}
        assert "MISSING_COLUMN" in codes
        assert preview.errors

    def test_unknown_column_is_a_warning(self, tmp_path):
        path = _write(
            tmp_path,
            "extra.csv",
            "position,sample_name,target_name,cq,mystery\nA1,S,G,18.5,?\n",
        )
        preview = preview_csv(path, schema="qpcr_tidy")
        warnings = {issue.code for issue in preview.warnings}
        assert "UNKNOWN_COLUMN" in warnings

    def test_non_numeric_value_is_reported_with_its_row(self, tmp_path):
        path = _write(
            tmp_path,
            "v.csv",
            "position,sample_name,target_name,cq\nA1,S,G,not-a-number\n",
        )
        preview = preview_csv(path, schema="qpcr_tidy")
        issue = next(i for i in preview.issues if i.code == "INVALID_NUMBER")
        assert issue.column == "cq"
        assert issue.row == 2

    def test_out_of_range_value_is_reported(self, tmp_path):
        path = _write(tmp_path, "e.csv", "compound,concentration,response\nA,-1,5\n")
        preview = preview_csv(path, schema="dose_response")
        assert any(issue.code == "OUT_OF_RANGE" for issue in preview.issues)

    def test_invalid_choice_is_reported(self, tmp_path):
        path = _write(
            tmp_path,
            "c.csv",
            "position,sample_name,target_name,cq,replicate_type\n"
            "A1,S,G,18.5,quadruplicate\n",
        )
        preview = preview_csv(path, schema="qpcr_tidy")
        assert any(issue.code == "INVALID_CHOICE" for issue in preview.issues)

    def test_undetermined_cq_is_accepted_as_missing(self, tmp_path):
        path = _write(
            tmp_path,
            "n.csv",
            "position,sample_name,target_name,cq\nA1,NTC,G,Undetermined\n",
        )
        preview = preview_csv(path, schema="qpcr_tidy")
        assert not [
            issue
            for issue in preview.issues
            if issue.code in {"MISSING_VALUE", "INVALID_NUMBER"}
        ]

    def test_blank_value_in_a_required_column_is_reported(self, tmp_path):
        table_path = _write(
            tmp_path,
            "b.csv",
            "position,sample_name,target_name,cq\nA1,,G,18.5\n",
        )
        table = read_csv_table(table_path)
        issues = validate_rows_against_schema(table, schema_for("qpcr_tidy"))
        assert any(
            issue.code == "MISSING_VALUE" and issue.column == "sample_name"
            for issue in issues
        )

    def test_duplicate_primary_key_is_warned_about(self, tmp_path):
        path = _write(
            tmp_path,
            "dup.csv",
            "position,sample_name,target_name,cq\nA1,S,G,18.5\nA1,S,G,18.6\n",
        )
        preview = preview_csv(path, schema="qpcr_tidy")
        assert any(issue.code == "DUPLICATE_ROW" for issue in preview.issues)

    def test_too_few_rows_is_reported(self, tmp_path):
        path = _write(tmp_path, "few.csv", "compound,concentration,response\nA,1,90\n")
        preview = preview_csv(path, schema="dose_response")
        assert any(issue.code == "TOO_FEW_ROWS" for issue in preview.issues)


class TestRegistry:
    def test_every_registry_entry_is_well_formed(self):
        for schema in list_schemas():
            assert isinstance(schema, CsvSchema)
            assert schema.key
            assert schema.title
            assert schema.summary
            assert schema.columns
            assert schema.min_rows >= 1

    def test_every_example_row_matches_its_header(self):
        # A short example row produces a template that no reader accepts, and
        # the short row only surfaces as a warning when it is previewed, so the
        # invariant is asserted directly.
        for schema in list_schemas():
            assert schema.example_width_error() is None, schema.key

    def test_every_schema_can_produce_a_valid_template(self, tmp_path):
        for schema in list_schemas():
            destination = tmp_path / f"{schema.key}.csv"
            write_template_csv(destination, schema)
            table = read_csv_table(destination)
            assert not [
                issue
                for issue in table.issues
                if issue.code in {"SHORT_ROW", "LONG_ROW"}
            ], schema.key

    def test_a_ragged_schema_is_refused(self, tmp_path):
        broken = CsvSchema(
            key="broken",
            title="Broken",
            summary="Rows narrower than the header.",
            columns=(
                ColumnSpec("a", "text", description="a"),
                ColumnSpec("b", "text", description="b"),
            ),
            example_rows=(("only-one",),),
        )
        with pytest.raises(ValueError, match="cannot produce a valid template"):
            write_template_csv(tmp_path / "broken.csv", broken)

    def test_registry_covers_every_supported_input(self):
        assert set(SCHEMAS) == {
            "amplification_traces",
            "dose_response",
            "dose_response_matrix",
            "enzyme_kinetics",
            "fermentation_kinetics",
            "qpcr_tidy",
            "qpcr_standard_curve",
        }

    def test_unknown_key_raises_with_the_available_keys(self):
        with pytest.raises(KeyError, match="Available|available"):
            schema_for("nope")

    def test_dose_response_alias_matches_the_registered_name(self):
        schema = schema_for("dose_response")
        assert schema.find("Concentration") is schema.columns[1]


class TestTemplates:
    def test_qpcr_template_passes_the_strict_reader(self, tmp_path):
        from bioeng_qpcr.io import read_tidy_csv

        destination = write_template_csv(tmp_path / "qpcr.csv", "qpcr_tidy")
        experiment = read_tidy_csv(destination)
        gapdh = experiment.assays["GAPDH"]
        assert gapdh.efficiency == pytest.approx(0.98)
        assert gapdh.lloq_cq <= gapdh.lod_cq

    def test_template_round_trips_through_validation(self, tmp_path):
        for key in SCHEMAS:
            destination = tmp_path / f"{key}.csv"
            write_template_csv(destination, key)
            preview = preview_csv(destination, schema=key)
            assert not preview.errors, (key, [i.message for i in preview.errors])

    def test_template_column_count_matches_the_header(self, tmp_path):
        destination = write_template_csv(tmp_path / "d.csv", "dose_response")
        table = read_csv_table(destination)
        assert table.header == ["compound", "concentration", "response", "replicate"]
        assert all(len(row) == 4 for row in table.rows)

    def test_row_count_can_be_requested(self, tmp_path):
        destination = write_template_csv(tmp_path / "d.csv", "dose_response", n_rows=30)
        assert len(read_csv_table(destination).rows) == 30

    def test_existing_file_can_be_protected(self, tmp_path):
        destination = write_template_csv(tmp_path / "d.csv", "dose_response")
        with pytest.raises(FileExistsError):
            write_template_csv(destination, "dose_response", overwrite=False)


class TestGeneratedSpecification:
    def test_markdown_covers_every_schema(self):
        markdown = import_spec_markdown()
        for key in SCHEMAS:
            assert f"`{key}`" in markdown

    def test_markdown_documents_the_required_columns(self):
        markdown = import_spec_markdown()
        assert "| 列名 | 类型 | 必填 |" in markdown
        assert "sample_name" in markdown
        assert "substrate_concentration" in markdown

    def test_markdown_explains_the_general_rules(self):
        markdown = import_spec_markdown()
        for phrase in ("编码", "分隔符", "表头", "缺失值", "不要用 0 表示缺失"):
            assert phrase in markdown

    def test_schema_markdown_lists_notes_and_examples(self):
        markdown = schema_for("enzyme_kinetics").to_markdown()
        assert "ENZ-002" in markdown
        assert "```csv" in markdown
