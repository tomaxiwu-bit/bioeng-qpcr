"""IO layer: input readers, previews and import specifications.

Submodules
----------
* :mod:`bioeng_qpcr.io.csv_reader` -- strict tidy qPCR CSV reader.
* :mod:`bioeng_qpcr.io.rdml_reader` -- RDML (1.2/1.3) reader.
* :mod:`bioeng_qpcr.io.amplification_csv` -- long-format trace reader.
* :mod:`bioeng_qpcr.io.vendor_export` -- vendor plate-export reader.
* :mod:`bioeng_qpcr.io.csv_preview` -- defensive CSV preview plus the
  registry of import formats every module consumes.
* :mod:`bioeng_qpcr.io.dose_response_csv` -- dose-response reader for the
  IC50/EC50 pipeline.
* :mod:`bioeng_qpcr.io.image_preview` -- multi-format image inspection and
  preview.
"""

from bioeng_qpcr.io.amplification_csv import (
    NotAmplificationFile,
    read_amplification_csv,
)
from bioeng_qpcr.io.csv_preview import (
    SCHEMAS,
    ColumnProfile,
    ColumnSpec,
    CsvPreview,
    CsvSchema,
    CsvTable,
    detect_schema,
    import_spec_markdown,
    list_schemas,
    preview_csv,
    read_csv_table,
    schema_for,
    validate_rows_against_schema,
    write_template_csv,
)
from bioeng_qpcr.io.csv_reader import read_tidy_csv
from bioeng_qpcr.io.diagnostics import CSVValidationError, CSVValidationIssue
from bioeng_qpcr.io.dose_response_csv import (
    NotDoseResponseFile,
    read_dose_response_csv,
)
from bioeng_qpcr.io.image_preview import (
    CHANNEL_PALETTES,
    ChannelImage,
    ImageInfo,
    ImagePreview,
    compose_channels,
    inspect_image,
    list_supported_formats,
    load_preview,
    save_preview_png,
    split_channels,
    to_data_uri,
)
from bioeng_qpcr.io.rdml_reader import read_rdml
from bioeng_qpcr.io.vendor_export import read_vendor_export

__all__ = [
    "CHANNEL_PALETTES",
    "SCHEMAS",
    "ChannelImage",
    "ColumnProfile",
    "ColumnSpec",
    "CsvPreview",
    "CsvSchema",
    "CsvTable",
    "CSVValidationError",
    "CSVValidationIssue",
    "ImageInfo",
    "ImagePreview",
    "NotAmplificationFile",
    "NotDoseResponseFile",
    "compose_channels",
    "detect_schema",
    "import_spec_markdown",
    "inspect_image",
    "list_schemas",
    "list_supported_formats",
    "load_preview",
    "preview_csv",
    "read_amplification_csv",
    "read_csv_table",
    "read_dose_response_csv",
    "read_rdml",
    "read_tidy_csv",
    "read_vendor_export",
    "save_preview_png",
    "schema_for",
    "split_channels",
    "to_data_uri",
    "validate_rows_against_schema",
    "write_template_csv",
]
