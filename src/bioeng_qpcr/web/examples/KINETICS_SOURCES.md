# Browser kinetics and FBA examples

These files are demonstration inputs, not measurements made by this application.

| File | Source and interpretation |
| --- | --- |
| `inhibition_example.csv` | Unmodified copy of this project's existing `examples/inhibition_example.csv`. Teaching rates, not an attributed measured experiment. |
| `fermentation_example.csv` | Unmodified copy of this project's existing `examples/fermentation_example.csv`. Teaching time series, not an attributed measured fermentation. |
| `e_coli_core.xml` | Decompressed `cobra.data/textbook.xml.gz` bundled with COBRApy **0.31.1**. Public E. coli core tutorial reconstruction; neither measured fluxes nor a model of the user's organism. |

COBRApy's tutorial describes the public textbook model:
https://cobrapy.readthedocs.io/en/0.27.0/getting_started.html

Upstream source and license:
https://github.com/opencobra/cobrapy/tree/0.31.1/src/cobra/data
https://github.com/opencobra/cobrapy/blob/0.31.1/LICENSE

SHA-256 of the decompressed model:
`0abebb806ce94922d7c7eb6cadca8c2fa5c4500d0db8e119d3c6224ae59e212d`.

REAL DATA REPLACEMENT: upload a CSV or SBML/JSON model in the corresponding
browser module. `web/kinetics.py` reads the uploaded temporary file through the
existing scientific readers. `web/records.py` saves the exact original upload,
SHA-256, parameters and complete result. Do not overwrite these examples to
represent a real experiment.

For the tutorial model only, `EX_o2_e,0,1000` closes oxygen uptake for a scenario
comparison. The application validates IDs against the uploaded model; it does not
apply this reaction ID to an arbitrary organism automatically. Bounds and
objective definitions belong to the model and require scientific review.
