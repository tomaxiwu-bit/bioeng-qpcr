"""MIQE 2.0 (2025) reporting checklist.

MIQE (Minimum Information for Publication of Quantitative Real-Time PCR
Experiments) was revised in 2025 ("MIQE 2.0", Bustin et al., Clinical
Chemistry) from the original 2009 checklist.  MIQE 2.0 organises the
reporting obligations into five categories backed by a Table 1 of items; the
authoritative list is mirrored here as :data:`MIQE2_FIELDS`.

This module does **not** re-implement bioeng-qpcr's executable validation
rules (see :mod:`bioeng_qpcr.validation.rules`).  Instead it provides a
*reporting completeness* checklist: for each item it tells you whether the
information required by that item is present to be reported, filling in the
fields that are genuinely attestable from an :class:`Experiment` and leaving
the rest for you to answer manually:

* ``reported`` -- the specific evidence the item demands is present in the
  data (or was answered ``True`` explicitly).
* ``missing`` -- the item is expected (essential) but has not been provided.
* ``na`` -- the item is not applicable (non-essential and/or labelled N/A).

The result is emitted as a structured :class:`Miqe2Report` (dict / JSON /
text) suitable for inclusion in a submission packet.

An item is auto-marked ``reported`` only when the evidence the item actually
requires is present.  In particular, incomplete proxies such as "there are
Cq values" do **not** satisfy items that demand a method, a species/accession,
a sample source, or a raw-data submission -- those stay ``missing`` rather
than being overstated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bioeng_qpcr.domain.entities import Experiment

__all__ = [
    "MIQE2_CATEGORIES",
    "MIQE2_FIELDS",
    "Miqe2ChecklistItem",
    "Miqe2Report",
    "miqe2_checklist",
]


# ---------------------------------------------------------------------------
# Static catalog
# ---------------------------------------------------------------------------

#: Ordered MIQE 2.0 categories: ``(category_id, label)``.
MIQE2_CATEGORIES: list[tuple[str, str]] = [
    ("reagent", "Reagent Preparation"),
    ("sample", "Sample Preparation"),
    ("rt", "Reverse Transcription"),
    ("protocol", "qPCR Protocol"),
    ("analysis", "Data Analysis"),
]


def _derive_field(field_id: str, experiment: Experiment) -> tuple[str, str] | None:
    """Auto-fill *field_id* from an :class:`Experiment`.

    Returns ``(status, evidence)`` when the evidence the item genuinely
    demands is present in the data, else ``None`` (the field stays unanswered
    rather than being overstated).
    """
    # Reagent Preparation ---------------------------------------------------
    # reagent_gene_symbols requires species + accession, which are never
    # captured, so it can never be auto-reported.
    if field_id == "reagent_sequences":
        with_seq = [
            a.name
            for a in experiment.assays.values()
            if a.primer_forward or a.primer_reverse or a.probe
        ]
        if with_seq:
            return "reported", f"sequences for: {', '.join(sorted(with_seq))}"
    # Sample Preparation ----------------------------------------------------
    # sample_types_quantity requires the sample *source*, which is not
    # captured, so well/sample counts alone cannot satisfy it.
    # qPCR Protocol ----------------------------------------------------------
    if field_id == "protocol_instrument":
        for key in ("instrument", "instrument_type", "Instrument Type"):
            if experiment.metadata.get(key):
                return "reported", str(experiment.metadata[key])
    if field_id == "protocol_dye":
        with_dye = sorted(
            a.name for a in experiment.assays.values() if a.dye_chemistry
        )
        if with_dye:
            return "reported", f"dye chemistry on: {', '.join(with_dye)}"
    # Data Analysis ----------------------------------------------------------
    if field_id == "analysis_raw_data":
        # The item demands raw fluorescence trajectories in an open format;
        # only an in-memory trace set is genuine evidence (Ct values are not).
        if experiment.has_traces:
            return "reported", "raw amplification traces present"
    # analysis_cq_method demands the baseline/Cq *method*, which is not
    # captured from the data; having Cq values is not the method.
    if field_id == "analysis_efficiency":
        with_eff = sorted(
            a.name for a in experiment.assays.values() if a.efficiency is not None
        )
        if with_eff:
            return "reported", f"efficiency for: {', '.join(with_eff)}"
    if field_id == "analysis_nc_frequency":
        if experiment.ntc_wells:
            return "reported", f"{len(experiment.ntc_wells)} NTC wells"
    if field_id == "analysis_repeats":
        tech = [
            w.position
            for w in experiment.wells
            if w.replicate_type.name == "TECHNICAL"
        ]
        if tech:
            return "reported", f"technical replicates: {len(tech)} wells"
    if field_id == "analysis_normalization":
        # Which genes are declared reference targets is attestable; the
        # wording states this is the declared normalisation approach, not a
        # full calculation.
        if experiment.has_reference_genes:
            return "reported", (
                f"reference genes declared: {', '.join(experiment.reference_targets)}"
            )
    # analysis_reference_validation demands the *validation method*, which the
    # data does not record; declaring reference genes is not validation.
    if field_id == "analysis_quantification_method":
        if experiment.method is not None:
            return "reported", f"method: {experiment.method.value}"
    if field_id == "analysis_lod":
        with_lod = sorted(
            a.name for a in experiment.assays.values() if a.lod_cq is not None
        )
        if with_lod:
            return "reported", f"LOD for: {', '.join(with_lod)}"
    return None


#: ``(field_id, category_id, item label, essential, helper-description)``.
MIQE2_FIELDS: list[tuple[str, str, str, bool, str]] = [
    # -- Reagent Preparation ------------------------------------------------
    ("reagent_gene_symbols", "reagent",
     "Gene symbol, species and accession number for each target", True,
     "Official gene symbol(s), target species and sequence accession."),
    ("reagent_amplicon_position", "reagent", "Amplicon position", True,
     "Position of the amplicon within the gene / transcript."),
    ("reagent_amplicon_length", "reagent", "Amplicon length", True,
     "Size of the PCR amplicon."),
    ("reagent_sequences", "reagent",
     "Primer and probe sequences (or assay context sequence)", False,
     "Full sequences for primers/probes, or the assay context sequence when "
     "pre-designed oligos are used."),
    ("reagent_modifications", "reagent", "Modification position and type", False,
     "Any chemical modifications to primers/probes."),
    ("reagent_oligo_manufacturer", "reagent", "Oligo manufacturer", False,
     "Supplier of primers/probes."),
    ("reagent_optimization", "reagent", "Optimization details", False,
     "How conditions were optimized."),
    # -- Sample Preparation -------------------------------------------------
    ("sample_types_quantity", "sample", "Sample type and quantity", True,
     "Sample source and number of samples."),
    ("sample_procedure", "sample", "Sampling procedure (incl. storage time)", True,
     "From sampling to storage."),
    ("sample_storage", "sample", "Aliquoting, storage conditions and duration", True,
     "Temperature, aliquoting and duration."),
    ("sample_extraction", "sample", "Extraction method (incl. amount processed)", True,
     "Reagent/kit and sample amount processed."),
    ("sample_spike_in", "sample", "Spike-in nucleic acids: source and amount", False,
     "External spike-in control(s), when used."),
    ("sample_elution", "sample", "Elution / re-suspension buffer volume", True,
     "Volume and buffer."),
    ("sample_extraction_replicates", "sample", "Extraction replicates", False,
     "Number of extraction replicates."),
    ("sample_blank_recovery", "sample", "Extraction blanks and recovery", False,
     "Blank controls and recovery rate."),
    ("sample_nucleic_assessment", "sample",
     "Nucleic acid quality/quantity method", True,
     "How concentration/quality were assessed."),
    ("sample_storage_conditions", "sample", "Nucleic acid storage conditions", True,
     "Temperature, buffer, concentration, duration."),
    ("sample_dilution_series", "sample", "Working-template dilution series", True,
     "Full dilution chain."),
    ("sample_template_modification", "sample", "Template modification", False,
     "e.g. DNase, sonication, pre-amplification."),
    ("sample_post_modification_cleanup", "sample", "Cleanup after modification", False,
     "Purification following any modification step."),
    # -- Reverse Transcription ---------------------------------------------
    ("rt_priming", "rt", "cDNA priming method and primer concentration", True,
     "GSP / oligo-dT / random hexamer priming and primer concentration."),
    ("rt_one_step_two_step", "rt", "One-step or two-step protocol", True,
     "Protocol architecture; reaction details if two-step."),
    ("rt_rna_amount", "rt", "RNA amount per reaction", True,
     "Template input (ng/µl)."),
    ("rt_components", "rt", "Detailed reaction components and conditions", True,
     "Enzymes, buffer, temperatures and times."),
    ("rt_no_rt_control", "rt", "Presence/absence of RT and estimated copies", False,
     "No-RT control and estimated copies (to assess DNA contamination)."),
    ("rt_manufacturer", "rt", "RT reagent manufacturer, catalog and lot", False,
     "Source details of RT reagents."),
    ("rt_cdna_storage", "rt", "cDNA storage conditions", False,
     "Temperature, dilution, duration and buffer."),
    # -- qPCR Protocol ------------------------------------------------------
    ("protocol_template_treatment", "protocol", "Template treatment", False,
     "Initial heating or chemical denaturation."),
    ("protocol_oligo_conc", "protocol", "Primer/probe concentration and source", True,
     "Final reaction concentrations and source."),
    ("protocol_reaction_chemistry", "protocol",
     "Polymerase, Mg2+ and dNTP concentrations", False,
     "Identities and concentrations (or kit reference)."),
    ("protocol_kit", "protocol", "Buffer/kit: manufacturer, catalog, lot", True,
     "Reaction chemistry source."),
    ("protocol_thermal_cycling", "protocol", "Full thermal cycling parameters", True,
     "All cycling steps and reaction volume."),
    ("protocol_instrument", "protocol", "qPCR instrument manufacturer and model", True,
     "Instrument identification."),
    ("protocol_dye", "protocol", "Detection chemistry (dye)", False,
     "RDML 1.3 dye chemistry when declared."),
    # -- Data Analysis ------------------------------------------------------
    ("analysis_raw_data", "analysis",
     "Raw fluorescence data storage and submission", True,
     "Submit raw data in an open format (RDML / RDES)."),
    ("analysis_standard", "analysis",
     "Standard identity and quantification method", True,
     "Synthetic, plasmid, gDNA, IVT, mRNA; calibration basis."),
    ("analysis_cq_method", "analysis", "Baseline correction and Cq determination", True,
     "Analysis methodology for Cq."),
    ("analysis_software", "analysis", "qPCR analysis software (source/version)", True,
     "Analysis program and version."),
    ("analysis_controls", "analysis", "Positive and negative control details", True,
     "How controls were set up."),
    ("analysis_nc_frequency", "analysis", "Negative control frequency and Cq", False,
     "Number of NTC wells and their Cq values."),
    ("analysis_examples", "analysis", "Positive/negative result examples", False,
     "Illustrative raw-data visualisation."),
    ("analysis_efficiency", "analysis", "PCR efficiency estimation and method", True,
     "Efficiency estimate and how it was derived."),
    ("analysis_quantification_method", "analysis", "Target quantity calculation", True,
     "Efficiency-corrected target quantity calculation."),
    ("analysis_repeats", "analysis", "Replicate description", True,
     "Biological/technical replicate structure."),
    ("analysis_repeatability", "analysis", "Repeatability (intra-experiment)", False,
     "Within-experiment variation."),
    ("analysis_reproducibility", "analysis",
     "Reproducibility (inter-experiment/user)", False,
     "Across-experiment/user/lab variation."),
    ("analysis_lod", "analysis", "Limit of detection calculation", False,
     "LOD as established."),
    ("analysis_loq_dynamic_range", "analysis",
     "Dynamic range / limits of quantification", True,
     "Quantification limits of the assay."),
    ("analysis_reference_validation", "analysis",
     "Reference gene validation method", True,
     "How reference genes were validated."),
    ("analysis_normalization", "analysis", "Normalization method / calculation", True,
     "Normalized expression calculation."),
    ("analysis_statistics", "analysis", "Statistical analysis method", True,
     "Statistical approach used."),
    ("analysis_significance_power", "analysis", "Significance level and power", False,
     "Chosen alpha and statistical power."),
    ("analysis_specificity", "analysis",
     "Specificity (rare-variant/pathogen assays)", False,
     "Specificity verification where relevant."),
]

_FIELD_BY_ID: dict[str, tuple[str, str, str, bool, str]] = {
    f[0]: f for f in MIQE2_FIELDS
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Miqe2ChecklistItem:
    """One MIQE 2.0 reporting item with its assessment."""

    field_id: str
    category: str
    label: str
    essential: bool
    description: str
    status: str  # "reported" | "missing" | "na"
    evidence: str


class Miqe2Report:
    """Completeness assessment of the MIQE 2.0 (2025) reporting checklist.

    Parameters
        items:
            Assessed :class:`Miqe2ChecklistItem` instances.
        experiment_name:
            Name of the experiment the report refers to.
    """

    def __init__(self, items: list[Miqe2ChecklistItem], experiment_name: str) -> None:
        self._items = list(items)
        self._experiment_name = experiment_name

    @property
    def items(self) -> list[Miqe2ChecklistItem]:
        return list(self._items)

    @property
    def reported(self) -> list[Miqe2ChecklistItem]:
        return [i for i in self._items if i.status == "reported"]

    @property
    def missing(self) -> list[Miqe2ChecklistItem]:
        return [i for i in self._items if i.status == "missing"]

    @property
    def summary(self) -> dict[str, int]:
        counts = {"reported": 0, "missing": 0, "na": 0}
        for i in self._items:
            counts[i.status] += 1
        return counts

    @property
    def essential_completeness(self) -> float:
        """Fraction of essential (non-NA) items that are reported."""
        essential = [i for i in self._items if i.essential and i.status != "na"]
        if not essential:
            return 1.0
        reported = sum(1 for i in essential if i.status == "reported")
        return reported / len(essential)

    def _category(self, category_id: str) -> str:
        for cid, label in MIQE2_CATEGORIES:
            if cid == category_id:
                return label
        return category_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard": "MIQE 2.0 (2025)",
            "experiment_name": self._experiment_name,
            "total_items": len(self._items),
            "summary": self.summary,
            "essential_completeness": round(self.essential_completeness, 3),
            "items": [
                {
                    "field_id": i.field_id,
                    "category": i.category,
                    "label": i.label,
                    "essential": i.essential,
                    "status": i.status,
                    "description": i.description,
                    "evidence": i.evidence,
                }
                for i in self._items
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        sep = "=" * 64
        sub = "-" * 64
        lines = [sep, "MIQE 2.0 (2025) REPORTING CHECKLIST", sep,
                 f"Experiment: {self._experiment_name}", ""]
        s = self.summary
        lines += [
            "SUMMARY",
            sub,
            f"  reported: {s['reported']}",
            f"  missing:  {s['missing']}",
            f"  n/a:      {s['na']}",
            f"  essential completeness: "
            f"{self.essential_completeness * 100:.1f}%",
            "",
        ]
        for category_id, _label in MIQE2_CATEGORIES:
            group = [i for i in self._items if i.category == category_id]
            if not group:
                continue
            lines += [f"[{self._category(category_id)}]", sub]
            for item in group:
                mark = {"reported": "OK ", "missing": "-- ", "na": "NA "}[item.status]
                lines.append(f"  {mark}{item.field_id} - {item.label}")
                if item.evidence:
                    lines.append(f"        {item.evidence}")
            lines.append("")
        lines += [
            sep,
            "This checklist records reporting completeness; it is not a "
            "certification of experimental quality.",
            sep,
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._items)


def miqe2_checklist(
    experiment: Experiment | None = None,
    answers: dict[str, bool] | None = None,
    *,
    experiment_name: str | None = None,
) -> Miqe2Report:
    """Assess MIQE 2.0 (2025) reporting completeness.

    Parameters
        experiment:
            An :class:`Experiment` whose fields auto-fill attestable items.
            ``None`` yields a fully manual checklist.
        answers:
            Optional overrides ``{field_id: reported_bool}``; explicit answers
            take precedence over auto-derived status.  ``False`` marks the item
            ``missing`` (or ``na`` when it is non-essential and the experiment
            gives no evidence).
        experiment_name:
            Name to label the report (defaults to ``experiment.name``).

    Returns
        :class:`Miqe2Report` spanning every item in :data:`MIQE2_FIELDS`.
    """
    answers = answers or {}
    name = experiment_name or (experiment.name if experiment is not None else "manual")
    items: list[Miqe2ChecklistItem] = []
    for field_id, category, label, essential, description in MIQE2_FIELDS:
        auto: tuple[str, str] | None = (
            _derive_field(field_id, experiment) if experiment is not None else None
        )

        if field_id in answers:
            if answers[field_id]:
                status, evidence = "reported", (auto[1] if auto else "")
            elif not essential:
                status, evidence = "na", (auto[1] if auto else "")
            else:
                status, evidence = "missing", (auto[1] if auto else "")
        elif auto is not None:
            status, evidence = auto
        elif essential:
            status, evidence = "missing", ""
        else:
            status, evidence = "na", ""

        items.append(
            Miqe2ChecklistItem(
                field_id=field_id,
                category=category,
                label=label,
                essential=essential,
                description=description,
                status=status,
                evidence=evidence,
            )
        )
    return Miqe2Report(items, name)