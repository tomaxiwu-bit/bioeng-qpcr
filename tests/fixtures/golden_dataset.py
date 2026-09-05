"""Synthetic golden datasets for validation rule testing.

Each function returns an :class:`~bioeng_qpcr.domain.entities.Experiment`
with *known-answer* properties -- i.e. the expected validation outcome
for every rule is pre-determined by construction.

Datasets
    * :func:`make_golden_pass` -- passes all three rules (EFF-001, REF-001,
      NTC-001).
    * :func:`make_golden_fail_eff` -- efficiency out of range (EFF-001 FAIL).
    * :func:`make_golden_no_ref` -- no reference gene (REF-001 NOT_ASSESSABLE).
    * :func:`make_golden_no_ntc` -- no NTC wells (NTC-001 NOT_ASSESSABLE).
    * :func:`make_golden_ntc_amplification` -- NTC with Cq < 35
      (NTC-001 FAIL).
    * :func:`make_golden_minimal` -- only Cq data; no efficiency, no
      reference, no NTC (all rules NOT_ASSESSABLE).
"""

from __future__ import annotations

from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
from bioeng_qpcr.domain.enums import InputLevel

__all__ = [
    "make_golden_fail_eff",
    "make_golden_minimal",
    "make_golden_no_ntc",
    "make_golden_no_ref",
    "make_golden_ntc_amplification",
    "make_golden_pass",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_wells() -> list[Well]:
    """Two samples x two targets with valid Cq values (no NTC)."""
    return [
        Well(position="A01", sample_name="S1", target_name="GAPDH", cq=20.5),
        Well(position="A02", sample_name="S1", target_name="TARGET", cq=18.3),
        Well(position="A03", sample_name="S2", target_name="GAPDH", cq=21.0),
        Well(position="A04", sample_name="S2", target_name="TARGET", cq=19.1),
    ]


def _clean_ntc_wells() -> list[Well]:
    """Two NTC wells with no amplification (cq=None)."""
    return [
        Well(
            position="A05",
            sample_name="NTC",
            target_name="GAPDH",
            cq=None,
            cq_status="no_template",
        ),
        Well(
            position="A06",
            sample_name="NTC",
            target_name="TARGET",
            cq=None,
            cq_status="no_template",
        ),
    ]


def _amplified_ntc_wells(cq: float) -> list[Well]:
    """Two NTC wells with amplification at the given Cq."""
    return [
        Well(
            position="A05",
            sample_name="NTC",
            target_name="GAPDH",
            cq=cq,
            cq_status="no_template",
        ),
        Well(
            position="A06",
            sample_name="NTC",
            target_name="TARGET",
            cq=cq,
            cq_status="no_template",
        ),
    ]


def _samples_dict() -> dict[str, Sample]:
    return {
        "S1": Sample(name="S1"),
        "S2": Sample(name="S2"),
    }


def _build_experiment(
    name: str,
    wells: list[Well],
    assays: dict[str, Assay],
    input_level: InputLevel = InputLevel.LEVEL_B,
    metadata: dict[str, object] | None = None,
) -> Experiment:
    """Assemble an Experiment with sensible defaults."""
    samples = {s.name: s for s in _samples_dict().values()}
    # Include NTC as a sample if NTC wells are present.
    if any(w.is_ntc for w in wells):
        samples["NTC"] = Sample(name="NTC")
    return Experiment(
        name=name,
        wells=wells,
        samples=samples,
        assays=assays,
        input_level=input_level,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Public golden datasets
# ---------------------------------------------------------------------------


def make_golden_pass() -> Experiment:
    """Experiment that passes all three MIQE rules.

    * EFF-001: both assays have valid efficiency (0.98 and 1.02).
    * REF-001: GAPDH is designated as reference gene.
    * NTC-001: two clean NTC wells (no amplification).
    """
    assays = {
        "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=True),
        "TARGET": Assay(name="TARGET", efficiency=1.02, is_reference=False),
    }
    wells = _sample_wells() + _clean_ntc_wells()
    return _build_experiment("golden_pass", wells, assays)


def make_golden_fail_eff() -> Experiment:
    """Experiment with efficiency out of range (TARGET = 1.20).

    * EFF-001: FAIL (1.20 > 1.10).
    * REF-001: PASS.
    * NTC-001: PASS.
    """
    assays = {
        "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=True),
        "TARGET": Assay(name="TARGET", efficiency=1.20, is_reference=False),
    }
    wells = _sample_wells() + _clean_ntc_wells()
    return _build_experiment("golden_fail_eff", wells, assays)


def make_golden_no_ref() -> Experiment:
    """Experiment with no reference gene designated.

    * EFF-001: PASS (valid efficiencies).
    * REF-001: NOT_ASSESSABLE.
    * NTC-001: PASS.
    """
    assays = {
        "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=False),
        "TARGET": Assay(name="TARGET", efficiency=1.02, is_reference=False),
    }
    wells = _sample_wells() + _clean_ntc_wells()
    return _build_experiment("golden_no_ref", wells, assays)


def make_golden_no_ntc() -> Experiment:
    """Experiment with no NTC wells.

    * EFF-001: PASS.
    * REF-001: PASS.
    * NTC-001: NOT_ASSESSABLE.
    """
    assays = {
        "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=True),
        "TARGET": Assay(name="TARGET", efficiency=1.02, is_reference=False),
    }
    wells = _sample_wells()  # no NTC wells
    return _build_experiment("golden_no_ntc", wells, assays)


def make_golden_ntc_amplification() -> Experiment:
    """Experiment with NTC showing early amplification (Cq = 30.0 < 35).

    * EFF-001: PASS.
    * REF-001: PASS.
    * NTC-001: FAIL (Cq < 35).
    """
    assays = {
        "GAPDH": Assay(name="GAPDH", efficiency=0.98, is_reference=True),
        "TARGET": Assay(name="TARGET", efficiency=1.02, is_reference=False),
    }
    wells = _sample_wells() + _amplified_ntc_wells(cq=30.0)
    return _build_experiment(
        "golden_ntc_amplification",
        wells,
        assays,
        metadata={"ntc_fail_threshold": 35.0},
    )


def make_golden_minimal() -> Experiment:
    """Minimal experiment: only Cq data, no efficiency, no ref, no NTC.

    All three rules should return NOT_ASSESSABLE.

    * EFF-001: NOT_ASSESSABLE (no efficiency data).
    * REF-001: NOT_ASSESSABLE (no reference gene).
    * NTC-001: NOT_ASSESSABLE (no NTC wells).
    """
    assays = {
        "GAPDH": Assay(name="GAPDH", efficiency=None, is_reference=False),
        "TARGET": Assay(name="TARGET", efficiency=None, is_reference=False),
    }
    wells = _sample_wells()  # no NTC wells
    return _build_experiment(
        "golden_minimal", wells, assays, input_level=InputLevel.LEVEL_A
    )
