"""NTC-001: No-Template Control Check.

Verifies that no-template control (NTC) wells are present in the
experiment and that they show no amplification.  Amplification in an NTC
well signals contamination or primer-dimer formation, which compromises
the quantitative results.

MIQE reference: §3.4

Decision logic
    1. If the experiment contains **no NTC wells** the rule is
       **NOT_ASSESSABLE** -- contamination cannot be evaluated.
    2. If NTC wells exist and **all** have ``cq=None`` (no amplification)
       the rule **PASS**es.
    3. If NTC wells exist and **some** have a Cq value, the rule warns unless
       the experiment supplies an explicit ``ntc_fail_threshold`` policy. A
       universal Cq cutoff is not scientifically defensible across assays.
"""

from __future__ import annotations

from bioeng_qpcr.domain.entities import Experiment, Well
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = ["METADATA", "check_ntc"]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

METADATA = RuleMetadata(
    rule_id="NTC-001",
    name="No-Template Control",
    description=("Verify NTC wells are present and show no amplification"),
    miqe_section="MIQE §3.4",
    min_input_level=InputLevel.LEVEL_A,
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Rule implementation
# ---------------------------------------------------------------------------


@validation_rule(METADATA)
def check_ntc(experiment: Experiment) -> RuleResult:
    """Check NTC wells for presence and amplification."""

    ntc_wells: list[Well] = experiment.ntc_wells

    # Case 1 -- no NTC wells at all.
    if not ntc_wells:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.NOT_ASSESSABLE,
            message="No NTC wells detected in experiment",
            evidence={
                "ntc_wells": [],
                "cq_values": {},
            },
        )

    positions: list[str] = [w.position for w in ntc_wells]
    cq_values: dict[str, float | None] = {w.position: w.cq for w in ntc_wells}

    # Wells with amplification (Cq is not None).
    amplified: dict[str, float] = {
        pos: cq for pos, cq in cq_values.items() if cq is not None
    }

    evidence: dict[str, object] = {
        "ntc_wells": positions,
        "cq_values": cq_values,
        "amplified_count": len(amplified),
    }

    # Case 2 -- NTC wells exist, none amplified.
    if not amplified:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.PASS,
            message="NTC wells show no amplification",
            evidence=evidence,
        )

    # Case 3 -- some NTC wells amplified. A hard fail requires an explicit,
    # assay/workflow-specific threshold declared in experiment metadata.
    threshold_raw = experiment.metadata.get("ntc_fail_threshold")
    threshold: float | None = None
    if threshold_raw is not None:
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = None
    evidence["fail_threshold"] = threshold
    evidence["threshold_source"] = (
        "experiment.metadata" if threshold is not None else "not_declared"
    )
    if threshold is None:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.WARN,
            message=(
                f"Amplification detected in {len(amplified)} NTC well(s); "
                "interpret against assay-specific LOD, melt specificity, and "
                "the declared contamination policy"
            ),
            evidence=evidence,
        )

    failing: dict[str, float] = {
        pos: cq for pos, cq in amplified.items() if cq < threshold
    }
    warning: dict[str, float] = {
        pos: cq for pos, cq in amplified.items() if cq >= threshold
    }

    evidence["failing_wells"] = failing
    evidence["warning_wells"] = warning

    if failing:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.FAIL,
            message=(
                f"NTC contamination detected: {len(failing)} well(s) "
                f"with Cq < declared threshold {threshold:g}"
            ),
            evidence=evidence,
        )

    # No hard fails, but late amplification present.
    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.WARN,
        message=(
            f"NTC wells with late amplification (Cq >= "
            f"declared threshold {threshold:g}): {len(warning)} well(s)"
        ),
        evidence=evidence,
    )
