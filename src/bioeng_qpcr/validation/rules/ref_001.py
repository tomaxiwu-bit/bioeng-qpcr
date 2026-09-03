"""REF-001: Reference Gene Presence.

Verifies that at least one reference (housekeeping) gene is designated in
the experiment.  Relative quantification methods (ΔCq, ΔΔCq, Pfaffl) all
require a stably-expressed reference gene for normalisation; the absence
of one makes the experiment scientifically uninterpretable.

MIQE reference: §4.1

Decision logic
    * **PASS** -- at least one assay is marked ``is_reference=True``.
    * **NOT_ASSESSABLE** -- no reference gene is designated.
"""

from __future__ import annotations

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = ["METADATA", "check_reference_gene"]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

METADATA = RuleMetadata(
    rule_id="REF-001",
    name="Reference Gene Presence",
    description=(
        "Verify that at least one reference (housekeeping) gene is designated"
    ),
    miqe_section="MIQE §4.1",
    min_input_level=InputLevel.LEVEL_A,
)


# ---------------------------------------------------------------------------
# Rule implementation
# ---------------------------------------------------------------------------


@validation_rule(METADATA)
def check_reference_gene(experiment: Experiment) -> RuleResult:
    """Check that the experiment designates at least one reference gene."""

    ref_targets: list[str] = experiment.reference_targets

    evidence: dict[str, object] = {
        "reference_targets": ref_targets,
        "total_assay_count": len(experiment.assays),
    }

    if ref_targets:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.PASS,
            message=(f"Reference gene(s) designated: {', '.join(ref_targets)}"),
            evidence=evidence,
        )

    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.NOT_ASSESSABLE,
        message=(
            "No reference gene designated; relative quantification "
            "requires at least one"
        ),
        evidence=evidence,
    )
