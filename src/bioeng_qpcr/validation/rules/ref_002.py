"""REF-002: Reference-gene multiplicity readiness check."""

from __future__ import annotations

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

METADATA = RuleMetadata(
    rule_id="REF-002",
    name="Reference Gene Multiplicity",
    description="Check whether normalization uses multiple designated references",
    miqe_section="MIQE 2.0: normalization",
    min_input_level=InputLevel.LEVEL_A,
    version="1.0.0",
)


@validation_rule(METADATA)
def check_reference_multiplicity(experiment: Experiment) -> RuleResult:
    references = experiment.reference_targets
    evidence = {
        "reference_targets": references,
        "reference_count": len(references),
        "stability_validated": False,
    }
    if not references:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.NOT_ASSESSABLE,
            message="No reference genes are designated",
            evidence=evidence,
        )
    if len(references) == 1:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.WARN,
            message=(
                "Only one reference gene is designated; document independent "
                "stability evidence or use multiple validated references"
            ),
            evidence=evidence,
        )
    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.PASS,
        message=(
            f"{len(references)} reference genes are designated; their stability "
            "still requires independent validation"
        ),
        evidence=evidence,
    )
