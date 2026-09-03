"""INPUT-001: Input ingestion diagnostics."""

from __future__ import annotations

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

METADATA = RuleMetadata(
    rule_id="INPUT-001",
    name="Input Data Integrity",
    description="Report coercions or exclusions performed during data ingestion",
    miqe_section="MIQE 2.0: data transparency",
    min_input_level=InputLevel.LEVEL_A,
    version="1.0.0",
)


@validation_rule(METADATA)
def check_input_diagnostics(experiment: Experiment) -> RuleResult:
    diagnostics = list(experiment.metadata.get("input_diagnostics") or [])
    parse_mode = experiment.metadata.get("parse_mode", "domain_object")
    evidence = {
        "parse_mode": parse_mode,
        "diagnostic_count": len(diagnostics),
        "diagnostics": diagnostics,
    }
    if diagnostics:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.WARN,
            message=f"Input was accepted with {len(diagnostics)} diagnostic issue(s)",
            evidence=evidence,
        )
    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.PASS,
        message="No input coercions or structural exclusions were recorded",
        evidence=evidence,
    )
