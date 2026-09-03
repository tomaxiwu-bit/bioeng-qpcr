"""ASSAY-001: Assay-characterization evidence coverage."""

from __future__ import annotations

import math

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

METADATA = RuleMetadata(
    rule_id="ASSAY-001",
    name="Assay Characterization",
    description="Check standard-curve and sensitivity evidence coverage",
    miqe_section="MIQE 2.0: efficiency, dynamic range, LOD and LLOQ",
    min_input_level=InputLevel.LEVEL_A,
    version="1.0.0",
)


@validation_rule(METADATA)
def check_assay_characterization(experiment: Experiment) -> RuleResult:
    missing: dict[str, list[str]] = {}
    low_r2: dict[str, float] = {}
    for name, assay in sorted(experiment.assays.items()):
        fields = [
            field_name
            for field_name in ("efficiency", "r2", "slope", "lod_cq", "lloq_cq")
            if getattr(assay, field_name) is None
        ]
        if fields:
            missing[name] = fields
        if assay.r2 is not None and assay.r2 < 0.98:
            low_r2[name] = assay.r2
    declared_minimum_raw = experiment.metadata.get("assay_min_r2")
    declared_minimum: float | None = None
    invalid_declared_minimum = False
    if declared_minimum_raw is not None:
        try:
            declared_minimum = float(declared_minimum_raw)
        except (TypeError, ValueError):
            invalid_declared_minimum = True
        else:
            if (
                not math.isfinite(declared_minimum)
                or not 0.0 <= declared_minimum <= 1.0
            ):
                declared_minimum = None
                invalid_declared_minimum = True
    policy_failures = {
        name: assay.r2
        for name, assay in experiment.assays.items()
        if declared_minimum is not None
        and assay.r2 is not None
        and assay.r2 < declared_minimum
    }
    evidence = {
        "missing_fields": missing,
        "r2_below_advisory_0_98": low_r2,
        "declared_minimum_r2": declared_minimum,
        "invalid_declared_minimum_r2": invalid_declared_minimum,
        "declared_policy_failures": policy_failures,
        "assay_count": len(experiment.assays),
    }
    if not experiment.assays:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.NOT_ASSESSABLE,
            message="No assay metadata are available",
            evidence=evidence,
        )
    if policy_failures:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.FAIL,
            message=(
                f"{len(policy_failures)} assay(s) have standard-curve R² below "
                f"the declared minimum {declared_minimum:g}"
            ),
            evidence=evidence,
        )
    if missing or low_r2 or invalid_declared_minimum:
        findings = []
        if missing:
            findings.append(f"{len(missing)} lack complete evidence")
        if low_r2:
            findings.append(f"{len(low_r2)} have R² below advisory 0.98")
        if invalid_declared_minimum:
            findings.append("declared minimum R² is invalid")
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.WARN,
            message="Assay characterization needs review: " + "; ".join(findings),
            evidence=evidence,
        )
    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.PASS,
        message="All assays include efficiency, curve-fit, LOD, and LLOQ evidence",
        evidence=evidence,
    )
