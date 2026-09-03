"""REP-001: Technical-replicate agreement."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, ReplicateType, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

METADATA = RuleMetadata(
    rule_id="REP-001",
    name="Technical Replicate Agreement",
    description="Check within-group technical-replicate Cq spread",
    miqe_section="MIQE 2.0: replicates and precision",
    min_input_level=InputLevel.LEVEL_A,
    version="1.0.0",
)


@validation_rule(METADATA)
def check_technical_replicates(experiment: Experiment) -> RuleResult:
    threshold_raw = experiment.metadata.get("technical_replicate_max_delta_cq", 0.5)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        threshold = math.nan
    if not math.isfinite(threshold) or threshold < 0:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.WARN,
            message="Technical-replicate spread threshold is invalid",
            evidence={
                "threshold_delta_cq": None,
                "declared_value": str(threshold_raw),
                "groups": {},
                "failing_groups": [],
            },
        )
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for well in experiment.wells:
        if well.replicate_type is ReplicateType.TECHNICAL and well.cq is not None:
            key = (
                well.sample_name,
                well.target_name,
                well.replicate_group or "ungrouped",
            )
            groups[key].append(float(well.cq))
    assessed = {key: values for key, values in groups.items() if len(values) >= 2}
    evidence_groups: dict[str, object] = {}
    failing: list[str] = []
    for key, values in sorted(assessed.items()):
        label = "/".join(key)
        spread = max(values) - min(values)
        evidence_groups[label] = {
            "n": len(values),
            "mean_cq": float(np.mean(values)),
            "delta_cq": spread,
        }
        if spread > threshold:
            failing.append(label)
    evidence = {
        "threshold_delta_cq": threshold,
        "groups": evidence_groups,
        "failing_groups": failing,
    }
    if not assessed:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.NOT_ASSESSABLE,
            message="No technical replicate group with at least two Cq values",
            evidence=evidence,
        )
    if failing:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.FAIL,
            message=(
                f"{len(failing)} technical replicate group(s) exceed the declared "
                "spread threshold"
            ),
            evidence=evidence,
        )
    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.PASS,
        message=(
            f"All {len(assessed)} technical replicate group(s) agree within threshold"
        ),
        evidence=evidence,
    )
