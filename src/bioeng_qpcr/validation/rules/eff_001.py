"""EFF-001: Amplification Efficiency Check.

Verifies that primer efficiencies are reported for every assay and that
all reported efficiencies fall within the MIQE-acceptable range of
90-110% (expressed as fractional efficiencies in ``[0.90, 1.10]``).

MIQE reference: §3.5

Decision logic
    1. If *no* assay carries efficiency data (all ``None``) the rule is
       **NOT_ASSESSABLE** -- the engine cannot judge what it cannot see.
    2. If *some* assays carry efficiency data and others do not, the rule
       issues a **WARN** listing the assays that are missing efficiency.
    3. When every assay has efficiency data:
       * **PASS** -- all efficiencies are within ``[0.90, 1.10]``.
       * **FAIL** -- one or more efficiencies are outside the range.
"""

from __future__ import annotations

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import InputLevel, RuleStatus
from bioeng_qpcr.validation.decorator import validation_rule
from bioeng_qpcr.validation.types import RuleMetadata, RuleResult

__all__ = ["METADATA", "check_efficiency"]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

METADATA = RuleMetadata(
    rule_id="EFF-001",
    name="Amplification Efficiency",
    description=(
        "Verify that primer efficiencies are reported and within "
        "acceptable range (90-110%)"
    ),
    miqe_section="MIQE §3.5",
    min_input_level=InputLevel.LEVEL_A,
)

#: Acceptable efficiency range as fractional values.
_EFF_MIN: float = 0.90
_EFF_MAX: float = 1.10


# ---------------------------------------------------------------------------
# Rule implementation
# ---------------------------------------------------------------------------


@validation_rule(METADATA)
def check_efficiency(experiment: Experiment) -> RuleResult:
    """Evaluate amplification efficiency for every assay in *experiment*."""

    assays = experiment.assays

    # Partition assays into those with and without efficiency data.
    with_eff: dict[str, float] = {
        name: assay.efficiency  # type: ignore[misc]
        for name, assay in assays.items()
        if assay.efficiency is not None
    }
    missing: list[str] = sorted(
        name for name, assay in assays.items() if assay.efficiency is None
    )

    # Among those *with* data, split into valid / invalid.
    valid: dict[str, float] = {
        name: eff for name, eff in with_eff.items() if _EFF_MIN <= eff <= _EFF_MAX
    }
    invalid: dict[str, float] = {
        name: eff for name, eff in with_eff.items() if not (_EFF_MIN <= eff <= _EFF_MAX)
    }

    evidence: dict[str, object] = {
        "efficiencies": dict(with_eff),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "missing_count": len(missing),
        "missing_assays": missing,
        "invalid_assays": sorted(invalid.keys()),
        "range": [_EFF_MIN, _EFF_MAX],
    }

    # Case 1 -- no efficiency data at all.
    if not with_eff:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.NOT_ASSESSABLE,
            message="No efficiency data provided for any assay",
            evidence=evidence,
        )

    # Invalid measured values are a failure even when other assays are
    # missing.  Evaluate the worst scientific outcome first so partial
    # coverage cannot hide an out-of-range efficiency.
    if invalid:
        offenders = sorted(invalid.keys())
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.FAIL,
            message=(
                f"Efficiency out of range (90-110%) for "
                f"{len(invalid)} assay(s): {', '.join(offenders)}"
                + (
                    f"; data also missing for {len(missing)} assay(s): "
                    f"{', '.join(missing)}"
                    if missing
                    else ""
                )
            ),
            evidence=evidence,
        )

    # Case 3 -- partial coverage (some assays missing efficiency).
    if missing:
        return RuleResult(
            metadata=METADATA,
            status=RuleStatus.WARN,
            message=(
                f"Efficiency data missing for {len(missing)} assay(s): "
                f"{', '.join(missing)}"
            ),
            evidence=evidence,
        )

    # Case 4 -- all assays have valid efficiency.
    return RuleResult(
        metadata=METADATA,
        status=RuleStatus.PASS,
        message=(
            f"All {len(with_eff)} assay(s) have efficiency within "
            f"acceptable range (90-110%)"
        ),
        evidence=evidence,
    )
