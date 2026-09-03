"""Chauvenet-criterion outlier detection for technical replicates.

Technical replicates of one sample/target pair measure the same template
solution; gross deviations usually have a physical cause (pipetting error,
a bubble, an edge-effect well).  This module *identifies* suspect wells
and reports the statistical evidence; it never modifies the experiment.
Acting on a recommendation (re-running the well, documenting an exclusion)
stays with the researcher, in line with the package's no-silent-discard
policy.

Two tiers of evidence
---------------------
**Chauvenet's criterion (statistical).**  For a group of ``n`` Cq values
with sample mean and standard deviation, a well is rejected when the
expected number of deviations at least that extreme falls below one half:
``n · P(|Z| ≥ z) < 0.5`` with ``P(|Z| ≥ z) = erfc(z / √2)``.  The test is
applied iteratively: the most extreme remaining well is examined, removed
if it qualifies, and the statistics are recomputed.  Note that with the
sample standard deviation (``ddof=1``) the criterion is very conservative
for small groups: three values can never produce a qualifying z-score
(maximum ≈ 1.155 versus a threshold of ≈ 1.383), and four values only in
the limit of two identical pairs.  Groups that small simply do not carry
enough information for statistical rejection; the review tier below
covers them.

**Review recommendation (heuristic).**  Groups whose Cq standard
deviation exceeds ``sd_trigger`` (default ``0.3``, a common technical
replicate tolerance) without a Chauvenet flag are marked for review and
report their most deviant well (largest distance from the group median)
as a candidate to inspect.  This tier is a documented heuristic, not
statistical evidence, and is labelled as such in the report.

Grouping follows REP-001: wells marked ``technical`` with a determined Cq,
keyed by sample name, target name, and replicate group.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.domain.enums import ReplicateType

__all__ = [
    "ChauvenetFlag",
    "ReplicateGroupReport",
    "ReplicateOutlierResult",
    "detect_replicate_outliers",
]


@dataclass(frozen=True)
class ChauvenetFlag:
    """One well rejected from its replicate group by Chauvenet's criterion.

    Attributes
        position:
            Plate position of the flagged well.
        cq:
            Cq value of the flagged well.
        deviation:
            ``cq - mean_cq``; the signed deviation at flagging time.
        z_score:
            ``abs(deviation) / std_cq`` at flagging time.
        tail_probability:
            Two-sided ``P(|Z| >= z_score)`` under normality.
        threshold_probability:
            ``1 / (2n)`` for the group size at flagging time; the flag is
            raised when ``tail_probability`` drops below it.
        group_mean_cq:
            Group mean (including this well) at flagging time.
        group_std_cq:
            Group sample standard deviation (ddof=1) at flagging time.
        removal_order:
            1 for the first well removed in the iterative procedure.
    """

    position: str
    cq: float
    deviation: float
    z_score: float
    tail_probability: float
    threshold_probability: float
    group_mean_cq: float
    group_std_cq: float
    removal_order: int


@dataclass(frozen=True)
class ReplicateGroupReport:
    """Outlier assessment of one technical-replicate group.

    ``status`` is one of:

    * ``"chauvenet_flagged"`` -- at least one well was rejected by
      Chauvenet's criterion (statistical evidence).
    * ``"review"`` -- no statistical flag, but the group Cq standard
      deviation exceeds ``sd_trigger``; inspect the candidate well.
    * ``"pass"`` -- no flag and the spread is within ``sd_trigger``.
    * ``"not_assessed"`` -- fewer than two determined Cq values.
    """

    sample_name: str
    target_name: str
    replicate_group: str
    positions: tuple[str, ...]
    n_wells: int
    mean_cq: float
    std_cq: float
    sd_trigger_exceeded: bool
    chauvenet_flags: tuple[ChauvenetFlag, ...] = ()
    review_candidate_position: str | None = None
    review_candidate_deviation: float | None = None
    retained_mean_cq: float | None = None
    retained_std_cq: float | None = None

    @property
    def status(self) -> str:
        if self.n_wells < 2:
            return "not_assessed"
        if self.chauvenet_flags:
            return "chauvenet_flagged"
        if self.sd_trigger_exceeded:
            return "review"
        return "pass"

    @property
    def label(self) -> str:
        return "/".join((self.sample_name, self.target_name, self.replicate_group))

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["status"] = self.status
        return data


@dataclass(frozen=True)
class ReplicateOutlierResult:
    """Aggregate outlier report over all technical-replicate groups.

    Attributes
        groups:
            One :class:`ReplicateGroupReport` per group with at least two
            determined Cq values, sorted by sample/target/group.
        sd_trigger:
            The Cq standard deviation above which unflagged groups are
            marked for review.
        min_group_size:
            Minimum group size for Chauvenet assessment.
    """

    groups: tuple[ReplicateGroupReport, ...] = ()
    sd_trigger: float = 0.3
    min_group_size: int = 3

    @property
    def total_flagged(self) -> int:
        """Total number of Chauvenet-rejected wells across all groups."""
        return sum(len(group.chauvenet_flags) for group in self.groups)

    @property
    def flagged_positions(self) -> set[str]:
        """Plate positions rejected by Chauvenet's criterion."""
        return {
            flag.position
            for group in self.groups
            for flag in group.chauvenet_flags
        }

    @property
    def chauvenet_groups(self) -> tuple[ReplicateGroupReport, ...]:
        """Groups with at least one statistical flag."""
        return tuple(g for g in self.groups if g.chauvenet_flags)

    @property
    def review_groups(self) -> tuple[ReplicateGroupReport, ...]:
        """Groups marked for manual review (no statistical flag)."""
        return tuple(g for g in self.groups if g.status == "review")

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_pass(self) -> int:
        return sum(1 for g in self.groups if g.status == "pass")

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.groups:
            return "No technical replicate group with at least two Cq values."
        parts = [
            f"{self.n_groups} group(s) assessed",
            f"{self.total_flagged} Chauvenet flag(s)",
            f"{len(self.review_groups)} review group(s)",
        ]
        return ", ".join(parts) + "."

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "n_groups": self.n_groups,
                "n_pass": self.n_pass,
                "n_chauvenet_flagged_groups": len(self.chauvenet_groups),
                "n_review_groups": len(self.review_groups),
                "total_flagged_wells": self.total_flagged,
                "flagged_positions": sorted(self.flagged_positions),
            },
            "parameters": {
                "sd_trigger": self.sd_trigger,
                "min_group_size": self.min_group_size,
            },
            "groups": [group.to_dict() for group in self.groups],
        }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _two_sided_tail_probability(z: float) -> float:
    """``P(|Z| >= z)`` for a standard normal variable."""
    return math.erfc(z / math.sqrt(2.0))


def _chauvenet_flags(
    wells: list[tuple[str, float]],
    *,
    min_group_size: int,
) -> list[ChauvenetFlag]:
    """Iteratively apply Chauvenet's criterion to one replicate group.

    *wells* holds ``(position, cq)`` pairs.  At each step the most extreme
    remaining value is tested; when its expected count of equally extreme
    deviations drops below 0.5 it is flagged and removed, and the group
    statistics are recomputed.  The loop stops when fewer than
    *min_group_size* values remain or no value qualifies.
    """
    flags: list[ChauvenetFlag] = []
    remaining = list(wells)
    while len(remaining) >= min_group_size:
        values = np.asarray([cq for _, cq in remaining], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if std <= 0.0:
            break
        worst_index = max(
            range(len(remaining)), key=lambda i: abs(remaining[i][1] - mean)
        )
        position, cq = remaining[worst_index]
        deviation = cq - mean
        z_score = abs(deviation) / std
        tail = _two_sided_tail_probability(z_score)
        threshold = 1.0 / (2.0 * len(remaining))
        if tail >= threshold:
            break
        flags.append(
            ChauvenetFlag(
                position=position,
                cq=cq,
                deviation=deviation,
                z_score=z_score,
                tail_probability=tail,
                threshold_probability=threshold,
                group_mean_cq=mean,
                group_std_cq=std,
                removal_order=len(flags) + 1,
            )
        )
        remaining.pop(worst_index)
    return flags


def _review_candidate(
    wells: list[tuple[str, float]],
) -> tuple[str | None, float | None]:
    """Most deviant well relative to the group median, with its distance.

    Returns ``(None, None)`` when the group has fewer than three wells
    (two values are symmetric around their mean, so neither is more
    deviant than the other).
    """
    if len(wells) < 3:
        return None, None
    median = float(np.median([cq for _, cq in wells]))
    position, cq = max(wells, key=lambda item: (abs(item[1] - median), item[0]))
    return position, abs(cq - median)


def detect_replicate_outliers(
    experiment: Experiment,
    *,
    min_group_size: int = 3,
    sd_trigger: float = 0.3,
) -> ReplicateOutlierResult:
    """Assess technical-replicate groups for outlier wells.

    Parameters
        experiment:
            The qPCR experiment to assess.
        min_group_size:
            Smallest group size Chauvenet's criterion is applied to;
            must be at least 3.  Smaller groups never receive a
            statistical flag (they carry a review recommendation instead
            when their spread exceeds ``sd_trigger``).
        sd_trigger:
            Cq standard deviation above which an unflagged group is
            marked for review with its most deviant well as candidate.

    Returns
        :class:`ReplicateOutlierResult` with one report per
        technical-replicate group holding at least two determined Cq
        values.  The experiment is never modified.

    Raises
        ValueError:
            If ``min_group_size`` is below 3 or ``sd_trigger`` is negative.
    """
    if min_group_size < 3:
        raise ValueError("min_group_size must be at least 3")
    if sd_trigger < 0:
        raise ValueError("sd_trigger must be non-negative")

    grouped: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for well in experiment.wells:
        if well.replicate_type is not ReplicateType.TECHNICAL or well.cq is None:
            continue
        key = (
            well.sample_name,
            well.target_name,
            well.replicate_group or "ungrouped",
        )
        grouped.setdefault(key, []).append((well.position, float(well.cq)))

    reports: list[ReplicateGroupReport] = []
    for (sample_name, target_name, replicate_group), wells in sorted(
        grouped.items()
    ):
        if len(wells) < 2:
            continue
        values = np.asarray([cq for _, cq in wells], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        flags = tuple(
            _chauvenet_flags(wells, min_group_size=min_group_size)
        )
        sd_exceeded = std > sd_trigger

        candidate_position: str | None = None
        candidate_deviation: float | None = None
        if not flags and sd_exceeded:
            candidate_position, candidate_deviation = _review_candidate(wells)

        retained_mean: float | None = None
        retained_std: float | None = None
        if flags:
            flagged_positions = {flag.position for flag in flags}
            retained = [cq for pos, cq in wells if pos not in flagged_positions]
            retained_mean = float(np.mean(retained))
            retained_std = (
                float(np.std(retained, ddof=1)) if len(retained) >= 2 else 0.0
            )

        reports.append(
            ReplicateGroupReport(
                sample_name=sample_name,
                target_name=target_name,
                replicate_group=replicate_group,
                positions=tuple(position for position, _ in wells),
                n_wells=len(wells),
                mean_cq=mean,
                std_cq=std,
                sd_trigger_exceeded=sd_exceeded,
                chauvenet_flags=flags,
                review_candidate_position=candidate_position,
                review_candidate_deviation=candidate_deviation,
                retained_mean_cq=retained_mean,
                retained_std_cq=retained_std,
            )
        )

    return ReplicateOutlierResult(
        groups=tuple(reports),
        sd_trigger=sd_trigger,
        min_group_size=min_group_size,
    )
