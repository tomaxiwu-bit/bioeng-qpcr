"""Domain core entities for bioeng-qpcr.

These dataclasses form the canonical in-memory representation of a qPCR
experiment.  All downstream modules (io, validation, calculations, etc.)
operate on these types — never on raw dicts or DataFrames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from bioeng_qpcr.domain.enums import InputLevel, MethodType, ReplicateType

# ---------------------------------------------------------------------------
# Atomic entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Well:
    """A single well in a qPCR plate.

    Attributes
        position:
            Well address, e.g. ``"A01"``.
        sample_name:
            Name of the sample loaded into this well.
        target_name:
            Name of the assay / primer pair targeting a specific sequence.
        cq:
            Quantification cycle (Ct).  ``None`` if undetermined.
        replicate_type:
            Whether this well is part of a technical or biological replicate set.
        replicate_group:
            Identifier grouping replicates (e.g. ``"s1_t1_r1"``).
        cq_status:
            One of ``"determined"``, ``"undetermined"``, ``"no_template"``.
    """

    position: str
    sample_name: str
    target_name: str
    cq: float | None = None
    replicate_type: ReplicateType = ReplicateType.SINGLE
    replicate_group: str | None = None
    cq_status: str = "determined"

    def __post_init__(self) -> None:
        if not self.position.strip():
            raise ValueError("Well position must not be empty")
        if not self.sample_name.strip():
            raise ValueError(f"Well {self.position}: sample_name must not be empty")
        if not self.target_name.strip():
            raise ValueError(f"Well {self.position}: target_name must not be empty")
        if self.cq_status not in {"determined", "undetermined", "no_template"}:
            raise ValueError(
                f"Well {self.position}: invalid cq_status {self.cq_status!r}"
            )
        if self.cq is not None and (not np.isfinite(self.cq) or self.cq < 0):
            raise ValueError(
                f"Well {self.position}: Cq must be finite and non-negative, "
                f"got {self.cq}"
            )
        if self.cq is None and self.cq_status == "determined":
            object.__setattr__(self, "cq_status", "undetermined")

    @property
    def is_ntc(self) -> bool:
        """True if this well is a no-template control."""
        return (
            self.sample_name.upper() in {"NTC", "NO_TEMPLATE", "BLANK"}
            or self.cq_status == "no_template"
        )


@dataclass(frozen=True)
class Sample:
    """A biological sample.

    Attributes
        name:
            Unique sample identifier.
        group:
            Experimental group / condition (e.g. ``"treated"``, ``"control"``).
        replicate_type:
            Default replicate type for wells belonging to this sample.
    """

    name: str
    group: str | None = None
    replicate_type: ReplicateType = ReplicateType.SINGLE

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sample name must not be empty")


@dataclass(frozen=True)
class Assay:
    """A target assay (primer pair / probe set).

    Attributes
        name:
            Unique target identifier (gene symbol or custom name).
        efficiency:
            Amplification efficiency as a fractional excess over one copy
            (1.0 = 100% efficiency = an amplification factor of 2 per cycle).
            ``None`` if not provided.  A value of 2.0 means 200% efficiency,
            not perfect doubling.
        r2:
            Coefficient of determination from standard curve.
        slope:
            Slope of the standard curve.
        is_reference:
            Whether this assay is a reference (housekeeping) gene.
        primer_forward:
            Forward primer sequence.
        primer_reverse:
            Reverse primer sequence.
        probe:
            Probe sequence (if applicable).
        lod_cq, lloq_cq:
            Assay-specific Cq at the limit of detection and lower limit of
            quantification, when experimentally established.
    """

    name: str
    efficiency: float | None = None
    r2: float | None = None
    slope: float | None = None
    is_reference: bool = False
    primer_forward: str | None = None
    primer_reverse: str | None = None
    probe: str | None = None
    lod_cq: float | None = None
    lloq_cq: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Assay name must not be empty")
        for field_name in ("efficiency", "r2", "slope", "lod_cq", "lloq_cq"):
            value = getattr(self, field_name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"Assay {self.name}: {field_name} must be finite")
        if self.efficiency is not None and self.efficiency < 0:
            raise ValueError(f"Assay {self.name}: efficiency must be non-negative")
        if self.r2 is not None and not 0 <= self.r2 <= 1:
            raise ValueError(f"Assay {self.name}: r2 must be between 0 and 1")
        if self.lod_cq is not None and self.lod_cq < 0:
            raise ValueError(f"Assay {self.name}: lod_cq must be non-negative")
        if self.lloq_cq is not None and self.lloq_cq < 0:
            raise ValueError(f"Assay {self.name}: lloq_cq must be non-negative")
        if (
            self.lod_cq is not None
            and self.lloq_cq is not None
            and self.lloq_cq > self.lod_cq
        ):
            raise ValueError(f"Assay {self.name}: lloq_cq must not exceed lod_cq")

    @property
    def efficiency_percent(self) -> float | None:
        """Efficiency as a percentage (e.g. 95.0 for 95%)."""
        return self.efficiency * 100 if self.efficiency is not None else None

    @property
    def is_efficiency_valid(self) -> bool | None:
        """True if efficiency is in the acceptable range [90%, 110%]."""
        if self.efficiency is None:
            return None
        return 0.90 <= self.efficiency <= 1.10


@dataclass(frozen=True)
class RawTrace:
    """Raw fluorescence amplification trace for a single well.

    Attributes
        position:
            Well address.
        cycles:
            Array of cycle numbers.
        fluorescence:
            Array of fluorescence readings, one per cycle.
    """

    position: str
    cycles: np.ndarray
    fluorescence: np.ndarray

    def __post_init__(self) -> None:
        if self.cycles.shape != self.fluorescence.shape:
            raise ValueError(
                f"Well {self.position}: cycles and fluorescence must have "
                "the same shape, "
                f"got {self.cycles.shape} vs {self.fluorescence.shape}"
            )
        cycles = np.array(self.cycles, dtype=float, copy=True)
        fluorescence = np.array(self.fluorescence, dtype=float, copy=True)
        if not np.all(np.isfinite(cycles)) or not np.all(np.isfinite(fluorescence)):
            raise ValueError(f"Well {self.position}: raw trace values must be finite")
        cycles.setflags(write=False)
        fluorescence.setflags(write=False)
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "fluorescence", fluorescence)

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)


# ---------------------------------------------------------------------------
# Aggregate root
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """Aggregate root representing a complete qPCR experiment.

    Attributes
        name:
            Experiment name / identifier.
        wells:
            List of all wells in the experiment.
        samples:
            Dict mapping sample name → :class:`Sample`.
        assays:
            Dict mapping target name → :class:`Assay`.
        traces:
            Dict mapping well position → :class:`RawTrace`.
        input_level:
            The richness level of the input data.
        method:
            The quantification method to be used.
        metadata:
            Free-form experimental metadata.
        created_at:
            UTC creation timestamp.
    """

    name: str
    wells: list[Well] = field(default_factory=list)
    samples: dict[str, Sample] = field(default_factory=dict)
    assays: dict[str, Assay] = field(default_factory=dict)
    traces: dict[str, RawTrace] = field(default_factory=dict)
    input_level: InputLevel = InputLevel.LEVEL_A
    method: MethodType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name must not be empty")
        positions = [well.position for well in self.wells]
        duplicates = sorted(
            position for position in set(positions) if positions.count(position) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate well position(s): {', '.join(duplicates)}")

    # -- query helpers ---------------------------------------------------

    def wells_for(self, sample_name: str, target_name: str | None = None) -> list[Well]:
        """Return wells for a given sample (and optionally target)."""
        result = [w for w in self.wells if w.sample_name == sample_name]
        if target_name is not None:
            result = [w for w in result if w.target_name == target_name]
        return result

    def targets_for(self, sample_name: str) -> list[str]:
        """Distinct target names measured for a sample."""
        return sorted(
            {w.target_name for w in self.wells if w.sample_name == sample_name}
        )

    def samples_for_target(self, target_name: str) -> list[str]:
        """Distinct sample names that measured a given target."""
        return sorted(
            {w.sample_name for w in self.wells if w.target_name == target_name}
        )

    @property
    def ntc_wells(self) -> list[Well]:
        """All no-template control wells."""
        return [w for w in self.wells if w.is_ntc]

    @property
    def reference_targets(self) -> list[str]:
        """Names of all reference-gene assays."""
        return sorted(name for name, a in self.assays.items() if a.is_reference)

    @property
    def target_names(self) -> list[str]:
        """All distinct target names across wells."""
        return sorted({w.target_name for w in self.wells})

    @property
    def sample_names(self) -> list[str]:
        """All distinct sample names (excluding NTC)."""
        return sorted({w.sample_name for w in self.wells if not w.is_ntc})

    @property
    def has_traces(self) -> bool:
        return len(self.traces) > 0

    @property
    def has_efficiency(self) -> bool:
        return any(a.efficiency is not None for a in self.assays.values())

    @property
    def has_reference_genes(self) -> bool:
        return len(self.reference_targets) > 0

    @property
    def has_ntc(self) -> bool:
        return len(self.ntc_wells) > 0

    def content_hash(self) -> str:
        """Stable SHA-256 hash of the complete normalized scientific state.

        ``created_at`` and the local ``source_file`` path are deliberately
        excluded because they do not change scientific content. The raw input
        byte hash stored as ``source_sha256`` remains included.
        """
        metadata = {
            key: value for key, value in self.metadata.items() if key != "source_file"
        }
        payload = {
            "name": self.name,
            "wells": [
                {
                    "position": w.position,
                    "sample_name": w.sample_name,
                    "target_name": w.target_name,
                    "cq": w.cq,
                    "replicate_type": w.replicate_type.value,
                    "replicate_group": w.replicate_group,
                    "cq_status": w.cq_status,
                }
                for w in sorted(
                    self.wells, key=lambda x: (x.sample_name, x.target_name, x.position)
                )
            ],
            "samples": {
                name: {
                    "name": sample.name,
                    "group": sample.group,
                    "replicate_type": sample.replicate_type.value,
                }
                for name, sample in sorted(self.samples.items())
            },
            "assays": {
                name: {
                    "name": assay.name,
                    "efficiency": assay.efficiency,
                    "r2": assay.r2,
                    "slope": assay.slope,
                    "is_reference": assay.is_reference,
                    "primer_forward": assay.primer_forward,
                    "primer_reverse": assay.primer_reverse,
                    "probe": assay.probe,
                    "lod_cq": assay.lod_cq,
                    "lloq_cq": assay.lloq_cq,
                }
                for name, assay in sorted(self.assays.items())
            },
            "traces": {
                position: {
                    "position": trace.position,
                    "cycles": trace.cycles.tolist(),
                    "fluorescence": trace.fluorescence.tolist(),
                }
                for position, trace in sorted(self.traces.items())
            },
            "input_level": self.input_level.value,
            "method": self.method.value if self.method is not None else None,
            "metadata": metadata,
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
