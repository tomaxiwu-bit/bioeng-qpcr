"""Long-format amplification-trace CSV reader.

Instruments that export per-cycle fluorescence in a tidy long format (one
row per well per cycle) can be ingested through :func:`read_amplification_csv`
into the same :class:`~bioeng_qpcr.domain.entities.Experiment` model that
RDML provides, so the amplification-curve pipeline runs on both.

Required columns
----------------
``position``, ``cycle``, ``fluorescence``.

Optional columns
----------------
``target_name``, ``sample_name``, ``cq``.  When present they build the
corresponding :class:`~bioeng_qpcr.domain.entities.Well` rows; a trace whose
position has no target/sample metadata is still retained (reporting it as a
trace without a well is left to the analysis layer, not silently dropped
here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bioeng_qpcr.domain.entities import Assay, Experiment, RawTrace, Sample, Well

__all__ = ["NotAmplificationFile", "read_amplification_csv"]

_REQUIRED = frozenset({"position", "cycle", "fluorescence"})


class NotAmplificationFile(ValueError):
    """Raised when a file is structurally not an amplification-trace CSV.

    Distinct from plain :class:`ValueError`, which signals a file that looks
    like an amplification CSV but is malformed.  Callers use this distinction
    to decide whether a tidy-CSV fallback is legitimate (no required columns
    present) or would silently reinterpret broken data (raise instead).
    """


def read_amplification_csv(
    csv_path: str | Path,
    *,
    reference_targets: list[str] | None = None,
) -> Experiment:
    """Read a tidy long-format amplification trace CSV into an Experiment.

    Parameters
        csv_path:
            Path to the long-format trace CSV.
        reference_targets:
            Optional target names to mark as reference assays.

    Returns
        :class:`~bioeng_qpcr.domain.entities.Experiment` with wells (when
        sample/target metadata is present), assays, and raw traces populated.

    Raises
        NotAmplificationFile:
            When the file has no trace columns (or is empty) and is therefore
            not an amplification CSV.
        ValueError:
            On a structurally amplification-like file that is malformed
            (non-finite values, non-increasing cycles, bad ``cq``).
    """
    df = pd.read_csv(csv_path)
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise NotAmplificationFile(
            f"Amplification CSV missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )
    if df.empty:
        raise NotAmplificationFile("Amplification CSV is empty.")

    positions = [str(p).strip() for p in df["position"]]
    cycles = np.asarray(df["cycle"], dtype=float)
    fluorescence = np.asarray(df["fluorescence"], dtype=float)
    if not np.all(np.isfinite(cycles)) or not np.all(np.isfinite(fluorescence)):
        raise ValueError("Amplification CSV contains non-finite cycle or fluorescence.")

    has_meta = "target_name" in df.columns or "sample_name" in df.columns
    wells: list[Well] = []
    assays: dict[str, Assay] = {}
    samples: dict[str, Sample] = {}
    reference = set(reference_targets or [])

    for position in sorted(set(positions)):
        mask = np.asarray([p == position for p in positions])
        cyc = cycles[mask]
        if not np.all(np.diff(cyc) > 0):
            raise ValueError(
                f"Amplification trace {position}: cycles must be strictly "
                "increasing and contiguous in the rows of this well."
            )

        if has_meta:
            sub = df.loc[mask, :]
            target_name = (
                str(sub["target_name"].iloc[0]).strip()
                if "target_name" in df.columns
                else "unknown"
            )
            sample_name = (
                str(sub["sample_name"].iloc[0]).strip()
                if "sample_name" in df.columns
                else "unknown"
            )
            cq_raw = (
                sub["cq"].iloc[0]
                if "cq" in df.columns and pd.notna(sub["cq"].iloc[0])
                else None
            )
            try:
                cq = None if cq_raw is None else float(cq_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Amplification trace {position}: 'cq' must be numeric or blank."
                ) from None
            wells.append(
                Well(
                    position=position,
                    sample_name=sample_name,
                    target_name=target_name,
                    cq=cq,
                    cq_status="undetermined" if cq is None else "determined",
                )
            )
            assays.setdefault(
                target_name,
                Assay(name=target_name, is_reference=target_name in reference),
            )
            samples.setdefault(sample_name, Sample(name=sample_name))

    experiment = Experiment(
        name=str(Path(csv_path).stem),
        wells=wells,
        samples=samples,
        assays=assays,
        traces={
            position: RawTrace(
                position=position,
                cycles=cycles[np.asarray([p == position for p in positions])],
                fluorescence=fluorescence[
                    np.asarray([p == position for p in positions])
                ],
            )
            for position in sorted(set(positions))
        },
    )
    return experiment