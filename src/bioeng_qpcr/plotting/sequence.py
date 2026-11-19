"""Plotting functions for sequencing-read quality control.

Provides the per-base Phred plot and a reference-coordinate overview of read
coverage and called variants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bioeng_qpcr.plotting.style import PlotStyle, apply_style, create_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_alignment_coverage",
    "plot_read_qualities",
    "plot_variant_map",
]


def plot_alignment_coverage(
    reference_length: int,
    ref_start: int,
    ref_end: int,
    variant_positions: list[int] | tuple[int, ...] = (),
    *,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Show aligned-read coverage and variants in reference coordinates."""
    if reference_length < 1:
        raise ValueError("reference_length must be positive.")
    if ref_start < 0 or ref_end < ref_start or ref_end > reference_length:
        raise ValueError("Alignment coordinates fall outside the reference.")

    figure, ax = create_figure(style)
    start = ref_start + 1
    ax.hlines(0, 1, reference_length, color="#d8d0c5", linewidth=8, label="reference")
    ax.hlines(0, start, ref_end, color="#2b6f5e", linewidth=8, label="aligned read")
    positions = [
        position
        for position in variant_positions
        if 1 <= position <= reference_length
    ]
    if positions:
        ax.scatter(
            positions,
            np.zeros(len(positions)),
            marker="D",
            s=48,
            color="#b85c3e",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="variant",
        )
    ax.set_xlim(1, reference_length)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("reference position (bp)")
    ax.set_title(f"Read coverage: {start}–{ref_end} of {reference_length} bp")
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax


def plot_variant_map(
    reference_length: int,
    variant_positions: list[int] | tuple[int, ...],
    *,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Show called variants as a reference-coordinate lollipop map."""
    if reference_length < 1:
        raise ValueError("reference_length must be positive.")
    positions = sorted(
        position
        for position in variant_positions
        if 1 <= position <= reference_length
    )

    figure, ax = create_figure(style)
    ax.hlines(0.12, 1, reference_length, color="#d8d0c5", linewidth=3)
    if positions:
        heights = np.resize(np.array([0.58, 0.86]), len(positions))
        ax.vlines(positions, 0.12, heights, color="#b85c3e", linewidth=1.4)
        ax.scatter(
            positions,
            heights,
            s=52,
            color="#b85c3e",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for position, height in zip(positions[:20], heights[:20], strict=True):
            ax.annotate(
                str(position),
                (position, height),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    else:
        ax.text(
            0.5,
            0.62,
            "No variants detected",
            transform=ax.transAxes,
            ha="center",
            color="#6f675f",
        )
    ax.set_xlim(1, reference_length)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("reference position (bp)")
    ax.set_title(f"Variant map ({len(positions)} called)")
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax


def plot_read_qualities(
    qualities: list[int] | np.ndarray,
    *,
    threshold: int = 20,
    window: int | None = None,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Render the per-base Phred score of a sequencing read.

    Parameters
        qualities:
            Phred scores in read order.
        threshold:
            Mean-quality threshold used for trimming; drawn as a reference
            line so the retained region can be judged visually.
        window:
            When given, the sliding-window mean is overlaid; that is the
            statistic a window-based trimmer actually thresholds on.
        style:
            Optional shared theme.

    Returns
        ``(figure, axes)`` tuple.

    Raises
        ValueError:
            If *qualities* is empty or *window* is not positive.
    """
    scores = np.asarray(qualities, dtype=float)
    if scores.size == 0:
        raise ValueError("At least one quality value is required to plot.")
    if window is not None and window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}.")

    figure, ax = create_figure(style)
    positions = np.arange(1, scores.size + 1)

    # Colour by whether each base clears the threshold: the failure region of
    # a read is usually one contiguous run, and that is what the eye needs.
    failing = scores < threshold
    ax.bar(
        positions[~failing],
        scores[~failing],
        width=1.0,
        label=f"Q ≥ {threshold}",
    )
    ax.bar(
        positions[failing],
        scores[failing],
        width=1.0,
        label=f"Q < {threshold}",
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"threshold Q{threshold}",
    )

    if window is not None and scores.size >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(scores, kernel, mode="valid")
        offsets = np.arange(window, scores.size + 1)
        ax.plot(
            offsets,
            smoothed,
            linewidth=1.8,
            label=f"{window}-base mean",
        )

    ax.set_xlabel("base position (5' → 3')")
    ax.set_ylabel("Phred quality score")
    ax.set_title("Per-base read quality")
    apply_style(ax, style)
    figure.tight_layout()
    return figure, ax
