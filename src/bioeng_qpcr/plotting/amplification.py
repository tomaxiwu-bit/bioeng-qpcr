"""Amplification curve plots for qPCR experiments.

These functions render the raw fluorescence-vs-cycle traces stored on an
:class:`~bioeng_qpcr.domain.entities.Experiment`.  Two views are provided:

* :func:`plot_amplification_curves` -- every trace on a single axes, coloured
  by *target* (assay).  Useful for an at-a-glance QC overview of the run.
* :func:`plot_amplification_by_target` -- traces for a single target, coloured
  by *sample*.  Useful for comparing biological replicates of one assay.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bioeng_qpcr.domain.entities import Experiment
from bioeng_qpcr.plotting.style import PlotStyle, apply_style, create_figure


def _category_colors(categories: list[str], style: PlotStyle) -> dict[str, str]:
    """Map each category name to a colour from the style palette (cyclic)."""
    palette = style.color_palette
    return {name: palette[i % len(palette)] for i, name in enumerate(categories)}


def plot_amplification_curves(
    experiment: Experiment,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Plot fluorescence vs cycle for every well that has a trace.

    Each trace is coloured by its well's ``target_name``; the legend shows
    one entry per target (duplicate labels are collapsed).

    Parameters
        experiment:
            The experiment containing ``traces`` and ``wells``.
        style:
            Optional :class:`PlotStyle`.  Defaults to the shared theme.

    Returns
        A ``(Figure, Axes)`` tuple with the amplification curves.

    Raises
        ValueError:
            If the experiment has no traces to plot.
    """
    if not experiment.has_traces:
        raise ValueError(
            f"Experiment '{experiment.name}' has no amplification traces to plot."
        )

    s = style if style is not None else PlotStyle()
    fig, ax = create_figure(style)

    wells_by_position = {w.position: w for w in experiment.wells}

    # Targets that actually have a trace, sorted for deterministic colours.
    plotted_targets = sorted(
        {
            wells_by_position[pos].target_name
            for pos in experiment.traces
            if pos in wells_by_position
        }
    )
    target_colors = _category_colors(plotted_targets, s)

    for position, trace in experiment.traces.items():
        well = wells_by_position.get(position)
        target = well.target_name if well is not None else "unknown"
        color = target_colors.get(target, s.color_palette[0])
        ax.plot(
            trace.cycles,
            trace.fluorescence,
            color=color,
            label=target,
        )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fluorescence (\u0394Rn)")
    apply_style(ax, style)
    return fig, ax


def plot_amplification_by_target(
    experiment: Experiment,
    target_name: str,
    style: PlotStyle | None = None,
) -> tuple[Figure, Axes]:
    """Plot amplification traces for wells of a single *target*.

    Traces are coloured by ``sample_name`` so biological/technical
    replicates of one assay can be compared side by side.

    Parameters
        experiment:
            The experiment containing ``traces`` and ``wells``.
        target_name:
            The target (assay) whose wells should be plotted.
        style:
            Optional :class:`PlotStyle`.

    Returns
        A ``(Figure, Axes)`` tuple.

    Raises
        ValueError:
            If the experiment has no traces, or no wells match *target_name*.
    """
    if not experiment.has_traces:
        raise ValueError(
            f"Experiment '{experiment.name}' has no amplification traces to plot."
        )

    target_wells = [w for w in experiment.wells if w.target_name == target_name]
    if not target_wells:
        raise ValueError(
            f"No wells found for target '{target_name}' in experiment "
            f"'{experiment.name}'."
        )

    s = style if style is not None else PlotStyle()
    fig, ax = create_figure(style)

    # Samples that have a trace for this target, sorted for determinism.
    samples_with_traces = sorted(
        {w.sample_name for w in target_wells if w.position in experiment.traces}
    )
    sample_colors = _category_colors(samples_with_traces, s)

    for well in target_wells:
        trace = experiment.traces.get(well.position)
        if trace is None:
            continue
        color = sample_colors.get(well.sample_name, s.color_palette[0])
        ax.plot(
            trace.cycles,
            trace.fluorescence,
            color=color,
            label=well.sample_name,
        )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fluorescence (\u0394Rn)")
    ax.set_title(f"{target_name} amplification")
    apply_style(ax, style)
    return fig, ax
