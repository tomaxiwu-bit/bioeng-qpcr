"""Consistent, exportable figure templates over existing measurement results."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from bioeng_qpcr.plotting.style import PlotStyle, apply_style
from bioeng_qpcr.workbench.model import Project
from bioeng_qpcr.workbench.templates import THEMES


def figure_for(
    project: Project, kind: str, *, well: str | None = None, assay: str | None = None
) -> Figure:
    theme = THEMES[project.presentation["theme"]]
    palette = theme["palette"]
    figure = Figure(
        figsize=(7.2, 4.2), facecolor=theme["background"], layout="constrained"
    )
    if kind == "standard":
        analysis = project.outputs.get("analysis", {})
        if analysis.get("mode") != "absolute":
            raise ValueError("请先完成标准曲线定量")
        ax, residual = figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [3.5, 1]}, sharex=True
        )
        x, y = np.asarray(analysis["standard_points"]).T
        x = np.log10(x)
        fit = analysis["standard_curve"]
        prediction = fit["slope"] * x + fit["intercept"]
        ax.scatter(
            x,
            y,
            s=31,
            color=palette[0],
            alpha=0.8,
            edgecolor=theme["background"],
            linewidth=0.6,
            label="Observed standards",
            zorder=3,
        )
        xx = np.array([min(x), max(x)])
        ax.plot(
            xx,
            fit["slope"] * xx + fit["intercept"],
            color=palette[1],
            linewidth=1.7,
            label="Linear fit",
        )
        ax.set(ylabel="Ct", title="Standard curve")
        ax.text(
            0.98,
            0.96,
            f"slope = {fit['slope']:.3f}\nR² = {fit['r_squared']:.4f}\n"
            f"E = {fit['efficiency'] * 100:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            color=theme["ink"],
            bbox={
                "facecolor": theme["background"],
                "edgecolor": theme["grid"],
                "boxstyle": "round,pad=.6",
            },
        )
        residual.axhline(0, color=palette[1], linewidth=1, linestyle="--", alpha=0.8)
        residual.scatter(x, y - prediction, s=15, color=palette[0], alpha=0.85)
        residual.set(
            ylabel="Residual", xlabel=f"log10(standard quantity / {analysis['unit']})"
        )
    elif kind == "relative":
        analysis = project.outputs.get("analysis", {})
        if analysis.get("mode") != "relative":
            raise ValueError("请先完成相对定量")
        rows = [r for r in analysis["rows"] if r["status"] == "ok"]
        if not rows:
            raise ValueError("没有可绘制的有效定量结果")
        figure.set_size_inches(7.2, max(2.8, min(7.0, len(rows) * 0.48 + 1.5)))
        ax = figure.subplots()
        values = [r["fold_change"] for r in rows]
        indices = np.arange(len(rows))
        colors = [palette[i % len(palette)] for i in range(len(rows))]
        chart = project.presentation["relative_chart"]
        if chart == "bar":
            ax.barh(indices, values, color=colors, height=0.5, alpha=0.9)
        else:
            if chart == "lollipop":
                ax.hlines(indices, 0, values, colors=colors, linewidth=3, alpha=0.5)
            for i, (value, color) in enumerate(zip(values, colors, strict=True)):
                ax.scatter(
                    value,
                    i,
                    s=75 if chart == "lollipop" else 55,
                    marker=("o", "s", "D", "^", "v", "P")[i % 6],
                    color=color,
                    edgecolor=theme["background"],
                    linewidth=1.2,
                    zorder=4,
                )
        ax.axvline(
            1, color=theme["ink"], alpha=0.45, linestyle=(0, (4, 4)), linewidth=0.9
        )
        ax.set_yticks(indices, [r["sample"] for r in rows])
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.annotate(
                f"{value:.3g}×",
                (value, i),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=theme["ink"],
                fontsize=10,
            )
        ax.set(
            xlim=(0, max(values) * 1.25),
            ylim=(len(rows) - 0.5, -0.5),
            xlabel=(
                "Relative quantity / calibrator (efficiency corrected)"
                if analysis.get("relative_method") == "efficiency_corrected"
                else "Relative expression · 2^(−ΔΔCt)"
            ),
            title="Relative quantification",
        )
    elif kind in {"amplification", "melting"}:
        run = project.active_run
        if not run or not run["measurements"]:
            raise ValueError("没有采集数据")
        measurements = run["measurements"]
        curves = {
            (r["well"], r["assay_id"]): r
            for r in project.outputs.get("curves", {}).get("rows", [])
        }
        if well is not None or assay is not None:
            chosen = next(
                (
                    m
                    for m in measurements
                    if m["well"] == well and m["assay_id"] == assay
                ),
                None,
            )
            if chosen is None:
                raise ValueError("指定的孔位 / 检测项目没有采集记录")
        else:
            chosen = next(
                (
                    m
                    for m in measurements
                    if (
                        m["amplification"]
                        if kind == "amplification"
                        else curves.get((m["well"], m["assay_id"]), {}).get(
                            "derivative"
                        )
                    )
                ),
                measurements[0],
            )
        assignments = {(a["well"], a["assay_id"]): a for a in project.plate}
        chosen_assignment = assignments[(chosen["well"], chosen["assay_id"])]
        scope = project.presentation["curve_scope"]
        if scope == "selected":
            measurements = [chosen]
        elif scope == "replicates":
            measurements = [
                m
                for m in measurements
                if assignments[(m["well"], m["assay_id"])]["sample_id"]
                == chosen_assignment["sample_id"]
                and m["assay_id"] == chosen["assay_id"]
            ]
        ax = figure.subplots()
        groups, used, count = {}, set(), 0
        for measurement in measurements:
            key = (measurement["well"], measurement["assay_id"])
            result = curves.get(key, {})
            points = (
                measurement["amplification"]
                if kind == "amplification"
                else result.get("derivative", [])
            )
            if not points:
                continue
            sid = assignments[key]["sample_id"]
            index = groups.setdefault(sid, len(groups))
            color = palette[index % len(palette)]
            x, y = np.asarray(points).T
            highlight = key == (chosen["well"], chosen["assay_id"])
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2 if highlight else 1.0,
                alpha=1 if highlight else 0.45,
                linestyle=("-", "--", "-.", ":")[index % 4],
                label=sid if sid not in used and len(used) < 6 else None,
            )
            used.add(sid)
            if highlight and kind == "melting" and result.get("tm") is not None:
                ax.axvline(
                    result["tm"], color=color, linewidth=0.9, linestyle=":", alpha=0.65
                )
                ax.fill_between(x, 0, y, where=y > 0, color=color, alpha=0.1)
            count += 1
        if not count:
            raise ValueError("当前选择没有可用曲线；缺失数据不生成占位图")
        if kind == "amplification":
            ax.set(
                xlabel="Cycle",
                ylabel="Raw fluorescence",
                title=f"Amplification · {count} trace(s)",
            )
        else:
            ax.set(
                xlabel="Temperature (°C)",
                ylabel="−dF/dT",
                title=f"Melting profile · {count} trace(s)",
            )
    else:
        raise ValueError("未知图表类型")

    style = PlotStyle(
        font_size=9, title_size=12, color_palette=palette, grid_alpha=0.65
    )
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    for axis in figure.axes:
        apply_style(axis, style)
        axis.set_facecolor(theme["background"])
        axis.grid(True, color=theme["grid"], linewidth=0.55, linestyle=(0, (2, 4)))
        axis.tick_params(colors=theme["ink"], length=0, pad=8)
        for spine in axis.spines.values():
            spine.set_color(theme["grid"])
        for label in [
            axis.title,
            axis.xaxis.label,
            axis.yaxis.label,
            *axis.get_xticklabels(),
            *axis.get_yticklabels(),
        ]:
            label.set_color(theme["ink"])
            if font_path.exists():
                size = label.get_fontsize()
                label.set_fontproperties(
                    FontProperties(fname=str(font_path), size=size)
                )
        legend = axis.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_color(theme["ink"])
    return figure


def figure_bytes(project: Project, kind: str, fmt: str, **selection) -> bytes:
    if fmt not in {"svg", "png", "pdf"}:
        raise ValueError("图表支持 SVG、PNG 和 PDF")
    figure = figure_for(project, kind, **selection)
    try:
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format=fmt,
            dpi=300,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        return buffer.getvalue()
    finally:
        figure.clear()
