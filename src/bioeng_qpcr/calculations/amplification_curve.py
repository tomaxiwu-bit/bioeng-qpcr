"""Amplification-curve (扩大增长曲线) fitting and per-well quality metrics.

This module implements a full, auditable per-well amplification-curve
analysis pipeline.  The existing :func:`fit_logistic_trace_shape` fits a raw
sigmoid; this module goes further by separating the well-established analysis
steps so that each can be validated independently:

1. **Baseline estimation** -- identify the pre-amplification background
   window and estimate the background signal (constant mean, or linear
   regression for a drifting baseline).
2. **Baseline correction** -- subtract the background to obtain the
   background-normalized signal *ΔRn*.
3. **Sigmoid (logistic) fit** on the corrected signal -- the quantification
   cycle *Cq* is read from the fitted inflection point (first-derivative
   maximum), and a second-derivative-maximum *Cq* is also reported.
4. **Per-well efficiency** -- ordinary least squares on the log-linear
   (exponential) window of the corrected curve (LinRegPCR-style).
5. **Quality metrics** -- fit R², plateau amplitude *ΔRn*, and an explicit
   ``status`` (``ok`` / ``degraded`` / ``non_amplifying`` / ``failed``).

``analyze_amplification_curves`` runs the pipeline across an entire
:class:`~bioeng_qpcr.domain.entities.Experiment`, aggregates per-assay
median efficiency, and -- following the project's no-silent-discard rule --
retains every well that could not be fitted instead of dropping it.

Established-method notes
------------------------
The exponential-window efficiency is the widely used individual-curve
estimate (LinRegPCR / qPCR curve analysis): within the early exponential
phase, ``dR(c) ≈ dR0 · (1 + E)^(c - c0)``, so ``log dR`` is linear in cycle
with slope ``log(1 + E)``.  This is a per-well estimate and should be
cross-checked against a standard-curve efficiency rather than used as the
sole MIQE evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "AmplificationAnalysisResult",
    "AmplificationFitResult",
    "AssayEffSummary",
    "BaselineEstimate",
    "analyze_amplification_curves",
    "apply_assay_efficiency",
    "baseline_correct",
    "estimate_baseline",
    "fit_amplification_curve",
]

#: Efficiency below this is flagged as a degraded / implausible per-well value.
_EFF_LOW: float = 0.30
#: Efficiency above this is flagged as a degraded / implausible per-well value.
_EFF_HIGH: float = 2.50
#: Minimum fit R² before a well is considered degraded on curve quality.
_R2_GOOD: float = 0.98
#: Smallest number of cycles for a standard amplification curve.
_MIN_CYCLES: int = 4
#: Minimum number of points in the exponential window for an efficiency fit.
_MIN_EFF_POINTS: int = 3


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineEstimate:
    """Estimated background fluorescence before amplification commences.

    Attributes
        method:
            ``"constant"`` or ``"linear"`` (how the background was modelled).
        value:
            The background level evaluated at the first cycle (constant mean,
            or the linear-fit intercept).
        slope:
            Background drift (cycles⁻¹); always zero for ``"constant"``.
        takeoff_cycle:
            First cycle whose baseline-corrected signal exceeds baseline plus
            the detection tolerance, or ``None`` when no amplification is
            detected.
        n_baseline_cycles:
            Number of leading cycles used to estimate the background.
        warnings:
            Any caveats raised while estimating the baseline.
    """

    method: str
    value: float
    slope: float
    takeoff_cycle: float | None
    n_baseline_cycles: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AmplificationFitResult:
    """Per-well amplification-curve fit.

    Attributes
        position:
            Well address.
        status:
            One of ``"ok"``, ``"degraded"``, ``"non_amplifying"`` or
            ``"failed"``.
        cycles, dR:
            Cycle numbers and background-corrected fluorescence (ΔRn).
        baseline, baseline_method:
            Estimated background and how it was modelled.
        baseline_takeoff_cycle:
            Estimated amplification onset cycle (first cycle above baseline).
        cq:
            Primary quantification cycle -- the fitted logistic inflection
            point (first-derivative maximum).
        cq_sdm:
            Second-derivative-maximum cycle, if it could be determined.
        eff:
            Per-well amplification efficiency (fractional; 1.0 = 100 %).
        eff_slope:
            Slope of ``log dR`` versus cycle in the exponential window.
        r_squared:
            R² of the logistic fit to the corrected signal.
        delta_rn:
            Plateau amplitude of the corrected signal (ΔRn at saturation).
        n_points, n_eff_points:
            Number of cycles used for the curve fit and for the efficiency
            window respectively.
        warnings:
            Any caveats raised while fitting this well.
    """

    position: str
    status: str
    cycles: np.ndarray
    dR: np.ndarray
    baseline: float
    baseline_method: str
    baseline_takeoff_cycle: float | None
    cq: float | None
    cq_sdm: float | None
    eff: float | None
    eff_slope: float | None
    r_squared: float | None
    delta_rn: float | None
    n_points: int
    n_eff_points: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssayEffSummary:
    """Median per-well amplification efficiency for one target assay."""

    target: str
    median: float | None
    n_wells: int
    values: tuple[float, ...]
    min: float | None
    max: float | None


@dataclass(frozen=True)
class AmplificationAnalysisResult:
    """Whole-experiment amplification-curve analysis.

    Attributes
        per_well:
            Mapping of well position to its :class:`AmplificationFitResult`.
        per_assay:
            Mapping of target name to :class:`AssayEffSummary`.
        trace_cq:
            Curve-derived quantification cycle by well position (``None`` for
            wells that could not be quantified).
        trace_target:
            Mapping of well position to the target assay it belongs to, when
            the well link is available.
        wells_without_trace:
            Wells with no raw fluorescence trace (retained, not discarded).
        traces_without_well:
            Raw traces with no matching well row (retained, not discarded).
        non_amplifying:
            Positions whose traces show no amplification.
        warnings:
            Any cross-well caveats.
    """

    per_well: dict[str, AmplificationFitResult]
    per_assay: dict[str, AssayEffSummary]
    trace_cq: dict[str, float | None]
    trace_target: dict[str, str]
    wells_without_trace: tuple[str, ...]
    traces_without_well: tuple[str, ...]
    non_amplifying: tuple[str, ...]
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Baseline estimation & correction
# ---------------------------------------------------------------------------


def _detection_tolerance(fluorescence: np.ndarray) -> float:
    """A scale above the minimum that indicates real signal, not noise."""
    fmin = float(np.min(fluorescence))
    fmax = float(np.max(fluorescence))
    lead_n = max(2, len(fluorescence) // 5)
    lead_noise = float(np.median(np.abs(np.diff(fluorescence[:lead_n]))))
    dynamic = fmax - fmin
    tol = max(8.0 * lead_noise, 0.08 * dynamic)
    if not np.isfinite(tol) or tol <= 0.0:
        tol = max(dynamic * 0.08, float(fmin) * 1e-4, 1e-12)
    return tol


def estimate_baseline(
    fluorescence: np.ndarray,
    *,
    method: str = "constant",
) -> BaselineEstimate:
    """Estimate the pre-amplification background of a raw fluorescence trace.

    Amplification onset is detected as the first cycle whose fluorescence
    exceeds the trace minimum by a noise-aware tolerance.  The background is
    then modelled over the leading window up to that onset either as a
    constant (its mean) or as a linear regression (for a drifting baseline).

    Parameters
        fluorescence:
            Raw fluorescence readings, one per cycle.
        method:
            ``"constant"`` (default) or ``"linear"`` background model.

    Returns
        :class:`BaselineEstimate`.

    Raises
        ValueError:
            If *fluorescence* has fewer than 3 points.
    """
    fluorescence = np.asarray(fluorescence, dtype=float)
    if fluorescence.size < 3:
        raise ValueError(
            f"Need at least 3 cycles to estimate a baseline, got {fluorescence.size}."
        )
    if not np.all(np.isfinite(fluorescence)):
        raise ValueError("Baseline estimation requires finite fluorescence values.")
    if method not in {"constant", "linear"}:
        raise ValueError(
            f"Unknown baseline method {method!r}; choose 'constant' or 'linear'."
        )

    warnings: list[str] = []
    fmin = float(np.min(fluorescence))
    tol = _detection_tolerance(fluorescence)

    # Amplification onset = first cycle clearly above the minimum.
    above = np.flatnonzero(fluorescence > fmin + tol)
    takeoff_index = int(above[0]) if above.size else None
    window_n = (
        max(2, takeoff_index) if takeoff_index is not None else len(fluorescence)
    )
    if takeoff_index is None or takeoff_index < 1:
        warnings.append(
            "No amplification detected above the fluctuation tolerance; "
            "baseline is taken over the whole trace."
        )
        takeoff_cycle: float | None = None
    else:
        takeoff_cycle = float(takeoff_index)

    window = fluorescence[:window_n]
    if method == "constant":
        value = float(np.mean(window))
        slope = 0.0
    else:
        x = np.arange(window_n, dtype=float)
        coeffs = np.polyfit(x, window, 1)
        value, slope = float(coeffs[1]), float(coeffs[0])
        if np.isclose(slope, 0.0, atol=1e-12):
            warnings.append(
                "Linear baseline fit produced a negligible slope; consider "
                "the constant baseline."
            )

    return BaselineEstimate(
        method=method,
        value=value,
        slope=slope,
        takeoff_cycle=takeoff_cycle,
        n_baseline_cycles=int(window_n),
        warnings=tuple(warnings),
    )


def baseline_correct(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    baseline: BaselineEstimate,
) -> tuple[np.ndarray, float | None]:
    """Return the background-corrected signal ``dR = F - baseline``.

    The baseline is evaluated at each cycle (constant, or the linear drift)
    and subtracted.  Returns ``(dR, takeoff_cycle)``.
    """
    cycles = np.asarray(cycles, dtype=float)
    fluorescence = np.asarray(fluorescence, dtype=float)
    if baseline.method == "linear":
        corrected = fluorescence - (baseline.value + baseline.slope * cycles)
    else:
        corrected = fluorescence - baseline.value
    return corrected, baseline.takeoff_cycle


# ---------------------------------------------------------------------------
# Sigmoid fit on the corrected signal
# ---------------------------------------------------------------------------


def _logistic(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """Three-parameter logistic on baseline-corrected signal."""
    return L / (1.0 + np.exp(-k * (x - x0)))


def _fit_logistic(
    cycles: np.ndarray, dR: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit the 3PL ``L/(1+exp(-k(x-x0)))`` to corrected signal."""
    L0 = float(np.max(dR) - np.min(dR))
    if L0 <= 0.0:
        raise ValueError("Baseline-corrected signal has no positive amplitude.")
    peak_index = int(np.argmax(dR))
    if np.max(dR) > 0:
        x0_0 = float(cycles[peak_index])
    else:
        x0_0 = float(cycles[len(cycles) // 2])
    k0 = 0.5
    p0 = [L0, k0, x0_0]
    pad = max(1.0, float(np.ptp(cycles)))
    cmin, cmax = float(np.min(cycles)), float(np.max(cycles))
    lower = [1e-9, 1e-6, cmin - pad]
    upper = [float(np.max(np.abs(dR))) * 50.0 + 1.0, 20.0, cmax + pad]
    try:
        popt, _ = curve_fit(
            _logistic,
            cycles,
            dR,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Logistic fit failed: {exc}") from exc
    L, k, x0 = popt
    if L <= 0.0:
        raise ValueError("Logistic fit returned a non-positive plateau amplitude.")
    predicted = _logistic(cycles, L, k, x0)
    ss_res = float(np.sum((dR - predicted) ** 2))
    ss_tot = float(np.sum((dR - np.mean(dR)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return np.array([L, k, x0]), r_squared


def _second_derivative_max_cycle(cycles: np.ndarray, params: np.ndarray) -> float:
    """Cycle where the fitted corrected signal's second derivative is maximal."""
    L, k, x0 = params
    span = float(cycles[-1] - cycles[0])
    grid = np.linspace(
        float(cycles[0]) - 0.5 * span,
        float(cycles[-1]) + 0.5 * span,
        4001,
    )
    e = np.exp(-k * (grid - x0))
    second = L * k**2 * e * (e - 1.0) / (1.0 + e) ** 3
    idx = int(np.argmax(second))
    peak = float(grid[idx])
    return float(np.clip(peak, float(np.min(cycles)), float(np.max(cycles))))


# ---------------------------------------------------------------------------
# Per-well efficiency from the exponential window
# ---------------------------------------------------------------------------


def _exponential_window_efficiency(
    cycles: np.ndarray, dR: np.ndarray, plateau: float
) -> tuple[float | None, float | None, int]:
    """OLS slope of ``log dR`` vs cycle inside the early exponential phase.

    Within the fixed, auditable window ``[0.05·plateau, 0.40·plateau]`` of the
    baseline-corrected signal, ``log dR`` is regressed on cycle number; the
    slope gives the per-cycle log amplification factor, so
    ``eff = exp(slope) - 1``.  The window is deliberately held in the early
    rising region so the estimate does not include plateau saturation, at the
    cost of underestimating the instantaneous slope for a symmetric logistic.

    Returns ``(eff, slope, n_points)``, or ``(None, None, 0)`` when fewer
    than :data:`_MIN_EFF_POINTS` corrected points fall inside the window.
    """
    idx = np.flatnonzero((dR > 0.05 * plateau) & (dR <= 0.40 * plateau))
    if idx.size < _MIN_EFF_POINTS:
        return None, None, 0
    xs = cycles[idx]
    ys = np.log(dR[idx])
    if not np.all(np.isfinite(ys)):
        return None, None, 0
    slope = float(np.polyfit(xs, ys, 1)[0])
    eff = float(np.exp(slope) - 1.0)
    return eff, slope, int(idx.size)


# ---------------------------------------------------------------------------
# Per-well public entry point
# ---------------------------------------------------------------------------


def fit_amplification_curve(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    *,
    position: str = "?",
    baseline_method: str = "constant",
    min_cycles: int = _MIN_CYCLES,
) -> AmplificationFitResult:
    """Fit a single amplification curve and return per-well metrics.

    Parameters
        cycles:
            Array of cycle numbers.
        fluorescence:
            Array of raw fluorescence readings, one per cycle.
        position:
            Well address (used in warnings and the returned result).
        baseline_method:
            ``"constant"`` (default) or ``"linear"`` background model.
        min_cycles:
            Minimum number of cycles required (default 4).

    Returns
        :class:`AmplificationFitResult`.

    Raises
        ValueError:
            On mismatched or insufficient data, or a totally flat trace.
    """
    cycles = np.asarray(cycles, dtype=float)
    fluorescence = np.asarray(fluorescence, dtype=float)
    if cycles.shape != fluorescence.shape:
        raise ValueError(
            f"Well {position}: cycles and fluorescence must have the same shape, "
            f"got {cycles.shape} vs {fluorescence.shape}."
        )
    if cycles.size < min_cycles:
        raise ValueError(
            f"Well {position}: need at least {min_cycles} cycles, got {cycles.size}."
        )
    if not np.all(np.isfinite(cycles)) or not np.all(np.isfinite(fluorescence)):
        raise ValueError(f"Well {position}: trace values must be finite.")

    warnings: list[str] = []

    # 1. Baseline.
    try:
        baseline_est = estimate_baseline(fluorescence, method=baseline_method)
    except ValueError as exc:
        return AmplificationFitResult(
            position=position,
            status="failed",
            cycles=cycles,
            dR=fluorescence - float(np.mean(fluorescence)),
            baseline=float(np.mean(fluorescence)),
            baseline_method=baseline_method,
            baseline_takeoff_cycle=None,
            cq=None,
            cq_sdm=None,
            eff=None,
            eff_slope=None,
            r_squared=None,
            delta_rn=None,
            n_points=int(cycles.size),
            n_eff_points=0,
            warnings=(str(exc),),
        )
    warnings.extend(baseline_est.warnings)

    # 2. Correct.
    dR, takeoff = baseline_correct(cycles, fluorescence, baseline_est)
    dynamic = float(np.max(dR) - np.min(dR))
    if takeoff is None:
        return AmplificationFitResult(
            position=position,
            status="non_amplifying",
            cycles=cycles,
            dR=dR,
            baseline=baseline_est.value,
            baseline_method=baseline_method,
            baseline_takeoff_cycle=None,
            cq=None,
            cq_sdm=None,
            eff=None,
            eff_slope=None,
            r_squared=None,
            delta_rn=dynamic if dynamic > 0 else None,
            n_points=int(cycles.size),
            n_eff_points=0,
            warnings=tuple(warnings),
        )

    # 3. Sigmoid fit + Cq.
    try:
        params, r_squared = _fit_logistic(cycles, dR)
    except ValueError as exc:
        return AmplificationFitResult(
            position=position,
            status="failed",
            cycles=cycles,
            dR=dR,
            baseline=baseline_est.value,
            baseline_method=baseline_method,
            baseline_takeoff_cycle=takeoff,
            cq=None,
            cq_sdm=None,
            eff=None,
            eff_slope=None,
            r_squared=None,
            delta_rn=dynamic if dynamic > 0 else None,
            n_points=int(cycles.size),
            n_eff_points=0,
            warnings=(str(exc),),
        )
    L, k, x0 = params
    cq = float(x0)
    delta_rn = float(L)
    cq_sdm = _second_derivative_max_cycle(cycles, params)

    # 4. Per-well efficiency.
    eff, eff_slope, n_eff_points = _exponential_window_efficiency(
        cycles, dR, plateau=float(L)
    )

    # 5. Classify.
    status = "ok"
    if eff is None:
        status = "degraded"
        warnings.append("Exponential window too short to estimate efficiency.")
    else:
        if not (_EFF_LOW <= eff <= _EFF_HIGH):
            status = "degraded"
            warnings.append(
                f"Per-well efficiency {eff:.3f} is outside the plausible "
                f"[{_EFF_LOW:.2f}, {_EFF_HIGH:.2f}] range."
            )
    if r_squared < _R2_GOOD:
        status = "degraded"
        warnings.append(f"Logistic fit R² ({r_squared:.3f}) is below {_R2_GOOD:.2f}.")

    return AmplificationFitResult(
        position=position,
        status=status,
        cycles=cycles,
        dR=dR,
        baseline=baseline_est.value,
        baseline_method=baseline_method,
        baseline_takeoff_cycle=takeoff,
        cq=cq,
        cq_sdm=cq_sdm,
        eff=eff,
        eff_slope=eff_slope,
        r_squared=r_squared,
        delta_rn=delta_rn,
        n_points=int(cycles.size),
        n_eff_points=int(n_eff_points),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Whole-experiment analysis
# ---------------------------------------------------------------------------


def analyze_amplification_curves(
    experiment,
    *,
    baseline_method: str = "constant",
    min_cycles: int = _MIN_CYCLES,
) -> AmplificationAnalysisResult:
    """Analyse every raw amplification trace in an experiment.

    Fits each available :class:`~...RawTrace`, links wells to targets where
    possible, aggregates per-assay median efficiency, and retains (rather than
    discards) wells without traces and traces without wells.

    Parameters
        experiment:
            An :class:`~...Experiment` whose ``traces`` dict is non-empty.
        baseline_method:
            Per-well baseline model (``"constant"`` or ``"linear"``).
        min_cycles:
            Minimum cycles required per well.

    Returns
        :class:`AmplificationAnalysisResult`.
    """
    from bioeng_qpcr.domain.entities import Experiment

    if not isinstance(experiment, Experiment):
        raise TypeError("analyze_amplification_curves expects an Experiment.")
    if not experiment.traces:
        return AmplificationAnalysisResult(
            per_well={},
            per_assay={},
            trace_cq={},
            trace_target={},
            wells_without_trace=tuple(w.position for w in experiment.wells),
            traces_without_well=(),
            non_amplifying=(),
            warnings=("No raw amplification traces were provided.",),
        )

    position_to_well = {w.position: w for w in experiment.wells}
    position_to_target: dict[str, str] = {}

    fits: dict[str, AmplificationFitResult] = {}
    non_amplifying: list[str] = []
    wells_without_trace: list[str] = []
    for well in experiment.wells:
        if well.position not in experiment.traces:
            wells_without_trace.append(well.position)
    traces_without_well: list[str] = []

    for position, trace in sorted(experiment.traces.items()):
        well = position_to_well.get(position)
        if well is None:
            traces_without_well.append(position)
            continue
        position_to_target[position] = well.target_name
        try:
            fit = fit_amplification_curve(
                trace.cycles,
                trace.fluorescence,
                position=position,
                baseline_method=baseline_method,
                min_cycles=min_cycles,
            )
        except ValueError as exc:
            fit = AmplificationFitResult(
                position=position,
                status="failed",
                cycles=trace.cycles,
                dR=trace.fluorescence - float(np.mean(trace.fluorescence)),
                baseline=float(np.mean(trace.fluorescence)),
                baseline_method=baseline_method,
                baseline_takeoff_cycle=None,
                cq=None,
                cq_sdm=None,
                eff=None,
                eff_slope=None,
                r_squared=None,
                delta_rn=None,
                n_points=int(trace.cycles.size),
                n_eff_points=0,
                warnings=(str(exc),),
            )
        fits[position] = fit
        if fit.status == "non_amplifying":
            non_amplifying.append(position)

    # Per-assay median efficiency over wells that were fitted and linked.
    assay_values: dict[str, list[float]] = {}
    for position, fit in fits.items():
        target = position_to_target.get(position)
        if target is None or fit.eff is None:
            continue
        assay_values.setdefault(target, []).append(fit.eff)

    per_assay: dict[str, AssayEffSummary] = {}
    for target, values in sorted(assay_values.items()):
        arr = np.asarray(values, dtype=float)
        per_assay[target] = AssayEffSummary(
            target=target,
            median=float(np.median(arr)),
            n_wells=int(arr.size),
            values=tuple(float(v) for v in np.sort(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )

    trace_cq: dict[str, float | None] = {
        position: (fits[position].cq if fits[position].status != "failed" else None)
        for position in fits
    }

    return AmplificationAnalysisResult(
        per_well=fits,
        per_assay=per_assay,
        trace_cq=trace_cq,
        trace_target=position_to_target,
        wells_without_trace=tuple(sorted(wells_without_trace)),
        traces_without_well=tuple(sorted(traces_without_well)),
        non_amplifying=tuple(sorted(non_amplifying)),
        warnings=(
            "Per-assay efficiency is a median of per-well estimates and should "
            "be cross-checked against a standard-curve efficiency.",
        ),
    )


def apply_assay_efficiency(
    experiment, result: AmplificationAnalysisResult, *, replace: bool = False
) -> None:
    """Copy curve-derived median efficiency into assay metadata when missing.

    ``replace=False`` (default) only fills assays that currently carry no
    efficiency, so pre-recorded standard-curve values are never silently
    overwritten.  Passing ``replace=True`` forces an explicit override.
    """
    from bioeng_qpcr.domain.entities import Experiment

    if not isinstance(experiment, Experiment):
        raise TypeError("apply_assay_efficiency expects an Experiment.")
    for target, summary in result.per_assay.items():
        assay = experiment.assays.get(target)
        if assay is None or summary.median is None:
            continue
        if replace or assay.efficiency is None:
            object.__setattr__(assay, "efficiency", summary.median)