"""Shared fixtures for the potency (IC50/EC50) test suite."""

from __future__ import annotations

import numpy as np

from bioeng_qpcr.potency.fitting import CurveDirection, DoseResponseDataset
from bioeng_qpcr.potency.models import (
    DoseResponseModel,
    evaluate_model,
    five_pl,
    four_pl,
)

#: Concentrations used throughout: 10 points spanning 4.5 log10 decades,
#: which is what a well-designed 96-well dose series looks like.
CONCENTRATIONS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])

#: Twelve points, for the tests that need more residuals than parameters.
DENSE_CONCENTRATIONS = np.array(
    [0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0]
)

#: True parameters of the reference inhibition curve.
TRUE_LOG_IC50 = float(np.log10(3.2))
TRUE_BOTTOM = 4.0
TRUE_TOP = 100.0
TRUE_HILL = -1.1


def make_inhibition_dataset(
    *,
    compound: str = "CompoundA",
    log_ic50: float = TRUE_LOG_IC50,
    hill: float = TRUE_HILL,
    bottom: float = TRUE_BOTTOM,
    top: float = TRUE_TOP,
    concentrations: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int = 7,
    replicates: int = 1,
    direction: CurveDirection = CurveDirection.AUTO,
) -> DoseResponseDataset:
    """Build a synthetic inhibition (descending) dose-response dataset.

    Parameters
        compound:
            Compound name.
        log_ic50:
            ``log10`` of the true IC50.
        hill:
            Signed Hill slope; negative for an inhibitor.
        bottom, top:
            Lower and upper asymptotes.
        concentrations:
            Concentration series; defaults to :data:`CONCENTRATIONS`.
        noise:
            Standard deviation of Gaussian noise added to the response.
        seed:
            Seed for the noise, so every test is reproducible.
        replicates:
            Number of observations per concentration level.
        direction:
            Direction hint stored on the dataset.

    Returns
        A :class:`~bioeng_qpcr.potency.DoseResponseDataset`.
    """
    series = CONCENTRATIONS if concentrations is None else concentrations
    generator = np.random.default_rng(seed)
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for concentration in series:
        predicted = float(
            four_pl(np.array([np.log10(concentration)]), bottom, top, log_ic50, hill)[0]
        )
        for replicate in range(replicates):
            value = predicted
            if noise:
                value += float(generator.normal(0.0, noise))
            xs.append(float(concentration))
            ys.append(value)
            labels.append(f"r{replicate + 1}")
    return DoseResponseDataset(
        compound=compound,
        concentration=np.array(xs),
        response=np.array(ys),
        concentration_unit="uM",
        response_unit="% activity",
        direction=direction,
        replicate_group=tuple(labels) if replicates > 1 else None,
    )


def make_activation_dataset(
    *,
    compound: str = "AgonistX",
    log_ec50: float = float(np.log10(0.8)),
    hill: float = 1.2,
    bottom: float = 2.0,
    top: float = 100.0,
    noise: float = 0.0,
    seed: int = 11,
) -> DoseResponseDataset:
    """Build a synthetic activation (ascending) dose-response dataset."""
    series = CONCENTRATIONS
    generator = np.random.default_rng(seed)
    responses = [
        float(four_pl(np.array([np.log10(c)]), bottom, top, log_ec50, hill)[0])
        + (float(generator.normal(0.0, noise)) if noise else 0.0)
        for c in series
    ]
    return DoseResponseDataset(
        compound=compound,
        concentration=series.copy(),
        response=np.array(responses),
        concentration_unit="uM",
        response_unit="luminescence",
    )


def make_asymmetric_dataset(
    *,
    compound: str = "Asymmetric",
    log_ec50: float = float(np.log10(3.0)),
    hill: float = -1.0,
    bottom: float = 3.0,
    top: float = 100.0,
    asym: float = 0.35,
    noise: float = 0.0,
    seed: int = 5,
) -> DoseResponseDataset:
    """Build a dataset generated from a genuinely 5PL (asymmetric) curve.

    A symmetric 4PL fit to this data leaves a systematic residual pattern, so
    the dataset is what the model-selection and residual-pattern tests need.
    """
    generator = np.random.default_rng(seed)
    responses = [
        float(five_pl(np.array([np.log10(c)]), bottom, top, log_ec50, hill, asym)[0])
        + (float(generator.normal(0.0, noise)) if noise else 0.0)
        for c in DENSE_CONCENTRATIONS
    ]
    return DoseResponseDataset(
        compound=compound,
        concentration=DENSE_CONCENTRATIONS.copy(),
        response=np.array(responses),
        concentration_unit="uM",
        response_unit="% activity",
    )


def make_hook_effect_dataset(
    *,
    compound: str = "HookEffect",
    log_ic50: float = float(np.log10(3.0)),
    hill: float = -1.4,
    bottom: float = 5.0,
    top: float = 100.0,
    hook_magnitude: float = 60.0,
    hook_onset_log10: float = 1.5,
    noise: float = 0.4,
    seed: int = 13,
) -> DoseResponseDataset:
    """Build a dataset showing the high-dose hook (prozone) effect.

    The response follows a normal descending sigmoid and then *rises again*
    at the highest concentrations.  This is a real pharmacological
    phenomenon (excess analyte saturating the capture reagent) and it
    invalidates a naive IC50, because the top of the curve is no longer a
    plateau.  The shape is not monotone, so no member of the sigmoid family
    can describe it and the fitted residuals keep an unmistakable pattern --
    which is exactly what the ``POT-009`` trend screen must catch.

    Note that a *linear* baseline drift is deliberately **not** used for this
    test: over a wide log-concentration range a sigmoid absorbs a linear ramp
    almost completely, leaving residuals with no detectable trend.

    Parameters
        compound:
            Compound name.
        log_ic50, hill, bottom, top:
            Parameters of the underlying descending sigmoid.
        hook_magnitude:
            Size of the rising term at the highest tested concentration.
        hook_onset_log10:
            ``log10`` concentration above which the hook starts to bite.
        noise:
            Standard deviation of the additive noise.
        seed:
            Noise seed.

    Returns
        A :class:`~bioeng_qpcr.potency.DoseResponseDataset`.
    """
    generator = np.random.default_rng(seed)
    log_series = np.log10(DENSE_CONCENTRATIONS)
    span = max(float(np.max(log_series)) - hook_onset_log10, 1e-9)
    responses = []
    for log_value in log_series:
        base = float(four_pl(np.array([log_value]), bottom, top, log_ic50, hill)[0])
        excess = max(log_value - hook_onset_log10, 0.0) / span
        responses.append(
            base + hook_magnitude * excess**2 + float(generator.normal(0.0, noise))
        )
    return DoseResponseDataset(
        compound=compound,
        concentration=DENSE_CONCENTRATIONS.copy(),
        response=np.array(responses),
        concentration_unit="uM",
        response_unit="% activity",
    )


def make_model_grid(
    model: DoseResponseModel, parameters: dict[str, float], x: np.ndarray
) -> np.ndarray:
    """Evaluate *model* at ``x = log10(concentration)`` for test convenience."""
    return evaluate_model(model, parameters, np.log10(x))
