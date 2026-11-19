"""Dose-response curve models used for potency (IC50/EC50) determination.

All models are written in terms of ``x = log10(concentration)`` rather than
the raw concentration.  Fitting in log space is not a cosmetic choice: it
makes the parameters far less correlated (so the covariance matrix and the
resulting confidence intervals are meaningful), and it makes the fitted
``log_ec50`` parameter *directly* the quantity of interest.

Sign convention for the Hill slope
----------------------------------
Every model uses a single **signed** Hill parameter ``hill``:

.. math::

    y(x) = \\mathrm{bottom} + \\frac{\\mathrm{top} - \\mathrm{bottom}}
           {1 + 10^{(\\log_{10}\\mathrm{EC50} - x)\\,\\mathrm{hill}}}

* ``hill > 0`` -- the response *increases* with concentration (agonist or
  activator assays; the reported potency is an EC50).
* ``hill < 0`` -- the response *decreases* with concentration (inhibitor or
  cytotoxicity assays; the reported potency is an IC50).

The sign therefore encodes the assay direction, which is exactly what
distinguishes an EC50 from an IC50 at the level of the *curve*.  When the
result is reported to a user, :attr:`DoseResponseFit.hill_slope` exposes the
conventional positive magnitude while :attr:`DoseResponseFit.direction`
carries the sign.

Half-maximal position is not always the fitted parameter
--------------------------------------------------------
For the symmetric models (3PL/4PL) the half-way point between the two
plateaus sits exactly at ``log_ec50``.  For the asymmetric models it does
not: the 5PL inflection is shifted by the asymmetry factor and the Gompertz
inflection sits at ~36.8 % of the window rather than 50 %.  Reporting the
fitted ``log_ec50`` parameter as "the EC50" would therefore be *wrong* for
those models.  :func:`half_max_log_concentration` derives the true
half-maximal position from the fitted parameters in closed form for every
model, and the fitting layer always reports the value it returns rather
than the raw parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, unique
from typing import Any

import numpy as np

__all__ = [
    "MODEL_PARAMETERS",
    "DoseResponseModel",
    "evaluate_model",
    "five_pl",
    "four_pl",
    "gompertz",
    "half_max_log_concentration",
    "solve_log_concentration_for_response",
    "three_pl",
]

#: ``ln(ln(2))``; the Gompertz inflection sits at ``c - ln(ln 2) / hill``.
_LN_LN2: float = float(np.log(np.log(2.0)))


@unique
class DoseResponseModel(str, Enum):
    """Supported sigmoid dose-response models.

    Attributes
        THREE_PL:
            3-parameter logistic with the lower asymptote fixed at zero.
            Appropriate for responses normalised so the baseline is known
            to be 0 (e.g. "% of control" after blank subtraction).
        FOUR_PL:
            4-parameter logistic.  The default and most widely used model;
            assumes symmetry about the inflection point.
        FIVE_PL:
            5-parameter logistic.  Adds an asymmetry factor for curves that
            are steeper on one side (common for allosteric modulators and
            for data spanning a wide concentration range).
        GOMPERTZ:
            Gompertz growth/inhibition model.  Asymmetric with a sharper
            approach to the upper asymptote.
    """

    THREE_PL = "3pl"
    FOUR_PL = "4pl"
    FIVE_PL = "5pl"
    GOMPERTZ = "gompertz"


#: Parameter names, in the order they are passed to each model function.
#: ``log_ec50`` is ``log10`` of the concentration at the inflection point.
MODEL_PARAMETERS: dict[DoseResponseModel, tuple[str, ...]] = {
    DoseResponseModel.THREE_PL: ("top", "log_ec50", "hill"),
    DoseResponseModel.FOUR_PL: ("bottom", "top", "log_ec50", "hill"),
    DoseResponseModel.FIVE_PL: ("bottom", "top", "log_ec50", "hill", "asym"),
    DoseResponseModel.GOMPERTZ: ("bottom", "top", "log_ec50", "hill"),
}


def three_pl(
    x: np.ndarray, top: float, log_ec50: float, hill: float
) -> np.ndarray:
    """Evaluate the 3-parameter logistic with a zero lower asymptote.

    Parameters
        x:
            ``log10`` of the concentration.
        top:
            Upper asymptote of the response.
        log_ec50:
            ``log10`` of the inflection-point concentration.
        hill:
            Signed Hill slope (see the module docstring).

    Returns
        Modelled responses, aligned with *x*.
    """
    exponent = np.clip((log_ec50 - x) * hill, -300.0, 300.0)
    return top / (1.0 + np.power(10.0, exponent))


def four_pl(
    x: np.ndarray, bottom: float, top: float, log_ec50: float, hill: float
) -> np.ndarray:
    """Evaluate the symmetric 4-parameter logistic.

    Parameters
        x:
            ``log10`` of the concentration.
        bottom:
            Lower asymptote of the response.
        top:
            Upper asymptote of the response.
        log_ec50:
            ``log10`` of the inflection-point concentration.
        hill:
            Signed Hill slope (see the module docstring).

    Returns
        Modelled responses, aligned with *x*.
    """
    exponent = np.clip((log_ec50 - x) * hill, -300.0, 300.0)
    return bottom + (top - bottom) / (1.0 + np.power(10.0, exponent))


def five_pl(
    x: np.ndarray,
    bottom: float,
    top: float,
    log_ec50: float,
    hill: float,
    asym: float,
) -> np.ndarray:
    """Evaluate the 5-parameter logistic (asymmetric) model.

    Parameters
        x:
            ``log10`` of the concentration.
        bottom:
            Lower asymptote of the response.
        top:
            Upper asymptote of the response.
        log_ec50:
            ``log10`` of the inflection-point concentration.
        hill:
            Signed Hill slope (see the module docstring).
        asym:
            Asymmetry factor.  ``asym == 1`` reproduces the 4PL exactly;
            ``asym > 1`` makes the transition sharper near the upper
            asymptote, ``asym < 1`` near the lower one.

    Returns
        Modelled responses, aligned with *x*.
    """
    exponent = np.clip((log_ec50 - x) * hill, -300.0, 300.0)
    base = 1.0 + np.power(10.0, exponent)
    return bottom + (top - bottom) * np.power(base, -asym)


def gompertz(
    x: np.ndarray, bottom: float, top: float, log_ec50: float, hill: float
) -> np.ndarray:
    """Evaluate the Gompertz dose-response model.

    The parameter ``log_ec50`` locates the *inflection point* of the curve,
    which for Gompertz is not the half-maximal response (see
    :func:`half_max_log_concentration`).

    Parameters
        x:
            ``log10`` of the concentration.
        bottom:
            Lower asymptote of the response.
        top:
            Upper asymptote of the response.
        log_ec50:
            ``log10`` of the inflection-point concentration.
        hill:
            Signed Hill-like steepness parameter.

    Returns
        Modelled responses, aligned with *x*.
    """
    exponent = np.clip(-hill * (x - log_ec50), -300.0, 300.0)
    return bottom + (top - bottom) * np.exp(-np.exp(exponent))


_MODEL_FUNCTIONS: dict[DoseResponseModel, Callable[..., np.ndarray]] = {
    DoseResponseModel.THREE_PL: three_pl,
    DoseResponseModel.FOUR_PL: four_pl,
    DoseResponseModel.FIVE_PL: five_pl,
    DoseResponseModel.GOMPERTZ: gompertz,
}


def evaluate_model(
    model: DoseResponseModel, parameters: dict[str, Any], x: np.ndarray
) -> np.ndarray:
    """Evaluate *model* at ``x = log10(concentration)``.

    Parameters
        model:
            Which dose-response model to evaluate.
        parameters:
            Mapping of parameter name to value; must contain every name in
            :data:`MODEL_PARAMETERS` for *model*.
        x:
            Array of ``log10`` concentrations.

    Returns
        Modelled responses, aligned with *x*.

    Raises
        KeyError:
            If a required parameter is missing from *parameters*.
    """
    names = MODEL_PARAMETERS[model]
    values = [float(parameters[name]) for name in names]
    return _MODEL_FUNCTIONS[model](np.asarray(x, dtype=float), *values)


def half_max_log_concentration(
    model: DoseResponseModel, parameters: dict[str, Any]
) -> float:
    """Return ``log10`` of the true half-maximal concentration.

    This is the concentration at which the response is exactly halfway
    between the fitted lower and upper asymptotes -- the *relative* EC50 or
    IC50.  For 3PL and 4PL this coincides with the fitted ``log_ec50``
    parameter; for 5PL and Gompertz it does not, and the closed-form
    correction below is applied instead.

    Derivation
    ----------
    *4PL/3PL*: by symmetry, :math:`x_{1/2} = \\log_{10}\\mathrm{EC50}`.

    *5PL*: setting the logistic factor to ``1/2`` gives
    :math:`2^{1/a} - 1 = 10^{(\\log_{10}\\mathrm{EC50} - x)\\,h}`, hence
    :math:`x_{1/2} = \\log_{10}\\mathrm{EC50}
    - \\log_{10}\\!\\left(2^{1/a} - 1\\right) / h`.

    *Gompertz*: :math:`\\exp(-\\exp(-h(x-c))) = 1/2` gives
    :math:`x_{1/2} = c - \\ln(\\ln 2)/h`.

    Parameters
        model:
            Fitted dose-response model.
        parameters:
            Fitted parameters for *model*.

    Returns
        ``log10`` of the half-maximal concentration.

    Raises
        KeyError:
            If a required parameter is missing from *parameters*.
        ValueError:
            If the fitted parameters make the correction undefined (for
            example a zero Hill slope).
    """
    log_ec50 = float(parameters["log_ec50"])
    hill = float(parameters["hill"])
    if hill == 0.0:
        raise ValueError(
            "Hill slope is zero; the half-maximal concentration is undefined."
        )
    if model in (DoseResponseModel.THREE_PL, DoseResponseModel.FOUR_PL):
        return log_ec50
    if model is DoseResponseModel.GOMPERTZ:
        return log_ec50 - _LN_LN2 / hill
    asym = float(parameters["asym"])
    if asym <= 0.0:
        raise ValueError(
            f"Asymmetry factor must be positive, got {asym!r}."
        )
    # 2**(1/asym) - 1 is positive for every finite asym > 0; clip only to
    # guard against overflow for pathologically small asym.
    factor = float(2.0 ** (1.0 / asym) - 1.0)
    if factor <= 0.0 or not np.isfinite(factor):
        raise ValueError(
            f"Asymmetry factor {asym!r} does not yield a finite "
            "half-maximal concentration."
        )
    return log_ec50 - float(np.log10(factor)) / hill


def solve_log_concentration_for_response(
    model: DoseResponseModel,
    parameters: dict[str, Any],
    target_response: float,
) -> float:
    """Solve ``y(x) = target_response`` for ``x = log10(concentration)``.

    This is the machinery behind *absolute* potency: while the relative
    IC50/EC50 is defined by the midpoint of the fitted window, the absolute
    IC50/EC50 is defined by a user-chosen response level (most commonly 50 %
    of the untreated control).  The two coincide only when the lower
    asymptote is zero.

    Every supported model admits a closed-form solution, given below for
    the fraction :math:`f = (t - b) / (T - b)` where :math:`b`/``bottom``
    and :math:`T`/``top`` are the fitted asymptotes and :math:`t` is the
    target:

    * 4PL -- :math:`x = c - \\log_{10}\\!\\big((T-t)/(t-b)\\big) / h`
    * 3PL -- the 4PL solution with :math:`b = 0`
    * 5PL -- :math:`x = c - \\log_{10}\\!\\Big(\\big((T-b)/(t-b)\\big)^{1/a}
      - 1\\Big) / h`
    * Gompertz -- :math:`x = c - \\ln(-\\ln f) / h`

    Parameters
        model:
            Fitted dose-response model.
        parameters:
            Fitted parameters for *model*.
        target_response:
            Response level to invert.  Must lie strictly between the two
            fitted asymptotes for a finite solution to exist.

    Returns
        ``log10`` of the concentration producing *target_response*.

    Raises
        ValueError:
            If *target_response* is not strictly inside the fitted window,
            or if the parameters make the inversion undefined.
    """
    names = MODEL_PARAMETERS[model]
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(
            f"Missing parameter(s) for model {model.value!r}: {missing}."
        )
    top = float(parameters["top"])
    bottom = 0.0 if model is DoseResponseModel.THREE_PL else float(
        parameters["bottom"]
    )
    log_ec50 = float(parameters["log_ec50"])
    hill = float(parameters["hill"])

    lo, hi = min(bottom, top), max(bottom, top)
    if not lo < target_response < hi:
        raise ValueError(
            f"Target response {target_response:g} lies outside the fitted "
            f"window ({lo:g} .. {hi:g}); the potency is undefined "
            "(extrapolation is not reported)."
        )
    if hill == 0.0:
        raise ValueError(
            "Hill slope is zero; the concentration is undefined."
        )

    if model is DoseResponseModel.GOMPERTZ:
        fraction = (target_response - bottom) / (top - bottom)
        inner = -float(np.log(fraction))
        if inner <= 0.0:
            raise ValueError(
                "Gompertz inversion is undefined for the given target "
                "response."
            )
        return log_ec50 - float(np.log(inner)) / hill

    if model is DoseResponseModel.FIVE_PL:
        asym = float(parameters["asym"])
        if asym <= 0.0:
            raise ValueError(f"Asymmetry factor must be positive, got {asym!r}.")
        ratio = (top - bottom) / (target_response - bottom)
        base = float(ratio ** (1.0 / asym)) - 1.0
    else:
        base = (top - target_response) / (target_response - bottom)

    if base <= 0.0 or not np.isfinite(base):
        raise ValueError(
            "The requested response level is not reachable by the fitted "
            "model."
        )
    return log_ec50 - float(np.log10(base)) / hill
