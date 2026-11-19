"""Potency analysis: IC50/EC50 determination from dose-response data.

This package fits sigmoid dose-response curves and derives half-maximal
potency with an explicit distinction between the *relative* potency (the
midpoint of the fitted window) and the *absolute* potency (the concentration
producing a chosen fraction of the control response).  See
:mod:`bioeng_qpcr.potency.fitting` for the full discussion.

Typical use::

    from bioeng_qpcr.potency import fit_dose_response, run_potency_validation

    result = fit_dose_response(dataset)
    for check in run_potency_validation(result, dataset):
        if not check.passed:
            print(check.rule_id, check.message)
"""

from __future__ import annotations

from bioeng_qpcr.potency.fitting import (
    CurveDirection,
    DoseResponseDataset,
    DoseResponseFit,
    ModelComparison,
    PotencyComparison,
    PotencyKind,
    PotencyResult,
    compare_dose_response_models,
    fit_dose_response,
    fit_dose_response_batch,
    fit_dose_response_curve,
    relative_potency,
)
from bioeng_qpcr.potency.models import (
    MODEL_PARAMETERS,
    DoseResponseModel,
    evaluate_model,
    five_pl,
    four_pl,
    gompertz,
    half_max_log_concentration,
    solve_log_concentration_for_response,
    three_pl,
)
from bioeng_qpcr.potency.validation import (
    PotencyValidationResult,
    run_potency_validation,
    validate_assay_window,
    validate_ci_width,
    validate_concentration_levels,
    validate_concentration_span,
    validate_fit_quality,
    validate_hill_slope,
    validate_model_adequacy,
    validate_monotonicity,
    validate_plateau_support,
    validate_replicate_precision,
    validate_within_tested_range,
)

__all__ = [
    "MODEL_PARAMETERS",
    # fitting
    "CurveDirection",
    "DoseResponseDataset",
    "DoseResponseFit",
    # models
    "DoseResponseModel",
    "ModelComparison",
    "PotencyComparison",
    "PotencyKind",
    "PotencyResult",
    # validation
    "PotencyValidationResult",
    "compare_dose_response_models",
    "evaluate_model",
    "five_pl",
    "fit_dose_response",
    "fit_dose_response_batch",
    "fit_dose_response_curve",
    "four_pl",
    "gompertz",
    "half_max_log_concentration",
    "relative_potency",
    "run_potency_validation",
    "solve_log_concentration_for_response",
    "three_pl",
    "validate_assay_window",
    "validate_ci_width",
    "validate_concentration_levels",
    "validate_concentration_span",
    "validate_fit_quality",
    "validate_hill_slope",
    "validate_model_adequacy",
    "validate_monotonicity",
    "validate_plateau_support",
    "validate_replicate_precision",
    "validate_within_tested_range",
]
