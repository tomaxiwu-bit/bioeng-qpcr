"""Calculations module: qPCR quantification methods.

This package provides four relative-quantification methods that operate on
the :class:`~bioeng_qpcr.domain.entities.Experiment` domain model:

* **ΔCq** (:mod:`~bioeng_qpcr.calculations.delta_cq`) -- simple normalisation
  to a reference gene.
* **ΔΔCq** (:mod:`~bioeng_qpcr.calculations.delta_delta_cq`) -- Livak method;
  fold change relative to a control sample (assumes 100 % efficiency).
* **Pfaffl** (:mod:`~bioeng_qpcr.calculations.pfaffl`) -- efficiency-corrected
  relative quantification.
* **Replicate outliers** (:mod:`~bioeng_qpcr.calculations.outliers`) --
  Chauvenet-criterion detection of suspect technical-replicate wells.
* **Efficiency fitting** (:mod:`~bioeng_qpcr.calculations.efficiency`) --
  estimate amplification efficiency from standard curves or amplification
  traces.
* **Amplification curves** (:mod:`~bioeng_qpcr.calculations.amplification_curve`) --
  full per-well curve fitting: baseline estimation and correction, sigmoid
  Cq and per-well efficiency, plus whole-experiment aggregation.
"""

from __future__ import annotations

from bioeng_qpcr.calculations.amplification_curve import (
    AmplificationAnalysisResult,
    AmplificationFitResult,
    AssayEffSummary,
    BaselineEstimate,
    analyze_amplification_curves,
    apply_assay_efficiency,
    baseline_correct,
    estimate_baseline,
    fit_amplification_curve,
)
from bioeng_qpcr.calculations.batch import AnalysisExclusion, BatchAnalysisResult
from bioeng_qpcr.calculations.delta_cq import (
    DeltaCqResult,
    delta_cq,
    delta_cq_all,
    delta_cq_batch,
)
from bioeng_qpcr.calculations.delta_delta_cq import (
    DeltaDeltaCqResult,
    delta_delta_cq,
    delta_delta_cq_all,
    delta_delta_cq_batch,
)
from bioeng_qpcr.calculations.efficiency import (
    EfficiencyResult,
    fit_efficiency_from_standard_curve,
    fit_efficiency_from_trace,
    fit_logistic_trace_shape,
)
from bioeng_qpcr.calculations.normalized_quantity import (
    GroupComparisonResult,
    NormalizedQuantityResult,
    compare_groups,
    normalized_quantity,
    normalized_quantity_all,
)
from bioeng_qpcr.calculations.outliers import (
    ChauvenetFlag,
    ReplicateGroupReport,
    ReplicateOutlierResult,
    detect_replicate_outliers,
)
from bioeng_qpcr.calculations.pfaffl import (
    PfafflResult,
    pfaffl,
    pfaffl_all,
    pfaffl_batch,
)

__all__ = [
    # Amplification curves
    "AmplificationAnalysisResult",
    "AmplificationFitResult",
    "AnalysisExclusion",
    "AssayEffSummary",
    "BaselineEstimate",
    "BatchAnalysisResult",
    # Technical-replicate outlier detection
    "ChauvenetFlag",
    # ΔCq
    "DeltaCqResult",
    # ΔΔCq
    "DeltaDeltaCqResult",
    # Efficiency
    "EfficiencyResult",
    "GroupComparisonResult",
    # Efficiency-corrected normalized quantities
    "NormalizedQuantityResult",
    # Pfaffl
    "PfafflResult",
    "ReplicateGroupReport",
    "ReplicateOutlierResult",
    "compare_groups",
    "delta_cq",
    "delta_cq_all",
    "delta_cq_batch",
    "delta_delta_cq",
    "delta_delta_cq_all",
    "delta_delta_cq_batch",
    "detect_replicate_outliers",
    "estimate_baseline",
    "fit_amplification_curve",
    "fit_efficiency_from_standard_curve",
    "fit_efficiency_from_trace",
    "fit_logistic_trace_shape",
    "normalized_quantity",
    "normalized_quantity_all",
    "pfaffl",
    "pfaffl_all",
    "pfaffl_batch",
    "analyze_amplification_curves",
    "apply_assay_efficiency",
    "baseline_correct",
]
