"""Tests for parameter identifiability analysis."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.fermentation_kinetics.identifiability import (
    IdentifiabilityResult,
    analyze_identifiability,
)
from bioeng_qpcr.fermentation_kinetics.monod import fit_monod

from .conftest import make_fermentation_data, make_noisy_fermentation_data


class TestAnalyzeIdentifiability:
    def test_returns_result_type(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert isinstance(id_result, IdentifiabilityResult)

    def test_parameter_names(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert id_result.parameter_names == ("mu_max", "ks", "yxs")

    def test_correlation_matrix_shape(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert len(id_result.correlation_matrix) == 3
        for row in id_result.correlation_matrix:
            assert len(row) == 3

    def test_correlation_diagonal_near_one(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        for i in range(3):
            assert abs(id_result.correlation_matrix[i][i]) == pytest.approx(
                1.0, abs=0.1
            )

    def test_condition_number_positive(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert id_result.condition_number > 0
        assert np.isfinite(id_result.condition_number)

    def test_eigenvalues_positive(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert len(id_result.eigenvalues) > 0
        for e in id_result.eigenvalues:
            assert e > 0

    def test_batch_data_identifiability_analysis(self):
        """Batch fermentation data should produce a valid identifiability analysis.

        Note: Monod parameters (μmax, Ks) are inherently highly correlated
        in single-batch experiments because the substrate monotonically
        decreases through the transition region (S ≈ Ks) only once.  This
        test verifies the analysis correctly detects this structural
        correlation rather than expecting full identifiability.
        """
        time, biomass, substrate = make_noisy_fermentation_data(
            n_points=20, noise_level=0.03
        )
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        # μmax and Ks are expected to be highly correlated in batch data.
        corr_mu_ks = abs(id_result.correlation_matrix[0][1])
        assert corr_mu_ks > 0.8
        # Yxs should be less correlated with μmax and Ks.
        corr_mu_yxs = abs(id_result.correlation_matrix[0][2])
        assert corr_mu_yxs < corr_mu_ks
        # The analysis should produce a valid confidence level.
        assert id_result.confidence in ("high", "medium", "low")
        # Condition number should be positive and finite.
        assert id_result.condition_number > 0
        assert np.isfinite(id_result.condition_number)

    def test_message_contains_condition_number(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert "condition number" in id_result.message.lower()

    def test_message_contains_correlation(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert "correlation" in id_result.message.lower()

    def test_confidence_is_valid(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        id_result = analyze_identifiability(result, time, biomass, substrate)
        assert id_result.confidence in ("high", "medium", "low")
