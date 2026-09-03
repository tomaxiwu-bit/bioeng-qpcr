"""Tests for fermentation kinetics validation rules."""

from __future__ import annotations

import numpy as np

from bioeng_qpcr.fermentation_kinetics.validation import (
    FermentationValidationResult,
    validate_data_sufficiency,
    validate_mu_max_range,
)


class TestValidateMuMaxRange:
    def test_valid_mu_max(self):
        result = validate_mu_max_range(0.5)
        assert isinstance(result, FermentationValidationResult)
        assert result.rule_id == "FERM-001"
        assert result.passed is True

    def test_zero_mu_max(self):
        result = validate_mu_max_range(0.0)
        assert result.passed is False
        assert "positive" in result.message

    def test_negative_mu_max(self):
        result = validate_mu_max_range(-0.5)
        assert result.passed is False
        assert "positive" in result.message

    def test_inf_mu_max(self):
        result = validate_mu_max_range(float("inf"))
        assert result.passed is False
        assert "finite" in result.message

    def test_nan_mu_max(self):
        result = validate_mu_max_range(float("nan"))
        assert result.passed is False
        assert "finite" in result.message

    def test_too_large_mu_max(self):
        result = validate_mu_max_range(10.0)
        assert result.passed is False
        assert "range" in result.message

    def test_tiny_mu_max_passes(self):
        result = validate_mu_max_range(0.001)
        assert result.passed is True

    def test_too_tiny_mu_max_fails(self):
        result = validate_mu_max_range(0.0001)
        assert result.passed is False

    def test_evidence_has_mu_max(self):
        result = validate_mu_max_range(0.5)
        assert result.evidence["mu_max"] == 0.5
        assert "range" in result.evidence


class TestValidateDataSufficiency:
    def test_sufficient_data(self):
        time = np.linspace(0, 24, 12)
        biomass = np.exp(0.1 * time) * 0.1
        substrate = 10.0 - 0.4 * time
        result = validate_data_sufficiency(time, biomass, substrate)
        assert isinstance(result, FermentationValidationResult)
        assert result.rule_id == "FERM-002"
        assert result.passed is True

    def test_too_few_points(self):
        time = np.array([0.0, 1.0, 2.0])
        biomass = np.array([0.1, 0.2, 0.3])
        substrate = np.array([10.0, 9.0, 8.0])
        result = validate_data_sufficiency(time, biomass, substrate)
        assert result.passed is False
        assert "5" in result.message

    def test_no_growth(self):
        time = np.linspace(0, 24, 10)
        biomass = np.full(10, 0.1)  # constant
        substrate = 10.0 - 0.4 * time
        result = validate_data_sufficiency(time, biomass, substrate)
        assert result.passed is False
        assert "growth" in result.message.lower()

    def test_no_consumption(self):
        time = np.linspace(0, 24, 10)
        biomass = np.exp(0.1 * time) * 0.1
        substrate = np.full(10, 10.0)  # constant
        result = validate_data_sufficiency(time, biomass, substrate)
        assert result.passed is False
        assert "consumption" in result.message.lower()

    def test_evidence_has_n_points(self):
        time = np.linspace(0, 24, 12)
        biomass = np.exp(0.1 * time) * 0.1
        substrate = 10.0 - 0.4 * time
        result = validate_data_sufficiency(time, biomass, substrate)
        assert result.evidence["n_points"] == 12
        assert result.evidence["has_growth"] is True
        assert result.evidence["has_consumption"] is True
