"""Tests for enzyme kinetics validation rules."""

from __future__ import annotations

import numpy as np

from bioeng_qpcr.enzyme_kinetics.validation import (
    KineticsValidationResult,
    validate_km_range,
    validate_substrate_span,
)


class TestValidateKmRange:
    def test_valid_km(self):
        result = validate_km_range(10.0)
        assert isinstance(result, KineticsValidationResult)
        assert result.passed is True
        assert result.rule_id == "ENZ-001"

    def test_zero_km(self):
        result = validate_km_range(0.0)
        assert result.passed is False
        assert "positive" in result.message

    def test_negative_km(self):
        result = validate_km_range(-5.0)
        assert result.passed is False

    def test_inf_km(self):
        result = validate_km_range(float("inf"))
        assert result.passed is False
        assert "finite" in result.message

    def test_nan_km(self):
        result = validate_km_range(float("nan"))
        assert result.passed is False

    def test_too_large_km(self):
        result = validate_km_range(1e6)
        assert result.passed is False
        assert "range" in result.message

    def test_tiny_km_passes(self):
        """1 nM is within the plausible range."""
        result = validate_km_range(1e-9)
        assert result.passed is True

    def test_evidence_has_km(self):
        result = validate_km_range(10.0)
        assert "km" in result.evidence
        assert result.evidence["km"] == 10.0


class TestValidateSubstrateSpan:
    def test_good_span(self):
        substrate = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        result = validate_substrate_span(substrate, km=10.0)
        assert isinstance(result, KineticsValidationResult)
        assert result.passed is True
        assert result.rule_id == "ENZ-002"

    def test_all_below_km(self):
        substrate = np.array([0.5, 1.0, 2.0])
        result = validate_substrate_span(substrate, km=10.0)
        assert result.passed is False
        assert "below" in result.message

    def test_all_above_km(self):
        substrate = np.array([50.0, 100.0, 200.0])
        result = validate_substrate_span(substrate, km=10.0)
        assert result.passed is False
        assert "above" in result.message

    def test_invalid_km(self):
        substrate = np.array([1, 2, 3])
        result = validate_substrate_span(substrate, km=0.0)
        assert result.passed is False

    def test_evidence_has_span_info(self):
        substrate = np.array([1.0, 10.0, 50.0])
        result = validate_substrate_span(substrate, km=10.0)
        assert "has_below_km" in result.evidence
        assert "has_above_km" in result.evidence
        assert result.evidence["n_points"] == 3
