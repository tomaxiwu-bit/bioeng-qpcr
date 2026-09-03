"""Tests for Michaelis-Menten equation fitting."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.enzyme_kinetics.michaelis_menten import (
    LineweaverBurkResult,
    MichaelisMentenResult,
    fit_lineweaver_burk,
    fit_michaelis_menten,
)


class TestFitMichaelisMenten:
    """Fit MM with known parameters (Vmax=100, Km=10)."""

    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        vmax, km = 100.0, 10.0
        v = vmax * s / (km + s)
        return s, v

    def test_returns_result_type(self, data):
        s, v = data
        result = fit_michaelis_menten(s, v)
        assert isinstance(result, MichaelisMentenResult)

    def test_vmax_recovered(self, data):
        s, v = data
        result = fit_michaelis_menten(s, v)
        assert result.vmax == pytest.approx(100.0, rel=0.01)

    def test_km_recovered(self, data):
        s, v = data
        result = fit_michaelis_menten(s, v)
        assert result.km == pytest.approx(10.0, rel=0.05)

    def test_r_squared_near_one(self, data):
        s, v = data
        result = fit_michaelis_menten(s, v)
        assert result.r_squared > 0.99

    def test_n_points(self, data):
        s, v = data
        result = fit_michaelis_menten(s, v)
        assert result.n_points == 6

    def test_ci_returned(self):
        """CI requires noisy data (perfect fit → zero covariance)."""
        rng = np.random.RandomState(42)
        s = np.geomspace(2.0, 100.0, 8)
        vmax, km = 100.0, 10.0
        v = vmax * s / (km + s)
        v += rng.normal(0, 2.0, 8)
        result = fit_michaelis_menten(s, v)
        assert result.vmax_ci is not None
        assert result.km_ci is not None
        assert len(result.vmax_ci) == 2
        assert result.vmax_ci[0] < result.vmax < result.vmax_ci[1]

    def test_residuals_length(self, data):
        s, v = data
        result = fit_michaelis_menten(s, v)
        assert len(result.residuals) == len(v)

    def test_with_noise(self):
        """Fit noisy data — should recover parameters approximately."""
        rng = np.random.RandomState(42)
        s = np.geomspace(2.0, 100.0, 8)
        vmax, km = 100.0, 10.0
        v = vmax * s / (km + s)
        v += rng.normal(0, 2.0, 8)
        result = fit_michaelis_menten(s, v)
        assert result.vmax == pytest.approx(100.0, rel=0.05)
        assert result.km == pytest.approx(10.0, rel=0.15)
        assert result.r_squared > 0.95

    def test_constant_velocity_has_undefined_r_squared(self):
        substrate = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        velocity = np.full(5, 3.0)

        result = fit_michaelis_menten(substrate, velocity)

        assert result.r_squared is None
        assert any("zero variance" in warning for warning in result.warnings)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            fit_michaelis_menten(np.array([1, 2, 3]), np.array([1, 2]))

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 4"):
            fit_michaelis_menten(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_nonpositive_substrate(self):
        with pytest.raises(ValueError, match="positive"):
            fit_michaelis_menten(
                np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4])
            )

    def test_negative_velocity(self):
        with pytest.raises(ValueError, match="non-negative"):
            fit_michaelis_menten(
                np.array([1, 2, 3, 4]), np.array([1, -1, 3, 4])
            )


class TestFitLineweaverBurk:
    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
        vmax, km = 100.0, 10.0
        v = vmax * s / (km + s)
        return s, v

    def test_returns_result_type(self, data):
        s, v = data
        result = fit_lineweaver_burk(s, v)
        assert isinstance(result, LineweaverBurkResult)

    def test_vmax_recovered(self, data):
        s, v = data
        result = fit_lineweaver_burk(s, v)
        assert result.vmax == pytest.approx(100.0, rel=0.01)

    def test_km_recovered(self, data):
        s, v = data
        result = fit_lineweaver_burk(s, v)
        assert result.km == pytest.approx(10.0, rel=0.05)

    def test_r_squared_near_one(self, data):
        s, v = data
        result = fit_lineweaver_burk(s, v)
        assert result.r_squared > 0.99

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 3"):
            fit_lineweaver_burk(np.array([1, 2]), np.array([1, 2]))

    def test_zero_velocity(self):
        with pytest.raises(ValueError, match="positive"):
            fit_lineweaver_burk(
                np.array([1, 2, 3]), np.array([0, 1, 2])
            )
