"""Tests for Monod equation, ODE solving, and parameter fitting."""

from __future__ import annotations

import numpy as np
import pytest

from bioeng_qpcr.fermentation_kinetics.monod import (
    MonodFitResult,
    batch_fermentation_ode,
    fit_monod,
    monod_equation,
    solve_batch_fermentation,
)

from .conftest import make_fermentation_data


class TestMonodEquation:
    def test_half_max_at_ks(self):
        """At S = Ks, μ = μmax / 2."""
        mu = monod_equation(s=2.0, mu_max=1.0, ks=2.0)
        assert mu == pytest.approx(0.5)

    def test_zero_substrate(self):
        """At S = 0, μ = 0."""
        mu = monod_equation(s=0.0, mu_max=1.0, ks=2.0)
        assert mu == pytest.approx(0.0)

    def test_high_substrate_approaches_vmax(self):
        """At S >> Ks, μ approaches μmax."""
        mu = monod_equation(s=1000.0, mu_max=1.0, ks=2.0)
        assert mu == pytest.approx(1.0, abs=0.01)

    def test_array_input(self):
        s = np.array([0.0, 1.0, 2.0, 4.0])
        mu = monod_equation(s, mu_max=1.0, ks=2.0)
        assert mu.shape == (4,)
        assert mu[0] == pytest.approx(0.0)
        assert mu[2] == pytest.approx(0.5)


class TestBatchFermentationOde:
    def test_returns_two_derivatives(self):
        result = batch_fermentation_ode(0.0, [0.1, 10.0], 0.5, 2.0, 0.5)
        assert len(result) == 2

    def test_biomass_increases(self):
        """dX/dt should be positive when X > 0 and S > 0."""
        dxdt, _ = batch_fermentation_ode(0.0, [0.1, 10.0], 0.5, 2.0, 0.5)
        assert dxdt > 0

    def test_substrate_decreases(self):
        """dS/dt should be negative when X > 0 and S > 0."""
        _, dsdt = batch_fermentation_ode(0.0, [0.1, 10.0], 0.5, 2.0, 0.5)
        assert dsdt < 0

    def test_zero_biomass(self):
        """When X = 0, both derivatives should be 0."""
        dxdt, dsdt = batch_fermentation_ode(0.0, [0.0, 10.0], 0.5, 2.0, 0.5)
        assert dxdt == pytest.approx(0.0)
        assert dsdt == pytest.approx(0.0)


class TestSolveBatchFermentation:
    def test_returns_arrays(self):
        time = np.linspace(0, 10, 10)
        x, s = solve_batch_fermentation(time, 0.1, 10.0, 0.5, 2.0, 0.5)
        assert x.shape == (10,)
        assert s.shape == (10,)

    def test_biomass_starts_at_x0(self):
        time = np.linspace(0, 10, 10)
        x, _ = solve_batch_fermentation(time, 0.1, 10.0, 0.5, 2.0, 0.5)
        assert x[0] == pytest.approx(0.1)

    def test_substrate_starts_at_s0(self):
        time = np.linspace(0, 10, 10)
        _, s = solve_batch_fermentation(time, 0.1, 10.0, 0.5, 2.0, 0.5)
        assert s[0] == pytest.approx(10.0)

    def test_biomass_increases_over_time(self):
        time = np.linspace(0, 20, 20)
        x, _ = solve_batch_fermentation(time, 0.1, 10.0, 0.5, 2.0, 0.5)
        assert x[-1] > x[0]

    def test_substrate_decreases_over_time(self):
        time = np.linspace(0, 20, 20)
        _, s = solve_batch_fermentation(time, 0.1, 10.0, 0.5, 2.0, 0.5)
        assert s[-1] < s[0]

    def test_mass_balance_conserved(self):
        """Yxs * (S0 - S) should approximately equal (X - X0)."""
        time = np.linspace(0, 20, 50)
        x0, s0, yxs = 0.1, 10.0, 0.5
        x, s = solve_batch_fermentation(time, x0, s0, 0.5, 2.0, yxs)
        delta_x = x[-1] - x0
        delta_s = s0 - s[-1]
        assert delta_x == pytest.approx(yxs * delta_s, rel=0.01)


class TestFitMonod:
    def test_returns_result_type(self):
        time, biomass, substrate = make_fermentation_data()
        result = fit_monod(time, biomass, substrate)
        assert isinstance(result, MonodFitResult)

    def test_mu_max_recovered(self):
        time, biomass, substrate = make_fermentation_data(mu_max=0.5)
        result = fit_monod(time, biomass, substrate)
        assert result.mu_max == pytest.approx(0.5, rel=0.15)

    def test_ks_recovered(self):
        time, biomass, substrate = make_fermentation_data(ks=2.0)
        result = fit_monod(time, biomass, substrate)
        assert result.ks == pytest.approx(2.0, rel=0.2)

    def test_yxs_recovered(self):
        time, biomass, substrate = make_fermentation_data(yxs=0.5)
        result = fit_monod(time, biomass, substrate)
        assert result.yxs == pytest.approx(0.5, rel=0.15)

    def test_r_squared_high(self):
        """Clean data should give a high R²."""
        time, biomass, substrate = make_fermentation_data(noise_level=0.01)
        result = fit_monod(time, biomass, substrate)
        assert result.r_squared > 0.95

    def test_constant_responses_have_undefined_r_squared(self):
        time = np.arange(6.0)
        biomass = np.full(6, 0.1)
        substrate = np.full(6, 10.0)

        result = fit_monod(time, biomass, substrate)

        assert result.r_squared is None
        assert result.biomass_r_squared is None
        assert result.substrate_r_squared is None
        assert len(result.warnings) == 2
        assert all("zero variance" in warning for warning in result.warnings)

    def test_n_points(self):
        time, biomass, substrate = make_fermentation_data(n_points=12)
        result = fit_monod(time, biomass, substrate)
        assert result.n_points == 12

    def test_residuals_length(self):
        time, biomass, substrate = make_fermentation_data(n_points=10)
        result = fit_monod(time, biomass, substrate)
        assert len(result.residuals) == 20  # 2 * n_points

    def test_ci_returned(self):
        """Confidence intervals should be returned for noisy data."""
        time, biomass, substrate = make_fermentation_data(
            noise_level=0.03, n_points=15
        )
        result = fit_monod(time, biomass, substrate)
        assert result.mu_max_ci is not None
        assert result.ks_ci is not None
        assert result.yxs_ci is not None

    def test_more_measurement_noise_widens_parameter_intervals(self):
        clean = make_fermentation_data(noise_level=0.005, n_points=20)
        noisy = make_fermentation_data(noise_level=0.06, n_points=20)

        clean_result = fit_monod(*clean)
        noisy_result = fit_monod(*noisy)

        assert clean_result.mu_max_ci is not None
        assert noisy_result.mu_max_ci is not None
        clean_width = clean_result.mu_max_ci[1] - clean_result.mu_max_ci[0]
        noisy_width = noisy_result.mu_max_ci[1] - noisy_result.mu_max_ci[0]
        assert noisy_width > clean_width

    def test_confidence_intervals_remain_positive(self):
        data = make_fermentation_data(noise_level=0.05, n_points=12)
        result = fit_monod(*data)

        for interval in (result.mu_max_ci, result.ks_ci, result.yxs_ci):
            assert interval is not None
            assert interval[0] > 0

    def test_response_scales_are_recorded(self):
        result = fit_monod(*make_fermentation_data())

        assert result.biomass_scale > 0
        assert result.substrate_scale > 0
        assert result.ci_method == "scaled-log-wald-t"

    def test_x0_s0_recorded(self):
        time, biomass, substrate = make_fermentation_data(
            x0=0.1, s0=10.0, noise_level=0.0
        )
        result = fit_monod(time, biomass, substrate)
        assert result.x0 == pytest.approx(0.1)
        assert result.s0 == pytest.approx(10.0)


class TestFitMonodErrors:
    def test_too_few_points(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        biomass = np.array([0.1, 0.2, 0.3, 0.4])
        substrate = np.array([10.0, 9.0, 8.0, 7.0])
        with pytest.raises(ValueError, match="5"):
            fit_monod(time, biomass, substrate)

    def test_mismatched_shapes(self):
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        biomass = np.array([0.1, 0.2, 0.3, 0.4])
        substrate = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        with pytest.raises(ValueError, match="same shape"):
            fit_monod(time, biomass, substrate)

    def test_negative_time(self):
        time = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        biomass = np.array([0.1, 0.1, 0.2, 0.3, 0.4])
        substrate = np.array([10.0, 10.0, 9.0, 8.0, 7.0])
        with pytest.raises(ValueError, match="non-negative"):
            fit_monod(time, biomass, substrate)

    def test_negative_biomass(self):
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        biomass = np.array([-0.1, 0.1, 0.2, 0.3, 0.4])
        substrate = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        with pytest.raises(ValueError, match="non-negative"):
            fit_monod(time, biomass, substrate)


class TestReadFermentationCsv:
    def test_read_valid_csv(self, tmp_path):
        import csv

        csv_path = tmp_path / "fermentation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "biomass", "substrate"])
            writer.writerow([0.0, 0.1, 10.0])
            writer.writerow([1.0, 0.15, 9.5])
            writer.writerow([2.0, 0.22, 8.8])

        from bioeng_qpcr.fermentation_kinetics.monod import (
            read_fermentation_csv,
        )

        time, biomass, substrate = read_fermentation_csv(csv_path)
        assert len(time) == 3
        assert time[0] == 0.0
        assert biomass[0] == 0.1
        assert substrate[0] == 10.0

    def test_missing_column_raises(self, tmp_path):
        import csv

        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "biomass"])
            writer.writerow([0.0, 0.1])

        from bioeng_qpcr.fermentation_kinetics.monod import (
            read_fermentation_csv,
        )

        with pytest.raises(ValueError, match="substrate"):
            read_fermentation_csv(csv_path)
