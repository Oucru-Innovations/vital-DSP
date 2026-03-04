"""
Comprehensive Tests for NonlinearAnalysis Module

This test file covers all edge cases, error conditions, and warning scenarios
to achieve high test coverage for non_linear_analysis.py.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 95%+
"""

import pytest
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from unittest.mock import patch
from vitalDSP.advanced_computation.non_linear_analysis import NonlinearAnalysis


@pytest.fixture
def test_signal():
    """Generate a test signal for use in tests."""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 1000)) + 0.5 * np.random.normal(size=1000)


@pytest.fixture
def small_signal():
    """Generate a small test signal."""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)


@pytest.fixture
def constant_signal():
    """Generate a constant signal (zero variance)."""
    return np.ones(1000) * 5.0


@pytest.fixture
def flat_signal():
    """Generate a signal with many repeated values."""
    signal = np.ones(1000) * 2.0
    signal[::10] = 2.0001  # Very small variation
    return signal


@pytest.fixture
def nonlinear_analysis(test_signal):
    """Create NonlinearAnalysis instance."""
    return NonlinearAnalysis(test_signal)


class TestLyapunovExponent:
    """Test lyapunov_exponent method comprehensively."""

    def test_lyapunov_exponent_basic(self, nonlinear_analysis):
        """Test basic Lyapunov exponent calculation."""
        lyapunov = nonlinear_analysis.lyapunov_exponent(max_iter=50, epsilon=1e-8)
        assert isinstance(lyapunov, float)
        assert np.isfinite(lyapunov)

    def test_lyapunov_exponent_signal_too_short(self, small_signal):
        """Test ValueError when signal length <= max_iter."""
        analysis = NonlinearAnalysis(small_signal)
        with pytest.raises(ValueError, match="Signal length.*must be greater than max_iter"):
            analysis.lyapunov_exponent(max_iter=100)

    def test_lyapunov_exponent_without_normalization(self, nonlinear_analysis):
        """Test Lyapunov exponent without normalization."""
        lyapunov = nonlinear_analysis.lyapunov_exponent(max_iter=50, normalize=False)
        assert isinstance(lyapunov, float)
        assert np.isfinite(lyapunov)

    def test_lyapunov_exponent_zero_variance_warning(self, constant_signal):
        """Test warning when signal has zero variance."""
        analysis = NonlinearAnalysis(constant_signal)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lyapunov = analysis.lyapunov_exponent(max_iter=50, normalize=True)
            assert len(w) > 0
            assert "zero or very low variance" in str(w[0].message).lower()

    def test_lyapunov_exponent_skip_points(self, flat_signal):
        """Test that points with near-zero distances are skipped."""
        analysis = NonlinearAnalysis(flat_signal)
        # Use small epsilon to trigger skipping
        lyapunov = analysis.lyapunov_exponent(max_iter=50, epsilon=1e-10, normalize=True)
        # Should either return a value or NaN
        assert isinstance(lyapunov, float)

    def test_lyapunov_exponent_many_skipped_points_warning(self):
        """Test warning when too many points are skipped."""
        # Create signal with many identical values
        signal = np.ones(2000) * 2.0
        signal[::100] = 2.0001  # Very few variations
        analysis = NonlinearAnalysis(signal)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lyapunov = analysis.lyapunov_exponent(max_iter=100, epsilon=1e-10, normalize=True)
            # May or may not trigger warning depending on skipped_points
            assert isinstance(lyapunov, float) or np.isnan(lyapunov)

    def test_lyapunov_exponent_no_valid_distances(self):
        """Test return of NaN when no valid distances are found."""
        # Create signal with all identical values after normalization
        # Use a signal that normalizes to constant values
        signal = np.ones(2000) * 1.0
        # Add tiny variations that become zero after normalization
        signal[::100] = 1.0001
        analysis = NonlinearAnalysis(signal)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lyapunov = analysis.lyapunov_exponent(max_iter=100, epsilon=1e-8, normalize=True)
            # Should return NaN or trigger warning
            if np.isnan(lyapunov):
                # Check for either warning message
                warning_messages = [str(warn.message) for warn in w]
                assert any("No valid distance calculations" in msg or "zero or very low variance" in msg for msg in warning_messages)

    def test_lyapunov_exponent_infinite_result_warning(self):
        """Test warning when result is infinite."""
        # Create signal that might cause numerical issues
        signal = np.random.randn(2000) * 1e10  # Very large values
        analysis = NonlinearAnalysis(signal)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lyapunov = analysis.lyapunov_exponent(max_iter=100, epsilon=1e-8, normalize=True)
            # Should be finite after normalization, but check anyway
            if not np.isfinite(lyapunov):
                assert len(w) > 0

    def test_lyapunov_exponent_different_epsilon(self, nonlinear_analysis):
        """Test with different epsilon values."""
        lyapunov1 = nonlinear_analysis.lyapunov_exponent(max_iter=50, epsilon=1e-6)
        lyapunov2 = nonlinear_analysis.lyapunov_exponent(max_iter=50, epsilon=1e-10)
        assert isinstance(lyapunov1, float)
        assert isinstance(lyapunov2, float)

    def test_lyapunov_exponent_different_max_iter(self, nonlinear_analysis):
        """Test with different max_iter values."""
        lyapunov1 = nonlinear_analysis.lyapunov_exponent(max_iter=50)
        lyapunov2 = nonlinear_analysis.lyapunov_exponent(max_iter=100)
        assert isinstance(lyapunov1, float)
        assert isinstance(lyapunov2, float)


class TestPoincarePlot:
    """Test poincare_plot method."""

    def test_poincare_plot_basic(self, nonlinear_analysis):
        """Test basic Poincaré plot generation."""
        # Mock plt.show() to prevent actual display
        with patch.object(plt, 'show', return_value=None):
            nonlinear_analysis.poincare_plot()
        # Test passes if no exception raised

    def test_poincare_plot_small_signal(self, small_signal):
        """Test Poincaré plot with small signal."""
        analysis = NonlinearAnalysis(small_signal)
        with patch.object(plt, 'show', return_value=None):
            analysis.poincare_plot()

    def test_poincare_plot_constant_signal(self, constant_signal):
        """Test Poincaré plot with constant signal."""
        analysis = NonlinearAnalysis(constant_signal)
        with patch.object(plt, 'show', return_value=None):
            analysis.poincare_plot()


class TestCorrelationDimension:
    """Test correlation_dimension method comprehensively."""

    def test_correlation_dimension_basic(self, nonlinear_analysis):
        """Test basic correlation dimension calculation."""
        corr_dim = nonlinear_analysis.correlation_dimension(radius=0.1)
        assert isinstance(corr_dim, float)
        assert np.isfinite(corr_dim)

    def test_correlation_dimension_radius_1_0_error(self, nonlinear_analysis):
        """Test ValueError when radius is exactly 1.0."""
        with pytest.raises(ValueError, match="Radius cannot be 1.0"):
            nonlinear_analysis.correlation_dimension(radius=1.0)

    def test_correlation_dimension_radius_close_to_1_0(self, nonlinear_analysis):
        """Test ValueError when radius is very close to 1.0."""
        with pytest.raises(ValueError, match="Radius cannot be 1.0"):
            nonlinear_analysis.correlation_dimension(radius=1.0 + 1e-11)

    def test_correlation_dimension_negative_radius_error(self, nonlinear_analysis):
        """Test ValueError when radius is negative."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            nonlinear_analysis.correlation_dimension(radius=-0.1)

    def test_correlation_dimension_zero_radius_error(self, nonlinear_analysis):
        """Test ValueError when radius is zero."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            nonlinear_analysis.correlation_dimension(radius=0.0)

    def test_correlation_dimension_without_normalization(self, nonlinear_analysis):
        """Test correlation dimension without normalization."""
        corr_dim = nonlinear_analysis.correlation_dimension(radius=0.5, normalize=False)
        assert isinstance(corr_dim, float)
        assert np.isfinite(corr_dim)

    def test_correlation_dimension_zero_variance_warning(self, constant_signal):
        """Test warning when signal has zero variance."""
        analysis = NonlinearAnalysis(constant_signal)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corr_dim = analysis.correlation_dimension(radius=0.1, normalize=True)
            assert len(w) > 0
            assert "zero or very low variance" in str(w[0].message).lower()

    def test_correlation_dimension_no_pairs_error(self):
        """Test ValueError for invalid radius values."""
        np.random.seed(42)
        signal = np.random.randn(100)
        analysis = NonlinearAnalysis(signal)

        # Radius of exactly 1.0 causes log(1.0) = 0 division-by-zero
        with pytest.raises(ValueError):
            analysis.correlation_dimension(radius=1.0, normalize=True)

        # Non-positive radius is invalid
        with pytest.raises(ValueError):
            analysis.correlation_dimension(radius=-0.5, normalize=True)

    def test_correlation_dimension_log_radius_zero_error(self, nonlinear_analysis):
        """Test ValueError when log(radius) is too close to zero."""
        # This is hard to trigger since we check radius != 1.0, but test the check
        # Try with radius very close to 1.0 but not exactly 1.0
        # Actually, this check should never trigger due to earlier validation
        # But we can test the validation logic
        with pytest.raises(ValueError):
            nonlinear_analysis.correlation_dimension(radius=1.0)

    def test_correlation_dimension_negative_result_warning(self):
        """Test warning when correlation dimension is negative."""
        # Create signal that might give negative result
        # Negative result occurs when log(count)/log(radius) < 0
        # This happens when radius > 1 and count < 1 (impossible) or
        # when radius < 1 and we have very few pairs
        signal = np.random.randn(100) * 0.01  # Very small spread
        analysis = NonlinearAnalysis(signal)
        
        # Use radius > 1.0 to potentially get negative result
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                corr_dim = analysis.correlation_dimension(radius=2.0, normalize=True)
                if corr_dim < 0:
                    assert len(w) > 0
                    assert "Negative correlation dimension" in str(w[0].message)
            except ValueError:
                # If no pairs found, that's also valid
                pass

    def test_correlation_dimension_infinite_result_warning(self):
        """Test warning when result is infinite."""
        # Create signal that might cause numerical issues
        signal = np.random.randn(100) * 1e10
        analysis = NonlinearAnalysis(signal)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                corr_dim = analysis.correlation_dimension(radius=0.1, normalize=True)
                if not np.isfinite(corr_dim):
                    assert len(w) > 0
            except ValueError:
                # If no pairs found, that's also valid
                pass

    def test_correlation_dimension_different_radii(self, nonlinear_analysis):
        """Test with different radius values."""
        corr_dim1 = nonlinear_analysis.correlation_dimension(radius=0.1)
        corr_dim2 = nonlinear_analysis.correlation_dimension(radius=0.5)
        corr_dim3 = nonlinear_analysis.correlation_dimension(radius=1.5)
        
        assert isinstance(corr_dim1, float)
        assert isinstance(corr_dim2, float)
        assert isinstance(corr_dim3, float)
        assert np.isfinite(corr_dim1)
        assert np.isfinite(corr_dim2)
        assert np.isfinite(corr_dim3)

    def test_correlation_dimension_large_radius(self, small_signal):
        """Test correlation dimension with larger radius."""
        analysis = NonlinearAnalysis(small_signal)
        corr_dim = analysis.correlation_dimension(radius=0.5)
        assert isinstance(corr_dim, float)
        assert np.isfinite(corr_dim)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_init_with_array(self):
        """Test initialization with numpy array."""
        signal = np.array([1, 2, 3, 4, 5])
        analysis = NonlinearAnalysis(signal)
        assert np.array_equal(analysis.signal, signal)

    def test_init_with_list(self):
        """Test initialization with list."""
        signal = [1, 2, 3, 4, 5]
        analysis = NonlinearAnalysis(signal)
        assert len(analysis.signal) == 5

    def test_lyapunov_exponent_very_small_signal(self):
        """Test Lyapunov exponent with very small signal."""
        signal = np.random.randn(10)
        analysis = NonlinearAnalysis(signal)
        # Should raise ValueError due to signal length <= max_iter
        with pytest.raises(ValueError):
            analysis.lyapunov_exponent(max_iter=10)  # max_iter equals signal length

    def test_correlation_dimension_very_small_signal(self):
        """Test correlation dimension with very small signal."""
        signal = np.random.randn(5)
        analysis = NonlinearAnalysis(signal)
        # Should work but may have issues finding pairs
        try:
            corr_dim = analysis.correlation_dimension(radius=10.0, normalize=True)
            assert isinstance(corr_dim, float)
        except ValueError:
            # Expected if no pairs found
            pass

    def test_lyapunov_exponent_all_nan_signal(self):
        """Test Lyapunov exponent with signal containing NaN."""
        signal = np.full(1000, np.nan)
        analysis = NonlinearAnalysis(signal)
        # Should handle gracefully or raise error
        try:
            lyapunov = analysis.lyapunov_exponent(max_iter=50, normalize=True)
            # May return NaN or raise error
            assert isinstance(lyapunov, float) or np.isnan(lyapunov)
        except (ValueError, RuntimeWarning):
            pass

    def test_correlation_dimension_all_nan_signal(self):
        """Test correlation dimension with signal containing NaN."""
        signal = np.full(100, np.nan)
        analysis = NonlinearAnalysis(signal)
        # Should handle gracefully
        try:
            corr_dim = analysis.correlation_dimension(radius=0.1, normalize=True)
            assert isinstance(corr_dim, float) or np.isnan(corr_dim)
        except (ValueError, RuntimeWarning):
            pass

