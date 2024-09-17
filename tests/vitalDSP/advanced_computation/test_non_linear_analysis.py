import pytest
import numpy as np
import matplotlib.pyplot as plt
from vitalDSP.advanced_computation.non_linear_analysis import NonlinearAnalysis


@pytest.fixture
def test_signal():
    # Generate a test signal for use in tests
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)


@pytest.fixture
def nonlinear_analysis(test_signal):
    return NonlinearAnalysis(test_signal)


def test_lyapunov_exponent(nonlinear_analysis):
    # Test Lyapunov exponent calculation
    lyapunov = nonlinear_analysis.lyapunov_exponent(max_iter=100, epsilon=1e-8)
    assert isinstance(lyapunov, float)
    # Ensure the exponent is calculated (it might be positive or negative depending on signal dynamics)
    assert lyapunov != 0


def test_lyapunov_exponent_small_signal():
    # Test Lyapunov exponent with a small signal (for robustness)
    small_signal = np.random.random(10)
    nonlinear_analysis_small = NonlinearAnalysis(small_signal)
    lyapunov = nonlinear_analysis_small.lyapunov_exponent(max_iter=5, epsilon=1e-8)
    assert isinstance(lyapunov, float)
    assert lyapunov != 0


def test_poincare_plot(nonlinear_analysis, monkeypatch):
    # Test Poincaré plot generation, ensuring the plot is generated
    # Use monkeypatch to prevent the plot from actually being shown during the test
    def mock_show():
        pass

    monkeypatch.setattr(plt, "show", mock_show)
    figure = nonlinear_analysis.poincare_plot()
    assert figure is None  # Since Poincaré plot doesn't return a figure, just plots


def test_correlation_dimension(nonlinear_analysis):
    # Test correlation dimension calculation
    correlation_dim = nonlinear_analysis.correlation_dimension(radius=0.1)
    assert isinstance(correlation_dim, float)
    # Ensure correlation dimension is calculated (could be positive or negative depending on signal complexity)
    assert correlation_dim != 0


def test_correlation_dimension_large_radius():
    # Test correlation dimension with a larger radius
    large_radius_signal = np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.2, 50)
    nonlinear_analysis_large = NonlinearAnalysis(large_radius_signal)
    correlation_dim = nonlinear_analysis_large.correlation_dimension(radius=0.5)
    assert isinstance(correlation_dim, float)
    assert correlation_dim != 0
