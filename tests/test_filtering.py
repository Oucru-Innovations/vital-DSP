import numpy as np
import pytest
from filtering import (MovingAverageFilter, GaussianFilter, ButterworthFilter, MedianFilter,
                       KalmanFilter, OptimizationBasedFilter, GradientDescentFilter, 
                       EnsembleFiltering, ConvolutionBasedFilter, AttentionBasedFilter)

@pytest.fixture
def sample_signal():
    """Fixture for creating a sample signal."""
    return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

# Test Basic Filters

def test_moving_average_filter(sample_signal):
    filter = MovingAverageFilter(window_size=5)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Moving average filtering failed"

def test_gaussian_filter(sample_signal):
    filter = GaussianFilter(sigma=1.0)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gaussian filtering failed"

def test_butterworth_filter(sample_signal):
    filter = ButterworthFilter(order=2, cutoff=0.3)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Butterworth filtering failed"

def test_median_filter(sample_signal):
    filter = MedianFilter(window_size=3)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Median filtering failed"

# Test Advanced Filters

def test_kalman_filter(sample_signal):
    filter = KalmanFilter(R=1e-3, Q=1e-4)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Kalman filtering failed"

def test_optimization_based_filter(sample_signal):
    filter = OptimizationBasedFilter(loss_function='mse')
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Optimization-based filtering failed"

def test_gradient_descent_filter(sample_signal):
    filter = GradientDescentFilter(learning_rate=0.01)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gradient Descent filtering failed"

def test_ensemble_filtering(sample_signal):
    filter = EnsembleFiltering(filters=[MovingAverageFilter(window_size=5), GaussianFilter(sigma=1.0)])
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Ensemble filtering failed"

def test_convolution_based_filter(sample_signal):
    kernel = np.array([0.25, 0.5, 0.25])
    filter = ConvolutionBasedFilter(kernel=kernel)
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Convolution-based filtering failed"

def test_attention_based_filter(sample_signal):
    filter = AttentionBasedFilter(attention_weights=np.array([0.1, 0.2, 0.3, 0.4]))
    filtered_signal = filter.apply(sample_signal)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Attention-based filtering failed"

if __name__ == "__main__":
    pytest.main()
