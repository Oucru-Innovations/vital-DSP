import numpy as np
import pytest
from filtering.signal_filtering import SignalFiltering
from filtering.advanced_signal_filtering import AdvancedSignalFiltering

@pytest.fixture
def sample_signal():
    """Fixture for creating a sample signal."""
    return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

def test_moving_average(sample_signal):
    filter = SignalFiltering(sample_signal)
    window_size = 5
    filtered_signal = filter.moving_average(window_size=window_size)
    assert len(filtered_signal) == (len(sample_signal)), "Filtered signal length mismatch"

def test_gaussian(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.gaussian(sigma=1.0)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gaussian filtering failed"

def test_butterworth(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.butterworth(fs=100,order=2, cutoff=0.3)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    # assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Butterworth filtering failed"

def test_median(sample_signal):
    filter = SignalFiltering(sample_signal)
    kernel_size = 3
    filtered_signal = filter.median(kernel_size=kernel_size)
    assert len(filtered_signal) == (len(sample_signal)), "Filtered signal length mismatch"
    # assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Median filtering failed"

def test_savgol_filter(sample_signal):
    filtered_signal = SignalFiltering.savgol_filter(sample_signal, window_length=5, polyorder=2)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Savitzky-Golay filtering failed"

def test_kalman_filter(sample_signal):
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.kalman_filter(R=1e-3, Q=1e-4)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Kalman filtering failed"

def test_optimization_based_filter(sample_signal):
    sample_signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])
    target = np.array([1, 1.5, 2.5, 3, 2.5, 2, 2.5, 3.5, 3, 2.5])
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.optimization_based_filtering(target,loss_type='mse')
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Optimization-based filtering failed"

def test_gradient_descent_filter(sample_signal):
    sample_signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])
    target = np.array([1, 1.5, 2.5, 3, 2.5, 2, 2.5, 3.5, 3, 2.5])
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.gradient_descent_filter(target,learning_rate=0.01,iterations=20)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    # assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gradient Descent filtering failed"

def test_ensemble_filtering(sample_signal):
    af = AdvancedSignalFiltering(sample_signal)
    filter = SignalFiltering(sample_signal)
    filters = [af.kalman_filter, af.kalman_filter]
    filtered_signal = af.ensemble_filtering(filters,method='mean')
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Ensemble filtering failed"

def test_convolution_based_filter(sample_signal):
    kernel = np.array([0.25, 0.5, 0.25])
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.convolution_based_filter(custom_kernel=kernel)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Convolution-based filtering failed"

def test_attention_based_filter(sample_signal):
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.attention_based_filter(attention_weights=np.array([0.1, 0.2, 0.3, 0.4]))
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Attention-based filtering failed"

if __name__ == "__main__":
    pytest.main()
