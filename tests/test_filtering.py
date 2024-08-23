import numpy as np
import pytest
from src.filtering.signal_filtering import SignalFiltering

@pytest.fixture
def sample_signal():
    """Fixture for creating a sample signal."""
    return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

def test_moving_average(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.moving_average(window_size=5)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Moving average filtering failed"

def test_gaussian(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.gaussian(sigma=1.0)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gaussian filtering failed"

def test_butterworth(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.butterworth(order=2, cutoff=0.3)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Butterworth filtering failed"

def test_median(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.median(window_size=3)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Median filtering failed"

def test_savgol_filter(sample_signal):
    filtered_signal = SignalFiltering.savgol_filter(sample_signal, window_length=5, polyorder=2)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Savitzky-Golay filtering failed"

if __name__ == "__main__":
    pytest.main()
