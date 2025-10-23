import pytest
import numpy as np
from vitalDSP.physiological_features.cross_correlation import CrossCorrelationFeatures
from vitalDSP.utils.signal_processing.peak_detection import PeakDetection


# Mock the PeakDetection class to avoid actual peak detection logic in the tests
class MockPeakDetection:
    def __init__(self, signal, method="ppg_first_derivative"):
        self.signal = signal
        self.method = method

    def detect_peaks(self):
        # Return mock foot points for testing
        return np.array([60, 160, 260, 360])


@pytest.fixture
def sample_signals():
    np.random.seed(42)
    signal1 = np.sin(np.linspace(0, 10, 500))  # Mock ECG signal
    signal2 = np.sin(np.linspace(0, 10, 500)) + np.random.normal(
        0, 0.1, 500
    )  # Mock PPG signal
    return signal1, signal2


@pytest.fixture
def cross_corr_features(sample_signals, monkeypatch):
    signal1, signal2 = sample_signals
    monkeypatch.setattr(PeakDetection, "__init__", MockPeakDetection.__init__)
    monkeypatch.setattr(PeakDetection, "detect_peaks", MockPeakDetection.detect_peaks)
    return CrossCorrelationFeatures(signal1, signal2)


def test_compute_cross_correlation(cross_corr_features):
    cross_corr, lag = cross_corr_features.compute_cross_correlation(mode="full")

    assert isinstance(cross_corr, np.ndarray)
    assert isinstance(lag, int)  # Now lag should always be an int

    # Length of cross-correlation should be signal1_len + signal2_len - 1 for 'full' mode
    expected_length = (
        len(cross_corr_features.signal1) + len(cross_corr_features.signal2) - 1
    )
    assert (
        len(cross_corr) == expected_length
    ), f"Expected {expected_length}, got {len(cross_corr)}"

    # Lag should be within the expected range
    assert lag >= -(len(cross_corr_features.signal1) - 1) and lag <= (
        len(cross_corr_features.signal1) - 1
    )


def test_compute_normalized_cross_correlation(cross_corr_features):
    norm_corr, lag = cross_corr_features.compute_normalized_cross_correlation()

    assert isinstance(norm_corr, np.ndarray)
    assert isinstance(lag, int)

    # Check if all values in normalized cross-correlation are between -1 and 1
    assert np.all(norm_corr >= -1.0) and np.all(
        norm_corr <= 1.0
    ), f"Normalized correlation values are out of bounds: {norm_corr}"

    # Lag should be within the expected range
    assert lag >= -(len(cross_corr_features.signal1) - 1) and lag <= (
        len(cross_corr_features.signal1) - 1
    )


def test_compute_pulse_transit_time(cross_corr_features):
    r_peaks = np.array([50, 150, 250])  # Mock R-peaks in ECG
    ptt = cross_corr_features.compute_pulse_transit_time(r_peaks)
    assert isinstance(ptt, float)
    assert ptt > 0


def test_compute_lag(cross_corr_features):
    r_peaks = np.array([50, 150, 250])  # Mock R-peaks in ECG
    lag = cross_corr_features.compute_lag(r_peaks)
    assert isinstance(lag, float)
    assert lag > 0


def test_empty_r_peaks_ptt(cross_corr_features):
    # Test case where no valid R-peaks are provided
    r_peaks = np.array([])  # Empty R-peaks
    ptt = cross_corr_features.compute_pulse_transit_time(r_peaks)
    assert ptt == 0.0  # No valid PTT


def test_empty_r_peaks_lag(cross_corr_features):
    # Test case where no valid R-peaks are provided
    r_peaks = np.array([])  # Empty R-peaks
    lag = cross_corr_features.compute_lag(r_peaks)
    assert lag == 0.0  # No valid lag
