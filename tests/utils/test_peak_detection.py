import pytest
import numpy as np
from vitalDSP.utils.common import find_peaks
from vitalDSP.filtering.signal_filtering import SignalFiltering
from vitalDSP.utils.scaler import StandardScaler
from vitalDSP.utils.peak_detection import PeakDetection

@pytest.fixture
def sample_signal():
    """Fixture for a sample signal array."""
    return np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])  # Expect peak at index 4


def test_threshold_based_detection(sample_signal, monkeypatch):
    """Test threshold-based peak detection."""
    def mock_find_peaks(signal, height, distance):
        return np.array([4])  # Expect peak at index 4

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    detector = PeakDetection(sample_signal, method="threshold", threshold=4, distance=2)
    peaks = detector.detect_peaks()
    print(f"Threshold-based detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))  # Expect the peak at index 4


def test_savgol_based_detection(sample_signal, monkeypatch):
    """Test Savitzky-Golay filter-based peak detection."""
    def mock_find_peaks(signal):
        return np.array([4])  # Expect peak at index 4

    def mock_savgol_filter(signal, window_length, polyorder):
        return signal  # Return the original signal for simplicity

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    monkeypatch.setattr("vitalDSP.filtering.signal_filtering.SignalFiltering.savgol_filter", mock_savgol_filter)

    detector = PeakDetection(sample_signal, method="savgol", window_length=11, polyorder=2)
    peaks = detector.detect_peaks()
    print(f"Savgol-based detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))


def test_gaussian_based_detection(sample_signal, monkeypatch):
    """Test Gaussian filter-based peak detection."""
    def mock_find_peaks(signal):
        return np.array([4])  # Expect peak at index 4

    def mock_gaussian_filter1d(signal, sigma):
        return signal  # Return the original signal

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    monkeypatch.setattr("vitalDSP.filtering.signal_filtering.SignalFiltering.gaussian_filter1d", mock_gaussian_filter1d)

    detector = PeakDetection(sample_signal, method="gaussian", sigma=1)
    peaks = detector.detect_peaks()
    print(f"Gaussian-based detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))


def test_relative_extrema_detection(sample_signal, monkeypatch):
    """Test relative extrema-based peak detection."""
    def mock_argrelextrema(signal, comparator, order):
        return np.array([4])  # Expect peak at index 4

    monkeypatch.setattr("vitalDSP.utils.common.argrelextrema", mock_argrelextrema)

    detector = PeakDetection(sample_signal, method="rel_extrema", order=1)
    peaks = detector.detect_peaks()
    print(f"Relative extrema detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))


def test_scaler_threshold_based_detection(sample_signal, monkeypatch):
    """Test scaler-based peak detection."""
    class MockScaler:
        def fit_transform(self, signal):
            return signal  # Return the original signal as if it was standardized

    def mock_find_peaks(signal, height, distance):
        return np.array([4])  # Expect peak at index 4

    monkeypatch.setattr("vitalDSP.utils.scaler.StandardScaler", MockScaler)
    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)

    detector = PeakDetection(sample_signal, method="scaler_threshold", threshold=1, distance=2)
    peaks = detector.detect_peaks()
    print(f"Scaler-threshold detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))

def test_ecg_r_peak_detection(sample_signal, monkeypatch):
    """Test ECG R-peak detection."""
    def mock_find_peaks(signal, height, distance):
        print(f"Signal passed to find_peaks: {signal}")
        # Assuming the real signal should generate a peak at index 4
        return np.array([4])  # Simulate detecting a peak at index 4

    def mock_convolve(signal, window, mode):
        # Return a signal where we expect a peak at index 4 after convolution
        return np.array([0, 0, 0, 0, 5, 0, 0])  # Peak at index 4

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    monkeypatch.setattr("numpy.convolve", mock_convolve)  # Mocking convolution

    detector = PeakDetection(sample_signal, method="ecg_r_peak")
    peaks = detector.detect_peaks()
    print(f"ECG R-peak detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))


def test_ecg_derivative_detection(sample_signal, monkeypatch):
    """Test ECG derivative-based peak detection."""
    def mock_find_peaks(signal, height):
        print(f"Signal passed to find_peaks: {signal}")
        # Simulate detecting a peak at index 4
        return np.array([4])

    def mock_diff(signal):
        # Return a derivative where a peak is expected at index 4
        return np.array([0, 1, 1, 1, 5, 1, 1, 1])  # Peak at index 4

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    monkeypatch.setattr("numpy.diff", mock_diff)  # Mock the derivative

    detector = PeakDetection(sample_signal, method="ecg_derivative")
    peaks = detector.detect_peaks()
    print(f"ECG derivative detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))


def test_ppg_first_derivative_detection(sample_signal, monkeypatch):
    """Test PPG first derivative-based peak detection."""
    def mock_find_peaks(signal, height):
        print(f"Signal passed to find_peaks: {signal}")
        # Simulate detecting a peak at index 4
        return np.array([4])

    def mock_diff(signal):
        # Return a first derivative signal where a peak is expected at index 4
        return np.array([0, 1, 1, 1, 5, 1, 1, 1])  # Peak at index 4

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    monkeypatch.setattr("numpy.diff", mock_diff)  # Mock the first derivative

    detector = PeakDetection(sample_signal, method="ppg_first_derivative")
    peaks = detector.detect_peaks()
    print(f"PPG first derivative detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))

def test_ppg_second_derivative_detection(sample_signal, monkeypatch):
    """Test PPG second derivative-based peak detection."""
    def mock_find_peaks(signal, height):
        print(f"Signal passed to find_peaks: {signal}")
        # Simulate detecting a peak only at index 4
        return np.array([4])

    def mock_diff(signal, n=1):
        if n == 2:
            # Return a second derivative signal that has a single peak at index 4
            return np.array([0, 0, 0, 0, 5, 0, 0])  # Peak at index 4
        return np.diff(signal)

    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", mock_find_peaks)
    monkeypatch.setattr("numpy.diff", mock_diff)  # Mock the second derivative

    detector = PeakDetection(sample_signal, method="ppg_second_derivative")
    peaks = detector.detect_peaks()
    print(f"PPG second derivative detection peaks: {peaks}")
    assert np.array_equal(peaks, np.array([4]))
