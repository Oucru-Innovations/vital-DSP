import pytest
import numpy as np
from vitalDSP.feature_engineering.ecg_autonomic_features import ECGExtractor


@pytest.fixture
def generate_ecg_signal():
    """
    Fixture to generate a simulated ECG signal for testing.
    """
    # Simulate an ECG signal (this is a simplified example)
    t = np.linspace(0, 1, 1000)
    ecg_signal = np.sin(2 * np.pi * 5 * t)  # Example sinusoidal signal
    return ecg_signal, 250  # Signal and sampling frequency of 250 Hz


def test_detect_r_peaks(generate_ecg_signal):
    """
    Test R-peak detection in the ECG signal.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    r_peaks = extractor.detect_r_peaks()

    assert isinstance(r_peaks, np.ndarray), "R-peaks should return a numpy array"
    assert len(r_peaks) > 0, "R-peaks should detect at least one peak"


def test_compute_qrs_duration(generate_ecg_signal):
    """
    Test QRS duration calculation.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    qrs_duration = extractor.compute_qrs_duration()

    assert isinstance(qrs_duration, float), "QRS duration should return a float value"
    assert qrs_duration > 0, "QRS duration should be a positive value"


def test_compute_p_wave_duration(generate_ecg_signal):
    """
    Test P-wave duration calculation.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    p_wave_duration = extractor.compute_p_wave_duration()

    assert isinstance(
        p_wave_duration, float
    ), "P-wave duration should return a float value"
    assert p_wave_duration > 0, "P-wave duration should be a positive value"


def test_compute_pr_interval(generate_ecg_signal):
    """
    Test PR interval calculation.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    pr_interval = extractor.compute_pr_interval()

    assert isinstance(pr_interval, float), "PR interval should return a float value"
    assert pr_interval > 0, "PR interval should be a positive value"


def test_compute_qt_interval(generate_ecg_signal):
    """
    Test QT interval calculation.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    qt_interval = extractor.compute_qt_interval()

    assert isinstance(qt_interval, float), "QT interval should return a float value"
    assert qt_interval > 0, "QT interval should be a positive value"


def test_compute_st_segment(generate_ecg_signal):
    """
    Test ST segment calculation.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    st_segment = extractor.compute_st_segment()

    assert isinstance(st_segment, float), "ST segment should return a float value"


def test_detect_arrhythmias(generate_ecg_signal):
    """
    Test arrhythmia detection including AFib, VTach, and Bradycardia.
    """
    ecg_signal, fs = generate_ecg_signal
    extractor = ECGExtractor(ecg_signal, fs)
    arrhythmias = extractor.detect_arrhythmias()

    assert isinstance(arrhythmias, dict), "Arrhythmias should return a dictionary"
    assert (
        "AFib" in arrhythmias
    ), "AFib detection should be part of the arrhythmia detection"
    assert (
        "VTach" in arrhythmias
    ), "VTach detection should be part of the arrhythmia detection"
    assert (
        "Bradycardia" in arrhythmias
    ), "Bradycardia detection should be part of the arrhythmia detection"


def test_empty_signal():
    """
    Test behavior with an empty ECG signal.
    """
    ecg_signal = np.array([])  # Empty ECG signal
    fs = 250

    with pytest.raises(ValueError, match="ECG signal is too short to compute features"):
        ECGExtractor(ecg_signal, fs)


def test_invalid_signal_type():
    """
    Test behavior when passing a non-numeric ECG signal.
    """
    ecg_signal = "invalid_signal"  # Non-numeric input
    fs = 250

    with pytest.raises(TypeError, match="Input signal must be a numpy array"):
        ECGExtractor(ecg_signal, fs)


def test_single_value_signal():
    """
    Test behavior with a single-value ECG signal.
    """
    ecg_signal = np.array([1])  # Single-point signal
    fs = 250

    with pytest.raises(ValueError, match="ECG signal is too short to compute features"):
        ECGExtractor(ecg_signal, fs)


def test_no_peak_signal():
    """
    Test behavior when no peaks are detected in the ECG signal.
    """
    ecg_signal = np.ones(1000)  # Flatline signal with no peaks
    fs = 250

    extractor = ECGExtractor(ecg_signal, fs)

    with pytest.raises(ValueError, match="No R-peaks detected in ECG signal"):
        extractor.detect_r_peaks()


def test_non_numeric_signal():
    """
    Test behavior with a non-numeric signal (NaN or Inf values).
    """
    ecg_signal = np.array([np.nan] * 1000)  # Signal with NaN values
    fs = 250

    with pytest.raises(ValueError, match="ECG signal contains invalid values"):
        ECGExtractor(ecg_signal, fs)


def test_infinite_values_signal():
    """
    Test behavior with infinite values in the ECG signal.
    """
    ecg_signal = np.array([np.inf] * 1000)  # Signal with infinite values
    fs = 250

    with pytest.raises(ValueError, match="ECG signal contains invalid values"):
        ECGExtractor(ecg_signal, fs)
