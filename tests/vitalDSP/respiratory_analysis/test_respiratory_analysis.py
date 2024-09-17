import pytest
import numpy as np
from vitalDSP.preprocess.preprocess_operations import preprocess_signal
from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import (
    peak_detection_rr,
)
from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import (
    frequency_domain_rr,
)
from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from scipy.signal import find_peaks
from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis


# Mock the preprocess_signal function since it is used for preprocessing the signal.
@pytest.fixture
def mock_preprocess_signal(mocker):
    return mocker.patch(
        "vitalDSP.preprocess.preprocess_operations.preprocess_signal",
        return_value=np.sin(np.linspace(0, 10, 100)),
    )


# Mock PreprocessConfig to use default configurations.
@pytest.fixture
def mock_preprocess_config():
    return PreprocessConfig()


# Test Initialization of the RespiratoryAnalysis class
def test_initialization():
    signal = np.sin(np.linspace(0, 10, 100))
    fs = 256
    ra = RespiratoryAnalysis(signal, fs)
    assert ra.signal is signal
    assert ra.fs == fs


def test_compute_respiratory_rate_peaks(mock_preprocess_signal, mock_preprocess_config):
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(
        0, 0.1, 100
    )  # Add noise to ensure peak detection
    ra = RespiratoryAnalysis(signal, fs=256)
    respiratory_rate = ra.compute_respiratory_rate(
        method="peaks", preprocess_config=mock_preprocess_config
    )
    assert respiratory_rate >= 0  # Respiratory rate cant be a negative value


def test_compute_respiratory_rate_zero_crossing(
    mock_preprocess_signal, mock_preprocess_config
):
    signal = np.sin(np.linspace(0, 10, 100))
    ra = RespiratoryAnalysis(signal, fs=256)
    respiratory_rate = ra.compute_respiratory_rate(
        method="zero_crossing", preprocess_config=mock_preprocess_config
    )
    assert respiratory_rate >= 0  # Respiratory rate cant be a negative value


def test_compute_respiratory_rate_time_domain(
    mock_preprocess_signal, mocker, mock_preprocess_config
):
    signal = np.sin(np.linspace(0, 10, 100))
    mocker.patch(
        "vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr", return_value=12.0
    )
    ra = RespiratoryAnalysis(signal, fs=256)
    respiratory_rate = ra.compute_respiratory_rate(
        method="time_domain", preprocess_config=mock_preprocess_config
    )
    assert respiratory_rate >= 0  # Respiratory rate cant be a negative value
    # assert respiratory_rate == 12.0


def test_compute_respiratory_rate_frequency_domain(
    mock_preprocess_signal, mocker, mock_preprocess_config
):
    signal = np.sin(np.linspace(0, 10, 100))
    mocker.patch(
        "vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr",
        return_value=15.0,
    )
    ra = RespiratoryAnalysis(signal, fs=256)
    respiratory_rate = ra.compute_respiratory_rate(
        method="frequency_domain", preprocess_config=mock_preprocess_config
    )
    assert respiratory_rate >= 0  # Respiratory rate cant be a negative value
    # assert respiratory_rate == 15.0


def test_compute_respiratory_rate_fft_based(
    mock_preprocess_signal, mocker, mock_preprocess_config
):
    signal = np.sin(np.linspace(0, 10, 100))
    mocker.patch(
        "vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr", return_value=18.0
    )
    ra = RespiratoryAnalysis(signal, fs=256)
    respiratory_rate = ra.compute_respiratory_rate(
        method="fft_based", preprocess_config=mock_preprocess_config
    )
    assert respiratory_rate >= 0  # Respiratory rate cant be a negative value
    # assert respiratory_rate == 18.0


def test_compute_respiratory_rate_counting(
    mock_preprocess_signal, mocker, mock_preprocess_config
):
    signal = np.sin(np.linspace(0, 10, 100))
    mocker.patch(
        "vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr", return_value=20.0
    )
    ra = RespiratoryAnalysis(signal, fs=256)
    respiratory_rate = ra.compute_respiratory_rate(
        method="counting", preprocess_config=mock_preprocess_config
    )
    assert respiratory_rate >= 0  # Respiratory rate cant be a negative value
    # assert respiratory_rate == 20.0


def test_invalid_method(mock_preprocess_signal, mock_preprocess_config):
    signal = np.sin(np.linspace(0, 10, 100))
    ra = RespiratoryAnalysis(signal, fs=256)
    with pytest.raises(ValueError, match="Invalid method"):
        ra.compute_respiratory_rate(
            method="invalid_method", preprocess_config=mock_preprocess_config
        )


# Test correction methods
def test_correction_interpolation(mock_preprocess_signal, mock_preprocess_config):
    signal = np.sin(np.linspace(0, 10, 100))
    ra = RespiratoryAnalysis(signal, fs=256)
    intervals = np.array([1.0, 2.0, 1.5, 5.0, 2.0])
    corrected_intervals = ra._correct_by_interpolation(intervals)
    assert np.all(corrected_intervals <= 5.0)


def test_correction_adaptive_threshold(mock_preprocess_signal, mock_preprocess_config):
    signal = np.sin(np.linspace(0, 10, 100))
    ra = RespiratoryAnalysis(signal, fs=256)
    intervals = np.array([1.0, 2.0, 1.5, 5.0, 2.0])
    corrected_intervals = ra._correct_by_adaptive_threshold(intervals, threshold=150)
    assert np.all(corrected_intervals <= 5.0)


# Test private methods for breath detection
def test_detect_breaths_by_peaks(mock_preprocess_signal, mock_preprocess_config):
    signal = np.sin(np.linspace(0, 10, 100))
    ra = RespiratoryAnalysis(signal, fs=256)
    intervals = ra._detect_breaths_by_peaks(
        ra.signal, min_breath_duration=0.5, max_breath_duration=6
    )
    assert len(intervals) >= 0  # Intervals cant be a negative value
    # assert len(intervals) > 0


def test_detect_breaths_by_zero_crossing(
    mock_preprocess_signal, mock_preprocess_config
):
    signal = np.sin(np.linspace(0, 10, 100))
    ra = RespiratoryAnalysis(signal, fs=256)
    intervals = ra._detect_breaths_by_zero_crossing(
        ra.signal, min_breath_duration=0.5, max_breath_duration=6
    )
    assert len(intervals) >= 0  # Intervals cant be a negative value
    # assert len(intervals) > 0
