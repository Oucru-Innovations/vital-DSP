import pytest
import numpy as np
from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis

@pytest.fixture
def test_signal():
    # Generating a sample ECG-like signal with some random noise
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    return signal

@pytest.fixture
def test_r_peaks():
    # Example R-peaks detected from an ECG or PPG signal (ensure variation in peaks)
    return np.array([10, 30, 52, 71, 98])

@pytest.fixture
def btb_analysis(test_signal, test_r_peaks):
    # Create an instance of BeatToBeatAnalysis
    return BeatToBeatAnalysis(signal=test_signal, r_peaks=test_r_peaks, fs=1000, signal_type="ECG")

def test_compute_rr_intervals(btb_analysis):
    # Test computation of R-R intervals without correction
    rr_intervals = btb_analysis.compute_rr_intervals()
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] == len(btb_analysis.r_peaks) - 1
    assert np.all(rr_intervals > 0)

def test_compute_rr_intervals_with_interpolation(btb_analysis):
    # Test R-R interval computation with interpolation correction
    rr_intervals = btb_analysis.compute_rr_intervals(correction_method='interpolation')
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] == len(btb_analysis.r_peaks) - 1

def test_compute_rr_intervals_with_resampling(btb_analysis):
    # Test R-R interval computation with resampling correction
    rr_intervals = btb_analysis.compute_rr_intervals(correction_method='resampling')
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] > 0

def test_compute_rr_intervals_with_adaptive_threshold(btb_analysis):
    # Test R-R interval computation with adaptive threshold correction
    rr_intervals = btb_analysis.compute_rr_intervals(correction_method='adaptive_threshold', threshold=100)
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] > 0

def test_compute_mean_rr(btb_analysis):
    # Test computation of mean R-R interval
    mean_rr = btb_analysis.compute_mean_rr()
    assert isinstance(mean_rr, float)
    assert mean_rr > 0

def test_compute_sdnn(btb_analysis):
    # Test computation of SDNN (standard deviation of R-R intervals)
    rr_intervals = btb_analysis.compute_rr_intervals()
    sdnn = btb_analysis.compute_sdnn()
    assert isinstance(sdnn, float)
    assert sdnn > 0  # SDNN should be positive if there's variability in R-R intervals
    assert np.std(rr_intervals) == sdnn  # Ensure the computed SDNN matches the standard deviation of R-R intervals

def test_compute_rmssd(btb_analysis):
    # Test computation of RMSSD (Root Mean Square of Successive Differences)
    rmssd = btb_analysis.compute_rmssd()
    assert isinstance(rmssd, float)
    assert rmssd > 0  # RMSSD should be positive for signals with variability

def test_compute_pnn50(btb_analysis):
    # Test computation of pNN50 (percentage of R-R intervals differing by more than 50 ms)
    pnn50 = btb_analysis.compute_pnn50()
    assert isinstance(pnn50, float)
    assert 0 <= pnn50 <= 100

def test_compute_hr(btb_analysis):
    # Test computation of heart rate from R-R intervals
    hr = btb_analysis.compute_hr()
    assert isinstance(hr, np.ndarray)
    assert hr.shape[0] == len(btb_analysis.r_peaks) - 1
    assert np.all(hr > 0)

def test_detect_arrhythmias(btb_analysis):
    # Test arrhythmia detection
    arrhythmias = btb_analysis.detect_arrhythmias(threshold=150)
    assert isinstance(arrhythmias, np.ndarray)
    assert arrhythmias.shape[0] >= 0  # Could have 0 or more arrhythmic beats
