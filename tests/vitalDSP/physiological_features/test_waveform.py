import pytest
import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.physiological_features.waveform import WaveformMorphology
from unittest.mock import MagicMock
from vitalDSP.utils.synthesize_data import generate_ecg_signal, generate_synthetic_ppg

@pytest.fixture
def sample_waveform():
    # Create a sample waveform with synthetic ECG-like features
    fs = 256  # Sampling frequency
    time = np.linspace(0, 2 * np.pi, fs)
    waveform = np.sin(2 * time) + 0.1 * np.random.randn(fs)
    return waveform, fs

@pytest.mark.parametrize("signal_type", ["ECG", "PPG", "EEG"])
def test_initialization(sample_waveform, signal_type):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type=signal_type)
    assert wm.waveform is not None
    assert wm.fs == fs
    assert wm.signal_type == signal_type

def test_compute_slope_ecg():
    fs = 256
    # waveform, fs = sample_waveform
    waveform = generate_ecg_signal(
        sfecg=fs, N=15, Anoise=0.01, hrmean=70
    )
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")

    # Now compute the slope at R peaks
    slopes = wm.compute_slope(option="r_peaks")
    assert slopes is not None
    assert isinstance(slopes, np.ndarray)

def test_compute_curvature_ppg(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    curvatures = wm.compute_curvature(option="systolic_peaks")
    assert curvatures is not None
    assert isinstance(curvatures, np.ndarray)

def test_detect_troughs(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    troughs = wm.detect_troughs()
    assert troughs is not None
    assert isinstance(troughs, np.ndarray)

def test_detect_q_valley(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    q_valleys = wm.detect_q_valley()
    assert q_valleys is not None
    assert isinstance(q_valleys, np.ndarray)

def test_detect_p_peak(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    p_peaks = wm.detect_p_peak()
    assert p_peaks is not None
    assert isinstance(p_peaks, np.ndarray)

def test_detect_s_valley(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    s_valleys = wm.detect_s_valley()
    assert s_valleys is not None
    assert isinstance(s_valleys, np.ndarray)

def test_compute_amplitude_r_to_s(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    amplitudes = wm.compute_amplitude(interval_type="R-to-S")
    assert amplitudes is not None
    assert isinstance(amplitudes, list)

def test_compute_volume_r_to_q(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    volumes = wm.compute_volume(interval_type="R-to-Q")
    assert volumes is not None
    assert isinstance(volumes, list)

def test_compute_duration_qrs(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    qrs_duration = wm.compute_duration(mode="QRS")
    assert qrs_duration is not None
    assert isinstance(qrs_duration, np.ndarray)

def test_compute_skewness(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    skewness_values = wm.compute_skewness()
    assert skewness_values is not None
    assert isinstance(skewness_values, list)

def test_invalid_signal_type():
    with pytest.raises(ValueError):
        WaveformMorphology(np.array([0.1, 0.2]), signal_type="Invalid")

def test_compute_slope_invalid_option(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_slope(option="invalid_option")

def test_compute_amplitude_invalid_interval(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_amplitude(interval_type="Invalid")

def test_compute_volume_invalid_interval(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_volume(interval_type="Invalid")

def test_compute_volume_invalid_mode(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_volume(mode="invalid_mode")
