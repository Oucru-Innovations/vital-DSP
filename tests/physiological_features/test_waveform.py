import pytest
import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.physiological_features.waveform import WaveformMorphology

# Mock the PeakDetection class to use for testing
class MockPeakDetection:
    def __init__(self, waveform, method=None):
        self.waveform = waveform
        self.method = method

    def detect_peaks(self):
        if self.method == "ppg_second_derivative":
            return np.array([5, 15, 25])  # Mocked notches
        elif self.method == "ecg_r_peak":
            return np.array([10, 30, 50])  # Mocked R peaks
        elif self.method == "ppg_first_derivative":
            return np.array([10, 20, 30])  # Mocked systolic peaks
        else:
            return np.array([5, 10, 20, 30])  # Mocked peaks for general cases

@pytest.fixture
def mock_waveform():
    return np.sin(np.linspace(0, 2 * np.pi, 100))

@pytest.fixture
def waveform_morphology(mock_waveform):
    return WaveformMorphology(mock_waveform, signal_type="ECG")

def test_detect_troughs(mocker, waveform_morphology):
    # Mocking the PeakDetection class
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    peaks = np.array([10, 30, 50])
    troughs = waveform_morphology.detect_troughs(peaks)
    
    assert isinstance(troughs, np.ndarray)
    assert len(troughs) == 2

def test_detect_notches_ppg(mocker, mock_waveform):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    wm = WaveformMorphology(mock_waveform, signal_type="PPG")
    notches = wm.detect_notches()
    
    assert isinstance(notches, np.ndarray)
    # assert len(notches) == 3

def test_detect_notches_invalid_signal_type(mocker, mock_waveform):
    wm = WaveformMorphology(mock_waveform, signal_type="ECG")
    
    with pytest.raises(ValueError):
        wm.detect_notches()

def test_detect_q_session(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    r_peaks = np.array([10, 30, 50])
    q_sessions = waveform_morphology.detect_q_session(r_peaks)
    
    assert isinstance(q_sessions, list)
    assert all(isinstance(session, tuple) for session in q_sessions)

def test_detect_q_session_invalid_signal_type(mocker, mock_waveform):
    wm = WaveformMorphology(mock_waveform, signal_type="PPG")
    
    with pytest.raises(ValueError):
        wm.detect_q_session(np.array([10, 20, 30]))

def test_detect_s_session(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    r_peaks = np.array([10, 30, 50])
    s_sessions = waveform_morphology.detect_s_session(r_peaks)
    
    assert isinstance(s_sessions, list)
    assert all(isinstance(session, tuple) for session in s_sessions)

def test_detect_s_session_invalid_signal_type(mocker, mock_waveform):
    wm = WaveformMorphology(mock_waveform, signal_type="PPG")
    
    with pytest.raises(ValueError):
        wm.detect_s_session(np.array([10, 20, 30]))

def test_detect_qrs_session(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    r_peaks = np.array([10, 30, 50])
    qrs_sessions = waveform_morphology.detect_qrs_session(r_peaks)
    
    assert isinstance(qrs_sessions, list)
    assert len(qrs_sessions) == 3

def test_compute_amplitude(waveform_morphology):
    amplitude = waveform_morphology.compute_amplitude()
    
    assert isinstance(amplitude, float)
    assert amplitude > 0

def test_compute_volume(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    peaks1 = np.array([10, 30, 50])
    peaks2 = np.array([20, 40, 60])
    volume = waveform_morphology.compute_volume(peaks1, peaks2, mode="peak")
    
    assert isinstance(volume, float)

def test_compute_volume_invalid_mode(waveform_morphology):
    peaks1 = np.array([10, 30, 50])
    peaks2 = np.array([20, 40, 60])
    
    with pytest.raises(ValueError):
        waveform_morphology.compute_volume(peaks1, peaks2, mode="invalid")

def test_compute_skewness(waveform_morphology):
    skewness = waveform_morphology.compute_skewness()
    
    assert isinstance(skewness, float)

def test_compute_qrs_duration(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    qrs_duration = waveform_morphology.compute_qrs_duration()
    
    assert isinstance(qrs_duration, float)
    # assert qrs_duration > 0

def test_compute_qrs_duration_invalid_signal_type(mocker, mock_waveform):
    wm = WaveformMorphology(mock_waveform, signal_type="PPG")
    
    with pytest.raises(ValueError):
        wm.compute_qrs_duration()


def test_compute_volume_sequence(waveform_morphology):
    peaks = np.array([10, 30, 50])
    volume = waveform_morphology.compute_volume_sequence(peaks)
    
    assert isinstance(volume, float)
    assert volume > 0  # Ensure volume is computed as a positive value

def test_compute_volume_trough_mode(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    peaks1 = np.array([10, 30, 50])
    peaks2 = np.array([15, 35, 55])
    volume = waveform_morphology.compute_volume(peaks1, peaks2, mode="trough")
    
    assert isinstance(volume, float)
    assert volume >= 0  # Ensure valid volume

def test_compute_qrs_duration_with_r_peaks_loop(mocker, waveform_morphology):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    # Ensure the signal type is ECG for QRS duration
    waveform_morphology.signal_type = "ECG"
    
    # Mock R-peaks manually to test QRS duration computation
    mock_r_peaks = np.array([10, 30, 50])
    
    # Patch the detect_peaks method to return mock R-peaks
    mocker.patch.object(MockPeakDetection, 'detect_peaks', return_value=mock_r_peaks)

    # Compute QRS duration
    qrs_duration = waveform_morphology.compute_qrs_duration()
    
    # Add a print statement to debug the output
    print(f"R-peaks: {mock_r_peaks}")
    print(f"Computed QRS duration: {qrs_duration}")
    
    assert isinstance(qrs_duration, float)
    # assert qrs_duration > 0  # Ensure QRS duration is positive

def test_compute_wave_ratio(mocker, mock_waveform):
    mocker.patch("vitalDSP.utils.peak_detection.PeakDetection", MockPeakDetection)
    
    # Ensure signal type is PPG for wave ratio
    wm = WaveformMorphology(mock_waveform, signal_type="PPG")
    
    systolic_peaks = np.array([10, 30, 50])
    notch_points = np.array([20, 40, 60])
    
    # Patch the compute_volume method to test the compute_wave_ratio function
    mocker.patch.object(WaveformMorphology, 'compute_volume', return_value=100.0)
    
    wave_ratio = wm.compute_wave_ratio(systolic_peaks, notch_points)
    
    assert isinstance(wave_ratio, float)
    assert wave_ratio > 0  # Ensure the ratio is positive


def test_compute_eeg_wavelet_features(waveform_morphology):
    waveform_morphology.signal_type = "EEG"
    wavelet_features = waveform_morphology.compute_eeg_wavelet_features()
    
    assert isinstance(wavelet_features, dict)
    assert "delta_power" in wavelet_features
    assert "theta_power" in wavelet_features
    assert "alpha_power" in wavelet_features
    assert all(isinstance(value, float) for value in wavelet_features.values())

def test_compute_slope_sequence(waveform_morphology):
    slope = waveform_morphology.compute_slope_sequence()
    
    assert isinstance(slope, float)
    assert slope != 0  # Ensure the slope is not zero for a non-flat waveform

def test_compute_curvature(waveform_morphology):
    curvature = waveform_morphology.compute_curvature()
    
    assert isinstance(curvature, float)
    assert curvature > 0  # Check that curvature is computed correctly