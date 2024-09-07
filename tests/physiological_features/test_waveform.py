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
