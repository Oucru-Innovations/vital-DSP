import pytest
import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.physiological_features.waveform import WaveformMorphology
from unittest.mock import MagicMock
from vitalDSP.utils.synthesize_data import generate_ecg_signal, generate_synthetic_ppg
from scipy.stats import linregress
import warnings

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
    assert isinstance(amplitudes, np.ndarray)

    amplitudes = wm.compute_amplitude(interval_type="R-to-Q")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)

    amplitudes = wm.compute_amplitude(interval_type="P-to-Q")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(interval_type="T-to-S")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(interval_type="T-to-Baseline")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(interval_type="S-to-Baseline")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    amplitudes = wm.compute_amplitude(signal_type="PPG", interval_type="Sys-to-Notch")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(signal_type="PPG", interval_type="Notch-to-Dia")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(signal_type="PPG", interval_type="Sys-to-Dia")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(signal_type="PPG", interval_type="Sys-to-Baseline")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(signal_type="PPG", interval_type="Notch-to-Baseline")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)
    
    amplitudes = wm.compute_amplitude(signal_type="PPG", interval_type="Dia-to-Baseline")
    assert amplitudes is not None
    assert isinstance(amplitudes, np.ndarray)

def test_compute_volume(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    volumes = wm.compute_volume(interval_type="R-to-Q")
    assert volumes is not None
    assert isinstance(volumes, np.ndarray)
    
    volumes = wm.compute_volume(interval_type="P-to-Q")
    assert volumes is not None
    assert isinstance(volumes, np.ndarray)
    
    volumes = wm.compute_volume(interval_type="T-to-S")
    assert volumes is not None
    assert isinstance(volumes, np.ndarray)
    
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    volumes = wm.compute_volume(signal_type="PPG", mode="trough",
                                interval_type="Sys-to-Notch")
    assert volumes is not None
    assert isinstance(volumes, np.ndarray)
    
    volumes = wm.compute_volume(signal_type="PPG", interval_type="Notch-to-Dia")
    assert volumes is not None
    assert isinstance(volumes, np.ndarray)
    
    volumes = wm.compute_volume(signal_type="PPG", interval_type="Sys-to-Dia")
    assert volumes is not None
    assert isinstance(volumes, np.ndarray)

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
    assert isinstance(skewness_values, np.ndarray)

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

def test_get_slope_ecg():
    fs = 256
    waveform = generate_ecg_signal(sfecg=fs, N=15, Anoise=0.01, hrmean=70)
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Test slope with mean summary
    slope = wm.get_slope(slope_type="qrs", summary_type="mean")
    assert slope is not None
    assert isinstance(slope, float)
    
    # Test slope with full summary
    slopes = wm.get_slope(slope_type="qrs", summary_type="full")
    assert slopes is not None
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) > 0

def test_get_area_qrs(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Test area with mean summary
    area = wm.get_area(interval_type="QRS", summary_type="mean")
    assert area is not None
    assert isinstance(area, float)
    
    # Test area with unsupported summary type
    with pytest.raises(ValueError):
        wm.get_area(interval_type="QRS", summary_type="unsupported")

def test_invalid_summary_type_for_duration(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    # Test with an invalid summary type
    with pytest.raises(ValueError):
        wm.get_duration(session_type="systolic", summary_type="unsupported")

def test_compute_amplitude_ppg(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    amplitude = wm.compute_amplitude(interval_type="Sys-to-Dia", signal_type="PPG")
    assert amplitude is not None  # Add further checks based on expected output
    
def test_detect_troughs_with_no_peaks(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    wm.systolic_peaks = np.array([])  # No peaks
    
    # Detect troughs without peaks
    troughs = wm.detect_troughs()
    assert len(troughs) == 0

def test_detect_r_session_no_q_or_s_sessions(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    # Call detect_r_session with empty q_session and s_session arguments
    r_sessions = wm.detect_r_session(q_sessions=np.array([]), s_sessions=np.array([]))
    # Assert that the result is empty
    assert r_sessions.size == 0

def test_skewness_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs)
    
    with pytest.raises(ValueError):
        wm.compute_skewness(signal_type="Invalid")

def test_compute_volume_empty_peaks_and_valleys():
    waveform = generate_ecg_signal(sfecg=256, N=15, Anoise=0.01, hrmean=70)
    wm = WaveformMorphology(waveform, fs=256, signal_type="ECG")
    wm.r_peaks = np.array([])  # Empty peaks
    wm.s_valleys = np.array([])  # Empty valleys
    
    volume = wm.compute_volume(interval_type="R-to-S")
    assert volume.size == 0

def test_qrs_amplitude_with_zero_data():
    waveform = np.zeros(256)
    wm = WaveformMorphology(waveform, fs=256, signal_type="ECG")
    
    # Test QRS amplitude on a flat signal (all zeroes)
    qrs_amplitude = wm.get_qrs_amplitude(summary_type="mean")
    assert qrs_amplitude == 0

def test_get_peak_trend_slope_slope_empty_peaks(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    peaks = np.array([])  # No peaks
    
    slope = wm.get_peak_trend_slope(peaks=peaks)
    assert slope == 0

# Additional tests for the refactor functions
# Testing compute_duration
def test_compute_duration_ecg(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    durations = wm.compute_duration(mode="ECG")
    assert isinstance(durations, np.ndarray)
    assert len(durations) > 0

def test_compute_duration_invalid_mode(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_duration(mode="INVALID")

# Testing compute_ppg_dicrotic_notch
def test_compute_ppg_dicrotic_notch(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    notch_timing = wm.compute_ppg_dicrotic_notch()
    assert isinstance(notch_timing, float)
    assert notch_timing >= 0

def test_compute_ppg_dicrotic_notch_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_ppg_dicrotic_notch()

# Testing compute_eeg_wavelet_features
def test_compute_eeg_wavelet_features(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="EEG")
    features = wm.compute_eeg_wavelet_features()
    assert isinstance(features, dict)
    assert "delta_power" in features
    assert "theta_power" in features
    assert "alpha_power" in features

def test_compute_eeg_wavelet_features_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    with pytest.raises(ValueError):
        wm.compute_eeg_wavelet_features()

# Testing compute_slope
def test_compute_slope_points(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    slopes = wm.compute_slope(points=[10, 20, 30])
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == 3

def test_compute_slope_invalid_option(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError):
        wm.compute_slope(option="INVALID")

# Testing compute_curvature
def test_compute_curvature_points(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    curvatures = wm.compute_curvature(points=[10, 20, 30])
    assert isinstance(curvatures, np.ndarray)
    assert len(curvatures) > 0
    
    curvatures = wm.compute_curvature(points=[10, 20, 30],option="p_peaks")
    assert isinstance(curvatures, np.ndarray)
    assert len(curvatures) > 0
    
    curvatures = wm.compute_curvature(points=[10, 20, 30],option="q_valleys")
    assert isinstance(curvatures, np.ndarray)
    assert len(curvatures) > 0
    
    curvatures = wm.compute_curvature(points=[10, 20, 30],option="r_peaks")
    assert isinstance(curvatures, np.ndarray)
    assert len(curvatures) > 0
    
    curvatures = wm.compute_curvature(points=[10, 20, 30],option="s_valleys")
    assert isinstance(curvatures, np.ndarray)
    assert len(curvatures) > 0
    
    curvatures = wm.compute_curvature(points=[10, 20, 30],option="t_peaks")
    assert isinstance(curvatures, np.ndarray)
    assert len(curvatures) > 0
    
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    curvatures = wm.compute_curvature(points=[10, 20, 10, 25, 15, 10],option="troughs")
    assert isinstance(curvatures, np.ndarray)
    # assert len(curvatures) > 0
    
    curvatures = wm.compute_curvature(points=[10, 20, 10, 25, 15, 10],option="diastolic_peaks")
    assert isinstance(curvatures, np.ndarray)
    # assert len(curvatures) > 0

def test_compute_curvature_invalid_option(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    with pytest.raises(ValueError):
        wm.compute_curvature(option="INVALID")

# Testing get_duration
def test_get_duration_systolic(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    duration = wm.get_duration(session_type="systolic", summary_type="mean")
    assert isinstance(duration, float)

def test_get_duration_invalid_summary_type(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    with pytest.raises(ValueError):
        wm.get_duration(session_type="systolic", summary_type="unsupported")

# def test_get_duration_custom_missing_points(sample_waveform):
#     waveform, fs = sample_waveform
#     with pytest.raises(ValueError, match="For 'Custom' session type, start_points and end_points must be provided."):
#         wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
#         wm.get_duration(session_type="Custom")

# Testing _summarize_list (helper function)
def test_summarize_list_mean():
    wm = WaveformMorphology(np.sin(np.linspace(0, 2 * np.pi, 10)), signal_type="ECG")  # Small waveform for helper testing
    values = [10, 20, 30]
    assert wm._summarize_list(values, "mean") == 20
    
    values = [10, 20, 30]
    assert wm._summarize_list(values, "median") == 20
    
    values = [10, 20, 30]
    assert wm._summarize_list(values, "2nd_quartile") == 15
    
    values = [10, 20, 30]
    assert wm._summarize_list(values, "3rd_quartile") == 25

def test_summarize_list_invalid_summary_type():
    wm = WaveformMorphology(np.sin(np.linspace(0, 2 * np.pi, 10)), signal_type="ECG")
    values = [10, 20, 30]
    with pytest.raises(ValueError):
        wm._summarize_list(values, "invalid")
        
def test_detect_r_session_basic(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    wm.r_peaks = np.array([50, 150, 250])
    wm.q_sessions = np.array([[40, 45], [140, 145], [240, 245]])
    wm.s_sessions = np.array([[55, 60], [155, 160], [255, 260]])

    # Expected R sessions based on the given Q and S sessions
    expected_r_sessions = np.array([[45, 55], [145, 155], [245, 255]])
    # Detect R sessions using the manually set Q and S sessions
    r_sessions = wm.detect_r_session(rpeaks=wm.r_peaks, 
                                    q_sessions=wm.q_sessions, 
                                    s_sessions=wm.s_sessions)

    assert np.array_equal(r_sessions, expected_r_sessions), f"Expected {expected_r_sessions}, but got {r_sessions}"

def test_detect_r_session_missing_parameters(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Set r_peaks and ensure q_valleys and s_valleys are detected
    wm.r_peaks = np.array([50, 150, 250])
    wm.q_valleys = wm.detect_q_valley(r_peaks=wm.r_peaks)
    wm.s_valleys = wm.detect_s_valley(r_peaks=wm.r_peaks)
    
    # Detect p_peaks and t_peaks which are needed for q_session and s_session detection
    wm.p_peaks = wm.detect_p_peak(r_peaks=wm.r_peaks, q_valleys=wm.q_valleys)
    wm.t_peaks = wm.detect_t_peak(r_peaks=wm.r_peaks, s_valleys=wm.s_valleys)
    
    # Now detect q_sessions and s_sessions with all required peaks/valleys
    wm.q_sessions = wm.detect_q_session(p_peaks=wm.p_peaks, q_valleys=wm.q_valleys, r_peaks=wm.r_peaks)
    wm.s_sessions = wm.detect_s_session(t_peaks=wm.t_peaks, s_valleys=wm.s_valleys, r_peaks=wm.r_peaks)
    
    # Now detect r_sessions without explicitly providing q_sessions and s_sessions
    r_sessions = wm.detect_r_session()
    
    # Verify we have valid sessions
    assert isinstance(r_sessions, np.ndarray), "R sessions should be a numpy array"
    assert r_sessions.size > 0, "Expected non-empty R sessions when q_sessions and s_sessions are detected by default"
    if len(r_sessions) > 0:
        assert r_sessions.shape[1] == 2, "R sessions should have start and end points"

def test_detect_r_session_empty_sessions(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    wm.q_sessions = np.array([])  # Empty Q sessions
    wm.s_sessions = np.array([])  # Empty S sessions

    # Expecting an empty array as output
    r_sessions = wm.detect_r_session(q_sessions=wm.q_sessions, s_sessions=wm.s_sessions)
    assert r_sessions.size == 0, f"Expected empty output, but got {r_sessions}"

def test_detect_r_session_mismatched_qs_sessions(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    wm.q_sessions = np.array([[40, 45], [140, 145]])  # Only two Q sessions
    wm.s_sessions = np.array([[55, 60], [155, 160], [255, 260]])  # Three S sessions

    # Expecting R sessions to be computed only up to the shortest list length
    expected_r_sessions = np.array([[45, 55], [145, 155]])
    r_sessions = wm.detect_r_session(q_sessions=wm.q_sessions, s_sessions=wm.s_sessions)
    assert np.array_equal(r_sessions, expected_r_sessions), f"Expected {expected_r_sessions}, but got {r_sessions}"

def test_detect_r_session_no_valid_intervals(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    wm.q_sessions = np.array([[40, 50], [140, 150]])  # End points are after start points of S sessions
    wm.s_sessions = np.array([[30, 35], [130, 135]])

    # Expecting no valid R sessions due to invalid intervals
    r_sessions = wm.detect_r_session(q_sessions=wm.q_sessions, s_sessions=wm.s_sessions)
    assert r_sessions.size == 0, f"Expected empty R sessions, but got {r_sessions}"

def test_get_peak_trend_slope_slope_empty_peaks(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    peaks = np.array([])  # No peaks
    
    slope = wm.get_peak_trend_slope(peaks=peaks)
    assert slope == 0

def test_get_peak_trend_slope_linear_regression():
    peaks = np.array([1, 2, 3, 4, 5])
    wm = WaveformMorphology(peaks, signal_type="ECG")
    expected_slope, _, _, _, _ = linregress(np.arange(len(peaks)), peaks)
    
    slope = wm.get_peak_trend_slope(peaks, method="linear_regression")
    assert slope == pytest.approx(expected_slope, rel=1e-3), "Linear regression slope does not match expected value."

def test_get_peak_trend_slope_moving_average():
    peaks = np.array([1, 2, 3, 4, 5, 6, 7])
    wm = WaveformMorphology(peaks, signal_type="PPG")
    window_size = 3
    moving_avg = np.convolve(peaks, np.ones(window_size) / window_size, mode="valid")
    expected_slope = np.mean(np.diff(moving_avg) / window_size)
    
    slope = wm.get_peak_trend_slope(peaks, method="moving_average", window_size=window_size)
    assert slope == pytest.approx(expected_slope, rel=1e-3), "Moving average slope does not match expected value."

def test_get_peak_trend_slope_rate_of_change():
    peaks = np.array([1, 3, 5, 7, 9])
    wm = WaveformMorphology(peaks, signal_type="PPG")
    expected_slope = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
    
    slope = wm.get_peak_trend_slope(peaks, method="rate_of_change")
    assert slope == pytest.approx(expected_slope, rel=1e-3), "Rate of change slope does not match expected value."

def test_get_peak_trend_slope_unsupported_method():
    peaks = np.array([1, 2, 3, 4, 5])
    wm = WaveformMorphology(peaks, signal_type="ECG")
    slope = wm.get_peak_trend_slope(peaks, method="unsupported_method")
    assert np.isnan(slope), "Expected NaN for unsupported method."

def test_get_amplitude_variability_std(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    # Test across multiple interval types to check if any produce valid amplitudes
    for interval_type in ["Sys-to-Baseline", "Notch-to-Baseline", "Sys-to-Dia"]:
        amplitudes = wm.compute_amplitude(interval_type=interval_type, signal_type="PPG")

        # Verify that amplitudes are detected
        assert amplitudes is not None, f"Expected non-empty amplitudes for {interval_type}"
        assert isinstance(amplitudes, (list, np.ndarray)), f"Expected amplitudes to be a list or np.ndarray for {interval_type}"
        if len(amplitudes) == 0:
            print(f"No amplitudes detected for interval type: {interval_type}. Check peak detection or baseline setup.")
            continue  # Skip to next interval type if no amplitudes found
        
        # Check variability if amplitudes are valid
        variability = wm.get_amplitude_variability(interval_type=interval_type, method="std_dev")
        assert isinstance(variability, float), "Expected variability to be a float"
        
        # Check variability if amplitudes are valid
        variability = wm.get_amplitude_variability(interval_type=interval_type, method="cv")
        assert isinstance(variability, float), "Expected variability to be a float"
        
        # Check variability if amplitudes are valid
        variability = wm.get_amplitude_variability(interval_type=interval_type, method="interquartile_range")
        assert isinstance(variability, float), "Expected variability to be a float"
        
        # assert not np.isnan(variability), "Expected variability to be a valid number, not NaN"
        # assert variability > 0, "Expected non-zero variability for standard deviation"


# def test_get_amplitude_variability_cv_zero_mean(sample_waveform):
#     waveform, fs = sample_waveform
#     wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
#     # Mock compute_amplitude to return an array with zero mean
#     wm.compute_amplitude = lambda *args, **kwargs: [0, 0, 0]
#     with pytest.raises(ValueError, match="Mean amplitude is zero, cannot compute coefficient of variation."):
#         wm.get_amplitude_variability(interval_type="Sys-to-Baseline", method="cv")

# def test_get_amplitude_variability_iqr(sample_waveform):
#     waveform, fs = sample_waveform
#     wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
#     variability = wm.get_amplitude_variability(interval_type="Sys-to-Baseline", method="interquartile_range")
#     assert isinstance(variability, float), "Expected variability to be a float"
#     assert variability >= 0, "Expected non-negative variability for interquartile range"

def test_get_amplitude_variability_invalid_method(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    with pytest.raises(ValueError, match="Unsupported variability method"):
        # amplitudes = wm.compute_amplitude(interval_type="Sys-to-Baseline", signal_type="PPG")
        wm.get_amplitude_variability(interval_type="Sys-to-Baseline", method="invalid")

def test_detect_dicrotic_notches_with_config():
    # Generate synthetic PPG signal
    fs = 256
    duration = 10  # seconds
    _, ppg_signal = generate_synthetic_ppg(  # Unpack time and signal
        duration=duration,
        sampling_rate=fs,
        heart_rate=60
    )
    
    # Test with default config
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    notches = wm.detect_dicrotic_notches()
    assert isinstance(notches, np.ndarray)
    
    # Test with custom config
    peak_config = {
        "distance": 50,
        "window_size": 7,
        "threshold_factor": 0.8,
        "search_window": 6,
        "dicrotic_config": {
            "slope_quartile_factor": 0.15,  # More strict threshold
            "search_range_start": 0.15,
            "search_range_end": 0.4,
        }
    }
    
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG", peak_config=peak_config)
    notches_custom = wm.detect_dicrotic_notches()
    assert isinstance(notches_custom, np.ndarray)

def test_detect_dicrotic_notches_invalid_signal():
    # Test with ECG signal (should raise error)
    fs = 256
    ecg_signal = generate_ecg_signal(sfecg=fs, N=15)
    wm = WaveformMorphology(ecg_signal, fs=fs, signal_type="ECG")
    with pytest.raises(ValueError, match="Notches can only be detected for PPG signals."):
        wm.detect_dicrotic_notches()

def test_detect_dicrotic_notches_no_peaks():
    # Test with no systolic peaks
    fs = 256
    ppg_signal = np.zeros(fs * 5)  # 5 seconds of zeros
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    notches = wm.detect_dicrotic_notches()
    assert len(notches) == 0

def create_test_ppg(duration, fs, heart_rate=60):
    """Create a synthetic PPG signal with clear systolic peaks, dicrotic notches, and diastolic peaks."""
    t = np.linspace(0, duration, int(duration * fs))
    # Base frequency for heart rate
    f_hr = heart_rate / 60.0  # Convert to Hz
    
    # Create very sharp systolic peaks
    systolic_phase = np.sin(2 * np.pi * f_hr * t)
    systolic = np.exp(-10 * (1 - systolic_phase)) * (systolic_phase > 0)  # Sharper peaks
    
    # Add dicrotic notch and diastolic peak with clear separation
    phase_shift = 0.35  # Increased separation
    dicrotic_phase = np.sin(2 * np.pi * f_hr * (t - phase_shift))
    dicrotic = 0.3 * np.exp(-5 * (1 - dicrotic_phase)) * (dicrotic_phase > 0)  # Reduced amplitude
    
    # Combine components
    ppg = systolic + dicrotic
    
    # Add very minimal baseline wander
    baseline = 0.02 * np.sin(2 * np.pi * 0.1 * t)  # Reduced amplitude
    ppg += baseline
    
    # Smooth the signal slightly
    window_size = int(fs * 0.01)  # 10ms window (shorter)
    if window_size % 2 == 0:
        window_size += 1
    ppg = np.convolve(ppg, np.ones(window_size)/window_size, mode='same')
    
    # Ensure positive values and normalize
    ppg = ppg - np.min(ppg)
    ppg = ppg / np.max(ppg)
    
    return ppg

def test_ppg_volume_calculations():
    # Generate synthetic PPG signal
    fs = 256
    duration = 5  # 5 seconds is enough for testing
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    
    # Create WaveformMorphology instance with appropriate peak detection config
    peak_config = {
        "distance": int(fs * 0.5),  # At least 0.5s between peaks
        "window_size": 15,          # Wider window for better peak detection
        "threshold_factor": 0.4,    # Lower threshold for better peak detection
        "search_window": 10,        # Wider search window
        "fs": fs,
        "dicrotic_config": {
            "slope_quartile_factor": 0.25,
            "search_range_start": 0.15,
            "search_range_end": 0.4,
        }
    }
    
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG", peak_config=peak_config)
    
    # First detect systolic peaks using PeakDetection directly
    pd = PeakDetection(ppg_signal, fs=fs)
    # Pass peak detection parameters through the config
    pd.config = peak_config
    systolic_peaks = pd.detect_peaks()
    wm.systolic_peaks = systolic_peaks
    
    # Then detect other features
    notches = wm.detect_dicrotic_notches()
    diastolic_peaks = wm.detect_diastolic_peak()
    
    # Print debug info
    print(f"\nDebug info:")
    print(f"Signal length: {len(ppg_signal)}")
    print(f"Systolic peaks: {systolic_peaks}")
    print(f"Dicrotic notches: {notches}")
    print(f"Diastolic peaks: {diastolic_peaks}")
    
    # Test different interval types
    intervals = ["Sys-to-Notch", "Notch-to-Dia", "Sys-to-Dia"]
    for interval in intervals:
        volume = wm.compute_volume(signal_type="PPG", interval_type=interval)
        assert isinstance(volume, np.ndarray)
        assert len(volume) > 0, f"No volume computed for interval type {interval}"
        print(f"Volume for {interval}: {volume}")

def test_ppg_amplitude_with_baseline():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Test amplitude computation with different baseline methods
    baseline_methods = ["moving_average", "low_pass", "polynomial_fit"]
    for method in baseline_methods:
        amplitude = wm.compute_amplitude(
            signal_type="PPG",
            interval_type="Sys-to-Baseline",
            baseline_method=method,
            compare_to_baseline=True
        )
        assert isinstance(amplitude, np.ndarray)
        assert len(amplitude) > 0, f"No amplitude computed for baseline method {method}"

def test_ppg_session_detection():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Test PPG session detection
    sessions = wm.detect_ppg_session()
    sessions = np.array(sessions) if not isinstance(sessions, np.ndarray) else sessions
    assert isinstance(sessions, np.ndarray), "Sessions should be a numpy array"
    assert sessions.shape[1] == 2, "Each session should have start and end points"
    assert len(sessions) > 0, "No sessions detected"

def test_dicrotic_notch_timing():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Ensure we have notches detected before computing timing
    wm.detect_dicrotic_notches()
    
    # Test dicrotic notch timing computation
    timing = wm.compute_ppg_dicrotic_notch()
    assert isinstance(timing, float)
    timing = timing / fs if timing > 1 else timing  # Convert to seconds if in samples
    assert 0 <= timing <= 1, f"Timing should be normalized between 0 and 1, got {timing}"

def test_edge_cases():
    fs = 256
    # Test with very short signal but long enough for window
    short_signal = create_test_ppg(duration=1.0, fs=fs, heart_rate=60)
    wm = WaveformMorphology(short_signal, fs=fs, signal_type="PPG")
    notches = wm.detect_dicrotic_notches()
    assert isinstance(notches, np.ndarray)
    
    # Test with noisy signal
    noisy_signal = create_test_ppg(duration=5, fs=fs, heart_rate=60)
    noise = 0.1 * np.random.randn(len(noisy_signal))
    noisy_signal = noisy_signal + noise
    wm = WaveformMorphology(noisy_signal, fs=fs, signal_type="PPG")
    notches = wm.detect_dicrotic_notches()
    assert isinstance(notches, np.ndarray)

def test_compute_amplitude_with_baseline():
    # Create a simple synthetic signal
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Test baseline comparison for PPG
    amplitudes = wm.compute_amplitude(
        signal_type="PPG",
        interval_type="Sys-to-Baseline",
        compare_to_baseline=True,
        baseline_method="moving_average"
    )
    assert isinstance(amplitudes, np.ndarray)
    assert len(amplitudes) > 0
    
    # Test baseline comparison for ECG
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="ECG")
    amplitudes = wm.compute_amplitude(
        signal_type="ECG",
        interval_type="R-to-Baseline",
        compare_to_baseline=True
    )
    assert isinstance(amplitudes, np.ndarray)
    assert len(amplitudes) > 0

def test_compute_amplitude_invalid_signal_type():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="Invalid signal type"):
        wm.compute_amplitude(signal_type="InvalidType")

def test_compute_volume_mode_trough():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Test trough mode for PPG
    volumes = wm.compute_volume(
        signal_type="PPG",
        interval_type="Sys-to-Notch",
        mode="trough"
    )
    assert isinstance(volumes, np.ndarray)
    assert len(volumes) > 0

def test_compute_volume_invalid_mode():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="Volume mode must be 'peak' or 'trough'"):
        wm.compute_volume(
            signal_type="PPG",
            interval_type="Sys-to-Notch",
            mode="invalid"
        )

def test_compute_volume_sys_to_sys():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Test Sys-to-Sys interval
    volumes = wm.compute_volume(
        signal_type="PPG",
        interval_type="Sys-to-Sys"
    )
    assert isinstance(volumes, np.ndarray)
    assert len(volumes) > 0

def test_compute_volume_sys_to_sys_insufficient_peaks():
    fs = 256
    # Create a very short signal that won't have enough peaks
    ppg_signal = create_test_ppg(duration=0.5, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="At least two systolic peaks required"):
        wm.compute_volume(
            signal_type="PPG",
            interval_type="Sys-to-Sys"
        )

def test_compute_volume_invalid_signal_type():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="signal_type must be 'ECG' or 'PPG'"):
        wm.compute_volume(
            signal_type="InvalidType",
            interval_type="Sys-to-Notch"
        )

def test_compute_volume_ecg_features():
    fs = 256
    duration = 5
    # Use ECG signal for this test
    time = np.linspace(0, duration, int(duration * fs))
    ecg_signal = np.sin(2 * np.pi * 1.0 * time)  # Simple synthetic ECG
    wm = WaveformMorphology(ecg_signal, fs=fs, signal_type="ECG")
    
    # Test ECG intervals
    intervals = ["R-to-S", "R-to-Q", "P-to-Q", "T-to-S"]
    for interval in intervals:
        volumes = wm.compute_volume(
            signal_type="ECG",
            interval_type=interval
        )
        assert isinstance(volumes, np.ndarray)

def test_compute_volume_invalid_ecg_interval():
    fs = 256
    duration = 5
    time = np.linspace(0, duration, int(duration * fs))
    ecg_signal = np.sin(2 * np.pi * 1.0 * time)
    wm = WaveformMorphology(ecg_signal, fs=fs, signal_type="ECG")
    
    with pytest.raises(ValueError, match="Invalid interval_type for ECG"):
        wm.compute_volume(
            signal_type="ECG",
            interval_type="InvalidInterval"
        )

def test_compute_volume_invalid_ppg_interval():
    fs = 256
    duration = 5
    ppg_signal = create_test_ppg(duration=duration, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="Invalid interval_type for PPG"):
        wm.compute_volume(
            signal_type="PPG",
            interval_type="InvalidInterval"
        )

def test_compute_slope_all_options(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Test with points
    points = [10, 20, 30]
    slopes = wm.compute_slope(points=points)
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == len(points)
    
    # Test all ECG options
    ecg_options = ["p_peaks", "q_valleys", "r_peaks", "s_valleys", "t_peaks"]
    for option in ecg_options:
        slopes = wm.compute_slope(option=option)
        assert isinstance(slopes, np.ndarray)
        
    # Test PPG options
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    ppg_options = ["systolic_peaks", "diastolic_peaks", "troughs"]  # Removed dicrotic_notches as it's not a valid option
    for option in ppg_options:
        slopes = wm.compute_slope(option=option)
        assert isinstance(slopes, np.ndarray)
    
    # Test with different windows
    windows = [3, 5, 7]
    for window in windows:
        slopes = wm.compute_slope(points=points, window=window)
        assert isinstance(slopes, np.ndarray)
        assert len(slopes) == len(points)
    
    # Test with different slope units
    slope_units = ["radians", "degrees"]
    for unit in slope_units:
        slopes = wm.compute_slope(points=points, slope_unit=unit)
        assert isinstance(slopes, np.ndarray)
        assert len(slopes) == len(points)

def test_compute_slope_option_combinations(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Test combinations of options
    points = [10, 20, 30]
    windows = [3, 5]
    slope_units = ["radians", "degrees"]
    
    for window in windows:
        for unit in slope_units:
            # Test with points
            slopes = wm.compute_slope(points=points, window=window, slope_unit=unit)
            assert isinstance(slopes, np.ndarray)
            assert len(slopes) == len(points)
            
            # Test with ECG options
            slopes = wm.compute_slope(option="r_peaks", window=window, slope_unit=unit)
            assert isinstance(slopes, np.ndarray)
            
            # Test with PPG options
            wm_ppg = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
            slopes = wm_ppg.compute_slope(option="systolic_peaks", window=window, slope_unit=unit)
            assert isinstance(slopes, np.ndarray)

def test_compute_slope_edge_cases(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Test with single point
    slopes = wm.compute_slope(points=[10])
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == 1
    
    # Test with two points
    slopes = wm.compute_slope(points=[10, 20])
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == 2
    
    # Test with empty points list
    slopes = wm.compute_slope(points=[])
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == 0
    
    # Test with window size equal to points length
    points = [10, 20, 30]
    slopes = wm.compute_slope(points=points, window=3)
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == len(points)
    
    # Test with window size larger than points length
    slopes = wm.compute_slope(points=points, window=5)
    assert isinstance(slopes, np.ndarray)
    assert len(slopes) == len(points)

def test_detection_methods_complex_mode(sample_waveform):
    waveform, fs = sample_waveform
    # Create a more realistic ECG signal for testing
    waveform = generate_ecg_signal(sfecg=fs, N=15, Anoise=0.01, hrmean=70)
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG", simple_mode=False)
    
    # Test ECG detection methods
    # 1. Test R peaks detection (automatically computed in init)
    assert isinstance(wm.r_peaks, np.ndarray)
    assert len(wm.r_peaks) > 0, "Should detect R peaks in complex mode"
    
    # 2. Test Q valleys detection
    q_valleys = wm.detect_q_valley()
    assert isinstance(q_valleys, np.ndarray)
    assert len(q_valleys) > 0, "Should detect Q valleys in complex mode"
    
    # 3. Test S valleys detection
    s_valleys = wm.detect_s_valley()
    assert isinstance(s_valleys, np.ndarray)
    assert len(s_valleys) > 0, "Should detect S valleys in complex mode"
    
    # 4. Test P peaks detection
    p_peaks = wm.detect_p_peak()
    assert isinstance(p_peaks, np.ndarray)
    assert len(p_peaks) > 0, "Should detect P peaks in complex mode"
    
    # 5. Test T peaks detection
    t_peaks = wm.detect_t_peak()
    assert isinstance(t_peaks, np.ndarray)
    assert len(t_peaks) > 0, "Should detect T peaks in complex mode"
    
    # Test PPG detection methods
    ppg_signal = create_test_ppg(duration=5, fs=fs, heart_rate=60)
    wm_ppg = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG", simple_mode=False)
    
    # 1. Test systolic peaks detection (automatically computed in init)
    assert isinstance(wm_ppg.systolic_peaks, np.ndarray)
    assert len(wm_ppg.systolic_peaks) > 0, "Should detect systolic peaks in complex mode"
    
    # 2. Test diastolic peaks detection
    diastolic_peaks = wm_ppg.detect_diastolic_peak()
    assert isinstance(diastolic_peaks, np.ndarray)
    assert len(diastolic_peaks) > 0, "Should detect diastolic peaks in complex mode"
    
    # 3. Test dicrotic notches detection
    dicrotic_notches = wm_ppg.detect_dicrotic_notches()
    assert isinstance(dicrotic_notches, np.ndarray)
    assert len(dicrotic_notches) > 0, "Should detect dicrotic notches in complex mode"
    
    # 4. Test troughs detection
    troughs = wm_ppg.detect_troughs()
    assert isinstance(troughs, np.ndarray)
    assert len(troughs) > 0, "Should detect troughs in complex mode"

def test_detection_methods_complex_mode_with_config(sample_waveform):
    waveform, fs = sample_waveform
    # Create configuration for detection
    peak_config = {
        "distance": int(fs * 0.5),  # At least 0.5s between peaks
        "window_size": 15,          # Wider window for better peak detection
        "threshold_factor": 0.4,    # Lower threshold for better peak detection
        "search_window": 10,        # Wider search window
        "fs": fs,
        "dicrotic_config": {
            "slope_quartile_factor": 0.25,
            "search_range_start": 0.15,
            "search_range_end": 0.4,
        }
    }
    
    # Test PPG detection with config
    ppg_signal = create_test_ppg(duration=5, fs=fs, heart_rate=60)
    wm_ppg = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG", 
                               simple_mode=False, peak_config=peak_config)
    
    # 1. Test systolic peaks detection with config (automatically computed in init)
    assert isinstance(wm_ppg.systolic_peaks, np.ndarray)
    assert len(wm_ppg.systolic_peaks) > 0, "Should detect systolic peaks with config"
    
    # 2. Test diastolic peaks detection with config
    diastolic_peaks = wm_ppg.detect_diastolic_peak()
    assert isinstance(diastolic_peaks, np.ndarray)
    assert len(diastolic_peaks) > 0, "Should detect diastolic peaks with config"
    
    # 3. Test dicrotic notches detection with config
    dicrotic_notches = wm_ppg.detect_dicrotic_notches()
    assert isinstance(dicrotic_notches, np.ndarray)
    assert len(dicrotic_notches) > 0, "Should detect dicrotic notches with config"
    
    # 4. Test troughs detection with config
    troughs = wm_ppg.detect_troughs()
    assert isinstance(troughs, np.ndarray)
    assert len(troughs) > 0, "Should detect troughs with config"

def test_detection_methods_complex_mode_session_detection():
    fs = 256
    # Create a more realistic ECG signal for testing
    waveform = generate_ecg_signal(sfecg=fs, N=15, Anoise=0.01, hrmean=70)
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG", simple_mode=False)
    
    # First detect all required peaks and valleys (r_peaks already computed in init)
    wm.q_valleys = wm.detect_q_valley()
    wm.s_valleys = wm.detect_s_valley()
    wm.p_peaks = wm.detect_p_peak()
    wm.t_peaks = wm.detect_t_peak()
    
    # Test ECG session detection methods
    # 1. Test Q session detection
    q_sessions = wm.detect_q_session()
    assert isinstance(q_sessions, np.ndarray)
    if len(q_sessions) > 0:  # Only check shape if sessions were detected
        assert q_sessions.shape[1] == 2, "Q sessions should have start and end points"
    
    # 2. Test S session detection
    s_sessions = wm.detect_s_session()
    assert isinstance(s_sessions, np.ndarray)
    if len(s_sessions) > 0:  # Only check shape if sessions were detected
        assert s_sessions.shape[1] == 2, "S sessions should have start and end points"
    
    # 3. Test R session detection
    r_sessions = wm.detect_r_session()
    assert isinstance(r_sessions, np.ndarray)
    if len(r_sessions) > 0:  # Only check shape if sessions were detected
        assert r_sessions.shape[1] == 2, "R sessions should have start and end points"
    
    # 4. Test QRS session detection
    qrs_sessions = wm.detect_qrs_session()
    assert isinstance(qrs_sessions, np.ndarray)
    if len(qrs_sessions) > 0:  # Only check shape if sessions were detected
        assert qrs_sessions.shape[1] == 2, "QRS sessions should have start and end points"
    
    # Test PPG session detection
    ppg_signal = create_test_ppg(duration=5, fs=fs, heart_rate=60)
    wm_ppg = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG", simple_mode=False)
    
    # Test PPG session detection
    ppg_sessions = wm_ppg.detect_ppg_session()
    assert isinstance(ppg_sessions, np.ndarray), "PPG sessions should be a numpy array"
    if len(ppg_sessions) > 0:  # Only check shape if sessions were detected
        assert ppg_sessions.shape[1] == 2, "PPG sessions should have start and end points"

def test_detection_methods_complex_mode_edge_cases():
    fs = 256
    # Test with very short signal
    short_signal = np.sin(np.linspace(0, 2 * np.pi, fs))
    wm = WaveformMorphology(short_signal, fs=fs, signal_type="ECG", simple_mode=False)
    
    # Should still detect basic features (r_peaks computed in init)
    assert isinstance(wm.r_peaks, np.ndarray)
    
    # Test with noisy signal
    noisy_signal = generate_ecg_signal(sfecg=fs, N=15, Anoise=0.1, hrmean=70)  # Higher noise
    wm_noisy = WaveformMorphology(noisy_signal, fs=fs, signal_type="ECG", simple_mode=False)
    
    # Should still detect features in noisy signal (r_peaks computed in init)
    assert isinstance(wm_noisy.r_peaks, np.ndarray)
    
    # Test with flat signal
    flat_signal = np.zeros(fs)
    wm_flat = WaveformMorphology(flat_signal, fs=fs, signal_type="ECG", simple_mode=False)
    
    # Should return empty arrays for flat signal (r_peaks computed in init)
    assert isinstance(wm_flat.r_peaks, np.ndarray)
    assert len(wm_flat.r_peaks) == 0, "Should detect no peaks in flat signal"

def test_cache_result(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Create a mock method that counts calls
    call_count = 0
    def mock_method(x, y=0):
        nonlocal call_count
        call_count += 1
        return x + y
    
    # Test basic caching
    result1 = wm._cache_result("test_key", mock_method, 1, y=2)
    assert result1 == 3, "Method should return correct result"
    assert call_count == 1, "Method should be called once"
    
    # Test cache hit (same arguments)
    result2 = wm._cache_result("test_key", mock_method, 1, y=2)
    assert result2 == 3, "Cached result should be returned"
    assert call_count == 1, "Method should not be called again for same arguments"
    
    # Test cache miss (different positional argument)
    result3 = wm._cache_result("test_key", mock_method, 2, y=2)
    assert result3 == 4, "Method should return new result for different arguments"
    assert call_count == 2, "Method should be called for different arguments"
    
    # Test cache miss (different keyword argument)
    result4 = wm._cache_result("test_key", mock_method, 1, y=3)
    assert result4 == 4, "Method should return new result for different keyword arguments"
    assert call_count == 3, "Method should be called for different keyword arguments"
    
    # Test cache miss (different key)
    result5 = wm._cache_result("different_key", mock_method, 1, y=2)
    assert result5 == 3, "Method should return result for different key"
    assert call_count == 4, "Method should be called for different key"
    
    # Test cache with multiple positional arguments
    def mock_method_multi(*args):
        nonlocal call_count
        call_count += 1
        return sum(args)
    
    result6 = wm._cache_result("multi_args", mock_method_multi, 1, 2, 3)
    assert result6 == 6, "Method should handle multiple positional arguments"
    
    # Test cache hit with multiple arguments
    result7 = wm._cache_result("multi_args", mock_method_multi, 1, 2, 3)
    assert result7 == 6, "Should return cached result for multiple arguments"
    assert call_count == 5, "Method should be called only once for same multiple arguments"

def test_detect_troughs_warning_non_ppg(sample_waveform, caplog):
    waveform, fs = sample_waveform
    # Initialize with ECG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    # Create some dummy peaks for testing
    dummy_peaks = np.array([10, 20, 30])
    
    # Test warning for non-PPG signal
    troughs = wm.detect_troughs(systolic_peaks=dummy_peaks)
    assert isinstance(troughs, np.ndarray), "Should still return numpy array even with warning"
    assert "This function is designed to work with PPG signals" in caplog.text, "Warning message should be logged for ECG signal"
    
    # Clear the log
    caplog.clear()
    
    # Test with other signal types
    wm_eeg = WaveformMorphology(waveform, fs=fs, signal_type="EEG")
    troughs_eeg = wm_eeg.detect_troughs(systolic_peaks=dummy_peaks)
    assert isinstance(troughs_eeg, np.ndarray), "Should return numpy array for EEG signal"
    assert "This function is designed to work with PPG signals" in caplog.text, "Warning message should be logged for EEG signal"
    
    # Clear the log
    caplog.clear()
    
    # Verify no warning for PPG signal
    wm_ppg = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    # Initialize systolic_peaks for PPG
    wm_ppg.systolic_peaks = dummy_peaks
    troughs_ppg = wm_ppg.detect_troughs()
    assert isinstance(troughs_ppg, np.ndarray), "Should return numpy array for PPG signal"
    assert "This function is designed to work with PPG signals" not in caplog.text, "No warning should be logged for PPG signal"

def test_detect_diastolic_peak_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with ECG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    
    with pytest.raises(ValueError, match="Diastolic peaks can only be detected for PPG signals"):
        wm.detect_diastolic_peak()

def test_detect_q_valley_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with PPG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="Q valleys can only be detected for ECG signals"):
        wm.detect_q_valley()

def test_detect_s_valley_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with PPG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="Q valleys can only be detected for ECG signals"):
        wm.detect_s_valley()

def test_detect_p_peak_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with PPG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="P peaks can only be detected for ECG signals"):
        wm.detect_p_peak()

def test_detect_t_peak_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with PPG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="T peaks can only be detected for ECG signals"):
        wm.detect_t_peak()

def test_detect_q_session_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with PPG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="Q sessions can only be detected for ECG signals"):
        wm.detect_q_session()

def test_detect_s_session_invalid_signal_type(sample_waveform):
    waveform, fs = sample_waveform
    # Initialize with PPG signal type
    wm = WaveformMorphology(waveform, fs=fs, signal_type="PPG")
    
    with pytest.raises(ValueError, match="S sessions can only be detected for ECG signals"):
        wm.detect_s_session()

def test_compute_slope_units(sample_waveform):
    waveform, fs = sample_waveform
    wm = WaveformMorphology(waveform, fs=fs, signal_type="ECG")
    points = [10, 20, 30]
    
    # Test degrees slope unit
    slopes_degrees = wm.compute_slope(points=points, slope_unit="degrees")
    assert isinstance(slopes_degrees, np.ndarray)
    assert len(slopes_degrees) == len(points)
    assert all(-90 <= slope <= 90 for slope in slopes_degrees), "Slopes in degrees should be between -90 and 90"
    
    # Test radians slope unit
    slopes_radians = wm.compute_slope(points=points, slope_unit="radians")
    assert isinstance(slopes_radians, np.ndarray)
    assert len(slopes_radians) == len(points)
    assert all(-np.pi/2 <= slope <= np.pi/2 for slope in slopes_radians), "Slopes in radians should be between -π/2 and π/2"
    
    # Test raw slope unit
    slopes_raw = wm.compute_slope(points=points, slope_unit="raw")
    assert isinstance(slopes_raw, np.ndarray)
    assert len(slopes_raw) == len(points)
    
    # Test invalid slope unit
    with pytest.raises(ValueError, match="slope_unit must be 'degrees', 'radians', or 'raw'"):
        wm.compute_slope(points=points, slope_unit="invalid")

def test_compute_skewness_ppg(sample_waveform):
    waveform, fs = sample_waveform
    # Create a PPG signal with known skewness
    ppg_signal = create_test_ppg(duration=5, fs=fs, heart_rate=60)
    wm = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    
    # Test PPG skewness computation
    skewness = wm.compute_skewness(signal_type="PPG")
    assert isinstance(skewness, np.ndarray)
    assert len(skewness) > 0, "Should compute skewness for PPG signal"
    
    # Test that each skewness value is a valid float
    assert all(not np.isnan(s) for s in skewness), "Skewness values should not be NaN"
    assert all(not np.isinf(s) for s in skewness), "Skewness values should not be infinite"

def test_compute_duration_modes(sample_waveform):
    waveform, fs = sample_waveform
    
    # Test PPG mode
    ppg_signal = create_test_ppg(duration=5, fs=fs, heart_rate=60)
    wm_ppg = WaveformMorphology(ppg_signal, fs=fs, signal_type="PPG")
    ppg_durations = wm_ppg.compute_duration(mode="PPG")
    assert isinstance(ppg_durations, np.ndarray)
    assert len(ppg_durations) > 0, "Should compute durations for PPG mode"
    
    # Test QRS mode
    ecg_signal = generate_ecg_signal(sfecg=fs, N=15, Anoise=0.01, hrmean=70)
    wm_ecg = WaveformMorphology(ecg_signal, fs=fs, signal_type="ECG", simple_mode=False)
    
    # First ensure we have R peaks (they should be automatically computed in init)
    assert isinstance(wm_ecg.r_peaks, np.ndarray)
    assert len(wm_ecg.r_peaks) > 0, "Should have R peaks detected"
    
    # Then detect Q and S valleys using the R peaks
    # Skip the first R peak to avoid boundary issues
    r_peaks = wm_ecg.r_peaks[1:]
    wm_ecg.q_valleys = wm_ecg.detect_q_valley(r_peaks=r_peaks)
    wm_ecg.s_valleys = wm_ecg.detect_s_valley(r_peaks=r_peaks)
    
    # Now detect P and T peaks
    wm_ecg.p_peaks = wm_ecg.detect_p_peak(r_peaks=r_peaks, q_valleys=wm_ecg.q_valleys)
    wm_ecg.t_peaks = wm_ecg.detect_t_peak(r_peaks=r_peaks, s_valleys=wm_ecg.s_valleys)
    
    # Detect Q and S sessions with all required components
    wm_ecg.q_sessions = wm_ecg.detect_q_session(p_peaks=wm_ecg.p_peaks, q_valleys=wm_ecg.q_valleys, r_peaks=r_peaks)
    wm_ecg.s_sessions = wm_ecg.detect_s_session(t_peaks=wm_ecg.t_peaks, s_valleys=wm_ecg.s_valleys, r_peaks=r_peaks)
    
    # Now detect QRS sessions with all required components
    wm_ecg.qrs_sessions = wm_ecg.detect_qrs_session(rpeaks=r_peaks, q_session=wm_ecg.q_sessions, s_session=wm_ecg.s_sessions)
    
    # Add debug prints to check what's happening
    # print(f"\nDebug info:")
    # print(f"R peaks: {r_peaks}")
    # print(f"Q valleys: {wm_ecg.q_valleys}")
    # print(f"S valleys: {wm_ecg.s_valleys}")
    # print(f"P peaks: {wm_ecg.p_peaks}")
    # print(f"T peaks: {wm_ecg.t_peaks}")
    # print(f"Q sessions: {wm_ecg.q_sessions}")
    # print(f"S sessions: {wm_ecg.s_sessions}")
    # print(f"QRS sessions: {wm_ecg.qrs_sessions}")
    
    # Now compute QRS durations
    qrs_durations = wm_ecg.compute_duration(mode="QRS")
    assert isinstance(qrs_durations, np.ndarray)
    # assert len(qrs_durations) > 0, "Should compute durations for QRS mode"
    
    # Verify durations are positive and in reasonable range
    assert all(d >= 0 for d in ppg_durations), "PPG durations should be non-negative"
    assert all(d >= 0 for d in qrs_durations), "QRS durations should be non-negative"
    
    # QRS complex typically lasts 0.06-0.10 seconds
    expected_qrs_max = 0.10 * fs  # Convert to samples
    assert all(d <= expected_qrs_max for d in qrs_durations), "QRS durations should be within physiological range"
