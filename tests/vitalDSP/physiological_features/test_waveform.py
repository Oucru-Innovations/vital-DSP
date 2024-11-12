import pytest
import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.physiological_features.waveform import WaveformMorphology
from unittest.mock import MagicMock
from vitalDSP.utils.synthesize_data import generate_ecg_signal, generate_synthetic_ppg
from scipy.stats import linregress

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

def test_get_peak_trend_slope_empty_peaks():
    waveform = generate_synthetic_ppg(256)
    wm = WaveformMorphology(waveform, fs=256, signal_type="PPG")
    peaks = np.array([])  # No peaks
    
    slope = wm.get_peak_trend_slope(peaks)
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
    wm.r_peaks = np.array([50, 150, 250])

    # Intentionally omit q_sessions and s_sessions to test default detection
    r_sessions = wm.detect_r_session()

    # Check that r_sessions were detected based on the default peak detection
    assert r_sessions.size > 0, "Expected non-empty R sessions when q_sessions and s_sessions are detected by default"

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
