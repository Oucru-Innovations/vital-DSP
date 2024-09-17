import pytest
import numpy as np
from scipy.stats import zscore
from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex


@pytest.fixture
def test_signal():
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def sqi_instance(test_signal):
    return SignalQualityIndex(test_signal)


def test_init_with_numpy_array():
    signal = np.array([1, 2, 3, 4, 5])
    sqi = SignalQualityIndex(signal)
    assert isinstance(sqi.signal, np.ndarray)
    assert sqi.signal is signal


def test_init_with_list():
    signal = [1, 2, 3, 4, 5]
    sqi = SignalQualityIndex(signal)
    assert isinstance(sqi.signal, np.ndarray)
    assert np.array_equal(sqi.signal, np.array(signal))


def test_scale_sqi_zscore(sqi_instance):
    sqi_values = np.array([1, 2, 3, 4, 5])
    scaled_sqi = sqi_instance._scale_sqi(sqi_values, scale="zscore")
    assert np.allclose(scaled_sqi, zscore(sqi_values))


def test_scale_sqi_iqr(sqi_instance):
    sqi_values = np.array([1, 2, 3, 4, 5])
    scaled_sqi = sqi_instance._scale_sqi(sqi_values, scale="iqr")
    median = np.median(sqi_values)
    iqr_value = np.subtract(*np.percentile(sqi_values, [75, 25]))
    assert np.allclose(scaled_sqi, (sqi_values - median) / iqr_value)


def test_scale_sqi_minmax(sqi_instance):
    sqi_values = np.array([1, 2, 3, 4, 5])
    scaled_sqi = sqi_instance._scale_sqi(sqi_values, scale="minmax")
    assert np.allclose(scaled_sqi, (sqi_values - 1) / (5 - 1))


def test_scale_sqi_invalid_scale(sqi_instance):
    with pytest.raises(ValueError):
        sqi_instance._scale_sqi(np.array([1, 2, 3]), scale="unknown")


def test_process_segments_below_threshold(sqi_instance):
    def mock_func(segment):
        return np.mean(segment)

    sqi_values, normal_segments, abnormal_segments = sqi_instance._process_segments(
        mock_func, window_size=5, step_size=2, threshold=0.0, threshold_type="below"
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_process_segments_above_threshold(sqi_instance):
    def mock_func(segment):
        return np.mean(segment)

    sqi_values, normal_segments, abnormal_segments = sqi_instance._process_segments(
        mock_func, window_size=5, step_size=2, threshold=0.0, threshold_type="above"
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_amplitude_variability_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = (
        sqi_instance.amplitude_variability_sqi(
            window_size=5, step_size=2, threshold=0.5
        )
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_baseline_wander_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.baseline_wander_sqi(
        window_size=5, step_size=2, threshold=0.5, moving_avg_window=3
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_zero_crossing_sqi():
    # A signal with clear zero crossings
    signal = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    sqi = SignalQualityIndex(signal)

    sqi_values, normal_segments, abnormal_segments = sqi.zero_crossing_sqi(
        window_size=3, step_size=1, threshold=0.5
    )

    assert len(sqi_values) > 0  # Ensure SQI values are computed
    assert len(normal_segments) > 0  # Ensure normal segments are found
    assert (
        len(abnormal_segments) >= 0
    )  # Abnormal segments could be 0 if none are detected


def test_waveform_similarity_sqi(sqi_instance):
    reference_waveform = sqi_instance.signal[:5]
    sqi_values, normal_segments, abnormal_segments = (
        sqi_instance.waveform_similarity_sqi(
            window_size=5,
            step_size=2,
            reference_waveform=reference_waveform,
            threshold=0.5,
        )
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_signal_entropy_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.signal_entropy_sqi(
        window_size=5, step_size=2, threshold=0.5
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_skewness_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.skewness_sqi(
        window_size=5, step_size=2, threshold=0.5
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_kurtosis_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.kurtosis_sqi(
        window_size=5, step_size=2, threshold=0.5
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_peak_to_peak_amplitude_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = (
        sqi_instance.peak_to_peak_amplitude_sqi(
            window_size=5, step_size=2, threshold=0.5
        )
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_snr_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.snr_sqi(
        window_size=5, step_size=2, threshold=0.5
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_energy_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.energy_sqi(
        window_size=5, step_size=2, threshold=0.5
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_heart_rate_variability_sqi():
    rr_intervals = np.array([0.8, 0.85, 0.9, 0.87, 0.89])
    sqi = SignalQualityIndex(rr_intervals)
    sqi_values, normal_segments, abnormal_segments = sqi.heart_rate_variability_sqi(
        rr_intervals, window_size=3, step_size=1, threshold=0.2
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_ppg_signal_quality_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = (
        sqi_instance.ppg_signal_quality_sqi(window_size=5, step_size=2, threshold=0.5)
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_eeg_band_power_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = sqi_instance.eeg_band_power_sqi(
        sqi_instance.signal, window_size=5, step_size=2, threshold=0.5
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0


def test_respiratory_signal_quality_sqi(sqi_instance):
    sqi_values, normal_segments, abnormal_segments = (
        sqi_instance.respiratory_signal_quality_sqi(
            window_size=5, step_size=2, threshold=0.5
        )
    )

    assert len(sqi_values) > 0
    assert len(normal_segments) > 0
    assert len(abnormal_segments) > 0
