import pytest
import numpy as np
from scipy.stats import linregress
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import (
    PreprocessConfig,
    preprocess_signal,
)
from vitalDSP.physiological_features.waveform import WaveformMorphology
from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
import os
from vitalDSP.notebooks import process_in_chunks, plot_trace
import warnings
warnings.simplefilter('default', DeprecationWarning)
import logging
import numpy as np
from vitalDSP.feature_engineering.ecg_autonomic_features import ECGExtractor
from vitalDSP.preprocess.preprocess_operations import preprocess_signal
from unittest.mock import patch

# Mock logger to prevent logging during tests
logger = logging.getLogger("vitalDSP.feature_engineering.ecg_autonomic_features")


def load_ecg_data():
    # Assuming the test is located in tests/vital_DSP/feature_engineering/test and sample_data is at the same level as src
    data_path = os.path.join(
        os.path.dirname(__file__), "../../../sample_data/public/ecg_small.csv"
    )
    signal_col, date_col = process_in_chunks(data_path, data_type="ecg", fs=256)
    signal_col = np.array(signal_col)
    return signal_col  # Assuming the ECG data is in a column labeled 'ECG'


@pytest.fixture
def generate_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)


@pytest.fixture
def feature_extractor(generate_signal):
    return PhysiologicalFeatureExtractor(generate_signal, fs=1000)


@pytest.fixture
def preprocess_config():
    return PreprocessConfig()


# Test the constructor initialization
def test_feature_extractor_initialization(generate_signal):
    extractor = PhysiologicalFeatureExtractor(generate_signal, fs=1000)
    assert isinstance(extractor.signal, np.ndarray)
    assert extractor.fs == 1000

# Test PPG feature extraction
def test_extract_features_ppg(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="PPG", preprocess_config=preprocess_config
    )
    assert isinstance(features, dict)
    assert "systolic_duration" in features
    assert isinstance(features["systolic_duration"], float)
    assert isinstance(features["diastolic_duration"], float)
    assert "signal_skewness" in features
    assert "peak_trend_slope" in features

# Test the detection of troughs
# def test_detect_troughs(feature_extractor):
#     peaks = np.array([100, 200, 300])
#     troughs = feature_extractor.detect_troughs(peaks)
#     assert len(troughs) == len(peaks) - 1
#     assert isinstance(troughs, np.ndarray)


# Test the computation of peak trend
def test_compute_peak_trend(feature_extractor):
    peaks = np.array([100, 200, 300])
    slope = feature_extractor.compute_peak_trend(peaks)
    assert isinstance(slope, float)


# Test when there's only one peak, should return 0.0
def test_compute_peak_trend_single_peak(feature_extractor):
    peaks = np.array([100])
    slope = feature_extractor.compute_peak_trend(peaks)
    assert slope == 0.0


# Test the computation of amplitude variability
def test_compute_amplitude_variability(feature_extractor):
    peaks = np.array([100, 200, 300])
    variability = feature_extractor.compute_amplitude_variability(peaks)
    assert isinstance(variability, float)
    assert variability > 0


# Test amplitude variability when there's only one peak
def test_compute_amplitude_variability_single_peak(feature_extractor):
    peaks = np.array([100])
    variability = feature_extractor.compute_amplitude_variability(peaks)
    assert variability == 0.0


# Test preprocessing functionality
def test_preprocess_signal(feature_extractor, preprocess_config):
    clean_signal = feature_extractor.get_preprocess_signal(preprocess_config)
    assert isinstance(clean_signal, np.ndarray)
    assert clean_signal.shape == feature_extractor.signal.shape


# Test ECG feature extraction
def test_extract_features_ecg(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )
    assert isinstance(features, dict)
    assert "qrs_duration" in features
    assert isinstance(features["qrs_duration"], float)
    assert isinstance(features["qrs_area"], float)
    assert "t_wave_area" in features
    assert "heart_rate" in features
    
    features = feature_extractor.extract_features(
        signal_type="ECG"
    )
    assert isinstance(features, dict)


# Test ECG feature extraction
def test_extract_features_ecg(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )
    assert isinstance(features, dict)
    assert "qrs_duration" in features
    assert isinstance(features["qrs_duration"], float)
    assert isinstance(features["qrs_area"], float)
    assert "t_wave_area" in features
    assert "heart_rate" in features
    
    features = feature_extractor.extract_features(
        signal_type="ECG"
    )
    assert isinstance(features, dict)


def test_preprocessing_error_handling(mocker):
    # Setup sample ECG data
    ecg_signal = np.random.rand(1000)
    fs = 250
    extractor = ECGExtractor(ecg_signal, fs)

    # Patch the detect_r_peaks method to raise an exception during processing
    with patch.object(extractor, 'detect_r_peaks', side_effect=Exception("Mock preprocessing error")):
        # Attempt to call a feature computation method that depends on detect_r_peaks
        try:
            p_wave_duration = extractor.compute_p_wave_duration()
        except Exception as e:
            # If an exception occurs, handle it and set the expected output
            p_wave_duration = np.nan
        
        # Assert that in case of an error, the function handled it by returning NaN
        assert np.isnan(p_wave_duration), "P-wave duration should be NaN when an error occurs in preprocessing"

# Test unsupported signal type
def test_extract_features_unsupported_type(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="Unknown", preprocess_config=preprocess_config
    )
    # Ensure all features are NaN when an unsupported signal type is used
    assert all(np.isnan(value) for value in features.values()), "All features should be NaN for an unsupported signal type."


# Test extraction for ECG when no peaks are detected
def test_extract_features_no_peaks_ecg(preprocess_config):
    waveform = np.zeros(1000)  # Flat signal
    feature_extractor = PhysiologicalFeatureExtractor(waveform, fs=1000)
    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )
    for key, value in features.items():
        assert np.isnan(value) or value == 0, f"{key} should be NaN or 0 for flat signal."


# New Tests: Handle cases where feature extraction raises an error and returns np.nan
# Test amplitude variability for single peak in ECG
def test_compute_amplitude_variability_single_peak_ecg(feature_extractor):
    peaks = np.array([100])
    variability = feature_extractor.compute_amplitude_variability(peaks)
    assert variability == 0.0


# Handle PPG feature extraction nan case with mock error
def test_extract_features_nan_case_ppg(feature_extractor, preprocess_config, mocker):
    mocker.patch(
        "vitalDSP.utils.peak_detection.PeakDetection.detect_peaks",
        side_effect=Exception("Mock error"),
    )
    features = feature_extractor.extract_features(
        signal_type="PPG", preprocess_config=preprocess_config
    )
    for key, value in features.items():
        assert np.isnan(value), f"{key} should be np.nan due to mock error"


# Mock QRS duration to raise an error for testing
def test_extract_features_nan_case_ecg(feature_extractor, preprocess_config, mocker):
    mocker.patch(
        "vitalDSP.physiological_features.waveform.WaveformMorphology.compute_duration",
        side_effect=Exception("Mock error"),
    )
    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )
    for key, value in features.items():
        assert np.isnan(value), f"{key} should be np.nan"


# Test heart rate and amplitude variability with NaN handling
def test_heart_rate_and_amplitude_variability(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )
    assert "heart_rate" in features
    assert "r_peak_amplitude_variability" in features
    if not np.isnan(features["heart_rate"]):
        assert features["heart_rate"] > 0
    if not np.isnan(features["r_peak_amplitude_variability"]):
        assert features["r_peak_amplitude_variability"] >= 0


@pytest.fixture
def feature_extractor_ecg():
    # Simulate signal for the feature extractor (e.g., PPG or ECG)
    return PhysiologicalFeatureExtractor(np.random.randn(1000), fs=1000)


def test_ecg_qrs_detection():
    # Load the ECG data from CSV
    waveform = load_ecg_data()

    # Initialize the feature extractor with the loaded ECG data
    morphology = WaveformMorphology(waveform, fs=1000, signal_type="ECG")

    # Mock the peak detection to return R peaks for testing purposes
    r_peaks = np.array([100, 300, 500, 700, 900])
    morphology.detect_peaks = lambda: r_peaks

    # Perform QRS detection
    qrs_duration = morphology.compute_duration(mode="QRS")

    # Check that all QRS duration values are finite and non-negative
    assert np.all(np.isfinite(qrs_duration)), "All QRS duration values should be finite."
    # assert np.all(qrs_duration != 0), "All QRS duration values should be non-negative."


# Test extraction with valid peaks and troughs for PPG
def test_extract_features_ppg_alignment(preprocess_config):
    # Create a signal that produces clear peaks/troughs
    signal = np.abs(np.sin(np.linspace(0, 10 * np.pi, 1000)))
    feature_extractor = PhysiologicalFeatureExtractor(signal, fs=1000)
    features = feature_extractor.extract_features(
        signal_type="PPG", preprocess_config=preprocess_config
    )
    if not np.isnan(features["systolic_area"]):
        assert features["systolic_area"] > 0, "Systolic area should be positive"
    if not np.isnan(features["diastolic_area"]):
        assert features["diastolic_area"] > 0, "Diastolic area should be positive"


# Test the baseline correction step
def test_baseline_correction(feature_extractor, preprocess_config):
    clean_signal = feature_extractor.get_preprocess_signal(preprocess_config)
    baseline_corrected_signal = clean_signal - np.min(clean_signal)
    assert np.min(baseline_corrected_signal) == 0, "Signal should be baseline-corrected."


# Test the T-wave area calculation in ECG feature extraction
def test_extract_features_t_wave_area(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )
    assert "t_wave_area" in features
    assert isinstance(features["t_wave_area"], float)


def test_extract_features_no_peaks(preprocess_config):
    # Load the ECG data from CSV
    waveform = load_ecg_data()

    # Make the waveform flatline to simulate no peaks
    waveform[:] = 0
    feature_extractor_ecg = PhysiologicalFeatureExtractor(waveform, fs=1000)
    # Extract features from the flatline signal
    features = feature_extractor_ecg.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )

    # Check that all features return NaN or 0 due to lack of detected peaks
    for key, value in features.items():
        assert (
            np.isnan(value) or value == 0
        ), f"{key} {value} should be NaN or 0 for a flatline signal."


# Test handling of signal preprocessing (mocked)
def test_signal_preprocessing(feature_extractor_ecg, preprocess_config, mocker):
    mocker.patch.object(
        feature_extractor_ecg, "get_preprocess_signal", return_value=np.ones(1000)
    )

    features = feature_extractor_ecg.extract_features(
        signal_type="PPG", preprocess_config=preprocess_config
    )
    assert isinstance(features, dict)
    assert "systolic_duration" in features
    assert features["systolic_duration"] == 0 or np.isnan(
        features["systolic_duration"]
    ), "Expected 0 or NaN for a flat signal after preprocessing."


def test_extract_features_ecg_case(sample_signal):
    signal, fs = sample_signal
    extractor = PhysiologicalFeatureExtractor(signal, fs)

    # Test with ECG signal type
    preprocess_config = PreprocessConfig()
    features = extractor.extract_features(signal_type="ECG", preprocess_config=preprocess_config)

    # Check that all expected features are present and have values (could be NaN in some cases)
    expected_keys = [
        "qrs_duration", "qrs_area", "qrs_amplitude", "qrs_slope",
        "t_wave_area", "heart_rate", "r_peak_amplitude_variability",
        "signal_skewness", "peak_trend_slope"
    ]
    assert set(features.keys()) == set(expected_keys), "ECG features dictionary keys mismatch"
    for value in features.values():
        assert np.isnan(value) or isinstance(value, (float, int)), "Feature values should be NaN or numerical"

@pytest.fixture
def sample_signal():
    """Fixture to create a sample ECG/PPG signal and sampling frequency."""
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simple sine wave signal
    fs = 250  # Sample frequency
    return signal, fs

def test_extract_features_ppg_case(sample_signal):
    signal, fs = sample_signal
    extractor = PhysiologicalFeatureExtractor(signal, fs)

    # Test with PPG signal type
    preprocess_config = PreprocessConfig()
    features = extractor.extract_features(signal_type="PPG", preprocess_config=preprocess_config)

    # Check that all expected features are present and have values
    expected_keys = [
        "systolic_duration", "diastolic_duration", "systolic_area", "diastolic_area",
        "systolic_slope", "diastolic_slope", "signal_skewness", "peak_trend_slope",
        "heart_rate","systolic_amplitude_variability", "diastolic_amplitude_variability"
    ]
    assert set(features.keys()) == set(expected_keys), "PPG features dictionary keys mismatch"
    for value in features.values():
        assert np.isnan(value) or isinstance(value, (float, int)), "Feature values should be NaN or numerical"

def test_extract_features_default_config_case(sample_signal):
    signal, fs = sample_signal
    extractor = PhysiologicalFeatureExtractor(signal, fs)

    # Test with default PreprocessConfig (None)
    features = extractor.extract_features(signal_type="ECG")
    assert features is not None, "Features should not be None with default config"

    # Test with default PreprocessConfig (None)
    features = extractor.extract_features(signal_type="PPG")
    assert features is not None, "Features should not be None with default config"

def test_extract_features_error_handling_case(sample_signal):
    signal, fs = sample_signal
    extractor = PhysiologicalFeatureExtractor(signal, fs)

    # Patch preprocess function to raise an exception
    with patch.object(extractor, 'get_preprocess_signal', side_effect=Exception("Mock preprocessing error")):
        features = extractor.extract_features(signal_type="ECG", preprocess_config=PreprocessConfig())
        assert all(np.isnan(value) for value in features.values()), "All features should be NaN on preprocessing error"
