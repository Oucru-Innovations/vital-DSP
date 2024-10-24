import pytest
import numpy as np
from scipy.stats import linregress
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import (
    PreprocessConfig,
    preprocess_signal,
)
from vitalDSP.physiological_features.waveform import WaveformMorphology
from vitalDSP.feature_engineering.morphology_features import (
    PhysiologicalFeatureExtractor,
)
import os
from vitalDSP.notebooks import process_in_chunks, plot_trace


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


# Test the detection of troughs
def test_detect_troughs(feature_extractor):
    peaks = np.array([100, 200, 300])
    troughs = feature_extractor.detect_troughs(peaks)
    assert len(troughs) == len(peaks) - 1
    assert isinstance(troughs, np.ndarray)


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
    # assert isinstance(features["t_wave_area"], float)


def test_extract_features_ppg(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="PPG", preprocess_config=preprocess_config
    )
    assert isinstance(features, dict)
    assert "systolic_duration" in features
    assert isinstance(features["systolic_duration"], float)
    assert isinstance(features["diastolic_duration"], float)


# Test feature extraction for unsupported signal type
def test_extract_features_unsupported_type(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(
        signal_type="Unknown", preprocess_config=preprocess_config
    )
    # Ensure all features are NaN when an unsupported signal type is used
    assert all(np.isnan(value) for value in features.values())


# New Tests: Handle cases where feature extraction raises an error and returns np.nan
def test_extract_features_nan_case_ppg(feature_extractor, preprocess_config, mocker):
    # Mock the PeakDetection to raise an error
    mocker.patch(
        "vitalDSP.utils.peak_detection.PeakDetection.detect_peaks",
        side_effect=Exception("Mock error"),
    )

    features = feature_extractor.extract_features(
        signal_type="PPG", preprocess_config=preprocess_config
    )

    # Check that all features are set to np.nan in case of an error
    for key, value in features.items():
        assert np.isnan(value), f"{key} should be np.nan"


def test_extract_features_nan_case_ecg(feature_extractor, preprocess_config, mocker):
    # Mock the compute_qrs_duration to raise an error
    mocker.patch(
        "vitalDSP.physiological_features.waveform.WaveformMorphology.compute_qrs_duration",
        side_effect=Exception("Mock error"),
    )

    features = feature_extractor.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )

    # Check that all features are set to np.nan in case of an error
    for key, value in features.items():
        assert np.isnan(value), f"{key} should be np.nan"


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
    qrs_duration = morphology.compute_qrs_duration()

    # Check that QRS duration is valid and non-negative
    assert np.isfinite(qrs_duration), "QRS duration should be a finite value."
    assert qrs_duration >= 0, "QRS duration should be non-negative."


# Test for T-wave area calculation in ECG
# def test_ecg_t_wave_area(feature_extractor_ecg, preprocess_config):
#     features = feature_extractor_ecg.extract_features(signal_type="ECG", preprocess_config=preprocess_config)
#     assert isinstance(features, dict)
#     assert "t_wave_area" in features
#     # assert np.isfinite(features["t_wave_area"]), "T-wave area should be a finite value."
#     # assert features["t_wave_area"] >= 0, f"T-wave {features['t_wave_area']} area should be non-negative."

# # Test handling of empty signal or no peaks detected
# def test_extract_features_no_peaks(feature_extractor_ecg, preprocess_config):
#     waveform = np.zeros(1000)  # Simulated flatline signal
#     features = feature_extractor_ecg.extract_features(signal_type="ECG", preprocess_config=preprocess_config)

#     # Check if all features return NaN or 0 due to lack of detected peaks
#     for key, value in features.items():
#         assert np.isnan(value) or value == 0, f"{key} should be NaN or 0 for a flatline signal."


def test_ecg_t_wave_area(feature_extractor_ecg, preprocess_config):
    # Load the ECG data from CSV
    waveform = load_ecg_data()

    # Ensure that the extractor can handle the real waveform data
    features = feature_extractor_ecg.extract_features(
        signal_type="ECG", preprocess_config=preprocess_config
    )

    # Check that features is a dictionary
    assert isinstance(features, dict)

    # Check that t_wave_area is in the features and is a valid number
    assert "t_wave_area" in features, "t_wave_area should be a key in the features."

    # Ensure the value is finite and non-negative
    assert np.isfinite(features["t_wave_area"]), "T-wave area should be a finite value."
    assert features["t_wave_area"] >= 0, "T-wave area should be non-negative."


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
