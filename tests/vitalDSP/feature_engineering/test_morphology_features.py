import pytest
import numpy as np
from scipy.stats import linregress
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import PreprocessConfig, preprocess_signal
from vitalDSP.physiological_features.waveform import WaveformMorphology
from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor

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
    features = feature_extractor.extract_features(signal_type="ECG", preprocess_config=preprocess_config)
    assert isinstance(features, dict)
    assert "qrs_duration" in features
    assert isinstance(features["qrs_duration"], float)
    assert isinstance(features["qrs_area"], float)
    # assert isinstance(features["t_wave_area"], float)

def test_extract_features_ppg(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(signal_type="PPG", preprocess_config=preprocess_config)
    assert isinstance(features, dict)
    assert "systolic_duration" in features
    assert isinstance(features["systolic_duration"], float)
    assert isinstance(features["diastolic_duration"], float)

# Test feature extraction for unsupported signal type
def test_extract_features_unsupported_type(feature_extractor, preprocess_config):
    features = feature_extractor.extract_features(signal_type="Unknown", preprocess_config=preprocess_config)
    # Ensure all features are NaN when an unsupported signal type is used
    assert all(np.isnan(value) for value in features.values())

# New Tests: Handle cases where feature extraction raises an error and returns np.nan
def test_extract_features_nan_case_ppg(feature_extractor, preprocess_config, mocker):
    # Mock the PeakDetection to raise an error
    mocker.patch('vitalDSP.utils.peak_detection.PeakDetection.detect_peaks', side_effect=Exception('Mock error'))
    
    features = feature_extractor.extract_features(signal_type="PPG", preprocess_config=preprocess_config)
    
    # Check that all features are set to np.nan in case of an error
    for key, value in features.items():
        assert np.isnan(value), f"{key} should be np.nan"

def test_extract_features_nan_case_ecg(feature_extractor, preprocess_config, mocker):
    # Mock the compute_qrs_duration to raise an error
    mocker.patch('vitalDSP.physiological_features.waveform.WaveformMorphology.compute_qrs_duration', side_effect=Exception('Mock error'))

    features = feature_extractor.extract_features(signal_type="ECG", preprocess_config=preprocess_config)

    # Check that all features are set to np.nan in case of an error
    for key, value in features.items():
        assert np.isnan(value), f"{key} should be np.nan"
