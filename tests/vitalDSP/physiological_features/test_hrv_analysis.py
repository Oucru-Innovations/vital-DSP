import pytest
import numpy as np
from unittest.mock import Mock, patch
from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
from vitalDSP.transforms.beats_transformation import RRTransformation
from vitalDSP.physiological_features.hrv_analysis import HRVFeatures


@pytest.fixture
def setup_data():
    """
    Fixture to provide common test data for HRV features.
    """
    nn_intervals = [800, 810, 790, 805, 795]
    signal = np.random.randn(1000)
    fs = 1000  # Sampling frequency
    return nn_intervals, signal, fs


@pytest.fixture
def sample_nn_intervals():
    return np.array([800, 810, 790, 805, 795])


# Fixture for invalid NN intervals (empty array)
@pytest.fixture
def invalid_nn_intervals():
    return np.array([])


@pytest.fixture
def sample_signal():
    return np.random.randn(1000)


@pytest.fixture
def hrv(sample_nn_intervals, sample_signal):
    return HRVFeatures(sample_signal, sample_nn_intervals, fs=1000, signal_type="ecg")


def test_hrv_features_initialization_with_nn_intervals(setup_data):
    """
    Test initialization with NN intervals provided.
    """
    nn_intervals, signal, fs = setup_data
    hrv = HRVFeatures(nn_intervals=nn_intervals, signals=signal, fs=fs)
    assert isinstance(hrv.nn_intervals, np.ndarray)
    assert hrv.fs == fs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_hrv_features_initialization_with_rr_transformation(setup_data):
    """
    Test initialization when RR intervals are computed from the signal using RRTransformation.
    """
    # Mock a signal that should not have any peaks (a flat signal)
    flat_signal = np.zeros(1000)  # Flat signal with no variation
    fs = 1000  # Sampling frequency

    # Expect RRTransformation to raise ValueError because no peaks can be found in a flat signal
    with pytest.raises(
        ValueError,
        match="No peaks detected in the signal. RR interval computation failed.",
    ):
        HRVFeatures(signals=flat_signal, fs=fs, signal_type="ppg")


def test_empty_nn_intervals(setup_data):
    """
    Test for empty NN intervals input.
    """
    _, signal, fs = setup_data
    with pytest.raises(ValueError):
        HRVFeatures(nn_intervals=[], signals=signal, fs=fs)


def test_invalid_nn_intervals(setup_data):
    """
    Test for invalid NN intervals (all zeros).
    """
    _, signal, fs = setup_data
    with pytest.raises(ValueError):
        HRVFeatures(nn_intervals=[0, 0, 0], signals=signal, fs=fs)


def test_compute_time_domain_features(setup_data):
    """
    Test the time-domain feature extraction from NN intervals.
    """
    nn_intervals, signal, fs = setup_data
    hrv = HRVFeatures(nn_intervals=nn_intervals, signals=signal, fs=fs)
    features = hrv.compute_all_features()

    assert "sdnn" in features
    assert "rmssd" in features
    assert "nn50" in features
    assert "pnn50" in features
    assert "mean_nn" in features

    # Check if values are correctly computed
    assert np.isclose(features["sdnn"], np.std(nn_intervals))
    assert np.isclose(features["rmssd"], np.sqrt(np.mean(np.diff(nn_intervals) ** 2)))


def test_compute_frequency_domain_features(setup_data):
    """
    Test the frequency-domain feature extraction.
    """
    nn_intervals, signal, fs = setup_data
    hrv = HRVFeatures(nn_intervals=nn_intervals, signals=signal, fs=fs)
    features = hrv.compute_all_features()

    assert "lf_power" in features
    assert "hf_power" in features
    assert "lf_hf_ratio" in features

    # Validate feature values are either numeric or NaN
    assert np.issubdtype(type(features["lf_power"]), np.number) or np.isnan(
        features["lf_power"]
    )
    assert np.issubdtype(type(features["hf_power"]), np.number) or np.isnan(
        features["hf_power"]
    )
    assert np.issubdtype(type(features["lf_hf_ratio"]), np.number) or np.isnan(
        features["lf_hf_ratio"]
    )


def test_compute_nonlinear_features(setup_data):
    """
    Test the nonlinear feature extraction.
    """
    nn_intervals, signal, fs = setup_data
    hrv = HRVFeatures(nn_intervals=nn_intervals, signals=signal, fs=fs)
    features = hrv.compute_all_features()

    assert "dfa" in features
    # assert "approx_entropy" in features
    assert "fractal_dimension" in features
    assert "lyapunov_exponent" in features

    # Validate feature values are either numeric or NaN
    # assert np.issubdtype(type(features["sample_entropy"]), np.number) or np.isnan(features["sample_entropy"])
    # assert np.issubdtype(type(features["approx_entropy"]), np.number) or np.isnan(features["approx_entropy"])
    assert features["dfa"] is not None
    assert features["lyapunov_exponent"] is not None
    assert np.issubdtype(type(features["fractal_dimension"]), np.number) or np.isnan(
        features["fractal_dimension"]
    )


def test_edge_case_zero_signal(setup_data):
    """
    Test the edge case where the signal is all zeros.
    """
    nn_intervals, _, fs = setup_data
    signal_zeros = np.zeros(1000)
    hrv = HRVFeatures(nn_intervals=nn_intervals, signals=signal_zeros, fs=fs)
    features = hrv.compute_all_features()
    # assert np.isclose(features["sample_entropy"], 0)
    # Ensure sample_entropy is not None
    assert features["dfa"] is not None
    # Now safely use np.isclose
    # assert np.isclose(features["dfa"], 0)


def test_edge_case_constant_nn_intervals(setup_data):
    """
    Test the edge case where NN intervals are constant.
    """
    _, signal, fs = setup_data
    constant_nn_intervals = [800] * 1000
    hrv = HRVFeatures(nn_intervals=constant_nn_intervals, signals=signal, fs=fs)
    features = hrv.compute_all_features()

    # Time-domain features should reflect constant intervals (low variation)
    assert np.isclose(features["sdnn"], 0)
    assert np.isclose(features["rmssd"], 0)


def test_missing_nn_intervals(setup_data):
    """
    Test the edge case when no NN intervals are provided and they are derived from the signal.
    """
    # Use a flat signal that does not produce any detectable peaks
    flat_signal = np.zeros(1000)  # Flat signal with no peaks
    fs = 1000  # Sampling frequency

    # Expect RRTransformation to raise ValueError due to no peaks being detected
    with pytest.raises(
        ValueError,
        match="No peaks detected in the signal. RR interval computation failed.",
    ):
        HRVFeatures(signals=flat_signal, fs=fs, signal_type="ecg")


def test_high_sampling_frequency(setup_data):
    """
    Test with a very high sampling frequency (stress testing).
    """
    nn_intervals, signal, _ = setup_data
    high_fs = 10000
    hrv = HRVFeatures(nn_intervals=nn_intervals, signals=signal, fs=high_fs)
    features = hrv.compute_all_features()

    # Ensure high-frequency sampling doesn't break frequency-domain feature extraction
    assert "lf_power" in features
    assert "hf_power" in features


def test_missing_signal(setup_data):
    """
    Test the edge case where no signal is provided.
    """
    nn_intervals, _, fs = setup_data
    with pytest.raises(TypeError):
        HRVFeatures(nn_intervals=nn_intervals)


# Test for invalid RR intervals transformation failure
def test_hrv_init_with_invalid_rr_intervals(sample_signal):
    with pytest.raises(
        ValueError,
        match="No peaks detected in the signal. RR interval computation failed.",
    ):
        waveform = np.array(sample_signal)
        waveform[:] = 0  # Simulate a flat signal to trigger the no peaks detected error
        HRVFeatures(waveform, nn_intervals=None, fs=100, signal_type="ecg")


# Test handling for time-domain feature exception (invalid NN intervals)
def test_hrv_time_features_exception_handling(sample_signal, invalid_nn_intervals):
    with pytest.raises(ValueError, match="NN intervals cannot be empty."):
        HRVFeatures(sample_signal, nn_intervals=invalid_nn_intervals, fs=1000)


# Test frequency-domain feature handling for invalid NN intervals
def test_hrv_frequency_features_exception_handling(sample_signal, invalid_nn_intervals):
    with pytest.raises(ValueError, match="NN intervals cannot be empty."):
        HRVFeatures(sample_signal, nn_intervals=invalid_nn_intervals, fs=1000)


# Test compute_all_features for valid input
def test_hrv_compute_all_features(hrv):
    features = hrv.compute_all_features()
    assert isinstance(features, dict)
    assert "sdnn" in features
    assert "rmssd" in features


# Test for frequency-domain features
def test_hrv_compute_frequency_domain_features(hrv):
    features = hrv.compute_all_features()
    assert "lf_power" in features
    assert "hf_power" in features
    assert features["lf_power"] is not None
    assert features["hf_power"] is not None


# Test nonlinear features
def test_hrv_compute_nonlinear_features(hrv):
    features = hrv.compute_all_features()
    assert "fractal_dimension" in features
    assert features["fractal_dimension"] is not None


# Test complex methods calculation
def test_hrv_compute_complex_methods(hrv):
    features = hrv.compute_all_features(include_complex_methods=True)
    assert "sample_entropy" in features
    assert features["sample_entropy"] is not None


# Test for exception in complex methods
def test_hrv_complex_methods_exception_handling(hrv):
    features = hrv.compute_all_features(include_complex_methods=True)
    assert "sample_entropy" in features
    assert features["sample_entropy"] is not None

# Update Test Cases
def test_rr_interval_transformation_fails():
    # Simulate signals and initialize HRVFeatures with rr_intervals being None
    signals = [1, 2, 3, 4, 5]  # Example signals (short signal that triggers filter error)
    fs = 100  # Sampling frequency
    signal_type = "ppg"
    # Expect the ValueError due to short signal length for the filter
    with pytest.raises(ValueError, match="Signal too short for the specified filter order."):
        HRVFeatures(signals=signals, fs=fs, signal_type=signal_type)

def test_rr_interval_transformation_fails_due_to_invalid_rr():
    signals = [1] * 100  # Longer signals to avoid filter error
    fs = 100
    signal_type = "ppg"

    # Mock RRTransformation to return None, simulating a failure in RR extraction
    with patch('vitalDSP.transforms.beats_transformation.RRTransformation.process_rr_intervals', return_value=None):
        with pytest.raises(ValueError, match="RR interval transformation failed to extract valid RR intervals"):
            HRVFeatures(signals=signals, fs=fs, signal_type=signal_type)

def test_time_domain_exception():
    # Mock TimeDomainFeatures method to raise an exception
    signals = [1, 2, 3, 4, 5]  # Example signals
    nn_intervals = [800, 810, 790, 805, 795]  # NN intervals in ms

    hrv = HRVFeatures(signals, nn_intervals)

    with patch('vitalDSP.physiological_features.time_domain.TimeDomainFeatures.compute_sdnn', side_effect=Exception('Test error')):
        features = hrv.compute_all_features()
        assert np.isnan(features['sdnn'])  # Assert that the feature is NaN when an exception occurs

def test_frequency_domain_exception():
    signals = [1, 2, 3, 4, 5]  # Example signals
    nn_intervals = [800, 810, 790, 805, 795]  # NN intervals in ms

    hrv = HRVFeatures(signals, nn_intervals)

    with patch('vitalDSP.physiological_features.frequency_domain.FrequencyDomainFeatures.compute_lf', side_effect=Exception('Test error')):
        features = hrv.compute_all_features()
        assert np.isnan(features['lf_power'])  # Assert that the feature is NaN when an exception occurs

def test_nonlinear_exception():
    signals = [1, 2, 3, 4, 5]  # Example signals
    nn_intervals = [800, 810, 790, 805, 795]  # NN intervals in ms

    hrv = HRVFeatures(signals, nn_intervals)

    with patch('vitalDSP.physiological_features.nonlinear.NonlinearFeatures.compute_fractal_dimension', side_effect=Exception('Test error')):
        features = hrv.compute_all_features()
        assert np.isnan(features['fractal_dimension'])  # Assert that the feature is NaN when an exception occurs
        
def test_complex_feature_exception():
    signals = [1, 2, 3, 4, 5]  # Example signals
    nn_intervals = [800, 810, 790, 805, 795]  # NN intervals in ms

    hrv = HRVFeatures(signals, nn_intervals)

    with patch('vitalDSP.physiological_features.nonlinear.NonlinearFeatures.compute_sample_entropy', side_effect=Exception('Test error')):
        features = hrv.compute_all_features(include_complex_methods=True)
        assert features['sample_entropy'] is None  # Assert that the feature is None when an exception occurs
        assert features['approximate_entropy'] is None
        assert features['recurrence_rate'] is None
        assert features['determinism'] is None
        assert features['laminarity'] is None
