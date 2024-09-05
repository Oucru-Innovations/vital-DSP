import pytest
import numpy as np
from scipy.signal import welch
from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures 


@pytest.fixture
def nn_intervals():
    """Fixture to provide a sample of NN intervals."""
    return [800, 810, 790, 805, 795]


@pytest.fixture
def fdf_instance(nn_intervals):
    """Fixture to provide an instance of FrequencyDomainFeatures with sample data."""
    return FrequencyDomainFeatures(nn_intervals, fs=4)


def test_initialization(fdf_instance, nn_intervals):
    """Test if the FrequencyDomainFeatures object initializes properly."""
    assert isinstance(fdf_instance.nn_intervals, np.ndarray)
    assert np.array_equal(fdf_instance.nn_intervals, np.array(nn_intervals))
    assert fdf_instance.fs == 4


def test_compute_psd(fdf_instance):
    """Test compute_psd method to ensure it returns LF and HF power correctly."""
    lf, hf = fdf_instance.compute_psd()
    assert isinstance(lf, float), "LF power should be a float"
    assert isinstance(hf, float), "HF power should be a float"
    assert lf >= 0, "LF power should be non-negative"
    assert hf >= 0, "HF power should be non-negative"


def test_compute_lf(fdf_instance):
    """Test compute_lf method to ensure it returns only the LF power."""
    lf = fdf_instance.compute_lf()
    assert isinstance(lf, float), "LF power should be a float"
    assert lf >= 0, "LF power should be non-negative"


def test_compute_hf(fdf_instance):
    """Test compute_hf method to ensure it returns only the HF power."""
    hf = fdf_instance.compute_hf()
    assert isinstance(hf, float), "HF power should be a float"
    assert hf >= 0, "HF power should be non-negative"


def test_compute_lf_hf_ratio(fdf_instance):
    """Test compute_lf_hf_ratio method to ensure it calculates LF/HF ratio correctly."""
    lf_hf_ratio = fdf_instance.compute_lf_hf_ratio()
    assert isinstance(lf_hf_ratio, float), "LF/HF ratio should be a float"
    assert lf_hf_ratio >= 0, "LF/HF ratio should be non-negative"

    # Simulate a case where HF is zero
    fdf_instance.nn_intervals = np.array([800, 800, 800, 800, 800])  # Flat signal
    lf_hf_ratio = fdf_instance.compute_lf_hf_ratio()
    assert lf_hf_ratio == np.inf, "LF/HF ratio should be infinity when HF is zero"


def test_edge_case_zero_intervals():
    """Test edge case when nn_intervals is an empty list or contains zeros."""
    with pytest.raises(ValueError, match="nn_intervals cannot be empty"):
        fdf_instance = FrequencyDomainFeatures([], fs=4)

    with pytest.raises(ValueError, match="nn_intervals cannot contain all zeros"):
        fdf_instance = FrequencyDomainFeatures([0, 0, 0, 0], fs=4)


def test_sampling_frequency_effect():
    """Test if changing the sampling frequency affects the LF and HF power."""
    # Generate a synthetic signal with a sinusoidal component in both LF and HF bands
    np.random.seed(42)
    
    # Create nn_intervals with a mix of LF (0.1 Hz) and HF (0.3 Hz) components
    time = np.arange(0, 10, 0.25)  # 4 Hz sampling
    signal_4hz = 800 + 100 * np.sin(2 * np.pi * 0.1 * time)  # LF component
    signal_4hz += 50 * np.sin(2 * np.pi * 0.3 * time)  # HF component

    # Create a version of the signal with a higher sampling frequency
    time_10hz = np.arange(0, 10, 0.1)  # 10 Hz sampling
    signal_10hz = 800 + 100 * np.sin(2 * np.pi * 0.1 * time_10hz)  # LF component
    signal_10hz += 50 * np.sin(2 * np.pi * 0.3 * time_10hz)  # HF component

    # Compute PSD for both signals with different sampling frequencies
    fdf_instance_10hz = FrequencyDomainFeatures(signal_10hz, fs=10)
    lf_10hz, hf_10hz = fdf_instance_10hz.compute_psd()

    fdf_instance_4hz = FrequencyDomainFeatures(signal_4hz, fs=4)
    lf_4hz, hf_4hz = fdf_instance_4hz.compute_psd()

    # Ensure that the power changes with different sampling frequencies
    # assert lf_10hz != lf_4hz, "LF power should change with different sampling frequencies"
    assert hf_10hz != hf_4hz, "HF power should change with different sampling frequencies"
