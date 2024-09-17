import pytest
import numpy as np
from vitalDSP.advanced_computation.harmonic_percussive_separation import (
    HarmonicPercussiveSeparation,
)


@pytest.fixture
def test_signal():
    # Fixture to create a sample signal for testing
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)


@pytest.fixture
def hps(test_signal):
    # Fixture to initialize HarmonicPercussiveSeparation
    return HarmonicPercussiveSeparation(test_signal)


def test_hps_init(hps, test_signal):
    # Test initialization
    assert isinstance(hps.signal, np.ndarray)
    assert np.array_equal(hps.signal, test_signal)


def test_hps_separate(hps):
    # Test the separation method for 1D signal
    harmonic, percussive = hps.separate(kernel_size=31)
    assert isinstance(harmonic, np.ndarray)
    assert isinstance(percussive, np.ndarray)
    assert harmonic.shape == hps.signal.shape
    assert percussive.shape == hps.signal.shape


def test_apply_median(hps, test_signal):
    # Test the _apply_median method for 1D signal
    filtered_arr = hps._apply_median(test_signal, kernel_size=31)
    assert isinstance(filtered_arr, np.ndarray)
    assert len(filtered_arr) == len(test_signal)
    assert np.all(filtered_arr >= np.min(test_signal))  # Check values are within range
    assert np.all(filtered_arr <= np.max(test_signal))  # Check values are within range


def test_invalid_signal_type():
    # Test the initialization with invalid signal input
    with pytest.raises(TypeError):
        HarmonicPercussiveSeparation("invalid_signal")
