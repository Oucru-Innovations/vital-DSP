import numpy as np
import pytest
from vitalDSP.feature_engineering.ppg_autonomic_features import PPGAutonomicFeatures


@pytest.fixture
def generate_ppg_signal():
    """
    Generate synthetic PPG signal for testing.
    """
    np.random.seed(42)
    ppg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.05 * np.random.randn(1000)
    fs = 100  # Sampling frequency of 100 Hz
    return ppg_signal, fs


def test_compute_rrv(generate_ppg_signal):
    """
    Test Respiratory Rate Variability (RRV) calculation.
    """
    ppg_signal, fs = generate_ppg_signal
    features = PPGAutonomicFeatures(ppg_signal, fs)
    rrv = features.compute_rrv()

    assert isinstance(rrv, float), "RRV should return a float value"
    assert rrv > 0, "RRV should be a positive value"


def test_compute_rsa(generate_ppg_signal):
    """
    Test Respiratory Sinus Arrhythmia (RSA) calculation.
    """
    ppg_signal, fs = generate_ppg_signal
    features = PPGAutonomicFeatures(ppg_signal, fs)
    rsa = features.compute_rsa()

    assert isinstance(rsa, float), "RSA should return a float value"
    assert rsa >= 0, "RSA should be a non-negative value"


def test_compute_fractal_dimension(generate_ppg_signal):
    """
    Test Fractal Dimension calculation.
    """
    ppg_signal, fs = generate_ppg_signal
    features = PPGAutonomicFeatures(ppg_signal, fs)
    fractal_dim = features.compute_fractal_dimension()
    assert isinstance(
        fractal_dim, float
    ), "Fractal Dimension should return a float value"
    assert fractal_dim > 0, "Fractal Dimension should be a positive value"


def test_compute_dfa(generate_ppg_signal):
    """
    Test DFA (Detrended Fluctuation Analysis) calculation.
    """
    ppg_signal, fs = generate_ppg_signal
    features = PPGAutonomicFeatures(ppg_signal, fs)
    dfa_value = features.compute_dfa()
    assert isinstance(dfa_value, float), "DFA should return a float value"
    assert dfa_value > 0, "DFA should be a positive value"


def test_empty_signal():
    ppg_signal = np.array([])  # Empty PPG signal
    fs = 100
    with pytest.raises(ValueError, match="PPG signal is too short to compute features"):
        features = PPGAutonomicFeatures(ppg_signal, fs)


def test_invalid_signal_type():
    ppg_signal = "invalid_signal"  # Non-numeric input
    fs = 100
    with pytest.raises(TypeError, match="Input signal must be a numpy array"):
        features = PPGAutonomicFeatures(ppg_signal, fs)


def test_single_value_signal():
    ppg_signal = np.array([1])  # Single-point signal
    fs = 100
    with pytest.raises(ValueError, match="PPG signal is too short to compute features"):
        features = PPGAutonomicFeatures(ppg_signal, fs)


def test_no_peak_signal():
    ppg_signal = np.ones(1000)  # Flatline signal with no peaks
    fs = 100
    features = PPGAutonomicFeatures(ppg_signal, fs)
    with pytest.raises(ValueError, match="No peaks detected in PPG signal"):
        features.compute_rrv()


def test_non_numeric_signal():
    ppg_signal = np.array([np.nan] * 1000)  # Signal with NaN values
    fs = 100
    with pytest.raises(ValueError, match="PPG signal contains invalid values"):
        features = PPGAutonomicFeatures(ppg_signal, fs)


def test_infinite_values_signal():
    ppg_signal = np.array([np.inf] * 1000)  # Signal with infinite values
    fs = 100
    with pytest.raises(ValueError, match="PPG signal contains invalid values"):
        features = PPGAutonomicFeatures(ppg_signal, fs)
