import pytest
import numpy as np
from vitalDSP.feature_engineering.ppg_light_features import (
    PPGLightFeatureExtractor,
)  # Assuming the class is in a file named ppg_feature_extractor.py


@pytest.fixture
def test_signals():
    ir_signal = np.random.randn(1000)  # Simulated IR signal
    red_signal = np.random.randn(1000)  # Simulated Red signal
    sampling_freq = 100  # Sampling frequency (Hz)
    return ir_signal, red_signal, sampling_freq


def test_calculate_spo2(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform SpO2 calculation
    spo2, times_spo2 = ppg_extractor.calculate_spo2()

    assert len(spo2) == len(
        times_spo2
    ), "SpO2 and time arrays must have the same length."
    assert spo2.min() >= 0, "SpO2 values should be non-negative."  # Allow 0 as valid
    assert spo2.max() <= 100, "SpO2 values should be within 0-100 range."


def test_calculate_spo2_no_red_signal(test_signals):
    ir_signal, _, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)

    with pytest.raises(ValueError, match="Red signal is required to compute SpO2."):
        ppg_extractor.calculate_spo2()


def test_calculate_perfusion_index(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform Perfusion Index calculation
    pi, times_pi = ppg_extractor.calculate_perfusion_index()

    assert len(pi) == len(times_pi), "PI and time arrays must have the same length."
    assert pi.min() >= 0, "Perfusion index values should be non-negative."
    assert (
        times_pi[-1] <= len(ir_signal) / sampling_freq
    ), "Last timestamp should not exceed the total signal duration."


def test_calculate_respiratory_rate(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform Respiratory Rate calculation
    rr, times_rr = ppg_extractor.calculate_respiratory_rate()

    assert len(rr) == len(times_rr), "RR and time arrays must have the same length."
    assert (
        len(rr) > 0
    ), "Respiratory rate array should not be empty."  # Ensure array is not empty
    assert rr.min() >= 0, "Respiratory rate values should be non-negative."


def test_calculate_ppr(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform PPR calculation
    ppr, times_ppr = ppg_extractor.calculate_ppr()

    assert len(ppr) == len(times_ppr), "PPR and time arrays must have the same length."
    assert ppr.min() >= 0, "PPR values should be non-negative."
    assert (
        times_ppr[-1] <= len(ir_signal) / sampling_freq
    ), "Last timestamp should not exceed the total signal duration."


def test_calculate_ppr_no_red_signal(test_signals):
    ir_signal, _, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)

    with pytest.raises(ValueError, match="Red signal is required to compute PPR."):
        ppg_extractor.calculate_ppr()
