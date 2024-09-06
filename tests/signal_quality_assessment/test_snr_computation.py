import pytest
import numpy as np
from vitalDSP.signal_quality_assessment.snr_computation import (
    snr_power_ratio,
    snr_peak_to_peak,
    snr_mean_square,
    crest_factor,
    harmonic_distortion,
    signal_to_noise_and_distortion_ratio,
    signal_to_noise_and_interference_ratio
)

@pytest.fixture
def sample_signal_and_noise():
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    noise = 0.1 * np.random.normal(size=signal.shape)
    return signal, noise

def test_snr_power_ratio(sample_signal_and_noise):
    signal, noise = sample_signal_and_noise
    snr = snr_power_ratio(signal, noise)
    assert isinstance(snr, float)
    assert snr > 0

    # Test edge case: zero noise
    zero_noise = np.zeros_like(noise)
    with pytest.raises(ZeroDivisionError):
        snr_power_ratio(signal, zero_noise)

def test_snr_peak_to_peak(sample_signal_and_noise):
    signal, noise = sample_signal_and_noise
    snr = snr_peak_to_peak(signal, noise)
    assert isinstance(snr, float)
    assert snr > 0

    # Test edge case: zero noise
    zero_signal = np.zeros_like(signal)
    zero_noise = np.zeros_like(noise)
    with pytest.raises(ZeroDivisionError):
        snr_peak_to_peak(zero_signal, zero_noise)

def test_snr_mean_square(sample_signal_and_noise):
    signal, noise = sample_signal_and_noise
    snr = snr_mean_square(signal, noise)
    assert isinstance(snr, float)
    assert snr > 0

    # Test edge case: zero noise
    zero_noise = np.zeros_like(noise)
    with pytest.raises(ZeroDivisionError):
        snr_mean_square(signal, zero_noise)

def test_crest_factor(sample_signal_and_noise):
    signal, _ = sample_signal_and_noise
    cf = crest_factor(signal)
    assert isinstance(cf, float)
    assert cf > 0

    # Test edge case: zero signal
    zero_signal = np.zeros_like(signal)
    with pytest.raises(ZeroDivisionError):
        crest_factor(zero_signal)

def test_harmonic_distortion():
    signal = np.sin(2 * np.pi * 50 * np.arange(0, 1, 1/1000))
    thd = harmonic_distortion(signal, fundamental_freq=50, sampling_rate=1000)
    assert isinstance(thd, float)
    assert thd >= 0

    # Test edge case: zero signal
    zero_signal = np.zeros_like(signal)
    thd_zero = harmonic_distortion(zero_signal, fundamental_freq=50, sampling_rate=1000)
    assert thd_zero == 0

def test_signal_to_noise_and_distortion_ratio(sample_signal_and_noise):
    signal, noise = sample_signal_and_noise
    sinad = signal_to_noise_and_distortion_ratio(signal, noise)
    assert isinstance(sinad, float)
    assert sinad > 0

    # Test edge case: zero noise
    zero_noise = np.zeros_like(noise)
    with pytest.raises(ZeroDivisionError):
        signal_to_noise_and_distortion_ratio(signal, zero_noise)

def test_signal_to_noise_and_interference_ratio(sample_signal_and_noise):
    signal, interference = sample_signal_and_noise
    snir = signal_to_noise_and_interference_ratio(signal, interference)
    assert isinstance(snir, float)
    assert snir > 0

    # Test edge case: zero interference
    zero_interference = np.zeros_like(interference)
    with pytest.raises(ZeroDivisionError):
        signal_to_noise_and_interference_ratio(signal, zero_interference)