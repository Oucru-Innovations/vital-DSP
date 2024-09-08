import numpy as np
import pytest
from vitalDSP.physiological_features.signal_power_analysis import SignalPowerAnalysis

@pytest.fixture
def sample_signal():
    return np.array([1, 2, 3, 4, 5])

@pytest.fixture
def sample_noise():
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1])

def test_compute_rmse(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    rmse = spa.compute_rmse()
    assert np.isclose(rmse, np.sqrt(np.mean(sample_signal**2)))

def test_compute_mean_square(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    mean_square = spa.compute_mean_square()
    assert np.isclose(mean_square, np.mean(sample_signal**2))

def test_compute_total_power(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    total_power = spa.compute_total_power()
    assert np.isclose(total_power, np.sum(sample_signal**2) / len(sample_signal))

def test_compute_peak_power(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    peak_power = spa.compute_peak_power()
    assert peak_power == np.max(sample_signal**2)

def test_compute_snr(sample_signal, sample_noise):
    spa = SignalPowerAnalysis(sample_signal)
    snr = spa.compute_snr(sample_noise)
    signal_power = np.mean(sample_signal**2)
    noise_power = np.mean(sample_noise**2)
    expected_snr = 10 * np.log10(signal_power / noise_power)
    assert np.isclose(snr, expected_snr)

def test_compute_snr_zero_noise(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    noise_signal = np.zeros_like(sample_signal)
    
    # Expect infinity when noise power is zero
    snr = spa.compute_snr(noise_signal)
    assert np.isinf(snr)

def test_compute_psd(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    fs = 10.0
    nperseg = 5
    freqs, psd = spa.compute_psd(fs=fs, nperseg=nperseg)
    
    assert len(freqs) == len(psd)
    assert np.all(psd >= 0)

def test_compute_band_power(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    band = (1, 3)
    fs = 10.0
    nperseg = len(sample_signal)  # Set nperseg to the length of the signal to avoid ValueError

    # Now compute the band power
    band_power = spa.compute_band_power(band, fs=fs, nperseg=nperseg)
    
    # Compute expected band power using the PSD method manually
    freqs, psd = spa.compute_psd(fs=fs, nperseg=nperseg)
    expected_band_power = np.trapz(
        psd[(freqs >= band[0]) & (freqs <= band[1])],
        freqs[(freqs >= band[0]) & (freqs <= band[1])]
    )
    assert np.isclose(band_power, expected_band_power)

    
    # Compute expected band power using the PSD method manually
    expected_band_power = np.trapz(
        psd[(freqs >= band[0]) & (freqs <= band[1])],
        freqs[(freqs >= band[0]) & (freqs <= band[1])]
    )
    assert np.isclose(band_power, expected_band_power)

def test_compute_energy(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    energy = spa.compute_energy()
    expected_energy = np.sum(sample_signal**2)
    assert np.isclose(energy, expected_energy)

def test_empty_signal():
    empty_signal = np.array([])

    with pytest.raises(ValueError):
        SignalPowerAnalysis(empty_signal)

def test_psd_invalid_nperseg(sample_signal):
    spa = SignalPowerAnalysis(sample_signal)
    fs = 10.0
    nperseg = 10  # nperseg larger than signal length
    with pytest.raises(ValueError):
        spa.compute_psd(fs=fs, nperseg=nperseg)