import pytest
import numpy as np
from vitalDSP.utils.synthesize_data import (
    generate_sinusoidal,
    generate_square_wave,
    generate_noisy_signal,
    generate_ecg_signal,
    generate_resp_signal,
    generate_synthetic_ppg,
    generate_synthetic_ppg_reversed,
)


# Test generate_sinusoidal
def test_generate_sinusoidal():
    frequency = 1.0
    sampling_rate = 100.0
    duration = 5.0
    signal = generate_sinusoidal(frequency, sampling_rate, duration)

    assert isinstance(signal, np.ndarray)
    assert len(signal) == int(sampling_rate * duration)
    assert np.isclose(np.max(signal), 1.0, atol=0.1)  # Max amplitude close to 1


# Test generate_square_wave
def test_generate_square_wave():
    frequency = 1.0
    sampling_rate = 100.0
    duration = 5.0
    signal = generate_square_wave(frequency, sampling_rate, duration)

    assert isinstance(signal, np.ndarray)
    assert len(signal) == int(sampling_rate * duration)
    assert np.all(
        np.unique(signal) == np.array([-1.0, 1.0])
    )  # Square wave with values -1 and 1


# Test generate_noisy_signal
def test_generate_noisy_signal():
    base_signal = generate_sinusoidal(1.0, 100.0, 5.0)
    noise_level = 0.2
    noisy_signal = generate_noisy_signal(base_signal, noise_level)

    assert isinstance(noisy_signal, np.ndarray)
    assert len(noisy_signal) == len(base_signal)
    assert np.any(np.abs(noisy_signal - base_signal) > 0.0)  # Noise added


# Test generate_ecg_signal
def test_generate_ecg_signal():
    sfecg = 256
    N = 256
    Anoise = 0.01
    hrmean = 70
    ecg_signal = generate_ecg_signal(
        sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
    )

    assert isinstance(ecg_signal, np.ndarray)
    assert len(ecg_signal) > 0

# Test generate_resp_signal
def test_generate_resp_signal():
    sampling_rate = 1000.0
    duration = 10.0
    resp_signal = generate_resp_signal(sampling_rate, duration)

    assert isinstance(resp_signal, np.ndarray)
    assert len(resp_signal) == int(sampling_rate * duration)


# Test generate_synthetic_ppg
def test_generate_synthetic_ppg():
    duration = 10.0
    sampling_rate = 1000
    heart_rate = 60
    ppg_signal = generate_synthetic_ppg(
        duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate
    )

    assert isinstance(ppg_signal, tuple)
    assert len(ppg_signal[1]) == int(sampling_rate * duration)


# Test generate_synthetic_ppg_reversed
def test_generate_synthetic_ppg_reversed():
    duration = 10.0
    sampling_rate = 1000
    heart_rate = 60
    ppg_signal = generate_synthetic_ppg_reversed(
        duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate
    )

    assert isinstance(ppg_signal, tuple)
    assert len(ppg_signal[1]) == int(sampling_rate * duration)
