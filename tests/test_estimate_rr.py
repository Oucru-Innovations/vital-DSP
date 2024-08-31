import numpy as np
import pytest
from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import peak_detection_rr
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import frequency_domain_rr

def generate_test_signal(frequency, sampling_rate, duration, noise_level=0.1):
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.sin(2 * np.pi * frequency * t) + noise_level * np.random.normal(size=len(t))
    return signal

def test_fft_based_rr():
    signal = generate_test_signal(frequency=0.25, sampling_rate=100, duration=60)
    rr = fft_based_rr(signal, sampling_rate=100)
    assert 12 <= rr <= 20  # Expected RR within a reasonable range

def test_peak_detection_rr():
    signal = generate_test_signal(frequency=0.5, sampling_rate=100, duration=60)
    rr = peak_detection_rr(signal, sampling_rate=100,
                           min_peak_distance=2.5, 
                        #    preprocess='bandpass', 
                        #    lowcut=0.1, highcut=4
                           )
    assert 12 <= rr <= 40

def test_time_domain_rr():
    signal = generate_test_signal(frequency=0.25, sampling_rate=100, duration=60)
    rr = time_domain_rr(signal, sampling_rate=100)
    assert 12 <= np.round(rr) <= 30

def test_frequency_domain_rr():
    signal = generate_test_signal(frequency=0.25, sampling_rate=100, duration=60)
    rr = frequency_domain_rr(signal, sampling_rate=100)
    assert 12 <= rr <= 40

if __name__ == "__main__":
    pytest.main()
