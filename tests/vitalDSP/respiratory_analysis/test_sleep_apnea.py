import numpy as np
import pytest
from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import detect_apnea_amplitude
from vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection import detect_apnea_pauses

def generate_apnea_signal(sampling_rate, duration, apnea_start, apnea_duration, noise_level=0.1):
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.sin(2 * np.pi * 0.2 * t) + noise_level * np.random.normal(size=len(t))
    signal[int(apnea_start*sampling_rate):int((apnea_start + apnea_duration)*sampling_rate)] = 0
    return signal

def test_detect_apnea_amplitude():
    signal = generate_apnea_signal(sampling_rate=100, duration=60, apnea_start=20, apnea_duration=10)
    apnea_events = detect_apnea_amplitude(signal, sampling_rate=100, threshold=0.1, min_duration=5)
    assert len(apnea_events) == 1
    assert 15 <= apnea_events[0][0] <= 25  # Start time
    assert 25 <= apnea_events[0][1] <= 35  # End time

def test_detect_apnea_pauses():
    signal = generate_apnea_signal(sampling_rate=100, duration=60, apnea_start=20, apnea_duration=10)
    apnea_events = detect_apnea_pauses(signal, sampling_rate=100, min_pause_duration=5)
    assert len(apnea_events) == 1
    assert 15 <= apnea_events[0][0] <= 25  # Start time
    assert 25 <= apnea_events[0][1] <= 35  # End time

if __name__ == "__main__":
    pytest.main()
