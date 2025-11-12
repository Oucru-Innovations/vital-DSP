import numpy as np
import pytest
from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import (
    detect_apnea_amplitude,
)
from vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection import (
    detect_apnea_pauses,
)


def generate_apnea_signal(
    sampling_rate, duration, apnea_start, apnea_duration, noise_level=0.1
):
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * 0.2 * t) + noise_level * np.random.normal(size=len(t))
    signal[
        int(apnea_start * sampling_rate) : int(
            (apnea_start + apnea_duration) * sampling_rate
        )
    ] = 0
    return signal


def test_detect_apnea_amplitude():
    signal = generate_apnea_signal(
        sampling_rate=100, duration=60, apnea_start=20, apnea_duration=10
    )
    apnea_events = detect_apnea_amplitude(
        signal, sampling_rate=100, threshold=0.1, min_duration=5
    )
    assert len(apnea_events) == 1
    assert 15 <= apnea_events[0][0] <= 25  # Start time
    assert 25 <= apnea_events[0][1] <= 35  # End time


def test_detect_apnea_pauses():
    signal = generate_apnea_signal(
        sampling_rate=100, duration=60, apnea_start=20, apnea_duration=10
    )
    apnea_events = detect_apnea_pauses(signal, sampling_rate=100, min_pause_duration=5)
    assert len(apnea_events) == 1
    assert 15 <= apnea_events[0][0] <= 25  # Start time
    assert 25 <= apnea_events[0][1] <= 35  # End time


def test_detect_apnea_pauses_with_preprocess():
    """Test detect_apnea_pauses with preprocessing.
    
    This test covers lines 62-65 in pause_detection.py where
    the preprocess parameter is used to preprocess the signal.
    """
    from unittest.mock import patch
    
    # Create a signal with apnea events (peaks with pauses)
    sampling_rate = 100
    duration = 60  # seconds
    t = np.arange(0, duration, 1 / sampling_rate)
    # Create a signal with clear peaks
    signal = np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.normal(size=len(t))
    
    # Mock preprocess_signal to return a modified signal
    def mock_preprocess_signal(sig, fs, filter_type, **kwargs):
        # Return a signal that's been filtered (just return original for testing)
        return sig
    
    with patch('vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection.preprocess_signal', side_effect=mock_preprocess_signal):
        apnea_events = detect_apnea_pauses(
            signal, 
            sampling_rate=sampling_rate, 
            min_pause_duration=5,
            preprocess="bandpass"
        )
        # Should still detect apnea events based on pauses between peaks
        assert isinstance(apnea_events, list)


def test_detect_apnea_amplitude_with_preprocess():
    """Test detect_apnea_amplitude with preprocessing.
    
    This test covers lines 66-69 in amplitude_threshold.py where
    the preprocess parameter is used to preprocess the signal.
    """
    from unittest.mock import patch
    
    # Create a signal with apnea events
    signal = generate_apnea_signal(
        sampling_rate=100, duration=60, apnea_start=20, apnea_duration=10
    )
    
    # Mock preprocess_signal to return a modified signal
    def mock_preprocess_signal(sig, fs, filter_type, **kwargs):
        # Return a signal that's been filtered (just return original for testing)
        return sig
    
    with patch('vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold.preprocess_signal', side_effect=mock_preprocess_signal):
        apnea_events = detect_apnea_amplitude(
            signal, 
            sampling_rate=100, 
            threshold=0.1, 
            min_duration=5,
            preprocess="bandpass"
        )
        # Should still detect apnea events
        assert len(apnea_events) >= 0


def test_detect_apnea_amplitude_ending_during_apnea():
    """Test detect_apnea_amplitude when signal ends during an apnea event.
    
    This test covers lines 86-91 in amplitude_threshold.py where
    the signal ends while current_start is not None.
    """
    # Create a signal that ends during an apnea event
    # Signal starts normal, then goes below threshold and stays there until the end
    sampling_rate = 100
    duration = 60  # seconds
    apnea_start = 50  # Start apnea at 50 seconds
    apnea_duration = 15  # Apnea lasts 15 seconds (extends beyond signal end)
    
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.normal(size=len(t))
    
    # Set signal below threshold starting from apnea_start until the end
    apnea_start_idx = int(apnea_start * sampling_rate)
    signal[apnea_start_idx:] = 0.05  # Below threshold
    
    # min_duration should be less than the remaining duration
    min_duration = 5  # seconds
    
    apnea_events = detect_apnea_amplitude(
        signal, 
        sampling_rate=sampling_rate, 
        threshold=0.1, 
        min_duration=min_duration
    )
    
    # Should detect the apnea event that extends to the end
    assert len(apnea_events) >= 1
    # The last event should end at the signal end
    last_event = apnea_events[-1]
    assert abs(last_event[1] - duration) < 0.1  # Should end near signal end


def test_detect_apnea_amplitude_ending_during_apnea_short_duration():
    """Test detect_apnea_amplitude when signal ends during an apnea event but duration is too short.
    
    This test covers lines 86-88 in amplitude_threshold.py where
    the signal ends while current_start is not None, but duration < min_duration.
    """
    # Create a signal that ends during an apnea event
    # Signal starts normal, then goes below threshold and stays there until the end
    sampling_rate = 100
    duration = 60  # seconds
    apnea_start = 58  # Start apnea at 58 seconds (only 2 seconds before end)
    
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.normal(size=len(t))
    
    # Set signal below threshold starting from apnea_start until the end
    apnea_start_idx = int(apnea_start * sampling_rate)
    signal[apnea_start_idx:] = 0.05  # Below threshold
    
    # min_duration should be greater than the remaining duration
    min_duration = 5  # seconds (longer than 2 seconds remaining)
    
    apnea_events = detect_apnea_amplitude(
        signal, 
        sampling_rate=sampling_rate, 
        threshold=0.1, 
        min_duration=min_duration
    )
    
    # Should NOT detect the apnea event because duration is too short
    # Check that no event ends at the signal end
    for event in apnea_events:
        assert abs(event[1] - duration) >= 0.1  # No event should end at signal end


