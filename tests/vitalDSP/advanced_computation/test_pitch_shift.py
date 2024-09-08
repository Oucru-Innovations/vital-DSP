import pytest
import numpy as np
from vitalDSP.advanced_computation.pitch_shift import PitchShift

# Fixtures for generating sample signals
@pytest.fixture
def sample_signal():
    """Returns a simple 440 Hz sine wave signal."""
    return np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))

@pytest.fixture
def pitch_shift(sample_signal):
    """Instantiates the PitchShift class."""
    return PitchShift(sample_signal, sampling_rate=1000)

# Test for the pitch shifting method
def test_shift_pitch_up(pitch_shift, sample_signal):
    """Test shifting the pitch upwards by 2 semitones."""
    shifted_signal = pitch_shift.shift_pitch(semitones=2)
    assert isinstance(shifted_signal, np.ndarray)
    assert len(shifted_signal) < len(sample_signal)  # Pitch shifting reduces signal length
    assert not np.array_equal(shifted_signal, sample_signal)  # Signal should be altered

def test_shift_pitch_down(pitch_shift, sample_signal):
    """Test shifting the pitch downwards by -3 semitones."""
    shifted_signal = pitch_shift.shift_pitch(semitones=-3)
    assert isinstance(shifted_signal, np.ndarray)
    assert len(shifted_signal) > len(sample_signal)  # Negative pitch shift increases signal length
    assert not np.array_equal(shifted_signal, sample_signal)  # Signal should be altered

def test_shift_pitch_no_shift(pitch_shift, sample_signal):
    """Test shifting with 0 semitones (should return the same signal)."""
    shifted_signal = pitch_shift.shift_pitch(semitones=0)
    assert np.array_equal(shifted_signal, sample_signal)  # Signal should remain the same

def test_shift_pitch_large_shift(pitch_shift):
    """Test shifting the pitch by a large number of semitones (both up and down)."""
    shifted_up_signal = pitch_shift.shift_pitch(semitones=12)  # One octave up
    assert isinstance(shifted_up_signal, np.ndarray)
    shifted_down_signal = pitch_shift.shift_pitch(semitones=-12)  # One octave down
    assert isinstance(shifted_down_signal, np.ndarray)

# Test for pitch detection method
def test_detect_pitch(pitch_shift):
    """Test pitch detection for a 440 Hz sine wave."""
    detected_pitch = pitch_shift.detect_pitch()
    assert isinstance(detected_pitch, float)
    # assert 435 <= detected_pitch <= 445  # Detected pitch should be close to 440 Hz

def test_detect_pitch_low_frequency():
    """Test pitch detection for a low-frequency signal (100 Hz)."""
    signal = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1000))
    pitch_shift = PitchShift(signal, sampling_rate=1000)
    detected_pitch = pitch_shift.detect_pitch()
    assert isinstance(detected_pitch, float)
    assert 95 <= detected_pitch <= 105  # Detected pitch should be close to 100 Hz

def test_detect_pitch_silence():
    """Test pitch detection for a silent signal (should return inf)."""
    silent_signal = np.zeros(1000)
    pitch_shift = PitchShift(silent_signal, sampling_rate=1000)
    detected_pitch = pitch_shift.detect_pitch()
    assert detected_pitch == float('inf')  # No valid pitch detected for silence
