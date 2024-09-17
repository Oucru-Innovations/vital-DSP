import pytest
import numpy as np
from vitalDSP.transforms.wavelet_transform import WaveletTransform
from vitalDSP.physiological_features.envelope_detection import EnvelopeDetection


# Mocking the WaveletTransform to avoid dependency during tests
class MockWaveletTransform:
    def __init__(self, signal, wavelet_name="db"):
        self.signal = signal
        self.wavelet_name = wavelet_name

    def perform_wavelet_transform(self, level=1):
        # Return a mocked wavelet decomposition (coeffs list)
        return [np.array([1, 2, 3]), np.array([4, 5, 6])]


@pytest.fixture
def sample_signal():
    # Simple test signal
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def envelope_detector(sample_signal, monkeypatch):
    # Replace the actual WaveletTransform with a mock to avoid unnecessary dependencies
    monkeypatch.setattr(
        WaveletTransform,
        "perform_wavelet_transform",
        MockWaveletTransform.perform_wavelet_transform,
    )
    return EnvelopeDetection(sample_signal)


def test_hilbert_envelope(envelope_detector):
    # Test hilbert_envelope method
    envelope = envelope_detector.hilbert_envelope()
    assert isinstance(envelope, np.ndarray)
    assert envelope.shape == envelope_detector.signal.shape


def test_moving_average_envelope(envelope_detector):
    # Test moving_average_envelope method
    window_size = 3
    envelope = envelope_detector.moving_average_envelope(window_size)
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == len(envelope_detector.signal) - window_size + 1


def test_absolute_value_envelope(envelope_detector):
    # Test absolute_value_envelope method
    smoothing_factor = 0.1
    envelope = envelope_detector.absolute_value_envelope(smoothing_factor)
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == len(envelope_detector.signal)


def test_peak_envelope(envelope_detector):
    # Use a signal that contains clear peaks
    signal_with_peaks = np.array([1, 3, 1, 4, 2, 1, 5, 1])
    envelope_detector.signal = signal_with_peaks
    envelope = envelope_detector.peak_envelope(interpolation_method="linear")
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == len(envelope_detector.signal)


def test_wavelet_envelope(envelope_detector):
    # Test wavelet_envelope method with a signal length divisible by the wavelet level
    signal_length = len(envelope_detector.signal)
    envelope = envelope_detector.wavelet_envelope(wavelet_name="db", level=1)

    # Assert that the length of the envelope matches the signal length after repeating
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == signal_length


def test_adaptive_filter_envelope(envelope_detector):
    # Test adaptive_filter_envelope method
    step_size = 0.01
    filter_order = 3
    envelope = envelope_detector.adaptive_filter_envelope(step_size, filter_order)
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == len(envelope_detector.signal)


def test_ml_based_envelope_default_model(envelope_detector):
    # Test ml_based_envelope with default model (None)
    envelope = envelope_detector.ml_based_envelope(model=None)
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == len(envelope_detector.signal)


def test_ml_based_envelope_custom_model(envelope_detector):
    # Test ml_based_envelope with custom model
    def mock_model(signal):
        return np.full_like(signal, 1)  # Custom model returns all ones

    envelope = envelope_detector.ml_based_envelope(model=mock_model)
    assert isinstance(envelope, np.ndarray)
    assert np.all(envelope == 1)
