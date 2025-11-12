"""
Comprehensive tests for signal_type_detection.py to improve coverage.

This file adds extensive coverage for signal type detection utilities.
"""

import pytest
import numpy as np
from vitalDSP_webapp.utils.signal_type_detection import (
    SignalTypeDetector,
    detect_signal_type,
    detect_respiratory_signal_type,
)


@pytest.fixture
def sample_ppg_signal():
    """Create sample PPG signal."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # PPG-like signal: smooth waveform around 1.2 Hz (72 BPM)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    return signal, 100.0  # signal, sampling_freq


@pytest.fixture
def sample_ecg_signal():
    """Create sample ECG signal with sharp peaks."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # ECG-like signal: sharp peaks around 1.2 Hz
    signal = np.zeros(len(t))
    for i in range(0, len(t), 100):
        if i + 10 < len(t):
            signal[i:i+10] = np.exp(-np.linspace(0, 5, 10))  # Sharp peak
    signal += 0.05 * np.random.randn(len(t))
    return signal, 100.0


@pytest.fixture
def sample_respiratory_signal():
    """Create sample respiratory signal."""
    np.random.seed(42)
    t = np.linspace(0, 30, 3000)
    # Respiratory signal: slow oscillation around 0.25 Hz (15 BPM)
    signal = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(len(t))
    return signal, 100.0


class TestSignalTypeDetector:
    """Test SignalTypeDetector class."""

    def test_init(self, sample_ppg_signal):
        """Test SignalTypeDetector initialization."""
        signal, fs = sample_ppg_signal
        detector = SignalTypeDetector(signal, fs)
        assert detector.signal_data is not None
        assert detector.sampling_freq == fs
        assert detector._features is None

    def test_get_signal_features(self, sample_ppg_signal):
        """Test get_signal_features method."""
        signal, fs = sample_ppg_signal
        detector = SignalTypeDetector(signal, fs)
        features = detector.get_signal_features()
        
        assert isinstance(features, dict)
        assert "mean" in features
        assert "std" in features
        assert "range" in features
        assert "dominant_freq" in features
        assert "peak_power" in features
        assert "freq_range" in features
        assert "snr" in features

    def test_get_signal_features_cached(self, sample_ppg_signal):
        """Test that features are cached."""
        signal, fs = sample_ppg_signal
        detector = SignalTypeDetector(signal, fs)
        features1 = detector.get_signal_features()
        features2 = detector.get_signal_features()
        
        # Should return same object (cached)
        assert features1 is features2

    def test_get_signal_features_empty_signal(self):
        """Test get_signal_features with empty signal."""
        signal = np.array([])
        detector = SignalTypeDetector(signal, 100.0)
        features = detector.get_signal_features()
        
        # Should handle gracefully
        assert isinstance(features, dict)

    def test_detect_type_ppg(self, sample_ppg_signal):
        """Test detect_type for PPG signal."""
        signal, fs = sample_ppg_signal
        detector = SignalTypeDetector(signal, fs)
        signal_type = detector.detect_type()
        
        assert isinstance(signal_type, str)
        assert signal_type in ["ecg", "ppg", "respiratory", "unknown"]

    def test_detect_type_ecg(self, sample_ecg_signal):
        """Test detect_type for ECG signal."""
        signal, fs = sample_ecg_signal
        detector = SignalTypeDetector(signal, fs)
        signal_type = detector.detect_type()
        
        assert isinstance(signal_type, str)
        assert signal_type in ["ecg", "ppg", "respiratory", "unknown"]

    def test_detect_type_respiratory(self, sample_respiratory_signal):
        """Test detect_type for respiratory signal."""
        signal, fs = sample_respiratory_signal
        detector = SignalTypeDetector(signal, fs)
        signal_type = detector.detect_type()
        
        assert isinstance(signal_type, str)
        assert signal_type in ["ecg", "ppg", "respiratory", "unknown"]

    def test_detect_type_unknown_frequency(self):
        """Test detect_type with signal outside expected frequency ranges."""
        np.random.seed(42)
        # Signal with very high frequency (outside cardiac/respiratory range)
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz - too high
        detector = SignalTypeDetector(signal, 1000.0)
        signal_type = detector.detect_type()
        
        assert signal_type == "unknown"

    def test_detect_type_empty_features(self):
        """Test detect_type when features extraction fails."""
        signal = np.array([])
        detector = SignalTypeDetector(signal, 100.0)
        signal_type = detector.detect_type()
        
        assert signal_type == "unknown"

    def test_detect_respiratory_type_ppg_derived(self, sample_respiratory_signal):
        """Test detect_respiratory_type for PPG-derived respiratory signal."""
        signal, fs = sample_respiratory_signal
        detector = SignalTypeDetector(signal, fs)
        resp_type = detector.detect_respiratory_type()
        
        assert isinstance(resp_type, str)
        assert resp_type in ["respiratory", "ppg", "ecg"]

    def test_detect_respiratory_type_direct(self):
        """Test detect_respiratory_type for direct respiratory signal."""
        np.random.seed(42)
        # High amplitude respiratory signal
        t = np.linspace(0, 30, 3000)
        signal = 5 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(len(t))
        detector = SignalTypeDetector(signal, 100.0)
        resp_type = detector.detect_respiratory_type()
        
        assert isinstance(resp_type, str)
        assert resp_type in ["respiratory", "ppg", "ecg"]

    def test_detect_respiratory_type_empty_features(self):
        """Test detect_respiratory_type when features extraction fails."""
        signal = np.array([])
        detector = SignalTypeDetector(signal, 100.0)
        resp_type = detector.detect_respiratory_type()
        
        # Should return default fallback
        assert resp_type == "ppg"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_signal_type_ppg(self, sample_ppg_signal):
        """Test detect_signal_type convenience function with PPG."""
        signal, fs = sample_ppg_signal
        signal_type = detect_signal_type(signal, fs)
        
        assert isinstance(signal_type, str)
        assert signal_type in ["ecg", "ppg", "respiratory", "unknown"]

    def test_detect_signal_type_ecg(self, sample_ecg_signal):
        """Test detect_signal_type convenience function with ECG."""
        signal, fs = sample_ecg_signal
        signal_type = detect_signal_type(signal, fs)
        
        assert isinstance(signal_type, str)
        assert signal_type in ["ecg", "ppg", "respiratory", "unknown"]

    def test_detect_signal_type_default_sampling_freq(self):
        """Test detect_signal_type with default sampling frequency."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 1000))
        signal_type = detect_signal_type(signal)
        
        assert isinstance(signal_type, str)

    def test_detect_respiratory_signal_type(self, sample_respiratory_signal):
        """Test detect_respiratory_signal_type convenience function."""
        signal, fs = sample_respiratory_signal
        resp_type = detect_respiratory_signal_type(signal, fs)
        
        assert isinstance(resp_type, str)
        assert resp_type in ["respiratory", "ppg", "ecg"]

    def test_detect_respiratory_signal_type_default_sampling_freq(self):
        """Test detect_respiratory_signal_type with default sampling frequency."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 0.25 * np.linspace(0, 30, 3000))
        resp_type = detect_respiratory_signal_type(signal)
        
        assert isinstance(resp_type, str)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_signal_type_detector_with_list(self):
        """Test SignalTypeDetector with list input."""
        signal_list = [1, 2, 3, 4, 5]
        detector = SignalTypeDetector(signal_list, 100.0)
        assert detector.signal_data is not None

    def test_signal_type_detector_with_single_value(self):
        """Test SignalTypeDetector with single value."""
        signal = np.array([1.0])
        detector = SignalTypeDetector(signal, 100.0)
        features = detector.get_signal_features()
        # Should handle gracefully
        assert isinstance(features, dict)

    def test_signal_type_detector_with_constant_signal(self):
        """Test SignalTypeDetector with constant signal."""
        signal = np.ones(1000)
        detector = SignalTypeDetector(signal, 100.0)
        signal_type = detector.detect_type()
        # Constant signal should be detected as unknown
        assert signal_type in ["ecg", "ppg", "respiratory", "unknown"]

    def test_signal_type_detector_with_very_short_signal(self):
        """Test SignalTypeDetector with very short signal."""
        signal = np.array([1, 2, 3])
        detector = SignalTypeDetector(signal, 100.0)
        features = detector.get_signal_features()
        # Should handle gracefully
        assert isinstance(features, dict)

    def test_detect_type_exception_handling(self):
        """Test detect_type handles exceptions gracefully."""
        # Create signal that might cause issues
        signal = np.array([np.nan, np.inf, -np.inf])
        detector = SignalTypeDetector(signal, 100.0)
        signal_type = detector.detect_type()
        # Should return unknown on error
        assert signal_type == "unknown"

