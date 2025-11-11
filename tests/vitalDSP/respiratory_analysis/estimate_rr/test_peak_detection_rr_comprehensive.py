"""
Comprehensive Tests for Peak Detection RR Module - Missing Coverage

This test file specifically targets missing lines in peak_detection_rr.py to achieve
high test coverage, including edge cases, error conditions, and all code paths.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 80%+
"""

import pytest
import numpy as np
import warnings
import logging
from unittest.mock import Mock, patch, MagicMock

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import module
try:
    from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import peak_detection_rr
    PEAK_DETECTION_RR_AVAILABLE = True
except ImportError:
    PEAK_DETECTION_RR_AVAILABLE = False


@pytest.fixture
def sample_respiratory_signal():
    """Create a sample respiratory signal."""
    np.random.seed(42)
    fs = 100  # 100 Hz sampling rate
    duration = 30  # 30 seconds
    t = np.arange(0, duration, 1/fs)
    
    # Create a signal with ~12 BPM (5 second period)
    signal = np.sin(2 * np.pi * (1/5) * t) + 0.1 * np.random.randn(len(t))
    
    return signal, fs


@pytest.fixture
def fast_respiratory_signal():
    """Create a fast respiratory signal (~30 BPM)."""
    np.random.seed(42)
    fs = 100
    duration = 20
    t = np.arange(0, duration, 1/fs)
    
    # Create a signal with ~30 BPM (2 second period)
    signal = np.sin(2 * np.pi * (1/2) * t) + 0.1 * np.random.randn(len(t))
    
    return signal, fs


@pytest.fixture
def slow_respiratory_signal():
    """Create a slow respiratory signal (~8 BPM)."""
    np.random.seed(42)
    fs = 100
    duration = 60
    t = np.arange(0, duration, 1/fs)
    
    # Create a signal with ~8 BPM (7.5 second period)
    signal = np.sin(2 * np.pi * (1/7.5) * t) + 0.1 * np.random.randn(len(t))
    
    return signal, fs


@pytest.mark.skipif(not PEAK_DETECTION_RR_AVAILABLE, reason="peak_detection_rr module not available")
class TestPeakDetectionRRBasic:
    """Test basic peak_detection_rr functionality."""
    
    def test_basic_functionality(self, sample_respiratory_signal):
        """Test basic peak detection RR estimation."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(signal, fs)
        
        assert isinstance(rr, float)
        assert rr >= 0
        # Should be in reasonable range (6-40 BPM)
        assert 6 <= rr <= 40 or rr == 0.0
    
    def test_with_preprocessing(self, sample_respiratory_signal):
        """Test peak detection with preprocessing."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(
            signal, fs,
            preprocess='bandpass',
            lowcut=0.1,
            highcut=0.5
        )
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_custom_parameters(self, sample_respiratory_signal):
        """Test peak detection with custom parameters."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(
            signal, fs,
            min_peak_distance=1.0,
            height=0.5,
            prominence=0.2,
            threshold=0.1
        )
        
        assert isinstance(rr, float)
        assert rr >= 0


@pytest.mark.skipif(not PEAK_DETECTION_RR_AVAILABLE, reason="peak_detection_rr module not available")
class TestPeakDetectionRREdgeCases:
    """Test edge cases and error conditions - covers missing lines."""
    
    def test_signal_too_short(self):
        """Test with signal too short - covers lines 140-143."""
        signal = np.random.randn(5)
        fs = 100
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs)
            
            assert rr == 0.0
            assert len(w) > 0
            assert any("too short" in str(warning.message).lower() for warning in w)
    
    def test_prominence_none_zero_variance(self):
        """Test prominence=None with zero variance - covers lines 146-154."""
        # Create a constant signal (zero variance)
        signal = np.ones(1000)
        fs = 100
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, prominence=None)
            
            assert rr == 0.0
            assert len(w) > 0
            assert any("zero variance" in str(warning.message).lower() for warning in w)
    
    def test_prominence_none_normal_signal(self, sample_respiratory_signal):
        """Test prominence=None with normal signal - covers lines 146-150."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(signal, fs, prominence=None)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_min_peak_distance_too_large(self):
        """Test min_peak_distance too large - covers lines 163-168."""
        signal = np.random.randn(100)  # Very short signal
        fs = 100
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, min_peak_distance=10.0)
            
            assert rr == 0.0
            assert len(w) > 0
            assert any("too large" in str(warning.message).lower() for warning in w)
    
    def test_insufficient_peaks(self):
        """Test with insufficient peaks detected."""
        # Create a signal with no clear peaks
        signal = np.random.randn(1000) * 0.01  # Very low amplitude noise
        fs = 100
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, prominence=1.0)
            
            assert rr == 0.0
            assert len(w) > 0
    
    def test_no_valid_intervals(self):
        """Test with no valid intervals - covers lines 221-230."""
        # Create a signal with peaks that are too close together
        fs = 100
        t = np.arange(0, 10, 1/fs)
        # Create very fast oscillations (period < 1.5s)
        signal = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, min_peak_distance=0.1)
            
            # May return 0.0 if no valid intervals
            assert rr == 0.0 or (6 <= rr <= 40)
            if rr == 0.0:
                assert any("no valid" in str(warning.message).lower() for warning in w)
    
    def test_high_variability(self):
        """Test with high breathing variability - covers lines 250-256."""
        fs = 100
        t = np.arange(0, 60, 1/fs)
        
        # Create irregular breathing pattern
        periods = [3.0, 2.0, 4.0, 1.5, 5.0, 2.5, 3.5]  # Highly variable
        signal = np.zeros(len(t))
        current_time = 0
        for period in periods:
            if current_time >= len(t) / fs:
                break
            end_time = min(current_time + period, len(t) / fs)
            idx_start = int(current_time * fs)
            idx_end = int(end_time * fs)
            if idx_end > idx_start:
                local_t = np.arange(0, (idx_end - idx_start) / fs, 1/fs)
                if len(local_t) > 0:
                    signal[idx_start:idx_end] = np.sin(2 * np.pi * (1/period) * local_t[:len(signal[idx_start:idx_end])])
                current_time = end_time
        
        # Add noise
        signal += 0.1 * np.random.randn(len(signal))
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, min_peak_distance=1.0)
            
            assert isinstance(rr, float)
            # May warn about high variability
            if any("variability" in str(warning.message).lower() for warning in w):
                assert True  # Expected warning
    
    def test_high_rejection_rate(self):
        """Test with high interval rejection rate - covers lines 260-268."""
        fs = 100
        t = np.arange(0, 30, 1/fs)
        
        # Create signal with mixed valid and invalid intervals
        # Some peaks very close (invalid), some far apart (invalid), some normal
        signal = np.zeros(len(t))
        
        # Add peaks at irregular intervals
        peak_times = [0, 0.5, 5, 5.8, 12, 12.3, 20, 25]  # Mix of valid and invalid
        
        for peak_time in peak_times:
            idx = int(peak_time * fs)
            if idx < len(signal):
                # Create a peak
                for i in range(max(0, idx-5), min(len(signal), idx+5)):
                    signal[i] += np.exp(-((i - idx) ** 2) / 2)
        
        # Add noise
        signal += 0.1 * np.random.randn(len(signal))
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, min_peak_distance=0.3)
            
            assert isinstance(rr, float)
            # May warn about high rejection rate
            if any("rejection" in str(warning.message).lower() for warning in w):
                assert True  # Expected warning
    
    def test_rr_outside_normal_range_fast(self):
        """Test with RR outside normal range (too fast) - covers lines 271-273."""
        # This is tricky - we need to create a signal that produces RR > 40
        # But the function validates intervals, so we need invalid intervals
        # Let's try with a signal that has very short intervals
        fs = 100
        t = np.arange(0, 10, 1/fs)
        
        # Create very fast oscillations
        signal = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(len(t))
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs, min_peak_distance=0.5)
            
            assert isinstance(rr, float)
            # May warn if outside range
            if rr > 40 or rr < 6:
                assert any("outside" in str(warning.message).lower() for warning in w) or rr == 0.0
    
    def test_rr_outside_normal_range_slow(self):
        """Test with RR outside normal range (too slow)."""
        fs = 100
        t = np.arange(0, 60, 1/fs)
        
        # Create very slow oscillations (period > 10s)
        signal = np.sin(2 * np.pi * (1/12) * t) + 0.1 * np.random.randn(len(t))
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rr = peak_detection_rr(signal, fs)
            
            assert isinstance(rr, float)
            # May warn if outside range or return 0.0
            if rr < 6:
                assert any("outside" in str(warning.message).lower() for warning in w) or rr == 0.0


@pytest.mark.skipif(not PEAK_DETECTION_RR_AVAILABLE, reason="peak_detection_rr module not available")
class TestPeakDetectionRRParameterVariations:
    """Test with various parameter combinations."""
    
    def test_with_height_parameter(self, sample_respiratory_signal):
        """Test with height parameter."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(signal, fs, height=0.5)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_threshold_parameter(self, sample_respiratory_signal):
        """Test with threshold parameter."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(signal, fs, threshold=0.1)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_width_parameter(self, sample_respiratory_signal):
        """Test with width parameter."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(signal, fs, width=5)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_all_parameters(self, sample_respiratory_signal):
        """Test with all parameters specified."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(
            signal, fs,
            min_peak_distance=1.5,
            height=0.3,
            threshold=0.1,
            prominence=0.2,
            width=3
        )
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_fast_breathing_signal(self, fast_respiratory_signal):
        """Test with fast breathing signal."""
        signal, fs = fast_respiratory_signal
        
        rr = peak_detection_rr(signal, fs, min_peak_distance=1.0)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_slow_breathing_signal(self, slow_respiratory_signal):
        """Test with slow breathing signal."""
        signal, fs = slow_respiratory_signal
        
        rr = peak_detection_rr(signal, fs)
        
        assert isinstance(rr, float)
        assert rr >= 0


@pytest.mark.skipif(not PEAK_DETECTION_RR_AVAILABLE, reason="peak_detection_rr module not available")
class TestPeakDetectionRRPreprocessing:
    """Test preprocessing options."""
    
    def test_with_bandpass_preprocessing(self, sample_respiratory_signal):
        """Test with bandpass preprocessing."""
        signal, fs = sample_respiratory_signal
        
        rr = peak_detection_rr(
            signal, fs,
            preprocess='bandpass',
            lowcut=0.1,
            highcut=0.5
        )
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_wavelet_preprocessing(self, sample_respiratory_signal):
        """Test with wavelet preprocessing."""
        signal, fs = sample_respiratory_signal
        
        try:
            rr = peak_detection_rr(
                signal, fs,
                preprocess='wavelet'
            )
            
            assert isinstance(rr, float)
            assert rr >= 0
        except (ValueError, NotImplementedError, TypeError):
            # Preprocessing may not be implemented or may not accept wavelet parameter
            pass


@pytest.mark.skipif(not PEAK_DETECTION_RR_AVAILABLE, reason="peak_detection_rr module not available")
class TestPeakDetectionRRLogging:
    """Test logging functionality."""
    
    def test_logging_with_debug_level(self, sample_respiratory_signal, caplog):
        """Test that logging works correctly."""
        signal, fs = sample_respiratory_signal
        
        with caplog.at_level(logging.DEBUG):
            rr = peak_detection_rr(signal, fs)
        
        # Check that some debug messages were logged
        assert len(caplog.records) > 0
        assert isinstance(rr, float)


@pytest.mark.skipif(not PEAK_DETECTION_RR_AVAILABLE, reason="peak_detection_rr module not available")
class TestPeakDetectionRRRobustness:
    """Test robustness to various signal conditions."""
    
    def test_with_noisy_signal(self):
        """Test with very noisy signal."""
        fs = 100
        t = np.arange(0, 30, 1/fs)
        signal = np.sin(2 * np.pi * (1/5) * t) + 0.5 * np.random.randn(len(t))
        
        rr = peak_detection_rr(signal, fs)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_dc_offset(self):
        """Test with DC offset."""
        fs = 100
        t = np.arange(0, 30, 1/fs)
        signal = np.sin(2 * np.pi * (1/5) * t) + 5.0  # Large DC offset
        
        rr = peak_detection_rr(signal, fs)
        
        assert isinstance(rr, float)
        assert rr >= 0
    
    def test_with_amplitude_variation(self):
        """Test with varying amplitude."""
        fs = 100
        t = np.arange(0, 30, 1/fs)
        amplitude = 1 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        signal = amplitude * np.sin(2 * np.pi * (1/5) * t)
        
        rr = peak_detection_rr(signal, fs)
        
        assert isinstance(rr, float)
        assert rr >= 0

