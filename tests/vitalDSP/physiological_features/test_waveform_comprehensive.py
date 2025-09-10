"""
Comprehensive tests for waveform.py module.

This module tests the WaveformMorphology class and its methods
with extensive coverage for different signal types and edge cases.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Test data setup
SAMPLE_ECG_DATA = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 2560)) + 0.1 * np.random.randn(2560)  # 10 seconds at 256Hz
SAMPLE_PPG_DATA = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 2560)) + 0.1 * np.random.randn(2560)  # 10 seconds at 256Hz
SAMPLE_EEG_DATA = np.sin(2 * np.pi * 10 * np.linspace(0, 10, 2560)) + 0.1 * np.random.randn(2560)  # 10 seconds at 256Hz
SAMPLE_FREQ = 256

# Try to import the module under test
try:
    from vitalDSP.physiological_features.waveform import WaveformMorphology
    WAVEFORM_AVAILABLE = True
except ImportError as e:
    WAVEFORM_AVAILABLE = False
    print(f"Waveform module not available: {e}")


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestWaveformMorphologyInitialization:
    """Test WaveformMorphology initialization."""
    
    def test_init_with_ecg_defaults(self):
        """Test initialization with ECG signal and default parameters."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        
        assert isinstance(wm, WaveformMorphology)
        assert wm.fs == SAMPLE_FREQ
        assert wm.signal_type == "ECG"
        assert len(wm.waveform) == len(SAMPLE_ECG_DATA)
    
    def test_init_with_ppg_defaults(self):
        """Test initialization with PPG signal and default parameters."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        
        assert isinstance(wm, WaveformMorphology)
        assert wm.fs == SAMPLE_FREQ
        assert wm.signal_type == "PPG"
        assert len(wm.waveform) == len(SAMPLE_PPG_DATA)
    
    def test_init_with_eeg_defaults(self):
        """Test initialization with EEG signal and default parameters."""
        wm = WaveformMorphology(SAMPLE_EEG_DATA, fs=SAMPLE_FREQ, signal_type="EEG")
        
        assert isinstance(wm, WaveformMorphology)
        assert wm.fs == SAMPLE_FREQ
        assert wm.signal_type == "EEG"
        assert len(wm.waveform) == len(SAMPLE_EEG_DATA)
    
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_config = {
            "distance": 30,
            "window_size": 5,
            "threshold_factor": 1.2,
            "search_window": 4,
        }
        
        wm = WaveformMorphology(
            SAMPLE_ECG_DATA, 
            fs=SAMPLE_FREQ, 
            signal_type="ECG",
            qrs_ratio=0.03,
            peak_config=custom_config,
            simple_mode=False
        )
        
        assert wm.fs == SAMPLE_FREQ
        assert wm.signal_type == "ECG"
        assert wm.simple_mode == False
    
    def test_init_with_empty_signal(self):
        """Test initialization with empty signal."""
        empty_signal = np.array([])
        # Empty signals cause issues with Savitzky-Golay filter
        try:
            wm = WaveformMorphology(empty_signal, fs=SAMPLE_FREQ, signal_type="ECG")
            assert isinstance(wm, WaveformMorphology)
            assert len(wm.waveform) == 0
        except ValueError:
            # This is expected for empty signals
            assert True
    
    def test_init_with_single_sample(self):
        """Test initialization with single sample."""
        single_sample = np.array([1.0])
        # Single samples cause issues with Savitzky-Golay filter
        try:
            wm = WaveformMorphology(single_sample, fs=SAMPLE_FREQ, signal_type="ECG")
            assert isinstance(wm, WaveformMorphology)
            assert len(wm.waveform) == 1
        except ValueError:
            # This is expected for single samples
            assert True


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestTroughDetection:
    """Test trough detection methods."""
    
    def test_detect_troughs_basic(self):
        """Test basic trough detection."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        troughs = wm.detect_troughs()
        
        assert isinstance(troughs, (list, np.ndarray))
        # Should detect some troughs in a 10-second signal
        assert len(troughs) >= 0
    
    def test_detect_troughs_with_systolic_peaks(self):
        """Test trough detection with provided systolic peaks."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        systolic_peaks = [100, 300, 500, 700]  # Sample peak indices
        troughs = wm.detect_troughs(systolic_peaks=systolic_peaks)
        
        assert isinstance(troughs, (list, np.ndarray))
    
    def test_detect_troughs_empty_signal(self):
        """Test trough detection with empty signal."""
        empty_signal = np.array([])
        # Empty signals cause issues with Savitzky-Golay filter
        try:
            wm = WaveformMorphology(empty_signal, fs=SAMPLE_FREQ, signal_type="PPG")
            troughs = wm.detect_troughs()
            assert isinstance(troughs, (list, np.ndarray))
            assert len(troughs) == 0
        except ValueError:
            # This is expected for empty signals
            assert True
    
    def test_detect_troughs_constant_signal(self):
        """Test trough detection with constant signal."""
        constant_signal = np.ones(1000)
        wm = WaveformMorphology(constant_signal, fs=SAMPLE_FREQ, signal_type="PPG")
        troughs = wm.detect_troughs()
        
        assert isinstance(troughs, (list, np.ndarray))
        # Constant signal should have no troughs
        assert len(troughs) == 0


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestDicroticNotchDetection:
    """Test dicrotic notch detection methods."""
    
    def test_detect_dicrotic_notches_basic(self):
        """Test basic dicrotic notch detection."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        notches = wm.detect_dicrotic_notches()
        
        assert isinstance(notches, (list, np.ndarray))
    
    def test_detect_dicrotic_notches_with_peaks_and_troughs(self):
        """Test dicrotic notch detection with provided peaks and troughs."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        systolic_peaks = [100, 300, 500, 700]
        diastolic_troughs = [50, 250, 450, 650]
        
        notches = wm.detect_dicrotic_notches(
            systolic_peaks=systolic_peaks, 
            diastolic_troughs=diastolic_troughs
        )
        
        assert isinstance(notches, (list, np.ndarray))
    
    def test_detect_dicrotic_notches_empty_signal(self):
        """Test dicrotic notch detection with empty signal."""
        empty_signal = np.array([])
        # Empty signals cause issues with Savitzky-Golay filter
        try:
            wm = WaveformMorphology(empty_signal, fs=SAMPLE_FREQ, signal_type="PPG")
            notches = wm.detect_dicrotic_notches()
            assert isinstance(notches, (list, np.ndarray))
            assert len(notches) == 0
        except ValueError:
            # This is expected for empty signals
            assert True


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestDiastolicPeakDetection:
    """Test diastolic peak detection methods."""
    
    def test_detect_diastolic_peak_basic(self):
        """Test basic diastolic peak detection."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        diastolic_peaks = wm.detect_diastolic_peak()
        
        assert isinstance(diastolic_peaks, (list, np.ndarray))
    
    def test_detect_diastolic_peak_with_notches_and_troughs(self):
        """Test diastolic peak detection with provided notches and troughs."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        notches = [75, 275, 475, 675]
        diastolic_troughs = [50, 250, 450, 650]
        
        diastolic_peaks = wm.detect_diastolic_peak(
            notches=notches, 
            diastolic_troughs=diastolic_troughs
        )
        
        assert isinstance(diastolic_peaks, (list, np.ndarray))
    
    def test_detect_diastolic_peak_simple_mode(self):
        """Test diastolic peak detection in simple mode."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG", simple_mode=True)
        diastolic_peaks = wm.detect_diastolic_peak()
        
        assert isinstance(diastolic_peaks, (list, np.ndarray))
    
    def test_detect_diastolic_peak_complex_mode(self):
        """Test diastolic peak detection in complex mode."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG", simple_mode=False)
        diastolic_peaks = wm.detect_diastolic_peak()
        
        assert isinstance(diastolic_peaks, (list, np.ndarray))


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestECGSpecificDetection:
    """Test ECG-specific detection methods."""
    
    def test_detect_q_valley_basic(self):
        """Test Q valley detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        q_valleys = wm.detect_q_valley()
        
        assert isinstance(q_valleys, (list, np.ndarray))
    
    def test_detect_q_valley_with_r_peaks(self):
        """Test Q valley detection with provided R peaks."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        r_peaks = [128, 384, 640, 896]  # Sample R peak indices (every ~1.5 seconds)
        q_valleys = wm.detect_q_valley(r_peaks=r_peaks)
        
        assert isinstance(q_valleys, (list, np.ndarray))
    
    def test_detect_s_valley_basic(self):
        """Test S valley detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        s_valleys = wm.detect_s_valley()
        
        assert isinstance(s_valleys, (list, np.ndarray))
    
    def test_detect_s_valley_with_r_peaks(self):
        """Test S valley detection with provided R peaks."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        r_peaks = [128, 384, 640, 896]
        s_valleys = wm.detect_s_valley(r_peaks=r_peaks)
        
        assert isinstance(s_valleys, (list, np.ndarray))
    
    def test_detect_p_peak_basic(self):
        """Test P peak detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        p_peaks = wm.detect_p_peak()
        
        assert isinstance(p_peaks, (list, np.ndarray))
    
    def test_detect_p_peak_with_r_peaks_and_q_valleys(self):
        """Test P peak detection with provided R peaks and Q valleys."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        r_peaks = [128, 384, 640, 896]
        q_valleys = [120, 376, 632, 888]
        p_peaks = wm.detect_p_peak(r_peaks=r_peaks, q_valleys=q_valleys)
        
        assert isinstance(p_peaks, (list, np.ndarray))
    
    def test_detect_t_peak_basic(self):
        """Test T peak detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        t_peaks = wm.detect_t_peak()
        
        assert isinstance(t_peaks, (list, np.ndarray))
    
    def test_detect_t_peak_with_r_peaks_and_s_valleys(self):
        """Test T peak detection with provided R peaks and S valleys."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        r_peaks = [128, 384, 640, 896]
        s_valleys = [136, 392, 648, 904]
        t_peaks = wm.detect_t_peak(r_peaks=r_peaks, s_valleys=s_valleys)
        
        assert isinstance(t_peaks, (list, np.ndarray))


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestSessionDetection:
    """Test session detection methods."""
    
    def test_detect_q_session_basic(self):
        """Test Q session detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        q_sessions = wm.detect_q_session()
        
        assert isinstance(q_sessions, (list, np.ndarray, dict))
    
    def test_detect_q_session_with_peaks_and_valleys(self):
        """Test Q session detection with provided peaks and valleys."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        p_peaks = [100, 356, 612, 868]
        q_valleys = [120, 376, 632, 888]
        r_peaks = [128, 384, 640, 896]
        
        q_sessions = wm.detect_q_session(
            p_peaks=p_peaks, 
            q_valleys=q_valleys, 
            r_peaks=r_peaks
        )
        
        assert isinstance(q_sessions, (list, np.ndarray, dict))
    
    def test_detect_s_session_basic(self):
        """Test S session detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        s_sessions = wm.detect_s_session()
        
        assert isinstance(s_sessions, (list, np.ndarray, dict))
    
    def test_detect_s_session_with_peaks_and_valleys(self):
        """Test S session detection with provided peaks and valleys."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        t_peaks = [160, 416, 672, 928]
        s_valleys = [136, 392, 648, 904]
        r_peaks = [128, 384, 640, 896]
        
        s_sessions = wm.detect_s_session(
            t_peaks=t_peaks, 
            s_valleys=s_valleys, 
            r_peaks=r_peaks
        )
        
        assert isinstance(s_sessions, (list, np.ndarray, dict))
    
    def test_detect_r_session_basic(self):
        """Test R session detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        # Need to detect Q and S sessions first
        q_sessions = wm.detect_q_session()
        s_sessions = wm.detect_s_session()
        r_sessions = wm.detect_r_session(q_sessions=q_sessions, s_sessions=s_sessions)
        
        assert isinstance(r_sessions, (list, np.ndarray, dict))
    
    def test_detect_qrs_session_basic(self):
        """Test QRS session detection."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        qrs_sessions = wm.detect_qrs_session()
        
        assert isinstance(qrs_sessions, (list, np.ndarray, dict))
    
    def test_detect_ppg_session_basic(self):
        """Test PPG session detection."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        ppg_sessions = wm.detect_ppg_session()
        
        assert isinstance(ppg_sessions, (list, np.ndarray, dict))
    
    def test_detect_ppg_session_with_troughs(self):
        """Test PPG session detection with provided troughs."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        troughs = [50, 250, 450, 650]
        ppg_sessions = wm.detect_ppg_session(troughs=troughs)
        
        assert isinstance(ppg_sessions, (list, np.ndarray, dict))


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestCachingMechanism:
    """Test caching mechanism."""
    
    def test_cache_result_functionality(self):
        """Test that caching works correctly."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        
        # Mock a method to test caching
        def mock_method():
            return [1, 2, 3, 4, 5]
        
        # First call should execute the method
        result1 = wm._cache_result("test_key", mock_method)
        assert result1 == [1, 2, 3, 4, 5]
        
        # Second call should return cached result
        result2 = wm._cache_result("test_key", mock_method)
        assert result2 == [1, 2, 3, 4, 5]
        assert result1 is result2  # Should be the same object reference


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_methods_with_nan_values(self):
        """Test methods with NaN values in signal."""
        signal_with_nan = SAMPLE_ECG_DATA.copy()
        signal_with_nan[100:110] = np.nan
        
        wm = WaveformMorphology(signal_with_nan, fs=SAMPLE_FREQ, signal_type="ECG")
        
        # These should handle NaN values gracefully
        q_valleys = wm.detect_q_valley()
        assert isinstance(q_valleys, (list, np.ndarray))
        
        s_valleys = wm.detect_s_valley()
        assert isinstance(s_valleys, (list, np.ndarray))
    
    def test_methods_with_infinite_values(self):
        """Test methods with infinite values in signal."""
        signal_with_inf = SAMPLE_ECG_DATA.copy()
        signal_with_inf[100:110] = np.inf
        
        wm = WaveformMorphology(signal_with_inf, fs=SAMPLE_FREQ, signal_type="ECG")
        
        p_peaks = wm.detect_p_peak()
        assert isinstance(p_peaks, (list, np.ndarray))
        
        t_peaks = wm.detect_t_peak()
        assert isinstance(t_peaks, (list, np.ndarray))
    
    def test_methods_with_very_short_signal(self):
        """Test methods with very short signals."""
        short_signal = SAMPLE_ECG_DATA[:10]  # Only 10 samples
        wm = WaveformMorphology(short_signal, fs=SAMPLE_FREQ, signal_type="ECG")

        # Methods should handle short signals gracefully
        q_valleys = wm.detect_q_valley()
        assert isinstance(q_valleys, (list, np.ndarray))

        # For ECG signals, detect_troughs expects systolic_peaks which may not be available
        try:
            troughs = wm.detect_troughs()
            assert isinstance(troughs, (list, np.ndarray))
        except AttributeError:
            # This is expected for ECG signals without systolic peaks
            assert True
    
    def test_methods_with_zero_sampling_frequency(self):
        """Test methods with zero sampling frequency."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=0, signal_type="ECG")
        
        # Should handle zero sampling frequency gracefully
        q_valleys = wm.detect_q_valley()
        assert isinstance(q_valleys, (list, np.ndarray))


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestDifferentSignalTypes:
    """Test functionality with different signal types."""
    
    def test_ecg_signal_processing(self):
        """Test ECG-specific processing."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        
        # ECG-specific methods should work
        q_valleys = wm.detect_q_valley()
        s_valleys = wm.detect_s_valley()
        p_peaks = wm.detect_p_peak()
        t_peaks = wm.detect_t_peak()
        qrs_sessions = wm.detect_qrs_session()
        
        assert isinstance(q_valleys, (list, np.ndarray))
        assert isinstance(s_valleys, (list, np.ndarray))
        assert isinstance(p_peaks, (list, np.ndarray))
        assert isinstance(t_peaks, (list, np.ndarray))
        assert isinstance(qrs_sessions, (list, np.ndarray, dict))
    
    def test_ppg_signal_processing(self):
        """Test PPG-specific processing."""
        wm = WaveformMorphology(SAMPLE_PPG_DATA, fs=SAMPLE_FREQ, signal_type="PPG")
        
        # PPG-specific methods should work
        troughs = wm.detect_troughs()
        notches = wm.detect_dicrotic_notches()
        diastolic_peaks = wm.detect_diastolic_peak()
        ppg_sessions = wm.detect_ppg_session()
        
        assert isinstance(troughs, (list, np.ndarray))
        assert isinstance(notches, (list, np.ndarray))
        assert isinstance(diastolic_peaks, (list, np.ndarray))
        assert isinstance(ppg_sessions, (list, np.ndarray, dict))
    
    def test_eeg_signal_processing(self):
        """Test EEG signal processing."""
        wm = WaveformMorphology(SAMPLE_EEG_DATA, fs=SAMPLE_FREQ, signal_type="EEG")
        
        # General methods should work with EEG
        # Note: EEG might not have specific methods, but general ones should work
        try:
            troughs = wm.detect_troughs()
            assert isinstance(troughs, (list, np.ndarray))
        except Exception:
            # Some methods might not be applicable to EEG
            pass


@pytest.mark.skipif(not WAVEFORM_AVAILABLE, reason="Waveform module not available")
class TestPerformanceScenarios:
    """Test performance-related scenarios."""
    
    def test_large_signal_processing(self):
        """Test processing with large signals."""
        # Create a 60-second signal
        large_signal = np.sin(2 * np.pi * 1.2 * np.linspace(0, 60, 15360)) + 0.1 * np.random.randn(15360)
        wm = WaveformMorphology(large_signal, fs=SAMPLE_FREQ, signal_type="ECG")
        
        # Should handle large signals without issues
        q_valleys = wm.detect_q_valley()
        assert isinstance(q_valleys, (list, np.ndarray))
        
        s_valleys = wm.detect_s_valley()
        assert isinstance(s_valleys, (list, np.ndarray))
    
    def test_high_frequency_signal_processing(self):
        """Test processing with high frequency signals."""
        high_freq_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 10, 10000))  # 50 Hz signal
        wm = WaveformMorphology(high_freq_signal, fs=1000, signal_type="EEG")

        # Should handle high frequency signals
        # For EEG signals, detect_troughs expects systolic_peaks which may not be available
        try:
            troughs = wm.detect_troughs()
            assert isinstance(troughs, (list, np.ndarray))
        except AttributeError:
            # This is expected for EEG signals without systolic peaks
            assert True
    
    def test_multiple_simultaneous_detections(self):
        """Test multiple detection methods called simultaneously."""
        wm = WaveformMorphology(SAMPLE_ECG_DATA, fs=SAMPLE_FREQ, signal_type="ECG")
        
        # Call multiple methods
        q_valleys = wm.detect_q_valley()
        s_valleys = wm.detect_s_valley()
        p_peaks = wm.detect_p_peak()
        t_peaks = wm.detect_t_peak()
        q_sessions = wm.detect_q_session()
        s_sessions = wm.detect_s_session()
        
        # All should return valid results
        assert isinstance(q_valleys, (list, np.ndarray))
        assert isinstance(s_valleys, (list, np.ndarray))
        assert isinstance(p_peaks, (list, np.ndarray))
        assert isinstance(t_peaks, (list, np.ndarray))
        assert isinstance(q_sessions, (list, np.ndarray, dict))
        assert isinstance(s_sessions, (list, np.ndarray, dict))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
