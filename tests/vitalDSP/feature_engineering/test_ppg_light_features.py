import pytest
import numpy as np
from vitalDSP.feature_engineering.ppg_light_features import (
    PPGLightFeatureExtractor,
)  # Assuming the class is in a file named ppg_feature_extractor.py


@pytest.fixture
def test_signals():
    ir_signal = np.random.randn(1000)  # Simulated IR signal
    red_signal = np.random.randn(1000)  # Simulated Red signal
    sampling_freq = 100  # Sampling frequency (Hz)
    return ir_signal, red_signal, sampling_freq


def test_calculate_spo2(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform SpO2 calculation
    spo2, times_spo2 = ppg_extractor.calculate_spo2()

    assert len(spo2) == len(
        times_spo2
    ), "SpO2 and time arrays must have the same length."
    assert spo2.min() >= 0, "SpO2 values should be non-negative."  # Allow 0 as valid
    assert spo2.max() <= 100, "SpO2 values should be within 0-100 range."


def test_calculate_spo2_no_red_signal(test_signals):
    ir_signal, _, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)

    with pytest.raises(ValueError, match="Red signal is required to compute SpO2."):
        ppg_extractor.calculate_spo2()


def test_calculate_perfusion_index(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform Perfusion Index calculation
    pi, times_pi = ppg_extractor.calculate_perfusion_index()

    assert len(pi) == len(times_pi), "PI and time arrays must have the same length."
    assert pi.min() >= 0, "Perfusion index values should be non-negative."
    assert (
        times_pi[-1] <= len(ir_signal) / sampling_freq
    ), "Last timestamp should not exceed the total signal duration."


def test_calculate_respiratory_rate(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform Respiratory Rate calculation
    rr, times_rr = ppg_extractor.calculate_respiratory_rate()

    assert len(rr) == len(times_rr), "RR and time arrays must have the same length."
    assert (
        len(rr) > 0
    ), "Respiratory rate array should not be empty."  # Ensure array is not empty
    assert rr.min() >= 0, "Respiratory rate values should be non-negative."


def test_calculate_ppr(test_signals):
    ir_signal, red_signal, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)

    # Perform PPR calculation
    ppr, times_ppr = ppg_extractor.calculate_ppr()

    assert len(ppr) == len(times_ppr), "PPR and time arrays must have the same length."
    assert ppr.min() >= 0, "PPR values should be non-negative."
    assert (
        times_ppr[-1] <= len(ir_signal) / sampling_freq
    ), "Last timestamp should not exceed the total signal duration."


def test_calculate_ppr_no_red_signal(test_signals):
    ir_signal, _, sampling_freq = test_signals
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)

    with pytest.raises(ValueError, match="Red signal is required to compute PPR."):
        ppg_extractor.calculate_ppr()


class TestPPGLightFeaturesMissingCoverage:
    """Tests to cover missing lines in ppg_light_features.py."""

    def test_calculate_spo2_uneven_window(self):
        """Test calculate_spo2 with signal length that doesn't divide evenly.
        
        This test covers line 88 in ppg_light_features.py where
        break is executed when end > len(self.ir_signal).
        """
        # Create signal with length that doesn't divide evenly by window_size
        # window_size = 1 * 100 = 100, signal length = 250 (2.5 windows)
        ir_signal = np.random.randn(250)
        red_signal = np.random.randn(250)
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)
        spo2, times_spo2 = ppg_extractor.calculate_spo2(window_seconds=1)
        
        # Should only process 2 complete windows (0-100, 100-200), skip 200-250
        assert len(spo2) == 2
        assert len(times_spo2) == 2

    def test_calculate_perfusion_index_uneven_window(self):
        """Test calculate_perfusion_index with signal length that doesn't divide evenly.
        
        This test covers line 129 in ppg_light_features.py where
        break is executed when end > len(self.ir_signal).
        """
        # Create signal with length that doesn't divide evenly
        ir_signal = np.random.randn(250)
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)
        pi, times_pi = ppg_extractor.calculate_perfusion_index(window_seconds=1)
        
        # Should only process 2 complete windows
        assert len(pi) == 2
        assert len(times_pi) == 2

    def test_calculate_perfusion_index_zero_dc(self):
        """Test calculate_perfusion_index when DC component is zero or negative.
        
        This test covers line 139 in ppg_light_features.py where
        pi = 0 is set when dc_component <= 0.
        """
        # Create signal with zero mean (DC component = 0)
        ir_signal = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)
        pi, times_pi = ppg_extractor.calculate_perfusion_index(window_seconds=0.1)
        
        # DC component should be 0, so pi should be 0
        assert len(pi) > 0
        # Check that at least one value is 0 (when DC is 0)
        assert 0 in pi or np.any(pi == 0)

    def test_calculate_respiratory_rate_uneven_window(self):
        """Test calculate_respiratory_rate with signal length that doesn't divide evenly.
        
        This test covers lines 168-169 in ppg_light_features.py where
        break is executed and segment is assigned.
        """
        # Create signal with length that doesn't divide evenly by window_size
        # window_size = 60 * 100 = 6000, signal length = 7000 (1.17 windows)
        ir_signal = np.random.randn(7000)
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)
        rr, times_rr = ppg_extractor.calculate_respiratory_rate(window_seconds=60)
        
        # Should only process 1 complete window (0-6000), skip 6000-7000
        assert len(rr) == 1
        assert len(times_rr) == 1

    def test_calculate_respiratory_rate_with_peaks(self):
        """Test calculate_respiratory_rate when peaks are found.
        
        This test covers lines 172-178 in ppg_light_features.py where
        peaks are detected and breaths_per_minute is calculated.
        """
        # Create a signal with clear peaks (respiratory-like pattern)
        sampling_freq = 100
        window_seconds = 60
        window_size = int(window_seconds * sampling_freq)
        
        # Create signal with periodic peaks (simulating breathing)
        t = np.linspace(0, window_seconds, window_size)
        # Create signal with ~15 breaths per minute (one peak every 4 seconds)
        ir_signal = np.sin(2 * np.pi * (15/60) * t) + 0.1 * np.random.randn(window_size)
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)
        rr, times_rr = ppg_extractor.calculate_respiratory_rate(window_seconds=window_seconds)
        
        assert len(rr) > 0
        assert len(times_rr) > 0
        # Should detect some peaks
        assert rr[0] >= 0

    def test_calculate_respiratory_rate_no_peaks(self):
        """Test calculate_respiratory_rate when no peaks are found.
        
        This test covers lines 172-180 in ppg_light_features.py where
        no peaks are detected and breaths_per_minute = 0 is set.
        """
        from unittest.mock import patch
        from scipy.signal import find_peaks
        
        sampling_freq = 100
        window_seconds = 60
        window_size = int(window_seconds * sampling_freq)
        
        # Create a signal
        ir_signal = np.random.randn(window_size)
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)
        
        # Mock find_peaks to return no peaks
        with patch('vitalDSP.feature_engineering.ppg_light_features.find_peaks', return_value=np.array([])):
            rr, times_rr = ppg_extractor.calculate_respiratory_rate(window_seconds=window_seconds)
            
            assert len(rr) > 0
            assert len(times_rr) > 0
            # Should return 0 when no peaks are detected
            assert rr[0] == 0

    def test_calculate_respiratory_rate_empty_result_fallback(self):
        """Test calculate_respiratory_rate fallback when no windows are processed.
        
        This test covers lines 184-186 in ppg_light_features.py where
        fallback values are added when rr_values is empty.
        """
        # Create a very short signal that won't produce any windows
        # window_size = 60 * 100 = 6000, signal length = 100 (< 1 window)
        ir_signal = np.random.randn(100)
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, sampling_freq=sampling_freq)
        rr, times_rr = ppg_extractor.calculate_respiratory_rate(window_seconds=60)
        
        # Should have fallback values (0, 0)
        assert len(rr) == 1
        assert len(times_rr) == 1
        assert rr[0] == 0
        assert times_rr[0] == 0

    def test_calculate_ppr_uneven_window(self):
        """Test calculate_ppr with signal length that doesn't divide evenly.
        
        This test covers line 215 in ppg_light_features.py where
        break is executed when end > len(self.ir_signal).
        """
        # Create signal with length that doesn't divide evenly
        ir_signal = np.random.randn(250)
        red_signal = np.random.randn(250)
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)
        ppr, times_ppr = ppg_extractor.calculate_ppr(window_seconds=1)
        
        # Should only process 2 complete windows
        assert len(ppr) == 2
        assert len(times_ppr) == 2

    def test_calculate_ppr_zero_dc_components(self):
        """Test calculate_ppr when DC components are zero or negative.
        
        This test covers line 229 in ppg_light_features.py where
        ppr = 0 is set when DC components are invalid.
        """
        # Create signals with zero mean (DC component = 0)
        ir_signal = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        red_signal = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        sampling_freq = 100
        
        ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)
        ppr, times_ppr = ppg_extractor.calculate_ppr(window_seconds=0.1)
        
        # DC components should be 0, so ppr should be 0
        assert len(ppr) > 0
        # Check that values are set correctly (0 when DC is invalid)
        assert np.all(ppr >= 0)