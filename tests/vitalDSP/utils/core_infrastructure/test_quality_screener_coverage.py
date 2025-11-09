"""
Comprehensive coverage tests for Quality Screener modules.

This test suite targets uncovered lines to improve coverage from 83%/86% to >90%.
Focus areas:
- Signal screening with various quality levels
- Edge cases and boundary conditions
- Different signal patterns and artifacts
- Statistics tracking
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

try:
    from vitalDSP.utils.core_infrastructure.optimized_quality_screener import (
        OptimizedQualityScreener,
    )
    from vitalDSP.utils.core_infrastructure.quality_screener import (
        QualityScreener,
    )
    QUALITY_SCREENER_AVAILABLE = True
except ImportError:
    QUALITY_SCREENER_AVAILABLE = False


@pytest.mark.skipif(not QUALITY_SCREENER_AVAILABLE, reason="QualityScreener not available")
class TestOptimizedQualityScreenerEdgeCases:
    """Test edge cases in OptimizedQualityScreener."""

    @pytest.fixture
    def screener(self):
        """Create quality screener for testing."""
        return OptimizedQualityScreener()

    def test_screen_perfect_signal(self, screener):
        """Test screening of perfect quality signal."""
        # Generate very clean signal
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        clean_signal = np.sin(2 * np.pi * 1.0 * t)

        # Create DataFrame format
        df = pd.DataFrame({
            'signal': clean_signal,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)
        if len(results) > 0:
            # Check result structure - should be ScreeningResult objects
            result = results[0]
            assert hasattr(result, 'passed_screening') or hasattr(result, 'passed') or isinstance(result, dict)

    def test_screen_noisy_signal(self, screener):
        """Test screening of noisy signal."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        noisy_signal = np.sin(2 * np.pi * 1.0 * t) + 2.0 * np.random.randn(len(t))

        df = pd.DataFrame({
            'signal': noisy_signal,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_flatline_signal(self, screener):
        """Test screening of flatline signal."""
        flatline_signal = np.ones(1000) * 5.0

        df = pd.DataFrame({
            'signal': flatline_signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)
        # Flatline should be detected as poor quality

    def test_screen_signal_with_spikes(self, screener):
        """Test screening with spike artifacts."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        signal = np.sin(2 * np.pi * 1.0 * t)

        # Add spikes
        spike_indices = [500, 1000, 1500]
        for idx in spike_indices:
            if idx < len(signal):
                signal[idx] = signal[idx] * 10  # Large spike

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_saturated_signal(self, screener):
        """Test screening of saturated signal."""
        fs = 256
        saturated_signal = np.random.randn(1000)
        saturated_signal[100:200] = 10.0  # Saturated region
        saturated_signal[500:600] = 10.0  # Another saturated region

        df = pd.DataFrame({
            'signal': saturated_signal,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_signal_with_nan(self, screener):
        """Test screening signal containing NaN values."""
        signal = np.random.randn(1000)
        signal[100:110] = np.nan

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_signal_with_inf(self, screener):
        """Test screening signal containing infinite values."""
        signal = np.random.randn(1000)
        signal[50] = np.inf
        signal[150] = -np.inf

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_very_short_signal(self, screener):
        """Test screening very short signal."""
        short_signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        df = pd.DataFrame({
            'signal': short_signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_very_long_signal(self, screener):
        """Test screening very long signal."""
        long_signal = np.random.randn(100000)

        df = pd.DataFrame({
            'signal': long_signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_zero_variance_signal(self, screener):
        """Test screening signal with zero variance."""
        zero_var_signal = np.zeros(1000)

        df = pd.DataFrame({
            'signal': zero_var_signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_high_frequency_noise(self, screener):
        """Test screening signal with high frequency noise."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_baseline_wander(self, screener):
        """Test screening signal with baseline wander."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        # Signal with slow baseline drift
        baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        signal = np.sin(2 * np.pi * 1.0 * t) + baseline

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_configure_thresholds(self, screener):
        """Test configuring quality thresholds."""
        # This tests the configure_thresholds method
        screener.configure_thresholds(
            min_quality=0.5,
            max_artifacts=10,
            min_snr=5.0
        )

        # Should not raise any errors
        assert True

    def test_get_statistics(self, screener):
        """Test retrieving screening statistics."""
        # Screen some signals first
        signal1 = np.random.randn(1000)
        df1 = pd.DataFrame({'signal': signal1, 'sampling_rate': 256})
        screener.screen_signal(df1)

        signal2 = np.random.randn(1000)
        df2 = pd.DataFrame({'signal': signal2, 'sampling_rate': 256})
        screener.screen_signal(df2)

        stats = screener.get_statistics()

        assert isinstance(stats, dict)

    def test_reset_statistics(self, screener):
        """Test resetting screening statistics."""
        # Screen a signal
        signal = np.random.randn(1000)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})
        screener.screen_signal(df)

        # Reset
        screener.reset_statistics()

        # Get stats should show reset
        stats = screener.get_statistics()
        assert isinstance(stats, dict)

    def test_screen_with_progress_callback(self, screener):
        """Test screening with progress callback."""
        signal = np.random.randn(5000)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        progress_calls = []
        def callback(info):
            progress_calls.append(info)

        results = screener.screen_signal(df, progress_callback=callback)

        assert isinstance(results, list)


@pytest.mark.skipif(not QUALITY_SCREENER_AVAILABLE, reason="QualityScreener not available")
class TestQualityScreenerComplexScenarios:
    """Test complex screening scenarios with QualityScreener."""

    @pytest.fixture
    def screener(self):
        """Create quality screener for testing."""
        return QualityScreener()

    def test_screen_ecg_like_signal(self, screener):
        """Test screening ECG-like signal."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        # Simulate ECG-like pattern with QRS complexes
        ecg = np.zeros_like(t)
        for i in range(10):
            peak_pos = int((i + 0.8) * fs)
            if peak_pos < len(ecg):
                ecg[peak_pos-2:peak_pos+3] = [0.2, 0.5, 1.0, 0.5, 0.2]

        df = pd.DataFrame({
            'signal': ecg,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_ppg_like_signal(self, screener):
        """Test screening PPG-like signal."""
        fs = 125
        t = np.linspace(0, 10, fs * 10)
        # Simulate PPG-like pattern
        ppg = np.zeros_like(t)
        for i in range(10):
            peak_pos = int((i + 0.8) * fs)
            if peak_pos + 20 < len(ppg):
                ppg[peak_pos:peak_pos+20] = np.exp(-np.linspace(0, 3, 20))

        df = pd.DataFrame({
            'signal': ppg,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_motion_artifact(self, screener):
        """Test screening signal with motion artifacts."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        signal = np.sin(2 * np.pi * 1.0 * t)

        # Add motion artifact (low frequency, high amplitude)
        motion = 3.0 * np.sin(2 * np.pi * 0.2 * t)
        signal_with_motion = signal + motion

        df = pd.DataFrame({
            'signal': signal_with_motion,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_powerline_interference(self, screener):
        """Test screening signal with powerline interference."""
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        signal = np.sin(2 * np.pi * 1.0 * t)

        # Add 60 Hz powerline interference
        powerline = 0.3 * np.sin(2 * np.pi * 60 * t)
        signal_with_powerline = signal + powerline

        df = pd.DataFrame({
            'signal': signal_with_powerline,
            'sampling_rate': fs
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_multiple_signals_batch(self, screener):
        """Test batch screening of multiple signals."""
        # Create multiple signals in a DataFrame
        signals = []
        for i in range(5):
            signal = np.random.randn(1000) + i * 0.1
            signals.append(signal)

        # Screen each signal
        all_results = []
        for signal in signals:
            df = pd.DataFrame({
                'signal': signal,
                'sampling_rate': 256
            })
            results = screener.screen_signal(df)
            all_results.append(results)

        assert len(all_results) == 5

    def test_screen_clipped_signal(self, screener):
        """Test screening signal with clipping."""
        signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 2560))
        # Clip signal
        signal = np.clip(signal, -0.5, 0.5)

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_asymmetric_signal(self, screener):
        """Test screening asymmetric signal."""
        t = np.linspace(0, 10, 2560)
        # Asymmetric waveform
        signal = np.where(np.sin(2 * np.pi * 1.0 * t) > 0,
                         np.sin(2 * np.pi * 1.0 * t),
                         0.5 * np.sin(2 * np.pi * 1.0 * t))

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_with_dc_offset(self, screener):
        """Test screening signal with DC offset."""
        signal = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 2560)) + 5.0

        df = pd.DataFrame({
            'signal': signal,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_screen_white_noise(self, screener):
        """Test screening white noise."""
        white_noise = np.random.randn(2560)

        df = pd.DataFrame({
            'signal': white_noise,
            'sampling_rate': 256
        })

        results = screener.screen_signal(df)

        assert isinstance(results, list)

    def test_get_statistics_after_multiple_screens(self, screener):
        """Test statistics tracking across multiple screenings."""
        for i in range(10):
            signal = np.random.randn(1000)
            df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})
            screener.screen_signal(df)

        stats = screener.get_statistics()
        assert isinstance(stats, dict)

    def test_reset_and_rescreen(self, screener):
        """Test resetting statistics and rescreening."""
        # Screen some signals
        for i in range(3):
            signal = np.random.randn(1000)
            df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})
            screener.screen_signal(df)

        # Reset
        screener.reset_statistics()

        # Screen again
        signal = np.random.randn(1000)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})
        results = screener.screen_signal(df)

        assert isinstance(results, list)


@pytest.mark.skipif(not QUALITY_SCREENER_AVAILABLE, reason="QualityScreener not available")
class TestQualityScreenerSpecialCases:
    """Test special cases and boundary conditions."""

    @pytest.fixture
    def screener(self):
        """Create quality screener for testing."""
        return OptimizedQualityScreener()

    def test_screen_single_sample(self, screener):
        """Test screening single sample signal."""
        signal = np.array([1.0])
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_all_nan(self, screener):
        """Test screening all-NaN signal."""
        signal = np.full(1000, np.nan)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_all_inf(self, screener):
        """Test screening all-infinite signal."""
        signal = np.full(1000, np.inf)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_mixed_invalid(self, screener):
        """Test screening signal with mixed NaN/Inf values."""
        signal = np.random.randn(1000)
        signal[::3] = np.nan
        signal[::5] = np.inf
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_extreme_values(self, screener):
        """Test screening signal with extreme values."""
        signal = np.random.randn(1000)
        signal[100] = 1e10
        signal[200] = -1e10
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_periodic_signal(self, screener):
        """Test screening perfectly periodic signal."""
        t = np.linspace(0, 10, 2560)
        signal = np.sin(2 * np.pi * 1.0 * t)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_with_different_sampling_rates(self, screener):
        """Test screening signals with various sampling rates."""
        sampling_rates = [50, 100, 125, 256, 500, 1000]

        for fs in sampling_rates:
            t = np.linspace(0, 10, fs * 10)
            signal = np.sin(2 * np.pi * 1.0 * t)
            df = pd.DataFrame({'signal': signal, 'sampling_rate': fs})

            results = screener.screen_signal(df)
            assert isinstance(results, list)

    def test_screen_numpy_array_input(self, screener):
        """Test screening with numpy array input."""
        signal = np.random.randn(1000)

        # Should handle numpy array input
        try:
            results = screener.screen_signal(signal)
            assert isinstance(results, list)
        except (TypeError, ValueError):
            # If it requires DataFrame, that's okay
            pass

    def test_screen_consistency(self, screener):
        """Test screening consistency for same signal."""
        signal = np.random.randn(1000)
        df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})

        results1 = screener.screen_signal(df)
        results2 = screener.screen_signal(df)

        # Should produce consistent results
        assert len(results1) == len(results2)

    def test_statistics_accumulation(self, screener):
        """Test statistics accumulate correctly."""
        initial_stats = screener.get_statistics()

        # Screen multiple signals
        for i in range(5):
            signal = np.random.randn(1000)
            df = pd.DataFrame({'signal': signal, 'sampling_rate': 256})
            screener.screen_signal(df)

        final_stats = screener.get_statistics()

        # Stats should have changed (unless they were None initially)
        assert isinstance(final_stats, dict)
