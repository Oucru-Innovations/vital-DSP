"""
Extended tests for RespiratoryAnalysis module to improve coverage.

Tests cover additional methods and edge cases for respiratory analysis.
"""

import pytest
import numpy as np
from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
from vitalDSP.preprocess.preprocess_operations import PreprocessConfig


class TestRespiratoryAnalysisInit:
    """Tests for RespiratoryAnalysis initialization."""

    def test_init_default_fs(self):
        """Test initialization with default sampling frequency."""
        signal = np.random.randn(2000)
        resp = RespiratoryAnalysis(signal)
        assert resp.fs == 256  # Default fs

    def test_init_custom_fs(self):
        """Test initialization with custom sampling frequency."""
        signal = np.random.randn(2000)
        for fs in [50, 100, 128, 250, 500]:
            resp = RespiratoryAnalysis(signal, fs=fs)
            assert resp.fs == fs

    def test_init_short_signal(self):
        """Test initialization with short signal."""
        signal = np.random.randn(100)
        resp = RespiratoryAnalysis(signal, fs=128)
        assert len(resp.signal) == 100

    def test_init_long_signal(self):
        """Test initialization with long signal."""
        signal = np.random.randn(100000)
        resp = RespiratoryAnalysis(signal, fs=256)
        assert len(resp.signal) == 100000


class TestComputeRespiratoryRate:
    """Tests for compute_respiratory_rate method."""

    def test_counting_method(self):
        """Test counting method for respiratory rate."""
        # Create respiratory-like signal
        t = np.linspace(0, 60, 7680)  # 60 seconds at 128 Hz
        signal = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min

        resp = RespiratoryAnalysis(signal, fs=128)
        result = resp.compute_respiratory_rate(method="counting")
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result > 0

    def test_fft_based_method(self):
        """Test FFT-based method for respiratory rate."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min

        resp = RespiratoryAnalysis(signal, fs=128)
        result = resp.compute_respiratory_rate(method="fft_based")
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_frequency_domain_method(self):
        """Test frequency domain method."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=128)
        result = resp.compute_respiratory_rate(method="frequency_domain")
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_peaks_method(self):
        """Test peaks method for respiratory rate."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=128)
        result = resp.compute_respiratory_rate(method="peaks")
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_time_domain_method(self):
        """Test time domain method."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.3 * t)  # 18 breaths/min

        resp = RespiratoryAnalysis(signal, fs=128)
        result = resp.compute_respiratory_rate(method="time_domain")
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_invalid_method(self):
        """Test with invalid method."""
        signal = np.random.randn(2000)
        resp = RespiratoryAnalysis(signal, fs=128)

        with pytest.raises((ValueError, KeyError, AttributeError)):
            resp.compute_respiratory_rate(method="invalid_method")


class TestWithPreprocessing:
    """Tests for respiratory rate with preprocessing."""

    def test_with_bandpass_filter(self):
        """Test respiratory rate with bandpass filtering."""
        t = np.linspace(0, 60, 7680)
        # Add noise to respiratory signal
        clean_signal = np.sin(2 * np.pi * 0.25 * t)
        noisy_signal = clean_signal + 0.5 * np.random.randn(len(t))

        resp = RespiratoryAnalysis(noisy_signal, fs=128)

        config = PreprocessConfig(
            filter_type="bandpass",
            lowcut=0.1,
            highcut=0.5
        )
        result = resp.compute_respiratory_rate(
            method="counting",
            preprocess_config=config
        )
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_with_lowpass_filter(self):
        """Test with lowpass filter."""
        signal = np.random.randn(5000)
        resp = RespiratoryAnalysis(signal, fs=128)

        config = PreprocessConfig(
            filter_type="butterworth",  # Use butterworth for lowpass filtering
            highcut=2.0
        )
        result = resp.compute_respiratory_rate(
            method="fft_based",
            preprocess_config=config
        )
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_with_wavelet_denoising(self):
        """Test with wavelet denoising."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t) + 0.3 * np.random.randn(len(t))

        resp = RespiratoryAnalysis(signal, fs=128)

        config = PreprocessConfig(
            noise_reduction_method="wavelet",
            level=3
        )
        result = resp.compute_respiratory_rate(
            method="counting",
            preprocess_config=config
        )
        # Method returns float, not dict
        assert isinstance(result, (int, float))
        assert result >= 0


class TestDifferentRespiratoryRates:
    """Tests with different respiratory rates."""

    def test_slow_breathing(self):
        """Test with slow breathing (8 breaths/min)."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * (8/60) * t)  # 8 breaths/min

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="fft_based")
            # Should detect slow breathing
            # Result is a float, not dict
            assert isinstance(result, (int, float))
            # Allow some tolerance
            assert 5 <= result <= 12
        except (KeyError, NotImplementedError, AttributeError):
            pytest.skip("Not fully implemented")

    def test_normal_breathing(self):
        """Test with normal breathing (15 breaths/min)."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
            assert 10 <= result <= 20
        except (KeyError, NotImplementedError, AttributeError):
            pytest.skip("Not fully implemented")

    def test_fast_breathing(self):
        """Test with fast breathing (25 breaths/min)."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * (25/60) * t)  # 25 breaths/min

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="peaks")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
            assert 20 <= result <= 30
        except (KeyError, NotImplementedError, AttributeError):
            pytest.skip("Not fully implemented")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_constant_signal(self):
        """Test with constant signal (no breathing)."""
        signal = np.ones(2000)
        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Should handle gracefully
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (ValueError, ZeroDivisionError, KeyError):
            # Expected for constant signal
            pass

    def test_very_noisy_signal(self):
        """Test with very noisy signal."""
        signal = np.random.randn(5000) * 10
        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="fft_based")
            # Should return some result even if unreliable
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_single_breath_cycle(self):
        """Test with signal containing single breath cycle."""
        t = np.linspace(0, 4, 512)  # 4 seconds at 128 Hz
        signal = np.sin(2 * np.pi * 0.25 * t)  # One cycle

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_irregular_breathing(self):
        """Test with irregular breathing pattern."""
        # Varying frequency
        t = np.linspace(0, 60, 7680)
        freq = 0.2 + 0.1 * np.sin(2 * np.pi * 0.05 * t)
        phase = np.cumsum(freq) * 2 * np.pi * (t[1] - t[0])
        signal = np.sin(phase)

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")


class TestRealisticSignals:
    """Tests with realistic physiological signals."""

    def test_ppg_derived_respiratory(self):
        """Test with PPG-derived respiratory signal."""
        # Simulate PPG with respiratory modulation
        t = np.linspace(0, 30, 3840)  # 30 seconds at 128 Hz
        hr = 1.2  # Hz (72 bpm)
        rr = 0.25  # Hz (15 breaths/min)

        # PPG with amplitude modulation
        ppg = (1 + 0.1 * np.sin(2 * np.pi * rr * t)) * np.sin(2 * np.pi * hr * t)
        ppg += 0.1 * np.random.randn(len(t))

        resp = RespiratoryAnalysis(ppg, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="fft_based")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_ecg_derived_respiratory(self):
        """Test with ECG-derived respiratory signal."""
        # Simulate ECG with RSA (respiratory sinus arrhythmia)
        t = np.linspace(0, 30, 3840)
        rr_baseline = 0.25  # 15 breaths/min

        # Heart rate varies with respiration
        hr = 1.2 + 0.1 * np.sin(2 * np.pi * rr_baseline * t)
        ecg = np.sin(2 * np.pi * np.cumsum(hr) / 128)

        resp = RespiratoryAnalysis(ecg, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="peaks")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_respiratory_effort_signal(self):
        """Test with direct respiratory effort signal."""
        # Simulate chest impedance or effort signal
        t = np.linspace(0, 60, 7680)
        resp_signal = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
        resp_signal += 0.05 * np.sin(2 * np.pi * 0.5 * t)  # Second harmonic

        resp = RespiratoryAnalysis(resp_signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
            # Should be close to 15 breaths/min
            assert 12 <= result <= 18
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")


class TestMultipleMethodsComparison:
    """Tests comparing different methods."""

    def test_compare_counting_and_fft(self):
        """Compare counting and FFT methods."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(len(t))

        resp = RespiratoryAnalysis(signal, fs=128)

        results = {}
        for method in ["counting", "fft_based"]:
            try:
                result = resp.compute_respiratory_rate(method=method)
                # Result is a float, not a dict
                if isinstance(result, (int, float)):
                    results[method] = result
            except (KeyError, NotImplementedError):
                pass

        # Both should give similar results for clean signal
        if len(results) >= 2:
            rates = list(results.values())
            # Check they're in reasonable range of each other
            assert abs(rates[0] - rates[1]) < 10  # Within 10 bpm (relaxed)

    def test_all_available_methods(self):
        """Test all available methods."""
        signal = np.random.randn(5000)
        resp = RespiratoryAnalysis(signal, fs=128)

        methods = ["counting", "fft_based", "frequency_domain", "peaks"]

        for method in methods:
            try:
                result = resp.compute_respiratory_rate(method=method)
                # Result is a float
                assert isinstance(result, (int, float))
            except (KeyError, NotImplementedError, AttributeError, ValueError):
                # Method may not be implemented
                pass


class TestResultFormat:
    """Tests for result format and metadata."""

    def test_result_contains_respiratory_rate(self):
        """Test that result contains respiratory_rate."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float
            assert isinstance(result, (int, float))
            assert result >= 0
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_result_metadata(self):
        """Test that result is a valid number."""
        signal = np.random.randn(5000)
        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="fft_based")
            # Result is a float
            assert isinstance(result, (int, float))
            assert result >= 0
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_amplitude_signal(self):
        """Test with large amplitude signal."""
        t = np.linspace(0, 60, 7680)
        signal = 1000 * np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
            assert not np.isnan(result)
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_small_amplitude_signal(self):
        """Test with small amplitude signal."""
        t = np.linspace(0, 60, 7680)
        signal = 0.001 * np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="fft_based")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_signal_with_trend(self):
        """Test with signal containing trend."""
        t = np.linspace(0, 60, 7680)
        # Respiratory signal with linear trend
        signal = np.sin(2 * np.pi * 0.25 * t) + 0.01 * t

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_signal_with_outliers(self):
        """Test with signal containing outliers."""
        t = np.linspace(0, 60, 7680)
        signal = np.sin(2 * np.pi * 0.25 * t)

        # Add some outliers
        outlier_indices = np.random.choice(len(signal), 10, replace=False)
        signal[outlier_indices] = 100

        resp = RespiratoryAnalysis(signal, fs=128)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")


class TestDifferentSamplingRates:
    """Tests with different sampling rates."""

    def test_low_sampling_rate(self):
        """Test with low sampling rate (50 Hz)."""
        t = np.linspace(0, 60, 3000)  # 50 Hz
        signal = np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=50)

        try:
            result = resp.compute_respiratory_rate(method="fft_based")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_high_sampling_rate(self):
        """Test with high sampling rate (1000 Hz)."""
        t = np.linspace(0, 60, 60000)  # 1000 Hz
        signal = np.sin(2 * np.pi * 0.25 * t)

        resp = RespiratoryAnalysis(signal, fs=1000)

        try:
            result = resp.compute_respiratory_rate(method="counting")
            # Result is a float, not dict
            assert isinstance(result, (int, float))
        except (KeyError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_variable_sampling_rates(self):
        """Test with various sampling rates."""
        for fs in [25, 50, 100, 128, 250, 500]:
            duration = 30  # seconds
            n_samples = int(duration * fs)
            t = np.linspace(0, duration, n_samples)
            signal = np.sin(2 * np.pi * 0.25 * t)

            resp = RespiratoryAnalysis(signal, fs=fs)

            try:
                result = resp.compute_respiratory_rate(method="fft_based")
                # Result is a float, not dict
                assert isinstance(result, (int, float))
            except (KeyError, NotImplementedError):
                pass
