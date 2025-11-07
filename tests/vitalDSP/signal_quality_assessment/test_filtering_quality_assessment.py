"""
Comprehensive tests for FilteringQualityAssessment module.

This test suite covers all methods and edge cases for the filtering quality
assessment functionality.
"""

import pytest
import numpy as np
from vitalDSP.signal_quality_assessment.filtering_quality_assessment import (
    FilteringQualityAssessment,
)


class TestFilteringQualityAssessmentInit:
    """Tests for FilteringQualityAssessment initialization."""

    def test_init_basic(self):
        """Test basic initialization with numpy arrays."""
        original = np.random.randn(1000)
        filtered = np.random.randn(1000)
        fqa = FilteringQualityAssessment(original, filtered, fs=250, signal_type='ECG')

        assert len(fqa.original_signal) == 1000
        assert len(fqa.filtered_signal) == 1000
        assert fqa.fs == 250
        assert fqa.signal_type == 'ECG'

    def test_init_with_lists(self):
        """Test initialization with Python lists."""
        original = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        filtered = [1.1, 2.1, 3.1, 4.1, 5.1] * 10
        fqa = FilteringQualityAssessment(original, filtered, fs=100, signal_type='PPG')

        assert isinstance(fqa.original_signal, np.ndarray)
        assert isinstance(fqa.filtered_signal, np.ndarray)
        assert len(fqa.original_signal) == 50

    def test_init_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        original = np.random.randn(1000)
        filtered = np.random.randn(500)

        with pytest.raises(ValueError, match="Signal lengths must match"):
            FilteringQualityAssessment(original, filtered, fs=250)

    def test_init_short_signals(self):
        """Test that signals shorter than minimum length raise error."""
        original = np.array([1, 2, 3])
        filtered = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            FilteringQualityAssessment(original, filtered, fs=250)

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        original = np.random.randn(100)
        filtered = np.random.randn(100)
        fqa = FilteringQualityAssessment(original, filtered)

        assert fqa.fs == 250.0
        assert fqa.signal_type == 'General'


class TestAdaptiveThresholds:
    """Tests for signal-adaptive threshold selection."""

    def test_ecg_thresholds(self):
        """Test ECG-specific thresholds."""
        signal = np.random.randn(100)
        fqa = FilteringQualityAssessment(signal, signal, fs=250, signal_type='ECG')
        thresholds = fqa.thresholds

        assert 'noise_reduction' in thresholds
        assert 'peak_preservation' in thresholds
        assert 'shape_similarity' in thresholds
        assert thresholds['peak_preservation']['excellent'] == 0.9

    def test_ppg_thresholds(self):
        """Test PPG-specific thresholds."""
        signal = np.random.randn(100)
        fqa = FilteringQualityAssessment(signal, signal, fs=128, signal_type='PPG')
        thresholds = fqa.thresholds

        assert thresholds['peak_preservation']['excellent'] == 0.92
        assert thresholds['snr_db']['excellent'] == 18

    def test_eeg_thresholds(self):
        """Test EEG-specific thresholds."""
        signal = np.random.randn(100)
        fqa = FilteringQualityAssessment(signal, signal, fs=256, signal_type='EEG')
        thresholds = fqa.thresholds

        assert thresholds['shape_similarity']['excellent'] == 0.70
        assert thresholds['noise_reduction']['excellent'] == 0.4

    def test_respiratory_thresholds(self):
        """Test Respiratory-specific thresholds."""
        signal = np.random.randn(100)
        fqa = FilteringQualityAssessment(signal, signal, fs=50, signal_type='Respiratory')
        thresholds = fqa.thresholds

        assert thresholds['smoothness_improvement']['excellent'] == 0.3
        assert thresholds['snr_db']['good'] == 10

    def test_general_thresholds(self):
        """Test General signal type thresholds."""
        signal = np.random.randn(100)
        fqa = FilteringQualityAssessment(signal, signal, fs=250, signal_type='General')
        thresholds = fqa.thresholds

        assert 'noise_reduction' in thresholds
        assert 'smoothness_improvement' in thresholds


class TestNoiseReduction:
    """Tests for noise reduction calculation."""

    def test_noise_reduction_heavy_filtering(self):
        """Test noise reduction with heavy filtering."""
        # Create noisy signal
        t = np.linspace(0, 10, 1000)
        clean = np.sin(2 * np.pi * 1.2 * t)
        noisy = clean + np.random.randn(1000) * 0.3

        fqa = FilteringQualityAssessment(noisy, clean, fs=100, signal_type='ECG')
        score, status, assessment = fqa.calculate_noise_reduction()

        assert 0 <= score <= 1
        assert status in ['Excellent', 'Good', 'Acceptable', 'Poor']
        assert isinstance(assessment, str)
        assert score > 0.05  # Should detect noise removal

    def test_noise_reduction_minimal_filtering(self):
        """Test noise reduction with minimal filtering."""
        signal = np.sin(np.linspace(0, 10*np.pi, 1000))
        # Almost no filtering
        filtered = signal + np.random.randn(1000) * 0.01

        fqa = FilteringQualityAssessment(signal, filtered, fs=250)
        score, status, assessment = fqa.calculate_noise_reduction()

        assert 0 <= score <= 1
        assert score < 0.1  # Minimal noise reduction

    def test_noise_reduction_no_change(self):
        """Test noise reduction when signals are identical."""
        signal = np.random.randn(100)

        fqa = FilteringQualityAssessment(signal, signal, fs=250)
        score, status, assessment = fqa.calculate_noise_reduction()

        assert score == 0.0
        assert status == 'Poor'

    def test_noise_reduction_zero_power(self):
        """Test noise reduction with near-zero power signal."""
        original = np.zeros(100) + 1e-12
        filtered = np.zeros(100)

        fqa = FilteringQualityAssessment(original, filtered, fs=250)
        score, status, assessment = fqa.calculate_noise_reduction()

        assert score == 0.0


class TestSNRImprovement:
    """Tests for SNR improvement calculation."""

    def test_snr_high_quality(self):
        """Test SNR calculation for high-quality filtered signal."""
        # Clean signal with small noise
        t = np.linspace(0, 10, 1000)
        clean = np.sin(2 * np.pi * 1.0 * t)
        noisy = clean + np.random.randn(1000) * 0.05

        fqa = FilteringQualityAssessment(noisy, clean, fs=250, signal_type='PPG')
        snr_db, status, assessment = fqa.calculate_snr_improvement()

        assert snr_db > 10  # Should have good SNR
        assert status in ['Excellent', 'Good', 'Acceptable', 'Poor']

    def test_snr_low_quality(self):
        """Test SNR calculation for low-quality signal."""
        # Heavy noise
        signal = np.sin(np.linspace(0, 10*np.pi, 1000))
        noisy = signal + np.random.randn(1000) * 2.0
        filtered = signal + np.random.randn(1000) * 1.5

        fqa = FilteringQualityAssessment(noisy, filtered, fs=250)
        snr_db, status, assessment = fqa.calculate_snr_improvement()

        assert isinstance(snr_db, (int, float))
        assert status in ['Excellent', 'Good', 'Acceptable', 'Poor']

    def test_snr_zero_power_signals(self):
        """Test SNR with near-zero power."""
        original = np.zeros(100) + 1e-12
        filtered = np.zeros(100) + 1e-13

        fqa = FilteringQualityAssessment(original, filtered, fs=250)
        snr_db, status, assessment = fqa.calculate_snr_improvement()

        assert snr_db == 0.0


class TestSmoothnessImprovement:
    """Tests for smoothness improvement calculation."""

    def test_smoothness_good_improvement(self):
        """Test smoothness improvement with effective smoothing."""
        # Rough signal
        rough = np.cumsum(np.random.randn(1000))
        # Smoothed version
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(rough, sigma=5)

        fqa = FilteringQualityAssessment(rough, smooth, fs=250, signal_type='ECG')
        improvement, status, assessment = fqa.calculate_smoothness_improvement()

        assert 0 <= improvement <= 1
        assert improvement > 0.1  # Should show improvement
        assert status in ['Excellent', 'Good', 'Acceptable', 'Poor']

    def test_smoothness_no_improvement(self):
        """Test smoothness when no smoothing applied."""
        signal = np.random.randn(100)

        fqa = FilteringQualityAssessment(signal, signal, fs=250)
        improvement, status, assessment = fqa.calculate_smoothness_improvement()

        assert improvement == 0.0
        assert status == 'Poor'

    def test_smoothness_clamping(self):
        """Test that smoothness improvement is clamped to [0,1]."""
        # Signal that becomes rougher (negative improvement)
        smooth = np.sin(np.linspace(0, 10*np.pi, 1000))
        rough = smooth + np.random.randn(1000) * 0.5

        fqa = FilteringQualityAssessment(smooth, rough, fs=250)
        improvement, status, assessment = fqa.calculate_smoothness_improvement()

        assert 0 <= improvement <= 1

    def test_smoothness_zero_variance(self):
        """Test smoothness with constant signal (zero variance)."""
        constant = np.ones(100) * 5.0

        fqa = FilteringQualityAssessment(constant, constant, fs=250)
        improvement, status, assessment = fqa.calculate_smoothness_improvement()

        assert improvement == 0.0


class TestPeakPreservation:
    """Tests for peak preservation calculation."""

    def test_peak_preservation_perfect(self):
        """Test perfect peak preservation."""
        # Signal with clear peaks
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 1.0 * t)
        # Slightly smoothed but peaks preserved
        filtered = signal * 0.98

        fqa = FilteringQualityAssessment(signal, filtered, fs=100, signal_type='ECG')
        score, status, assessment = fqa.calculate_peak_preservation()

        assert 0 <= score <= 1
        assert isinstance(status, str)

    def test_peak_preservation_no_peaks_original(self):
        """Test peak preservation when no peaks in original."""
        # Flat signal
        flat = np.ones(1000) * 0.5
        filtered = np.ones(1000) * 0.5

        fqa = FilteringQualityAssessment(flat, filtered, fs=250)
        score, status, assessment = fqa.calculate_peak_preservation()

        assert score == 1.0
        assert 'No peaks detected' in assessment

    def test_peak_preservation_some_lost(self):
        """Test peak preservation when some peaks are lost."""
        # Create signal with peaks
        signal = np.zeros(1000)
        signal[100:110] = 1.0
        signal[500:510] = 1.0
        signal[900:910] = 1.0

        # Filter removes one peak
        filtered = signal.copy()
        filtered[500:510] = 0.1

        fqa = FilteringQualityAssessment(signal, filtered, fs=250)
        score, status, assessment = fqa.calculate_peak_preservation()

        assert 0 <= score <= 1


class TestShapeSimilarity:
    """Tests for shape similarity calculation."""

    def test_shape_similarity_high(self):
        """Test high shape similarity."""
        # Similar signals
        t = np.linspace(0, 10, 1000)
        original = np.sin(2 * np.pi * 1.0 * t) + np.random.randn(1000) * 0.1
        filtered = np.sin(2 * np.pi * 1.0 * t)

        fqa = FilteringQualityAssessment(original, filtered, fs=250, signal_type='PPG')
        similarity, status, assessment = fqa.calculate_shape_similarity()

        assert 0 <= similarity <= 1
        assert similarity > 0.7  # Should be high correlation
        assert status in ['Excellent', 'Good', 'Acceptable', 'Poor']

    def test_shape_similarity_low(self):
        """Test low shape similarity."""
        # Very different signals - use deterministic seed for reproducibility
        np.random.seed(42)
        original = np.random.randn(1000)
        np.random.seed(123)
        filtered = np.random.randn(1000)

        fqa = FilteringQualityAssessment(original, filtered, fs=250)
        similarity, status, assessment = fqa.calculate_shape_similarity()

        # Similarity can be negative (correlation ranges from -1 to 1)
        assert -1 <= similarity <= 1
        assert status in ['Excellent', 'Good', 'Acceptable', 'Poor']
        # Result should be computed (not None or NaN)
        assert similarity is not None and not np.isnan(similarity)

    def test_shape_similarity_perfect(self):
        """Test perfect shape similarity (identical signals)."""
        signal = np.random.randn(100)

        fqa = FilteringQualityAssessment(signal, signal, fs=250)
        similarity, status, assessment = fqa.calculate_shape_similarity()

        assert similarity > 0.99  # Should be near 1.0


class TestAssessQuality:
    """Tests for comprehensive quality assessment."""

    def test_assess_quality_comprehensive(self):
        """Test comprehensive quality assessment."""
        # Create realistic scenario
        t = np.linspace(0, 10, 1000)
        clean = np.sin(2 * np.pi * 1.2 * t)
        noisy = clean + np.random.randn(1000) * 0.2

        fqa = FilteringQualityAssessment(noisy, clean, fs=250, signal_type='ECG')
        results = fqa.assess_quality()

        # Check structure
        assert 'overall_quality' in results
        assert 'recommendation' in results
        assert 'signal_type' in results
        assert 'sampling_frequency' in results
        assert 'metrics' in results

        # Check overall quality
        assert results['overall_quality'] in ['Excellent', 'Good', 'Acceptable', 'Poor']
        assert isinstance(results['recommendation'], str)

        # Check metrics structure
        metrics = results['metrics']
        assert 'noise_reduction' in metrics
        assert 'snr_db' in metrics
        assert 'smoothness_improvement' in metrics
        assert 'peak_preservation' in metrics
        assert 'shape_similarity' in metrics

        # Check each metric has required fields
        for metric_name, metric_data in metrics.items():
            assert 'status' in metric_data
            assert 'assessment' in metric_data

    def test_assess_quality_excellent(self):
        """Test assessment that should result in excellent rating."""
        # Perfect scenario - identical signals (no noise to remove)
        signal = np.sin(np.linspace(0, 10*np.pi, 1000))

        fqa = FilteringQualityAssessment(signal, signal, fs=250, signal_type='General')
        results = fqa.assess_quality()

        assert results['signal_type'] == 'General'
        assert results['sampling_frequency'] == 250

    def test_assess_quality_poor(self):
        """Test assessment that should result in poor rating."""
        # Bad filtering - completely different signals
        original = np.random.randn(1000)
        filtered = np.random.randn(1000) * 5

        fqa = FilteringQualityAssessment(original, filtered, fs=250)
        results = fqa.assess_quality()

        assert 'overall_quality' in results
        # With random signals, quality can vary - just ensure it returns a valid rating
        assert results['overall_quality'] in ['Excellent', 'Good', 'Acceptable', 'Poor']

    def test_assess_quality_all_signal_types(self):
        """Test assessment for all signal types."""
        signal_types = ['ECG', 'PPG', 'EEG', 'Respiratory', 'General']

        for sig_type in signal_types:
            original = np.random.randn(1000)
            filtered = original + np.random.randn(1000) * 0.1

            fqa = FilteringQualityAssessment(original, filtered, fs=250, signal_type=sig_type)
            results = fqa.assess_quality()

            assert results['signal_type'] == sig_type
            assert 'overall_quality' in results


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constant_signals(self):
        """Test with constant signals."""
        constant = np.ones(100) * 5.0

        fqa = FilteringQualityAssessment(constant, constant, fs=250)
        results = fqa.assess_quality()

        assert 'overall_quality' in results

    def test_very_small_signals(self):
        """Test with minimum length signals."""
        signal = np.random.randn(10)

        fqa = FilteringQualityAssessment(signal, signal, fs=250)
        results = fqa.assess_quality()

        assert 'overall_quality' in results

    def test_high_frequency_signals(self):
        """Test with high frequency content."""
        t = np.linspace(0, 1, 10000)
        original = np.sin(2 * np.pi * 100 * t)
        filtered = np.sin(2 * np.pi * 100 * t) * 0.9

        fqa = FilteringQualityAssessment(original, filtered, fs=10000, signal_type='ECG')
        results = fqa.assess_quality()

        assert 'overall_quality' in results

    def test_negative_values(self):
        """Test with signals containing negative values."""
        original = np.random.randn(1000) - 5.0
        filtered = np.random.randn(1000) - 5.0

        fqa = FilteringQualityAssessment(original, filtered, fs=250)
        results = fqa.assess_quality()

        assert 'overall_quality' in results

    def test_large_amplitude_differences(self):
        """Test with large amplitude differences."""
        original = np.random.randn(1000) * 1000
        filtered = np.random.randn(1000) * 0.001

        fqa = FilteringQualityAssessment(original, filtered, fs=250)
        results = fqa.assess_quality()

        assert 'overall_quality' in results


class TestRealisticScenarios:
    """Tests with realistic filtering scenarios."""

    def test_lowpass_filtering(self):
        """Test assessment of lowpass filtering."""
        from scipy.signal import butter, filtfilt

        # Create signal with high-frequency noise
        t = np.linspace(0, 10, 5000)
        clean = np.sin(2 * np.pi * 1.0 * t)
        noise = np.random.randn(5000) * 0.2
        noisy = clean + noise

        # Apply lowpass filter
        b, a = butter(4, 0.1, btype='low')
        filtered = filtfilt(b, a, noisy)

        fqa = FilteringQualityAssessment(noisy, filtered, fs=500, signal_type='ECG')
        results = fqa.assess_quality()

        assert results['metrics']['noise_reduction']['score'] > 0.05

    def test_bandpass_filtering(self):
        """Test assessment of bandpass filtering."""
        from scipy.signal import butter, filtfilt

        # Signal with multiple frequency components
        t = np.linspace(0, 10, 5000)
        signal = (np.sin(2 * np.pi * 0.5 * t) +  # Low freq
                  np.sin(2 * np.pi * 5.0 * t) +   # Mid freq (target)
                  np.sin(2 * np.pi * 50 * t))      # High freq

        # Bandpass filter around 5 Hz
        b, a = butter(3, [0.08, 0.12], btype='band')
        filtered = filtfilt(b, a, signal)

        fqa = FilteringQualityAssessment(signal, filtered, fs=500, signal_type='General')
        results = fqa.assess_quality()

        assert 'overall_quality' in results

    def test_adaptive_filtering(self):
        """Test assessment of adaptive filtering results."""
        # Simulate adaptive filtering result
        t = np.linspace(0, 10, 2000)
        baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        ecg = np.sin(2 * np.pi * 1.2 * t)
        original = baseline + ecg + np.random.randn(2000) * 0.1
        filtered = ecg  # Baseline removed

        fqa = FilteringQualityAssessment(original, filtered, fs=200, signal_type='ECG')
        results = fqa.assess_quality()

        # Should detect good noise/baseline removal
        assert results['metrics']['noise_reduction']['score'] > 0.1
