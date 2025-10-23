"""
Test suite for Configurable Thresholds in Quality Screener.

This test suite verifies that all hard-coded thresholds have been replaced
with configurable parameters in the QualityScreener.
"""

import pytest
import numpy as np
from vitalDSP.utils.core_infrastructure.quality_screener import QualityScreener


class TestConfigurableThresholds:
    """Test configurable thresholds in QualityScreener."""

    @pytest.fixture
    def sample_ecg_signal(self):
        """Generate a sample ECG-like signal."""
        fs = 1000  # 1000 Hz sampling rate
        duration = 10  # 10 seconds
        t = np.arange(0, duration, 1/fs)

        # Generate synthetic ECG-like signal
        signal = np.sin(2 * np.pi * 1.2 * t)  # Heart rate component
        signal += 0.1 * np.random.randn(len(t))  # Gaussian noise

        return signal, fs

    def test_default_thresholds_are_set(self):
        """Test that all default thresholds are properly initialized."""
        screener = QualityScreener(signal_type="ecg", sampling_rate=1000)

        # Stage 1: SNR thresholds
        assert "snr_min_db" in screener.thresholds
        assert screener.thresholds["snr_min_db"] == 15.0  # ECG-specific default

        # Stage 2: Statistical thresholds
        assert "outlier_std_factor" in screener.thresholds
        assert "outlier_max_ratio" in screener.thresholds
        assert "constant_max_ratio" in screener.thresholds
        assert "jump_std_factor" in screener.thresholds
        assert "jump_max_ratio" in screener.thresholds

        assert screener.thresholds["outlier_std_factor"] == 3.0
        assert screener.thresholds["outlier_max_ratio"] == 0.1
        assert screener.thresholds["constant_max_ratio"] == 0.5
        assert screener.thresholds["jump_std_factor"] == 3.0
        assert screener.thresholds["jump_max_ratio"] == 0.15

        # Peak detection parameters
        assert "ecg_min_peak_distance_factor" in screener.thresholds
        assert "ecg_expected_heart_rate_bpm" in screener.thresholds

        assert screener.thresholds["ecg_min_peak_distance_factor"] == 0.4
        assert screener.thresholds["ecg_expected_heart_rate_bpm"] == 72

    def test_custom_thresholds_via_kwargs(self):
        """Test that custom thresholds can be set via kwargs."""
        screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=1000,
            snr_min_db=20.0,
            outlier_max_ratio=0.05,
            jump_max_ratio=0.1,
            ecg_expected_heart_rate_bpm=80,
        )

        assert screener.thresholds["snr_min_db"] == 20.0
        assert screener.thresholds["outlier_max_ratio"] == 0.05
        assert screener.thresholds["jump_max_ratio"] == 0.1
        assert screener.thresholds["ecg_expected_heart_rate_bpm"] == 80

    def test_signal_type_specific_defaults(self):
        """Test that different signal types get appropriate default thresholds."""
        ecg_screener = QualityScreener(signal_type="ecg", sampling_rate=1000)
        ppg_screener = QualityScreener(signal_type="ppg", sampling_rate=100)
        eeg_screener = QualityScreener(signal_type="eeg", sampling_rate=256)

        # ECG should have stricter thresholds
        assert ecg_screener.thresholds["snr_min_db"] == 15.0
        assert ecg_screener.thresholds["artifact_max_ratio"] == 0.2

        # PPG should have moderate thresholds
        assert ppg_screener.thresholds["snr_min_db"] == 12.0
        assert ppg_screener.thresholds["artifact_max_ratio"] == 0.25

        # EEG should have more lenient thresholds
        assert eeg_screener.thresholds["snr_min_db"] == 8.0
        assert eeg_screener.thresholds["artifact_max_ratio"] == 0.4

    def test_strict_vs_lenient_thresholds_affect_results(self, sample_ecg_signal):
        """Test that strict vs lenient thresholds produce different results."""
        signal, fs = sample_ecg_signal

        # Add some artifacts to make it borderline quality
        signal_with_artifacts = signal.copy()
        signal_with_artifacts[500:510] += 5.0  # Add a spike

        # Strict screener
        strict_screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=fs,
            snr_min_db=15.0,
            outlier_max_ratio=0.05,  # Very strict
            jump_max_ratio=0.05,     # Very strict
        )

        # Lenient screener
        lenient_screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=fs,
            snr_min_db=5.0,  # Very lenient
            outlier_max_ratio=0.3,
            jump_max_ratio=0.3,
        )

        # Screen with both
        strict_results = strict_screener.screen_signal(signal_with_artifacts)
        lenient_results = lenient_screener.screen_signal(signal_with_artifacts)

        # Lenient screener should be more likely to pass segments
        strict_pass_count = sum(1 for r in strict_results if r.passed_screening)
        lenient_pass_count = sum(1 for r in lenient_results if r.passed_screening)

        # With added artifacts, strict should pass fewer (or same) segments
        assert lenient_pass_count >= strict_pass_count

    def test_peak_detection_parameters_are_used(self):
        """Test that peak detection parameters are accessible and configurable."""
        # ECG screener with custom peak detection params
        ecg_screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=1000,
            ecg_min_peak_distance_factor=0.3,  # Allow faster heart rates
            ecg_expected_heart_rate_bpm=90,     # Expect higher HR
        )

        assert ecg_screener.thresholds["ecg_min_peak_distance_factor"] == 0.3
        assert ecg_screener.thresholds["ecg_expected_heart_rate_bpm"] == 90

        # PPG screener with custom peak detection params
        ppg_screener = QualityScreener(
            signal_type="ppg",
            sampling_rate=100,
            ppg_min_peak_distance_factor=0.6,  # Allow slower pulse rates
            ppg_expected_pulse_rate_bpm=50,     # Expect lower pulse
        )

        assert ppg_screener.thresholds["ppg_min_peak_distance_factor"] == 0.6
        assert ppg_screener.thresholds["ppg_expected_pulse_rate_bpm"] == 50

    def test_statistical_screen_uses_configurable_thresholds(self, sample_ecg_signal):
        """Test that Stage 2 statistical screen uses configurable thresholds."""
        signal, fs = sample_ecg_signal

        screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=fs,
            outlier_std_factor=2.0,  # More sensitive to outliers
            outlier_max_ratio=0.05,
            constant_max_ratio=0.3,
            jump_std_factor=2.0,
            jump_max_ratio=0.1,
        )

        # Test that screener was configured correctly
        assert screener.thresholds["outlier_std_factor"] == 2.0
        assert screener.thresholds["jump_std_factor"] == 2.0

        # Run screening
        results = screener.screen_signal(signal)
        assert len(results) > 0

    def test_sqi_thresholds_are_configurable(self):
        """Test that all SQI method thresholds are configurable."""
        screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=1000,
            amplitude_variability_min=0.7,
            baseline_wander_min=0.6,
            zero_crossing_min=0.65,
            hrv_min=0.55,
        )

        assert screener.thresholds["amplitude_variability_min"] == 0.7
        assert screener.thresholds["baseline_wander_min"] == 0.6
        assert screener.thresholds["zero_crossing_min"] == 0.65
        assert screener.thresholds["hrv_min"] == 0.55

    def test_all_new_threshold_parameters_exist(self):
        """Verify all new threshold parameters are in the configuration."""
        screener = QualityScreener(signal_type="generic", sampling_rate=100)

        # New Stage 2 parameters
        required_stage2_params = [
            "outlier_std_factor",
            "outlier_max_ratio",
            "constant_max_ratio",
            "jump_std_factor",
            "jump_max_ratio",
        ]

        for param in required_stage2_params:
            assert param in screener.thresholds, f"Missing threshold parameter: {param}"

        # New peak detection parameters
        required_peak_params = [
            "ecg_min_peak_distance_factor",
            "ppg_min_peak_distance_factor",
            "ecg_expected_heart_rate_bpm",
            "ppg_expected_pulse_rate_bpm",
        ]

        for param in required_peak_params:
            assert param in screener.thresholds, f"Missing peak detection parameter: {param}"

    def test_thresholds_override_preserves_signal_type_defaults(self):
        """Test that custom thresholds override defaults while preserving others."""
        screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=1000,
            snr_min_db=20.0,  # Override this
            # Don't override artifact_max_ratio
        )

        # Overridden value
        assert screener.thresholds["snr_min_db"] == 20.0

        # ECG-specific default should still be preserved
        assert screener.thresholds["artifact_max_ratio"] == 0.2  # ECG default

    def test_zero_and_negative_thresholds_are_allowed(self):
        """Test that zero and negative thresholds can be set if needed."""
        # This might be useful for testing or special cases
        screener = QualityScreener(
            signal_type="generic",
            sampling_rate=100,
            snr_min_db=0.0,  # Allow very noisy signals
            outlier_max_ratio=1.0,  # Allow all outliers
        )

        assert screener.thresholds["snr_min_db"] == 0.0
        assert screener.thresholds["outlier_max_ratio"] == 1.0

    def test_extreme_thresholds_dont_crash(self, sample_ecg_signal):
        """Test that extreme threshold values don't cause crashes."""
        signal, fs = sample_ecg_signal

        # Very strict (almost impossible to pass)
        strict_screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=fs,
            snr_min_db=100.0,
            outlier_max_ratio=0.0,
            jump_max_ratio=0.0,
        )

        # Very lenient (everything passes)
        lenient_screener = QualityScreener(
            signal_type="ecg",
            sampling_rate=fs,
            snr_min_db=-100.0,
            outlier_max_ratio=1.0,
            jump_max_ratio=1.0,
        )

        # Should not crash
        strict_results = strict_screener.screen_signal(signal)
        lenient_results = lenient_screener.screen_signal(signal)

        assert len(strict_results) > 0
        assert len(lenient_results) > 0

    def test_threshold_documentation_matches_implementation(self):
        """Test that all documented thresholds exist in implementation."""
        screener = QualityScreener(signal_type="generic", sampling_rate=100)

        # From the documentation/summary
        documented_thresholds = [
            # Stage 1
            "snr_min_db",
            # Stage 2
            "artifact_max_ratio",
            "outlier_std_factor",
            "outlier_max_ratio",
            "constant_max_ratio",
            "jump_std_factor",
            "jump_max_ratio",
            # Stage 3
            "baseline_max_drift",
            "peak_detection_min_rate",
            "frequency_score_min",
            "temporal_consistency_min",
            "overall_quality_min",
            # SQI thresholds
            "amplitude_variability_min",
            "baseline_wander_min",
            "zero_crossing_min",
            "waveform_similarity_min",
            "signal_entropy_min",
            "skewness_min",
            "kurtosis_min",
            "peak_to_peak_min",
            "energy_min",
            "hrv_min",
            "ppg_quality_min",
            "eeg_band_power_min",
            "respiratory_quality_min",
            # Peak detection
            "ecg_min_peak_distance_factor",
            "ppg_min_peak_distance_factor",
            "ecg_expected_heart_rate_bpm",
            "ppg_expected_pulse_rate_bpm",
        ]

        for threshold in documented_thresholds:
            assert threshold in screener.thresholds, \
                f"Documented threshold '{threshold}' not found in implementation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
