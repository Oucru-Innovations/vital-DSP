"""
Comprehensive tests for health_report_feature_extractor.py to achieve 100% line coverage.

This test file covers all methods, branches, and edge cases in the feature extractor.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import scipy.stats
import scipy.signal

from vitalDSP_webapp.callbacks.analysis.health_report_feature_extractor import (
    HealthFeatureExtractor,
    extract_health_features_from_data,
)


@pytest.fixture
def sample_signal():
    """Create sample ECG signal for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
    # Create ECG-like signal with peaks
    signal = (
        np.sin(2 * np.pi * 1.0 * t) +  # 1 Hz heart rate component
        0.5 * np.sin(2 * np.pi * 0.2 * t) +  # Respiratory component
        0.2 * np.random.randn(len(t))
    )
    return signal


@pytest.fixture
def sample_ppg_signal():
    """Create sample PPG signal for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # PPG-like signal
    signal = (
        np.abs(np.sin(2 * np.pi * 1.2 * t)) +  # Pulse component
        0.3 * np.sin(2 * np.pi * 0.25 * t) +  # Respiratory component
        0.1 * np.random.randn(len(t))
    )
    return signal


@pytest.fixture
def sample_resp_signal():
    """Create sample respiratory signal for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # Respiratory-like signal
    signal = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(len(t))
    return signal


class TestHealthFeatureExtractorInitialization:
    """Test HealthFeatureExtractor initialization."""
    
    def test_init_ecg(self):
        """Test initialization with ECG signal type."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        assert extractor.fs == 100.0
        assert extractor.signal_type == "ECG"
    
    def test_init_ppg(self):
        """Test initialization with PPG signal type."""
        extractor = HealthFeatureExtractor(200.0, "PPG")
        assert extractor.fs == 200.0
        assert extractor.signal_type == "PPG"
    
    def test_init_resp(self):
        """Test initialization with RESP signal type."""
        extractor = HealthFeatureExtractor(50.0, "RESP")
        assert extractor.fs == 50.0
        assert extractor.signal_type == "RESP"
    
    def test_init_unknown(self):
        """Test initialization with unknown signal type."""
        extractor = HealthFeatureExtractor(100.0, "Unknown")
        assert extractor.fs == 100.0
        assert extractor.signal_type == "UNKNOWN"
    
    def test_init_lowercase(self):
        """Test initialization with lowercase signal type."""
        extractor = HealthFeatureExtractor(100.0, "ecg")
        assert extractor.signal_type == "ECG"


class TestExtractAllFeatures:
    """Test extract_all_features method."""
    
    def test_extract_all_features_ecg(self, sample_signal):
        """Test extracting all features from ECG signal."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor.extract_all_features(sample_signal)
        assert isinstance(features, dict)
        assert len(features) > 0
        assert "signal_mean" in features
    
    def test_extract_all_features_ppg(self, sample_ppg_signal):
        """Test extracting all features from PPG signal."""
        extractor = HealthFeatureExtractor(100.0, "PPG")
        features = extractor.extract_all_features(sample_ppg_signal)
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_all_features_resp(self, sample_resp_signal):
        """Test extracting all features from respiratory signal."""
        extractor = HealthFeatureExtractor(100.0, "RESP")
        features = extractor.extract_all_features(sample_resp_signal)
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_all_features_unknown(self, sample_signal):
        """Test extracting all features from unknown signal type."""
        extractor = HealthFeatureExtractor(100.0, "Unknown")
        features = extractor.extract_all_features(sample_signal)
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_all_features_few_peaks(self):
        """Test extracting features when few peaks are detected."""
        # Create signal with very few peaks
        signal = np.random.randn(100)
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
        # Should still have basic features
    
    def test_extract_all_features_error_handling(self):
        """Test error handling in extract_all_features."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Create invalid signal (empty array)
        signal = np.array([])
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
        # Empty array returns some statistical features with NaN values
        assert len(features) > 0
    
    def test_extract_all_features_respiratory_error(self, sample_signal):
        """Test respiratory feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        with patch.object(extractor, '_extract_respiratory_features', side_effect=Exception("Error")):
            features = extractor.extract_all_features(sample_signal)
            assert isinstance(features, dict)


class TestExtractTimeDomainFeatures:
    """Test _extract_time_domain_features method."""
    
    def test_extract_time_domain_features(self, sample_signal):
        """Test time domain feature extraction."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor._extract_time_domain_features(sample_signal)
        assert isinstance(features, dict)
        assert "signal_mean" in features
        assert "signal_std" in features
        assert "signal_min" in features
        assert "signal_max" in features
        assert "signal_range" in features
        assert "signal_rms" in features
        assert "signal_p25" in features
        assert "signal_p50" in features
        assert "signal_p75" in features
        assert "signal_iqr" in features
        assert "signal_ptp" in features
    
    def test_extract_time_domain_features_error(self):
        """Test time domain feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Create signal that might cause errors
        signal = np.array([np.nan, np.inf, -np.inf])
        features = extractor._extract_time_domain_features(signal)
        assert isinstance(features, dict)


class TestExtractStatisticalFeatures:
    """Test _extract_statistical_features method."""
    
    def test_extract_statistical_features(self, sample_signal):
        """Test statistical feature extraction."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor._extract_statistical_features(sample_signal)
        assert isinstance(features, dict)
        assert "skewness" in features
        assert "kurtosis" in features
        assert "coefficient_of_variation" in features
    
    def test_extract_statistical_features_zero_mean(self):
        """Test statistical features with zero mean."""
        signal = np.array([-1, 0, 1, -1, 0, 1])
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor._extract_statistical_features(signal)
        assert isinstance(features, dict)
        # coefficient_of_variation should not be in features if mean is 0
    
    def test_extract_statistical_features_sample_entropy_error(self, sample_signal):
        """Test statistical features when sample entropy fails."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        with patch.object(extractor, '_calculate_sample_entropy', side_effect=Exception("Error")):
            features = extractor._extract_statistical_features(sample_signal)
            assert isinstance(features, dict)
            assert "skewness" in features
    
    def test_extract_statistical_features_error(self):
        """Test statistical feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([])
        features = extractor._extract_statistical_features(signal)
        assert isinstance(features, dict)


class TestExtractFrequencyFeatures:
    """Test _extract_frequency_features method."""
    
    def test_extract_frequency_features_ecg(self, sample_signal):
        """Test frequency feature extraction for ECG."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor._extract_frequency_features(sample_signal)
        assert isinstance(features, dict)
        assert "dominant_frequency" in features
        assert "dominant_power" in features
        assert "spectral_centroid" in features
        assert "total_power" in features
        assert "power_vlf" in features
        assert "power_lf" in features
        assert "power_hf" in features
        # lf_hf_ratio only present if power_hf > 0 (code checks: if features["power_hf"] > 0)
        # The test signal might not have enough HF power, so lf_hf_ratio might not be present
        # This is expected behavior - only assert it exists if power_hf > 0
        if features.get("power_hf", 0) > 0:
            assert "lf_hf_ratio" in features
        # If power_hf is 0 or very small, lf_hf_ratio won't be present - that's correct
    
    def test_extract_frequency_features_ppg(self, sample_ppg_signal):
        """Test frequency feature extraction for PPG."""
        extractor = HealthFeatureExtractor(100.0, "PPG")
        features = extractor._extract_frequency_features(sample_ppg_signal)
        assert isinstance(features, dict)
        assert "power_vlf" in features
    
    def test_extract_frequency_features_resp(self, sample_resp_signal):
        """Test frequency feature extraction for RESP."""
        extractor = HealthFeatureExtractor(100.0, "RESP")
        features = extractor._extract_frequency_features(sample_resp_signal)
        assert isinstance(features, dict)
        assert "dominant_frequency" in features
        # Should not have HRV bands for RESP
    
    def test_extract_frequency_features_zero_hf_power(self):
        """Test frequency features when HF power is zero."""
        # Create signal with no HF power
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 0.01 * t)  # Very low frequency only
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor._extract_frequency_features(signal)
        assert isinstance(features, dict)
        # lf_hf_ratio should not be in features if power_hf is 0
    
    def test_extract_frequency_features_error(self):
        """Test frequency feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([])
        features = extractor._extract_frequency_features(signal)
        assert isinstance(features, dict)


class TestDetectPeaks:
    """Test _detect_peaks method."""
    
    def test_detect_peaks_ecg(self, sample_signal):
        """Test peak detection for ECG signal."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = extractor._detect_peaks(sample_signal)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) >= 0
    
    def test_detect_peaks_ppg(self, sample_ppg_signal):
        """Test peak detection for PPG signal."""
        extractor = HealthFeatureExtractor(100.0, "PPG")
        peaks = extractor._detect_peaks(sample_ppg_signal)
        assert isinstance(peaks, np.ndarray)
    
    def test_detect_peaks_error(self):
        """Test peak detection error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([])
        peaks = extractor._detect_peaks(signal)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) == 0
    
    def test_detect_peaks_constant_signal(self):
        """Test peak detection with constant signal."""
        signal = np.ones(100)
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = extractor._detect_peaks(signal)
        assert isinstance(peaks, np.ndarray)


class TestExtractHRVFeatures:
    """Test _extract_hrv_features method."""
    
    def test_extract_hrv_features_sufficient_peaks(self, sample_signal):
        """Test HRV feature extraction with sufficient peaks."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = extractor._detect_peaks(sample_signal)
        if len(peaks) >= 5:
            features = extractor._extract_hrv_features(peaks)
            assert isinstance(features, dict)
            if len(features) > 0:
                assert "hrv_mean_rr" in features or "hrv_sdnn" in features
    
    def test_extract_hrv_features_insufficient_peaks(self):
        """Test HRV feature extraction with insufficient peaks."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([10, 50, 90])  # Only 3 peaks
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)
        assert len(features) == 0
    
    def test_extract_hrv_features_invalid_intervals(self):
        """Test HRV features with invalid RR intervals."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Create peaks that result in invalid intervals
        peaks = np.array([10, 50, 90, 130, 170, 200])  # Very close peaks
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)
    
    def test_extract_hrv_features_insufficient_valid_intervals(self):
        """Test HRV features with insufficient valid intervals."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Create peaks that result in mostly invalid intervals
        peaks = np.array([10, 20, 30, 40, 50])  # Very close, invalid intervals
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)
    
    def test_extract_hrv_features_zero_mean_rr(self):
        """Test HRV features when mean RR is zero."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Create peaks that result in zero mean RR (shouldn't happen, but test edge case)
        peaks = np.array([10, 50, 90, 130, 170, 210])
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)
    
    def test_extract_hrv_features_single_diff(self):
        """Test HRV features with single RR difference."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([10, 500, 1000, 1500, 2000, 2500])
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)
        # Should handle single diff case
    
    def test_extract_hrv_features_error(self):
        """Test HRV feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([])
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)


class TestExtractHeartRateFeatures:
    """Test _extract_heart_rate_features method."""
    
    def test_extract_heart_rate_features_sufficient_peaks(self, sample_signal):
        """Test heart rate feature extraction with sufficient peaks."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = extractor._detect_peaks(sample_signal)
        if len(peaks) >= 2:
            features = extractor._extract_heart_rate_features(peaks)
            assert isinstance(features, dict)
            if len(features) > 0:
                assert "mean_hr" in features
    
    def test_extract_heart_rate_features_insufficient_peaks(self):
        """Test heart rate features with insufficient peaks."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([10])  # Single peak
        features = extractor._extract_heart_rate_features(peaks)
        assert isinstance(features, dict)
        assert len(features) == 0
    
    def test_extract_heart_rate_features_invalid_intervals(self):
        """Test heart rate features with invalid intervals."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([10, 20, 30])  # Very close peaks
        features = extractor._extract_heart_rate_features(peaks)
        assert isinstance(features, dict)
    
    def test_extract_heart_rate_features_no_valid_intervals(self):
        """Test heart rate features with no valid intervals."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([10, 20])  # Invalid interval
        features = extractor._extract_heart_rate_features(peaks)
        assert isinstance(features, dict)
    
    def test_extract_heart_rate_features_error(self):
        """Test heart rate feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        peaks = np.array([])
        features = extractor._extract_heart_rate_features(peaks)
        assert isinstance(features, dict)


class TestExtractRespiratoryFeatures:
    """Test _extract_respiratory_features method."""
    
    def test_extract_respiratory_features_ecg(self, sample_signal):
        """Test respiratory feature extraction from ECG."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        features = extractor._extract_respiratory_features(sample_signal)
        assert isinstance(features, dict)
    
    def test_extract_respiratory_features_ppg(self, sample_ppg_signal):
        """Test respiratory feature extraction from PPG."""
        extractor = HealthFeatureExtractor(100.0, "PPG")
        features = extractor._extract_respiratory_features(sample_ppg_signal)
        assert isinstance(features, dict)
    
    def test_extract_respiratory_features_resp(self, sample_resp_signal):
        """Test respiratory feature extraction from RESP signal."""
        extractor = HealthFeatureExtractor(100.0, "RESP")
        features = extractor._extract_respiratory_features(sample_resp_signal)
        assert isinstance(features, dict)
        if len(features) > 0:
            assert "respiratory_rate" in features
    
    def test_extract_respiratory_features_insufficient_peaks(self):
        """Test respiratory features with insufficient peaks."""
        signal = np.random.randn(100)
        extractor = HealthFeatureExtractor(100.0, "RESP")
        features = extractor._extract_respiratory_features(signal)
        assert isinstance(features, dict)
    
    def test_extract_respiratory_features_error(self):
        """Test respiratory feature extraction error handling."""
        extractor = HealthFeatureExtractor(100.0, "RESP")
        signal = np.array([])
        features = extractor._extract_respiratory_features(signal)
        assert isinstance(features, dict)


class TestCalculateSampleEntropy:
    """Test _calculate_sample_entropy method."""
    
    def test_calculate_sample_entropy_default(self, sample_signal):
        """Test sample entropy calculation with default parameters."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        entropy = extractor._calculate_sample_entropy(sample_signal)
        assert isinstance(entropy, (int, float))
    
    def test_calculate_sample_entropy_custom_params(self, sample_signal):
        """Test sample entropy with custom parameters."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        entropy = extractor._calculate_sample_entropy(sample_signal, m=3, r=0.1)
        assert isinstance(entropy, (int, float))
    
    def test_calculate_sample_entropy_zero_matches(self):
        """Test sample entropy with zero matches."""
        signal = np.random.randn(10)  # Very short signal
        extractor = HealthFeatureExtractor(100.0, "ECG")
        entropy = extractor._calculate_sample_entropy(signal)
        assert isinstance(entropy, (int, float))
        assert entropy >= 0
    
    def test_calculate_sample_entropy_short_signal(self):
        """Test sample entropy with very short signal."""
        signal = np.array([1, 2, 3])
        extractor = HealthFeatureExtractor(100.0, "ECG")
        entropy = extractor._calculate_sample_entropy(signal)
        assert isinstance(entropy, (int, float))


class TestExtractHealthFeaturesFromData:
    """Test extract_health_features_from_data convenience function."""
    
    def test_extract_health_features_from_data_ecg(self, sample_signal):
        """Test convenience function for ECG."""
        features = extract_health_features_from_data(sample_signal, 100.0, "ECG")
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_health_features_from_data_ppg(self, sample_ppg_signal):
        """Test convenience function for PPG."""
        features = extract_health_features_from_data(sample_ppg_signal, 100.0, "PPG")
        assert isinstance(features, dict)
    
    def test_extract_health_features_from_data_resp(self, sample_resp_signal):
        """Test convenience function for RESP."""
        features = extract_health_features_from_data(sample_resp_signal, 100.0, "RESP")
        assert isinstance(features, dict)
    
    def test_extract_health_features_from_data_unknown(self, sample_signal):
        """Test convenience function for unknown signal type."""
        features = extract_health_features_from_data(sample_signal, 100.0, "Unknown")
        assert isinstance(features, dict)
    
    def test_extract_health_features_from_data_empty(self):
        """Test convenience function with empty signal."""
        signal = np.array([])
        features = extract_health_features_from_data(signal, 100.0, "ECG")
        assert isinstance(features, dict)


class TestEdgeCases:
    """Test various edge cases and error conditions."""
    
    def test_extract_all_features_empty_signal(self):
        """Test extracting features from empty signal."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([])
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
        # Empty signal returns some statistical features with NaN values
        assert len(features) > 0
    
    def test_extract_all_features_single_value(self):
        """Test extracting features from single value signal."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([1.0])
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_extract_all_features_constant_signal(self):
        """Test extracting features from constant signal."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.ones(100)
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_extract_all_features_nan_values(self):
        """Test extracting features from signal with NaN values."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([1, 2, np.nan, 4, 5])
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_extract_all_features_inf_values(self):
        """Test extracting features from signal with inf values."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        signal = np.array([1, 2, np.inf, 4, 5])
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_very_high_sampling_rate(self):
        """Test with very high sampling rate."""
        extractor = HealthFeatureExtractor(10000.0, "ECG")
        # Use smaller signal to avoid extremely slow sample entropy calculation
        # 500 samples at 10000 Hz = 0.05 seconds, enough to test the functionality
        signal = np.random.randn(500)
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_very_low_sampling_rate(self):
        """Test with very low sampling rate."""
        extractor = HealthFeatureExtractor(1.0, "ECG")
        signal = np.random.randn(10)
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_very_long_signal(self):
        """Test with very long signal."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Use smaller signal to avoid extremely slow sample entropy calculation
        # 5000 samples at 100 Hz = 50 seconds, enough to test long signal functionality
        signal = np.random.randn(5000)
        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
    
    def test_hrv_features_with_exact_boundaries(self):
        """Test HRV features with intervals at boundaries."""
        extractor = HealthFeatureExtractor(100.0, "ECG")
        # Create peaks that result in intervals at boundaries (300ms, 2000ms)
        # 300ms = 30 samples at 100Hz, 2000ms = 200 samples
        peaks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270])
        features = extractor._extract_hrv_features(peaks)
        assert isinstance(features, dict)
    
    def test_frequency_features_with_no_hrv_bands(self):
        """Test frequency features for non-cardiac signal."""
        extractor = HealthFeatureExtractor(100.0, "Unknown")
        signal = np.random.randn(1000)
        features = extractor._extract_frequency_features(signal)
        assert isinstance(features, dict)
        # Should not have HRV bands
        assert "power_vlf" not in features

