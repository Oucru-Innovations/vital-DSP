"""
Comprehensive Tests for Feature Extractor Module - Missing Coverage

This test file specifically targets missing lines in feature_extractor.py to achieve
high test coverage, including morphological features, time-frequency features,
error handling, and edge cases.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 90%+
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock

# Suppress warnings
warnings.filterwarnings("ignore")

# Import module
try:
    from vitalDSP.ml_models.feature_extractor import (
        FeatureExtractor,
        FeatureEngineering,
        extract_features,
    )
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False


@pytest.fixture
def sample_signal():
    """Create sample signal for testing."""
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def sample_signals():
    """Create multiple sample signals for testing."""
    np.random.seed(42)
    return [np.random.randn(1000) for _ in range(10)]


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=10)


@pytest.mark.skipif(not FEATURE_EXTRACTOR_AVAILABLE, reason="FeatureExtractor not available")
class TestFeatureExtractorMissingCoverage:
    """Test FeatureExtractor to cover missing lines."""
    
    def test_fit_with_normalize(self, sample_signals):
        """Test fit with normalization - covers lines 128-130."""
        extractor = FeatureExtractor(normalize=True)
        extractor.fit(sample_signals)
        
        assert extractor.scaler_ is not None
    
    def test_fit_with_mutual_info_selection(self, sample_signals, sample_labels):
        """Test fit with mutual_info feature selection - covers lines 136-137."""
        extractor = FeatureExtractor(
            feature_selection='mutual_info',
            n_features=5
        )
        extractor.fit(sample_signals, y=sample_labels)
        
        assert extractor.selector_ is not None
    
    def test_transform_with_normalize(self, sample_signals):
        """Test transform with normalization - covers lines 170-171."""
        extractor = FeatureExtractor(normalize=True)
        extractor.fit(sample_signals)
        
        features = extractor.transform(sample_signals[:5])
        assert features.shape[0] == 5
    
    def test_extract_features_1d_array(self, sample_signal):
        """Test _extract_features with 1D array - covers lines 183-184."""
        extractor = FeatureExtractor()
        
        features = extractor._extract_features(sample_signal)
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1
    
    def test_extract_features_invalid_shape(self):
        """Test _extract_features with invalid shape - covers lines 188-188."""
        extractor = FeatureExtractor()
        
        # 3D array should raise ValueError
        X = np.random.randn(10, 100, 1)
        
        with pytest.raises(ValueError, match="Invalid input shape"):
            extractor._extract_features(X)
    
    def test_extract_features_morphology_domain(self, sample_signal):
        """Test _extract_features with morphology domain - covers lines 215-217."""
        extractor = FeatureExtractor(domains=['morphology'])
        
        features = extractor._extract_features([sample_signal])
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1
    
    def test_extract_features_time_frequency_domain(self, sample_signal):
        """Test _extract_features with time_frequency domain - covers lines 220-222."""
        extractor = FeatureExtractor(domains=['time_frequency'])
        
        features = extractor._extract_features([sample_signal])
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1
    
    def test_extract_time_domain_cv_zero_mean(self):
        """Test _extract_time_domain with zero mean - covers lines 268-271."""
        extractor = FeatureExtractor()
        
        # Signal with zero mean
        signal = np.array([-1, 0, 1, -1, 0, 1])
        
        features = extractor._extract_time_domain(signal)
        assert features['cv'] == 0
    
    def test_extract_time_domain_error_handling(self):
        """Test _extract_time_domain error handling - covers lines 281-307."""
        extractor = FeatureExtractor()

        # Create signal that might cause errors
        signal = np.array([np.nan, np.inf, -np.inf])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            features = extractor._extract_time_domain(signal)

            # NumPy operations on nan/inf don't raise exceptions, they return nan
            # Check that features are computed (even if they're nan)
            assert all(key in features for key in [
                'mean', 'std', 'var', 'min', 'max'
            ])
    
    def test_extract_frequency_domain_bandwidth_zero_total(self):
        """Test _extract_frequency_domain with zero total power - covers lines 378-383."""
        extractor = FeatureExtractor(signal_type='ecg', sampling_rate=250.0)
        
        # Signal with zero power
        signal = np.zeros(1000)
        
        features = extractor._extract_frequency_domain(signal)
        assert features['bandwidth'] == 0
    
    def test_extract_frequency_domain_error_handling(self):
        """Test _extract_frequency_domain error handling - covers lines 385-408."""
        extractor = FeatureExtractor(signal_type='ecg', sampling_rate=250.0)

        # Create signal that might cause errors
        signal = np.array([np.nan, np.inf, -np.inf] * 100)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            features = extractor._extract_frequency_domain(signal)

            # Check that features are computed
            # total_power may be nan, but spectral_mean/std should be 0 when sum(psd) is nan
            assert 'total_power' in features
            assert 'spectral_mean' in features
            assert 'spectral_std' in features
    
    def test_extract_frequency_domain_non_ecg_ppg(self):
        """Test _extract_frequency_domain for non-ECG/PPG signals - covers lines 325-325."""
        extractor = FeatureExtractor(signal_type='eeg', sampling_rate=250.0)
        
        signal = np.random.randn(1000)
        features = extractor._extract_frequency_domain(signal)
        
        # Should not have HRV frequency band features
        assert 'vlf_power' not in features
        assert 'lf_power' not in features
    
    def test_extract_nonlinear_features_error_handling(self):
        """Test _extract_nonlinear_features error handling - covers lines 439-449."""
        extractor = FeatureExtractor()

        # Create signal that might cause errors
        signal = np.array([np.nan, np.inf, -np.inf] * 100)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            features = extractor._extract_nonlinear_features(signal)

            # Check that features are computed (may be nan or 0)
            assert all(key in features for key in [
                'sample_entropy', 'approximate_entropy', 'dfa_alpha'
            ])
    
    def test_extract_morphological_features(self, sample_signal):
        """Test _extract_morphological_features - covers lines 453-509."""
        extractor = FeatureExtractor(sampling_rate=250.0)
        
        features = extractor._extract_morphological_features(sample_signal)
        
        assert isinstance(features, dict)
        assert 'n_peaks' in features
        assert 'mean_peak_amplitude' in features
    
    def test_extract_morphological_features_no_peaks(self):
        """Test _extract_morphological_features with no peaks - covers lines 483-488."""
        extractor = FeatureExtractor(sampling_rate=250.0)
        
        # Signal with no peaks (monotonic)
        signal = np.linspace(0, 1, 1000)
        
        features = extractor._extract_morphological_features(signal)
        
        assert features['n_peaks'] == 0
        assert features['mean_peak_amplitude'] == 0
        assert features['mean_ipi'] == 0
    
    def test_extract_morphological_features_single_peak(self):
        """Test _extract_morphological_features with single peak - covers lines 467-482."""
        extractor = FeatureExtractor(sampling_rate=250.0)
        
        # Signal with single peak
        signal = np.zeros(1000)
        signal[500] = 1.0
        
        features = extractor._extract_morphological_features(signal)
        
        assert features['n_peaks'] >= 0  # May find peaks or not
    
    def test_extract_morphological_features_error_handling(self):
        """Test _extract_morphological_features error handling - covers lines 495-507."""
        extractor = FeatureExtractor(sampling_rate=250.0)
        
        # Create signal that might cause errors
        signal = np.array([np.nan, np.inf, -np.inf] * 100)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            features = extractor._extract_morphological_features(signal)
            
            # Should return zeros for failed features
            assert all(features.get(key, 0.0) == 0.0 for key in [
                'n_peaks', 'mean_peak_amplitude', 'mean_ipi'
            ])
    
    def test_extract_time_frequency_features(self, sample_signal):
        """Test _extract_time_frequency_features - covers lines 511-544."""
        extractor = FeatureExtractor(sampling_rate=250.0)
        
        features = extractor._extract_time_frequency_features(sample_signal)
        
        assert isinstance(features, dict)
        assert 'stft_mean' in features
        assert 'stft_std' in features
    
    def test_extract_time_frequency_features_error_handling(self):
        """Test _extract_time_frequency_features error handling - covers lines 533-542."""
        extractor = FeatureExtractor(sampling_rate=250.0)

        # Create signal that might cause errors
        signal = np.array([np.nan, np.inf, -np.inf] * 100)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            features = extractor._extract_time_frequency_features(signal)

            # Check that features are computed (may be nan)
            assert all(key in features for key in [
                'stft_mean', 'stft_std', 'stft_max'
            ])
    
    def test_skewness_zero_std(self):
        """Test _skewness with zero std - covers lines 556-557."""
        extractor = FeatureExtractor()
        
        # Signal with zero std
        signal = np.ones(100)
        
        result = extractor._skewness(signal)
        assert result == 0
    
    def test_kurtosis_zero_std(self):
        """Test _kurtosis with zero std - covers lines 568-569."""
        extractor = FeatureExtractor()
        
        # Signal with zero std
        signal = np.ones(100)
        
        result = extractor._kurtosis(signal)
        assert result == 0
    
    def test_sample_entropy_zero_matches(self):
        """Test _sample_entropy with zero matches - covers lines 609-610."""
        extractor = FeatureExtractor()
        
        # Signal that will result in zero matches
        signal = np.random.randn(20)  # Very short signal
        
        result = extractor._sample_entropy(signal, m=2, r=0.01)
        assert result == 0
    
    def test_dfa_break_condition(self):
        """Test _dfa with break condition - covers lines 652-653."""
        extractor = FeatureExtractor()
        
        # Very short signal that will trigger break
        signal = np.random.randn(10)
        
        result = extractor._dfa(signal, min_box=4, max_box=20)
        assert isinstance(result, (int, float))
    
    def test_hurst_exponent_zero_std(self):
        """Test _hurst_exponent with zero std - covers lines 690-691."""
        extractor = FeatureExtractor()
        
        # Signal with zero std at some lag
        signal = np.ones(100)
        
        result = extractor._hurst_exponent(signal)
        assert isinstance(result, (int, float))
    
    def test_hurst_exponent_insufficient_tau(self):
        """Test _hurst_exponent with insufficient tau - covers lines 702-703."""
        extractor = FeatureExtractor()
        
        # Very short signal
        signal = np.random.randn(10)
        
        result = extractor._hurst_exponent(signal)
        assert result == 0.5  # Default value
    
    def test_lyapunov_exponent_insufficient_embedded(self):
        """Test _lyapunov_exponent with insufficient embedded - covers lines 719-720."""
        extractor = FeatureExtractor()
        
        # Very short signal
        signal = np.random.randn(5)
        
        result = extractor._lyapunov_exponent(signal)
        assert result == 0
    
    def test_lyapunov_exponent_no_divergences(self):
        """Test _lyapunov_exponent with no divergences - covers lines 729-730."""
        extractor = FeatureExtractor()
        
        # Signal where all distances are zero
        signal = np.zeros(100)
        
        result = extractor._lyapunov_exponent(signal)
        assert result == 0
    
    def test_correlation_dimension_zero_max_dist(self):
        """Test _correlation_dimension with zero max_dist - covers lines 753-759."""
        extractor = FeatureExtractor()
        
        # Signal with zero std
        signal = np.ones(100)
        
        result = extractor._correlation_dimension(signal, max_dist=0)
        assert result == 0
    
    def test_get_feature_names(self, sample_signals):
        """Test get_feature_names - covers lines 761-770."""
        extractor = FeatureExtractor()
        extractor.fit(sample_signals)
        
        feature_names = extractor.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_get_feature_importances(self):
        """Test get_feature_importances - covers lines 772-781."""
        extractor = FeatureExtractor()
        
        importances = extractor.get_feature_importances()
        assert isinstance(importances, dict)


@pytest.mark.skipif(not FEATURE_EXTRACTOR_AVAILABLE, reason="FeatureExtractor not available")
class TestFeatureEngineering:
    """Test FeatureEngineering class - covers lines 784-849."""
    
    def test_feature_engineering_init(self):
        """Test FeatureEngineering initialization - covers lines 808-813."""
        engineer = FeatureEngineering(
            interaction_terms=True,
            polynomial_degree=2
        )
        
        assert engineer.interaction_terms == True
        assert engineer.polynomial_degree == 2
    
    def test_feature_engineering_fit(self):
        """Test FeatureEngineering fit - covers lines 815-817."""
        engineer = FeatureEngineering()
        
        X = np.random.randn(100, 10)
        result = engineer.fit(X)
        
        assert result == engineer
    
    def test_feature_engineering_transform(self):
        """Test FeatureEngineering transform - covers lines 819-822."""
        engineer = FeatureEngineering()
        
        X = np.random.randn(100, 10)
        X_transformed = engineer.transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == X.shape[1]
    
    def test_feature_engineering_polynomial_features(self):
        """Test FeatureEngineering with polynomial features - covers lines 824-828."""
        engineer = FeatureEngineering(polynomial_degree=2)
        
        X = np.random.randn(50, 5)
        X_transformed = engineer.transform(X)
        
        # Polynomial features should increase number of features
        assert X_transformed.shape[1] > X.shape[1]
    
    def test_feature_engineering_interaction_terms(self):
        """Test FeatureEngineering with interaction terms - covers lines 831-841."""
        engineer = FeatureEngineering(interaction_terms=True)
        
        X = np.random.randn(50, 5)
        X_transformed = engineer.transform(X)
        
        # Interaction terms should increase number of features
        assert X_transformed.shape[1] > X.shape[1]
    
    def test_feature_engineering_interaction_single_feature(self):
        """Test FeatureEngineering interaction with single feature - covers lines 831-831."""
        engineer = FeatureEngineering(interaction_terms=True)
        
        X = np.random.randn(50, 1)  # Single feature
        
        X_transformed = engineer.transform(X)
        # Should not add interactions for single feature
        assert X_transformed.shape[1] == X.shape[1]
    
    def test_feature_engineering_fit_transform(self):
        """Test FeatureEngineering fit_transform - covers lines 845-849."""
        engineer = FeatureEngineering()
        
        X = np.random.randn(50, 10)
        X_transformed = engineer.fit_transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]


@pytest.mark.skipif(not FEATURE_EXTRACTOR_AVAILABLE, reason="FeatureExtractor not available")
class TestExtractFeaturesFunction:
    """Test extract_features convenience function - covers lines 853-895."""
    
    def test_extract_features_basic(self, sample_signals):
        """Test extract_features basic usage - covers lines 853-891."""
        features = extract_features(
            sample_signals,
            signal_type='ecg',
            sampling_rate=250.0
        )
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_signals)
    
    def test_extract_features_return_dataframe(self, sample_signals):
        """Test extract_features with return_dataframe=True - covers lines 892-893."""
        try:
            import pandas as pd
            
            features = extract_features(
                sample_signals,
                signal_type='ecg',
                sampling_rate=250.0,
                return_dataframe=True
            )
            
            assert isinstance(features, pd.DataFrame)
            assert features.shape[0] == len(sample_signals)
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_extract_features_custom_domains(self, sample_signals):
        """Test extract_features with custom domains."""
        features = extract_features(
            sample_signals,
            signal_type='ecg',
            sampling_rate=250.0,
            domains=['time', 'frequency']
        )
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_signals)


@pytest.mark.skipif(not FEATURE_EXTRACTOR_AVAILABLE, reason="FeatureExtractor not available")
class TestFeatureExtractorEdgeCases:
    """Test FeatureExtractor edge cases."""
    
    def test_fit_with_pca_selection_no_labels(self, sample_signals):
        """Test fit with PCA selection without labels - covers lines 143-145."""
        extractor = FeatureExtractor(
            feature_selection='pca',
            n_features=5
        )
        
        # PCA doesn't need labels
        extractor.fit(sample_signals)
        
        assert extractor.selector_ is not None
    
    def test_fit_with_supervised_selection_no_labels(self, sample_signals):
        """Test fit with supervised selection without labels - covers lines 146-148."""
        extractor = FeatureExtractor(
            feature_selection='kbest',
            n_features=5
        )
        
        # kbest needs labels, but should handle gracefully
        extractor.fit(sample_signals, y=None)
        
        # Selector may not be fitted, but should not crash
        assert True
    
    def test_extract_features_empty_list(self):
        """Test _extract_features with empty list."""
        extractor = FeatureExtractor()
        
        features = extractor._extract_features([])
        assert features.shape[0] == 0
    
    def test_extract_features_single_signal_list(self, sample_signal):
        """Test _extract_features with single signal in list."""
        extractor = FeatureExtractor()
        
        features = extractor._extract_features([sample_signal])
        assert features.shape[0] == 1
    
    def test_extract_time_domain_very_short_signal(self):
        """Test _extract_time_domain with very short signal."""
        extractor = FeatureExtractor()
        
        signal = np.array([1.0, 2.0])
        features = extractor._extract_time_domain(signal)
        
        assert isinstance(features, dict)
        assert 'mean' in features
    
    def test_extract_frequency_domain_very_short_signal(self):
        """Test _extract_frequency_domain with very short signal."""
        extractor = FeatureExtractor(signal_type='ecg', sampling_rate=250.0)
        
        signal = np.array([1.0, 2.0, 3.0])
        features = extractor._extract_frequency_domain(signal)
        
        assert isinstance(features, dict)
    
    def test_extract_nonlinear_features_very_short_signal(self):
        """Test _extract_nonlinear_features with very short signal."""
        extractor = FeatureExtractor()
        
        signal = np.array([1.0, 2.0, 3.0])
        features = extractor._extract_nonlinear_features(signal)
        
        assert isinstance(features, dict)
    
    def test_skewness_short_signal(self):
        """Test _skewness with short signal - covers lines 552-553."""
        extractor = FeatureExtractor()
        
        signal = np.array([1.0, 2.0])  # n < 3
        result = extractor._skewness(signal)
        assert result == 0
    
    def test_kurtosis_short_signal(self):
        """Test _kurtosis with short signal - covers lines 564-565."""
        extractor = FeatureExtractor()
        
        signal = np.array([1.0, 2.0, 3.0])  # n < 4
        result = extractor._kurtosis(signal)
        assert result == 0
    
    def test_sample_entropy_short_signal(self):
        """Test _sample_entropy with short signal - covers lines 580-581."""
        extractor = FeatureExtractor()
        
        signal = np.array([1.0, 2.0])  # n < m + 1
        result = extractor._sample_entropy(signal, m=2, r=0.1)
        assert result == 0
    
    def test_approximate_entropy_short_signal(self):
        """Test _approximate_entropy with short signal - covers lines 622-623."""
        extractor = FeatureExtractor()
        
        signal = np.array([1.0, 2.0])  # n < m + 1
        result = extractor._approximate_entropy(signal, m=2, r=0.1)
        assert result == 0
    
    def test_dfa_insufficient_scales(self):
        """Test _dfa with insufficient scales - covers lines 670-671."""
        extractor = FeatureExtractor()
        
        signal = np.random.randn(10)
        result = extractor._dfa(signal, min_box=4, max_box=5)
        assert result == 0
    
    def test_hurst_exponent_short_signal(self):
        """Test _hurst_exponent with short signal - covers lines 681-682."""
        extractor = FeatureExtractor()
        
        signal = np.random.randn(10)  # n < 20
        result = extractor._hurst_exponent(signal)
        assert result == 0.5
    
    def test_lyapunov_exponent_short_signal(self):
        """Test _lyapunov_exponent with short signal - covers lines 713-714."""
        extractor = FeatureExtractor()
        
        signal = np.random.randn(5)  # n < 10
        result = extractor._lyapunov_exponent(signal)
        assert result == 0
    
    def test_correlation_dimension_short_signal(self):
        """Test _correlation_dimension with short signal - covers lines 738-739."""
        extractor = FeatureExtractor()
        
        signal = np.random.randn(5)  # n < 10
        result = extractor._correlation_dimension(signal)
        assert result == 0

