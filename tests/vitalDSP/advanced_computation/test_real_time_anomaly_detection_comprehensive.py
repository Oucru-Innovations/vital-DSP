"""
Comprehensive Tests for RealTimeAnomalyDetection Module

This test file covers all edge cases, error conditions, and missing lines
to achieve high test coverage for real_time_anomaly_detection.py.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 95%+
"""

import pytest
import numpy as np
from vitalDSP.advanced_computation.real_time_anomaly_detection import (
    RealTimeAnomalyDetection,
    SimpleSVM,
    SimpleAutoencoder,
    SimpleLSTM,
)


@pytest.fixture
def detector():
    """Create a RealTimeAnomalyDetection instance."""
    return RealTimeAnomalyDetection(window_size=10)


@pytest.fixture
def training_data():
    """Generate training data."""
    np.random.seed(42)
    return np.random.randn(50)


@pytest.fixture
def test_data():
    """Generate test data."""
    np.random.seed(123)
    return np.random.randn(20)


class TestDetectStatistical:
    """Test detect_statistical method comprehensively."""

    def test_detect_statistical_z_score(self, detector):
        """Test Z-score detection method."""
        # Fill window first
        for i in range(10):
            detector.detect_statistical(i * 0.1, method='z_score', threshold=2.0)
        
        # Test with normal point
        result = detector.detect_statistical(0.5, method='z_score', threshold=2.0)
        assert isinstance(result, (bool, np.bool_))
        
        # Test with anomaly (very large value)
        result = detector.detect_statistical(100.0, method='z_score', threshold=2.0)
        assert bool(result) == True

    def test_detect_statistical_moving_average(self, detector):
        """Test moving average detection method - covers line 115."""
        # Fill window first
        for i in range(10):
            detector.detect_statistical(i * 0.1, method='moving_average', threshold=1.0)
        
        # Test with normal point
        result = detector.detect_statistical(0.5, method='moving_average', threshold=1.0)
        assert isinstance(result, (bool, np.bool_))
        
        # Test with anomaly
        result = detector.detect_statistical(100.0, method='moving_average', threshold=1.0)
        assert bool(result) == True

    def test_detect_statistical_unknown_method(self, detector):
        """Test ValueError for unknown statistical method - covers line 118."""
        # Fill window first
        for i in range(10):
            detector.detect_statistical(i * 0.1)
        
        with pytest.raises(ValueError, match="Unknown statistical method"):
            detector.detect_statistical(0.5, method='unknown_method')

    def test_detect_statistical_insufficient_data(self, detector):
        """Test that False is returned when window is not full."""
        # Add only a few points
        result = detector.detect_statistical(0.5)
        assert result == False
        
        # Add more points but still not enough
        for i in range(5):
            result = detector.detect_statistical(i * 0.1)
        assert result == False


class TestKNN:
    """Test k-NN methods."""

    def test_train_knn(self, detector, training_data):
        """Test training k-NN model."""
        detector.train_knn(training_data, k=5)
        assert "knn" in detector.models
        assert detector.models["knn"]["k"] == 5

    def test_detect_knn(self, detector, training_data):
        """Test k-NN detection."""
        detector.train_knn(training_data, k=5)
        result = detector.detect_knn(0.5)
        assert isinstance(result, (bool, np.bool_))

    def test_detect_knn_not_trained(self, detector):
        """Test error when k-NN not trained."""
        with pytest.raises(ValueError, match="k-NN model has not been trained"):
            detector.detect_knn(0.5)


class TestSVM:
    """Test SVM methods."""

    def test_train_svm(self, detector, training_data):
        """Test training SVM model."""
        detector.train_svm(training_data, kernel='rbf')
        assert "svm" in detector.models

    def test_train_svm_different_kernels(self, detector, training_data):
        """Test training SVM with different kernels."""
        for kernel in ['linear', 'poly', 'rbf']:
            detector.train_svm(training_data, kernel=kernel)
            assert detector.models["svm"].kernel == kernel

    def test_detect_svm(self, detector, training_data):
        """Test SVM detection."""
        detector.train_svm(training_data)
        result = detector.detect_svm(0.5)
        assert isinstance(result, (bool, np.bool_))

    def test_detect_svm_not_trained(self, detector):
        """Test error when SVM not trained."""
        with pytest.raises(ValueError, match="SVM model has not been trained"):
            detector.detect_svm(0.5)

    def test_simple_svm_1d_data(self):
        """Test SimpleSVM with 1D data."""
        data = np.random.randn(50)
        svm = SimpleSVM(data, kernel='rbf')
        assert svm.kernel == 'rbf'
        assert len(svm.support_vectors) > 0

    def test_simple_svm_2d_data(self):
        """Test SimpleSVM with 2D data - covers line 403-404."""
        data = np.random.randn(50, 3)  # 2D data
        svm = SimpleSVM(data, kernel='rbf')
        assert svm.kernel == 'rbf'
        assert len(svm.support_vectors) > 0

    def test_simple_svm_predict(self, training_data):
        """Test SimpleSVM prediction."""
        svm = SimpleSVM(training_data)
        result = svm.predict(0.5)
        assert isinstance(result, (bool, np.bool_))

    def test_simple_svm_update(self, training_data):
        """Test SimpleSVM update."""
        svm = SimpleSVM(training_data)
        original_len = len(svm.training_data)
        svm.update(1.5)
        assert len(svm.training_data) == original_len + 1


class TestAutoencoder:
    """Test Autoencoder methods."""

    def test_train_autoencoder_1d(self, detector):
        """Test training autoencoder with 1D data - covers line 439."""
        training_data = np.random.randn(50)  # 1D data
        detector.train_autoencoder(training_data, encoding_dim=3)
        assert "autoencoder" in detector.models

    def test_train_autoencoder_2d(self, detector):
        """Test training autoencoder with 2D data."""
        training_data = np.random.randn(50, 5)  # 2D data
        detector.train_autoencoder(training_data, encoding_dim=3)
        assert "autoencoder" in detector.models

    def test_detect_autoencoder(self, detector):
        """Test autoencoder detection."""
        training_data = np.random.randn(50, 3)
        detector.train_autoencoder(training_data, encoding_dim=2)
        result = detector.detect_autoencoder(np.array([1.0, 2.0, 3.0]), threshold=0.1)
        assert isinstance(result, (bool, np.bool_))

    def test_detect_autoencoder_not_trained(self, detector):
        """Test error when autoencoder not trained."""
        with pytest.raises(ValueError, match="Autoencoder model has not been trained"):
            detector.detect_autoencoder(np.array([1.0, 2.0]))

    def test_simple_autoencoder_1d_init(self):
        """Test SimpleAutoencoder with 1D training data - covers line 439."""
        training_data = np.random.randn(50)  # 1D
        ae = SimpleAutoencoder(training_data, encoding_dim=3)
        assert ae.training_data.ndim == 2  # Should be reshaped to 2D
        assert ae.training_data.shape[1] == 1

    def test_simple_autoencoder_2d_init(self):
        """Test SimpleAutoencoder with 2D training data."""
        training_data = np.random.randn(50, 5)  # 2D
        ae = SimpleAutoencoder(training_data, encoding_dim=3)
        assert ae.training_data.ndim == 2
        assert ae.training_data.shape[1] == 5

    def test_simple_autoencoder_reconstruction_error(self):
        """Test reconstruction error calculation."""
        training_data = np.random.randn(50, 3)
        ae = SimpleAutoencoder(training_data, encoding_dim=2)
        error = ae.reconstruction_error(np.array([1.0, 2.0, 3.0]))
        assert isinstance(error, float)
        assert error >= 0

    def test_simple_autoencoder_update_1d(self):
        """Test autoencoder update with 1D data point - covers line 487-488."""
        training_data = np.random.randn(50, 3)
        ae = SimpleAutoencoder(training_data, encoding_dim=2)
        original_len = len(ae.training_data)
        
        # Update with 1D array
        ae.update(np.array([1.0, 2.0, 3.0]))  # 1D array
        assert len(ae.training_data) == original_len + 1
        assert ae.training_data.ndim == 2

    def test_simple_autoencoder_update_2d(self):
        """Test autoencoder update with 2D data point."""
        training_data = np.random.randn(50, 3)
        ae = SimpleAutoencoder(training_data, encoding_dim=2)
        original_len = len(ae.training_data)
        
        # Update with 2D array
        ae.update(np.array([[1.0, 2.0, 3.0]]))  # 2D array
        assert len(ae.training_data) == original_len + 1


class TestLSTM:
    """Test LSTM methods."""

    def test_train_lstm(self, detector):
        """Test training LSTM model."""
        training_data = np.random.randn(50, 5)  # Must be 2D for LSTM
        detector.train_lstm(training_data, hidden_units=50)
        assert "lstm" in detector.models

    def test_detect_lstm(self, detector):
        """Test LSTM detection."""
        training_data = np.random.randn(50, 3)
        detector.train_lstm(training_data, hidden_units=20)
        result = detector.detect_lstm(np.array([1.0, 2.0, 3.0]), threshold=0.1)
        assert isinstance(result, (bool, np.bool_))

    def test_detect_lstm_not_trained(self, detector):
        """Test error when LSTM not trained."""
        with pytest.raises(ValueError, match="LSTM model has not been trained"):
            detector.detect_lstm(np.array([1.0, 2.0]))

    def test_simple_lstm_init(self):
        """Test SimpleLSTM initialization."""
        training_data = np.random.randn(50, 5)
        lstm = SimpleLSTM(training_data, hidden_units=30)
        assert lstm.hidden_units == 30
        assert lstm.W.shape == (5, 30)
        assert lstm.U.shape == (30, 30)
        assert lstm.V.shape == (30, 5)

    def test_simple_lstm_prediction_error(self):
        """Test LSTM prediction error."""
        training_data = np.random.randn(50, 3)
        lstm = SimpleLSTM(training_data, hidden_units=20)
        error = lstm.prediction_error(np.array([1.0, 2.0, 3.0]))
        assert isinstance(error, float)
        assert error >= 0

    def test_simple_lstm_update(self):
        """Test LSTM update - covers lines 514-516."""
        training_data = np.random.randn(50, 3)
        lstm = SimpleLSTM(training_data, hidden_units=20)
        
        # Store original weights
        W_orig = lstm.W.copy()
        U_orig = lstm.U.copy()
        V_orig = lstm.V.copy()
        
        # Update
        lstm.update(np.array([1.0, 2.0, 3.0]))
        
        # Weights should have changed
        assert not np.array_equal(lstm.W, W_orig)
        assert not np.array_equal(lstm.U, U_orig)
        assert not np.array_equal(lstm.V, V_orig)


class TestWavelet:
    """Test wavelet detection method."""

    def test_detect_wavelet(self, detector):
        """Test wavelet detection."""
        # Fill window first
        for i in range(10):
            detector.detect_wavelet(i * 0.1, wavelet_name='haar', level=1, threshold=0.1)
        
        result = detector.detect_wavelet(0.5)
        assert isinstance(result, (bool, np.bool_))

    def test_detect_wavelet_insufficient_data(self, detector):
        """Test wavelet detection with insufficient data."""
        result = detector.detect_wavelet(0.5)
        assert result == False


class TestUpdateModel:
    """Test update_model method comprehensively."""

    def test_update_model_knn(self, detector, training_data):
        """Test updating k-NN model."""
        detector.train_knn(training_data)
        original_len = len(detector.models["knn"]["training_data"])
        detector.update_model(1.5, model_type='knn')
        assert len(detector.models["knn"]["training_data"]) == original_len + 1

    def test_update_model_svm(self, detector, training_data):
        """Test updating SVM model."""
        detector.train_svm(training_data)
        original_len = len(detector.models["svm"].training_data)
        detector.update_model(1.5, model_type='svm')
        assert len(detector.models["svm"].training_data) == original_len + 1

    def test_update_model_autoencoder(self, detector):
        """Test updating autoencoder model."""
        training_data = np.random.randn(50, 3)
        detector.train_autoencoder(training_data)
        original_len = len(detector.models["autoencoder"].training_data)
        detector.update_model(np.array([1.0, 2.0, 3.0]), model_type='autoencoder')
        assert len(detector.models["autoencoder"].training_data) == original_len + 1

    def test_update_model_lstm(self, detector):
        """Test updating LSTM model - covers line 348."""
        training_data = np.random.randn(50, 3)
        detector.train_lstm(training_data)
        
        # Store original weights
        W_orig = detector.models["lstm"].W.copy()
        
        detector.update_model(np.array([1.0, 2.0, 3.0]), model_type='lstm')
        
        # Weights should have changed
        assert not np.array_equal(detector.models["lstm"].W, W_orig)

    def test_update_model_unknown_type(self, detector):
        """Test ValueError for unknown model type - covers line 351."""
        with pytest.raises(ValueError, match="Unknown model type"):
            detector.update_model(1.5, model_type='unknown')


class TestEvaluate:
    """Test evaluate method comprehensively."""

    def test_evaluate_knn(self, detector, training_data, test_data):
        """Test evaluation with k-NN."""
        detector.train_knn(training_data)
        accuracy = detector.evaluate(test_data, model_type='knn')
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_evaluate_svm(self, detector, training_data, test_data):
        """Test evaluation with SVM - covers line 374."""
        detector.train_svm(training_data)
        accuracy = detector.evaluate(test_data, model_type='svm')
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_evaluate_autoencoder(self, detector, test_data):
        """Test evaluation with autoencoder - covers line 376."""
        training_data = np.random.randn(50, 3)
        detector.train_autoencoder(training_data)
        accuracy = detector.evaluate(test_data, model_type='autoencoder')
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_evaluate_lstm(self, detector, test_data):
        """Test evaluation with LSTM - covers line 378."""
        training_data = np.random.randn(50, 3)
        detector.train_lstm(training_data)
        # Convert test_data to 2D for LSTM
        test_data_2d = test_data.reshape(-1, 1)
        accuracy = detector.evaluate(test_data_2d, model_type='lstm')
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_evaluate_unknown_type(self, detector, test_data):
        """Test ValueError for unknown model type in evaluate - covers line 380."""
        with pytest.raises(ValueError, match="Unknown model type"):
            detector.evaluate(test_data, model_type='unknown')


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_init_custom_window_size(self):
        """Test initialization with custom window size."""
        detector = RealTimeAnomalyDetection(window_size=20)
        assert detector.window_size == 20
        assert len(detector.data_window) == 0

    def test_data_window_maxlen(self, detector):
        """Test that data_window respects maxlen."""
        # Add more points than window_size
        for i in range(20):
            detector.detect_statistical(i * 0.1)
        
        # Window should only contain last window_size points
        assert len(detector.data_window) == detector.window_size

    def test_z_score_detection_zero_std(self, detector):
        """Test Z-score detection with zero standard deviation."""
        # Fill window with identical values
        for i in range(10):
            detector.detect_statistical(5.0)
        
        # This will cause division by zero, but should handle gracefully
        result = detector.detect_statistical(5.0, method='z_score', threshold=2.0)
        # Result may be False, True, NaN, or Inf depending on implementation
        assert isinstance(result, (bool, np.bool_)) or np.isnan(result) or np.isinf(result)

    def test_detect_knn_different_k(self, detector, training_data):
        """Test k-NN with different k values."""
        for k in [1, 3, 5, 10]:
            detector.train_knn(training_data, k=k)
            result = detector.detect_knn(0.5)
            assert isinstance(result, (bool, np.bool_))

    def test_detect_autoencoder_different_thresholds(self, detector):
        """Test autoencoder with different thresholds."""
        training_data = np.random.randn(50, 3)
        detector.train_autoencoder(training_data)
        
        for threshold in [0.01, 0.1, 0.5, 1.0]:
            result = detector.detect_autoencoder(np.array([1.0, 2.0, 3.0]), threshold=threshold)
            assert isinstance(result, (bool, np.bool_))

    def test_detect_lstm_different_thresholds(self, detector):
        """Test LSTM with different thresholds."""
        training_data = np.random.randn(50, 3)
        detector.train_lstm(training_data)
        
        for threshold in [0.01, 0.1, 0.5, 1.0]:
            result = detector.detect_lstm(np.array([1.0, 2.0, 3.0]), threshold=threshold)
            assert isinstance(result, (bool, np.bool_))

    def test_detect_wavelet_different_wavelets(self, detector):
        """Test wavelet detection with different wavelets."""
        # Fill window first
        for i in range(10):
            detector.detect_wavelet(i * 0.1, wavelet_name='haar')
        
        for wavelet in ['haar', 'db4', 'coif2']:
            try:
                result = detector.detect_wavelet(0.5, wavelet_name=wavelet, level=1, threshold=0.1)
                assert isinstance(result, (bool, np.bool_))
            except (ValueError, AttributeError):
                # Some wavelets may not be available
                pass

    def test_detect_wavelet_different_levels(self, detector):
        """Test wavelet detection with different decomposition levels."""
        # Fill window first
        for i in range(10):
            detector.detect_wavelet(i * 0.1)
        
        for level in [1, 2, 3]:
            try:
                result = detector.detect_wavelet(0.5, level=level, threshold=0.1)
                assert isinstance(result, (bool, np.bool_))
            except (ValueError, AttributeError):
                # Some levels may not be valid
                pass

    def test_simple_svm_different_kernels(self, training_data):
        """Test SimpleSVM with different kernels."""
        for kernel in ['linear', 'poly', 'rbf']:
            svm = SimpleSVM(training_data, kernel=kernel)
            assert svm.kernel == kernel

    def test_simple_autoencoder_different_encoding_dims(self):
        """Test SimpleAutoencoder with different encoding dimensions."""
        training_data = np.random.randn(50, 5)
        for encoding_dim in [1, 3, 5, 10]:
            ae = SimpleAutoencoder(training_data, encoding_dim=encoding_dim)
            assert ae.encoding_dim == encoding_dim
            assert ae.weights.shape == (5, encoding_dim)

    def test_simple_lstm_different_hidden_units(self):
        """Test SimpleLSTM with different hidden units."""
        training_data = np.random.randn(50, 5)
        for hidden_units in [10, 20, 50, 100]:
            lstm = SimpleLSTM(training_data, hidden_units=hidden_units)
            assert lstm.hidden_units == hidden_units
            assert lstm.W.shape == (5, hidden_units)
            assert lstm.U.shape == (hidden_units, hidden_units)
            assert lstm.V.shape == (hidden_units, 5)

    def test_evaluate_empty_test_data(self, detector, training_data):
        """Test evaluate with empty test data."""
        detector.train_knn(training_data)
        with pytest.raises((ZeroDivisionError, ValueError)):
            detector.evaluate(np.array([]), model_type='knn')

    def test_is_anomaly_placeholder(self, detector):
        """Test _is_anomaly placeholder method."""
        result = detector._is_anomaly(0.5)
        assert result == False  # Always returns False as placeholder

