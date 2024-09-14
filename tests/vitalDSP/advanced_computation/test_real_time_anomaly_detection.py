import pytest
import numpy as np
from collections import deque
from vitalDSP.transforms.wavelet_transform import WaveletTransform
from vitalDSP.advanced_computation.real_time_anomaly_detection import RealTimeAnomalyDetection, SimpleSVM, SimpleAutoencoder, SimpleLSTM

@pytest.fixture
def anomaly_detector():
    return RealTimeAnomalyDetection(window_size=5)

def test_statistical_detection_z_score(anomaly_detector):
    # Adjusting the data to ensure the last point is a clear anomaly
    data_stream = [1, 1, 1, 1, 10]  # Last point is a noticeable outlier with lower threshold
    anomalies = []
    for point in data_stream:
        anomaly = anomaly_detector.detect_statistical(point, method='z_score', threshold=1.0)
        anomalies.append(anomaly)
    assert anomalies.count(True) == 1  # Only the last point should be an anomaly

def test_statistical_detection_moving_average(anomaly_detector):
    # Adjust the threshold to ensure the last point is detected as an anomaly
    data_stream = [1, 1, 1, 1, 10]  # Last point is an anomaly
    anomalies = []
    for point in data_stream:
        anomaly = anomaly_detector.detect_statistical(point, method='moving_average', threshold=1.0)
        anomalies.append(anomaly)
    assert anomalies.count(True) == 1  # Only the last point should be flagged

def test_knn_training_and_detection(anomaly_detector):
    # Train the k-NN model with normal data and test a clear outlier
    training_data = np.array([1, 2, 3, 4, 5])
    anomaly_detector.train_knn(training_data)
    anomaly = anomaly_detector.detect_knn(100)  # Clear outlier
    assert anomaly == True  # Anomaly expected

def test_knn_detection_no_training(anomaly_detector):
    with pytest.raises(ValueError):
        anomaly_detector.detect_knn(100)  # Should raise error if k-NN model not trained

def test_svm_training_and_detection(anomaly_detector):
    # Train SVM with normal data and detect an outlier
    training_data = np.random.randn(100)
    anomaly_detector.train_svm(training_data)
    anomaly = anomaly_detector.detect_svm(10)  # Detecting an outlier in standard normal data
    assert anomaly == True  # Large values should be an anomaly

def test_svm_detection_no_training(anomaly_detector):
    with pytest.raises(ValueError):
        anomaly_detector.detect_svm(100)  # Should raise error if SVM not trained

def test_autoencoder_training_and_detection(anomaly_detector):
    # Train Autoencoder and detect an outlier
    training_data = np.random.randn(100, 3)
    anomaly_detector.train_autoencoder(training_data)
    anomaly = anomaly_detector.detect_autoencoder(np.random.randn(3) + 10)  # Outlier
    assert anomaly == True  # Expect large reconstruction error as anomaly

def test_autoencoder_detection_no_training(anomaly_detector):
    with pytest.raises(ValueError):
        anomaly_detector.detect_autoencoder(np.random.randn(3))  # Error if autoencoder not trained

def test_lstm_training_and_detection(anomaly_detector):
    # Train LSTM and detect an outlier
    training_data = np.random.randn(100, 3)
    anomaly_detector.train_lstm(training_data)
    anomaly = anomaly_detector.detect_lstm(np.random.randn(3) + 10)  # Clear anomaly
    assert anomaly == True  # Expect large prediction error as anomaly

def test_lstm_detection_no_training(anomaly_detector):
    with pytest.raises(ValueError):
        anomaly_detector.detect_lstm(np.random.randn(3))  # Error if LSTM not trained

def test_wavelet_detection(anomaly_detector):
    # Test wavelet detection with a normal signal followed by a clear anomaly
    data_stream = np.concatenate([np.sin(np.linspace(0, 10, 5)), [50]])  # Last point is a large outlier
    anomalies = []
    for point in data_stream:
        anomaly = anomaly_detector.detect_wavelet(point, wavelet_name='haar', level=1, threshold=10.0)
        anomalies.append(anomaly)
    assert anomalies.count(True) == 1  # Only the last point should be detected as an anomaly

def test_update_model_knn(anomaly_detector):
    # Test k-NN model update with new data point
    training_data = np.array([1, 2, 3, 4, 5])
    anomaly_detector.train_knn(training_data)
    anomaly_detector.update_model(6, model_type='knn')
    assert 6 in anomaly_detector.models['knn']['training_data']

def test_update_model_svm(anomaly_detector):
    # Test SVM model update with new data point
    training_data = np.array([1, 2, 3, 4, 5])
    anomaly_detector.train_svm(training_data)
    anomaly_detector.update_model(6, model_type='svm')
    # Assert that the SVM model has been updated with the new data
    assert hasattr(anomaly_detector.models['svm'], 'training_data'), "SVM model should have 'training_data' attribute"
    assert 6 in anomaly_detector.models['svm'].training_data, "New data point should be in the SVM training data"

def test_update_model_autoencoder(anomaly_detector):
    """
    Test AutoEncoder model update with a new data point.
    """
    training_data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshape to 2D array
    anomaly_detector.train_autoencoder(training_data)

    new_data_point = np.array([6]).reshape(1, -1)  # Ensure new data point is 2D
    anomaly_detector.update_model(new_data_point, model_type='autoencoder')

    # Assert that the training data includes the new data point
    autoencoder_model = anomaly_detector.models['autoencoder']
    assert autoencoder_model.training_data[-1] == new_data_point, "The new data point should be appended."


def test_update_model_invalid(anomaly_detector):
    # Test invalid model type during update
    with pytest.raises(ValueError):
        anomaly_detector.update_model(6, model_type='invalid')

def test_evaluate_knn(anomaly_detector):
    # Test evaluation of k-NN model
    training_data = np.array([1, 2, 3, 4, 5])
    anomaly_detector.train_knn(training_data)
    test_data = np.array([1, 2, 3, 100])  # 100 is a clear anomaly
    accuracy = anomaly_detector.evaluate(test_data, model_type='knn')
    assert 0 <= accuracy <= 1  # Ensure accuracy is a valid percentage

def test_is_anomaly(anomaly_detector):
    result = anomaly_detector._is_anomaly(100)
    assert result is False  # Default implementation should return False
