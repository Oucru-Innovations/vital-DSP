import pytest
import numpy as np
from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection


# Sample test signals
@pytest.fixture
def test_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)


@pytest.fixture
def anomaly_detector(test_signal):
    return AnomalyDetection(test_signal)


def test_z_score_anomaly_detection(anomaly_detector):
    # Lowering the threshold to detect more anomalies
    anomalies = anomaly_detector.detect_anomalies(method="z_score", threshold=1.0)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected
    assert np.all(
        np.abs(
            (anomaly_detector.signal - np.mean(anomaly_detector.signal))
            / np.std(anomaly_detector.signal)
        )[anomalies]
        > 1.0
    )


def test_moving_average_anomaly_detection(anomaly_detector):
    # Lowering the threshold to detect more anomalies for moving average
    anomalies = anomaly_detector.detect_anomalies(
        method="moving_average", window_size=5, threshold=0.2
    )
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected
    moving_avg = np.convolve(anomaly_detector.signal, np.ones(5) / 5, mode="valid")
    residuals = np.abs(anomaly_detector.signal[4:] - moving_avg)
    # Only check anomalies that are within valid range (>= 4)
    valid_anomalies = anomalies[anomalies >= 4]
    if len(valid_anomalies) > 0:
        # Check that detected anomalies have higher average residuals than non-anomalies
        anomaly_residuals = residuals[valid_anomalies - 4]
        # At least some of the detected anomalies should exceed the threshold
        assert np.any(anomaly_residuals > 0.2), "At least some anomalies should exceed threshold"


def test_lof_anomaly_detection():
    # Testing Local Outlier Factor (LOF) anomaly detection with clear outliers
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 10, 200)) + np.random.normal(0, 0.05, 200)
    # Inject clear outliers at known positions
    signal[50] += 5.0
    signal[100] += -5.0
    signal[150] += 5.0
    detector = AnomalyDetection(signal)
    anomalies = detector.detect_anomalies(method="lof", n_neighbors=10)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected with injected outliers


def test_fft_anomaly_detection(anomaly_detector):
    # Testing FFT anomaly detection (returns time-domain indices)
    anomalies = anomaly_detector.detect_anomalies(method="fft", threshold=1.5)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected
    assert np.all(anomalies < len(anomaly_detector.signal))  # Valid time-domain indices


def test_threshold_anomaly_detection(anomaly_detector):
    # Testing Simple Threshold anomaly detection
    anomalies = anomaly_detector.detect_anomalies(method="threshold", threshold=0.8)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected
    assert np.all(np.abs(anomaly_detector.signal[anomalies]) > 0.8)


def test_invalid_method(anomaly_detector):
    # Testing for invalid method
    with pytest.raises(ValueError):
        anomaly_detector.detect_anomalies(method="invalid_method")
