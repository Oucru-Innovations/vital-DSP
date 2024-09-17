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
    assert np.all(residuals[anomalies - 4] > 0.2)


def test_lof_anomaly_detection(anomaly_detector):
    # Testing Local Outlier Factor (LOF) anomaly detection
    anomalies = anomaly_detector.detect_anomalies(method="lof", n_neighbors=10)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected


def test_fft_anomaly_detection(anomaly_detector):
    # Testing FFT anomaly detection
    anomalies = anomaly_detector.detect_anomalies(method="fft", threshold=1.5)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected
    fft_result = np.fft.fft(anomaly_detector.signal)
    assert np.any(np.abs(fft_result[anomalies]) > 1.5 * np.mean(np.abs(fft_result)))


def test_threshold_anomaly_detection(anomaly_detector):
    # Testing Simple Threshold anomaly detection
    anomalies = anomaly_detector.detect_anomalies(method="threshold", threshold=0.8)
    assert isinstance(anomalies, np.ndarray)
    assert anomalies.shape[0] > 0  # Ensure anomalies are detected
    assert np.all(anomaly_detector.signal[anomalies] > 0.8)


def test_invalid_method(anomaly_detector):
    # Testing for invalid method
    with pytest.raises(ValueError):
        anomaly_detector.detect_anomalies(method="invalid_method")
