import numpy as np

class AnomalyDetection:
    """
    Comprehensive Anomaly Detection for detecting anomalies in real-time from streaming data.

    Methods:
    - detect_anomalies: Detects anomalies using various methods including z-score, moving average, custom LOF, and more.

    Example Usage:
    --------------
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    anomaly_detector = AnomalyDetection(signal)
    anomalies_z_score = anomaly_detector.detect_anomalies(method="z_score", threshold=2.0)
    anomalies_moving_avg = anomaly_detector.detect_anomalies(method="moving_average", window_size=5, threshold=0.5)
    anomalies_lof = anomaly_detector.detect_anomalies(method="lof", n_neighbors=20)
    print("Anomalies (Z-Score):", anomalies_z_score)
    print("Anomalies (Moving Average):", anomalies_moving_avg)
    print("Anomalies (LOF):", anomalies_lof)
    """

    def __init__(self, signal):
        self.signal = signal

    def detect_anomalies(self, method="z_score", **kwargs):
        """
        Detect anomalies in the signal using the specified method.

        Parameters:
        method (str): The method to use for detecting anomalies. Options include "z_score", "moving_average", "lof", "fft", "threshold".
        kwargs: Additional parameters depending on the method.

        Returns:
        numpy.ndarray: Indices of the detected anomalies.
        """
        if method == "z_score":
            return self._z_score_anomaly_detection(kwargs.get('threshold', 3.0))
        elif method == "moving_average":
            return self._moving_average_anomaly_detection(kwargs.get('window_size', 5), kwargs.get('threshold', 1.0))
        elif method == "lof":
            return self._lof_anomaly_detection(kwargs.get('n_neighbors', 20))
        elif method == "fft":
            return self._fft_anomaly_detection(kwargs.get('threshold', 1.5))
        elif method == "threshold":
            return self._threshold_anomaly_detection(kwargs.get('threshold', 1.0))
        else:
            raise ValueError("Unknown method: {}".format(method))

    def _z_score_anomaly_detection(self, threshold):
        """
        Z-Score based anomaly detection.

        Parameters:
        threshold (float): The z-score threshold for anomaly detection.

        Returns:
        numpy.ndarray: Indices of the detected anomalies.
        """
        mean = np.mean(self.signal)
        std = np.std(self.signal)
        z_scores = (self.signal - mean) / std
        anomalies = np.where(np.abs(z_scores) > threshold)[0]
        return anomalies

    def _moving_average_anomaly_detection(self, window_size, threshold):
        """
        Moving average based anomaly detection.

        Parameters:
        window_size (int): The window size for the moving average.
        threshold (float): The threshold for detecting anomalies based on the deviation from the moving average.

        Returns:
        numpy.ndarray: Indices of the detected anomalies.
        """
        moving_avg = np.convolve(self.signal, np.ones(window_size) / window_size, mode='valid')
        residuals = np.abs(self.signal[window_size - 1:] - moving_avg)
        anomalies = np.where(residuals > threshold)[0] + window_size - 1
        return anomalies

    def _lof_anomaly_detection(self, n_neighbors):
        """
        Local Outlier Factor (LOF) based anomaly detection.

        Parameters:
        n_neighbors (int): Number of neighbors to use for LOF.

        Returns:
        numpy.ndarray: Indices of the detected anomalies.
        """
        n_points = len(self.signal)
        distances = np.zeros((n_points, n_points))
        
        # Compute distance matrix
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distances[i, j] = np.abs(self.signal[i] - self.signal[j])
                distances[j, i] = distances[i, j]

        # Sort distances for each point and compute reachability distance
        sorted_distances = np.sort(distances, axis=1)
        reachability_distances = np.zeros_like(distances)
        for i in range(n_points):
            reachability_distances[i, :] = np.maximum(distances[i, :], sorted_distances[i, n_neighbors])

        # Compute local reachability density (LRD)
        lrd = np.zeros(n_points)
        for i in range(n_points):
            lrd[i] = 1 / (np.mean(reachability_distances[i, np.argsort(distances[i, :])[:n_neighbors]]) + 1e-10)

        # Compute LOF
        lof = np.zeros(n_points)
        for i in range(n_points):
            lof[i] = np.mean(lrd[np.argsort(distances[i, :])[:n_neighbors]]) / lrd[i]

        # Anomalies are those points with LOF > 1 (indicating a potential outlier)
        anomalies = np.where(lof > 1)[0]
        return anomalies

    def _fft_anomaly_detection(self, threshold):
        """
        FFT based anomaly detection by analyzing the frequency domain.

        Parameters:
        threshold (float): The threshold for detecting anomalies based on frequency domain components.

        Returns:
        numpy.ndarray: Indices of the detected anomalies.
        """
        fft_result = np.fft.fft(self.signal)
        magnitude = np.abs(fft_result)
        anomalies = np.where(magnitude > threshold * np.mean(magnitude))[0]
        return anomalies

    def _threshold_anomaly_detection(self, threshold):
        """
        Simple threshold based anomaly detection.

        Parameters:
        threshold (float): The threshold value for detecting anomalies.

        Returns:
        numpy.ndarray: Indices of the detected anomalies.
        """
        anomalies = np.where(self.signal > threshold)[0]
        return anomalies
