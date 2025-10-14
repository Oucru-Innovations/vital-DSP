import numpy as np
import warnings
from ..utils.quality_performance.performance_monitoring import (
    monitor_analysis_operation,
)


class AnomalyDetection:
    """
    Comprehensive Anomaly Detection for detecting anomalies in real-time from streaming data.

    This class offers multiple methods to detect anomalies in a given signal, including statistical methods,
    moving averages, Local Outlier Factor (LOF), and Fourier-based methods.

    Methods
    -------
    detect_anomalies : function
        Detects anomalies using various methods including z-score, moving average, custom LOF, and more.

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
    >>>
    >>> # Example 1: Z-score anomaly detection
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    >>> anomaly_detector = AnomalyDetection(signal)
    >>> anomalies_z_score = anomaly_detector.detect_anomalies(method="z_score", threshold=2.0)
    >>> print(f"Z-score anomalies: {len(anomalies_z_score)}")
    >>>
    >>> # Example 2: Moving average anomaly detection
    >>> anomalies_moving_avg = anomaly_detector.detect_anomalies(method="moving_average", window_size=5, threshold=0.5)
    >>> print(f"Moving average anomalies: {len(anomalies_moving_avg)}")
    >>>
    >>> # Example 3: LOF anomaly detection
    >>> anomalies_lof = anomaly_detector.detect_anomalies(method="lof", n_neighbors=20)
    >>> print(f"LOF anomalies: {len(anomalies_lof)}")
    >>>
    >>> # Example 4: FFT-based anomaly detection
    >>> anomalies_fft = anomaly_detector.detect_anomalies(method="fft", threshold=0.1)
    >>> print(f"FFT anomalies: {len(anomalies_fft)}")
    """

    def __init__(self, signal):
        """
        Initialize the AnomalyDetection class with the given signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal on which anomaly detection is to be performed.
        """
        self.signal = signal

    @monitor_analysis_operation
    def detect_anomalies(self, method="z_score", **kwargs):
        """
        Detect anomalies in the signal using the specified method.

        Parameters
        ----------
        method : str, optional
            The method to use for detecting anomalies. Options: 'z_score', 'moving_average', 'lof', 'fft', 'threshold'.
            Default is 'z_score'.
        **kwargs : additional arguments
            Additional parameters depending on the chosen method.

        Returns
        -------
        numpy.ndarray
            Indices of the detected anomalies.

        Raises
        ------
        ValueError
            If the specified method is unknown.

        Examples
        --------
        >>> anomalies_z_score = anomaly_detector.detect_anomalies(method="z_score", threshold=2.0)
        >>> anomalies_moving_avg = anomaly_detector.detect_anomalies(method="moving_average", window_size=5, threshold=0.5)
        """
        if method == "z_score":
            return self._z_score_anomaly_detection(kwargs.get("threshold", 3.0))
        elif method == "moving_average":
            return self._moving_average_anomaly_detection(
                kwargs.get("window_size", 5), kwargs.get("threshold", 1.0)
            )
        elif method == "lof":
            return self._lof_anomaly_detection(kwargs.get("n_neighbors", 20))
        elif method == "fft":
            return self._fft_anomaly_detection(kwargs.get("threshold", 1.5))
        elif method == "threshold":
            return self._threshold_anomaly_detection(kwargs.get("threshold", 1.0))
        else:
            raise ValueError(f"Unknown method: {method}")

    def _z_score_anomaly_detection(self, threshold):
        """
        Z-Score based anomaly detection.

        Anomalies are detected by calculating the z-score for each data point and identifying those
        that exceed the specified threshold.

        Parameters
        ----------
        threshold : float
            The z-score threshold for anomaly detection.

        Returns
        -------
        numpy.ndarray
            Indices of the detected anomalies.

        Examples
        --------
        >>> anomalies = anomaly_detector._z_score_anomaly_detection(threshold=2.0)
        >>> print(anomalies)
        """
        mean = np.mean(self.signal)
        std = np.std(self.signal)
        # Avoid divide by zero warning
        if std > 0:
            z_scores = (self.signal - mean) / std
        else:
            z_scores = np.zeros_like(self.signal)  # If std is 0, all z-scores are 0
        anomalies = np.where(np.abs(z_scores) > threshold)[0]
        return anomalies

    def _moving_average_anomaly_detection(self, window_size, threshold):
        """
        Moving average based anomaly detection.

        Anomalies are detected by calculating the moving average of the signal and identifying points
        where the deviation from the moving average exceeds the specified threshold.

        Parameters
        ----------
        window_size : int
            The window size for the moving average.
        threshold : float
            The threshold for detecting anomalies based on the deviation from the moving average.

        Returns
        -------
        numpy.ndarray
            Indices of the detected anomalies.

        Examples
        --------
        >>> anomalies = anomaly_detector._moving_average_anomaly_detection(window_size=5, threshold=0.5)
        >>> print(anomalies)
        """
        moving_avg = np.convolve(
            self.signal, np.ones(window_size) / window_size, mode="valid"
        )
        residuals = np.abs(self.signal[window_size - 1 :] - moving_avg)
        anomalies = np.where(residuals > threshold)[0] + window_size - 1
        return anomalies

    def _lof_anomaly_detection(self, n_neighbors):
        """
        Local Outlier Factor (LOF) based anomaly detection - OPTIMIZED VERSION.

        LOF identifies anomalies by comparing the local density of a data point to that of its neighbors.
        Points with significantly lower density are considered anomalies.

        OPTIMIZATION: Uses spatial data structures for O(n log n) complexity instead of O(n²).

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to use for LOF.

        Returns
        -------
        numpy.ndarray
            Indices of the detected anomalies.

        Examples
        --------
        >>> anomalies = anomaly_detector._lof_anomaly_detection(n_neighbors=20)
        >>> print(anomalies)
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            # Fallback to original implementation if sklearn not available
            return self._lof_anomaly_detection_fallback(n_neighbors)

        n_points = len(self.signal)

        # Input validation
        if n_points < n_neighbors + 1:
            return np.array([])  # Not enough points for LOF

        # Create phase space (embedding dimension = 2)
        phase_space = np.column_stack((self.signal[:-1], self.signal[1:]))

        # Use spatial data structure for efficient neighbor search
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
        nbrs.fit(phase_space)

        # Find neighbors for each point
        distances, indices = nbrs.kneighbors(phase_space)

        # Compute reachability distances
        reachability_distances = np.zeros((len(phase_space), n_neighbors))
        for i in range(len(phase_space)):
            for j in range(n_neighbors):
                # Reachability distance is max of distance and k-distance
                reachability_distances[i, j] = max(
                    distances[i, j + 1],  # j+1 because we exclude the point itself
                    distances[indices[i, j + 1], n_neighbors],  # k-distance of neighbor
                )

        # Compute local reachability density (LRD)
        lrd = np.zeros(len(phase_space))
        for i in range(len(phase_space)):
            avg_reach_dist = np.mean(reachability_distances[i])
            lrd[i] = 1 / (avg_reach_dist + 1e-10)

        # Compute LOF
        lof = np.zeros(len(phase_space))
        for i in range(len(phase_space)):
            neighbor_lrd = lrd[indices[i, 1 : n_neighbors + 1]]  # Exclude self
            lof[i] = np.mean(neighbor_lrd) / lrd[i]

        # Anomalies are those points with LOF > 1 (indicating a potential outlier)
        anomalies = np.where(lof > 1)[0]
        return anomalies

    def _lof_anomaly_detection_fallback(self, n_neighbors):
        """
        Fallback LOF implementation for when sklearn is not available.
        Uses sampling to reduce computational complexity.
        """
        n_points = len(self.signal)

        if n_points < n_neighbors + 1:
            return np.array([])

        # Sample-based approach to reduce complexity
        sample_size = min(1000, n_points)  # Limit sample size
        if n_points > sample_size:
            # Random sampling
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_signal = self.signal[sample_indices]
        else:
            sample_signal = self.signal
            sample_indices = np.arange(n_points)

        # Create phase space
        phase_space = np.column_stack((sample_signal[:-1], sample_signal[1:]))

        # Compute distances only for sampled points
        n_sample = len(phase_space)
        distances = np.zeros((n_sample, n_sample))

        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                dist = np.linalg.norm(phase_space[i] - phase_space[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Rest of LOF computation on sampled data
        sorted_distances = np.sort(distances, axis=1)
        reachability_distances = np.zeros_like(distances)

        for i in range(n_sample):
            reachability_distances[i, :] = np.maximum(
                distances[i, :], sorted_distances[i, n_neighbors]
            )

        # Compute LRD
        lrd = np.zeros(n_sample)
        for i in range(n_sample):
            lrd[i] = 1 / (
                np.mean(
                    reachability_distances[i, np.argsort(distances[i, :])[:n_neighbors]]
                )
                + 1e-10
            )

        # Compute LOF
        lof = np.zeros(n_sample)
        for i in range(n_sample):
            lof[i] = np.mean(lrd[np.argsort(distances[i, :])[:n_neighbors]]) / lrd[i]

        # Find anomalies in sample
        sample_anomalies = np.where(lof > 1)[0]

        # Map back to original indices
        if n_points > sample_size:
            anomalies = sample_indices[sample_anomalies]
        else:
            anomalies = sample_anomalies

        return anomalies

    def _fft_anomaly_detection(self, threshold):
        """
        FFT based anomaly detection by analyzing the frequency domain.

        Anomalies are detected by transforming the signal to the frequency domain using FFT and
        identifying components whose magnitude exceeds the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for detecting anomalies based on frequency domain components.

        Returns
        -------
        numpy.ndarray
            Indices of the detected anomalies.

        Examples
        --------
        >>> anomalies = anomaly_detector._fft_anomaly_detection(threshold=1.5)
        >>> print(anomalies)
        """
        fft_result = np.fft.fft(self.signal)
        magnitude = np.abs(fft_result)
        anomalies = np.where(magnitude > threshold * np.mean(magnitude))[0]
        return anomalies

    def _threshold_anomaly_detection(self, threshold):
        """
        Simple threshold based anomaly detection.

        Anomalies are detected by identifying points in the signal that exceed the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for detecting anomalies.

        Returns
        -------
        numpy.ndarray
            Indices of the detected anomalies.

        Examples
        --------
        >>> anomalies = anomaly_detector._threshold_anomaly_detection(threshold=1.0)
        >>> print(anomalies)
        """
        anomalies = np.where(self.signal > threshold)[0]
        return anomalies
