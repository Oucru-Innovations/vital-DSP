import numpy as np


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
        z_scores = (self.signal - mean) / std
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
        Local Outlier Factor (LOF) based anomaly detection.

        LOF identifies anomalies by comparing the local density of a data point to that of its neighbors.
        Points with significantly lower density are considered anomalies.

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
            reachability_distances[i, :] = np.maximum(
                distances[i, :], sorted_distances[i, n_neighbors]
            )

        # Compute local reachability density (LRD)
        lrd = np.zeros(n_points)
        for i in range(n_points):
            lrd[i] = 1 / (
                np.mean(
                    reachability_distances[i, np.argsort(distances[i, :])[:n_neighbors]]
                )
                + 1e-10
            )

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
