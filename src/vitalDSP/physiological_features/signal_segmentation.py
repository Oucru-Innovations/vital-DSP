"""
Physiological Features Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.signal_segmentation import SignalSegmentation
    >>> signal = np.random.randn(1000)
    >>> processor = SignalSegmentation(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class SignalSegmentation:
    """
    A comprehensive class for segmenting physiological signals.

    Methods
    -------
    fixed_size_segmentation : function
        Segments the signal into fixed-size segments.
    adaptive_segmentation : function
        Segments the signal based on adaptive criteria.
    threshold_based_segmentation : function
        Segments the signal based on a threshold.
    variance_based_segmentation : function
        Segments the signal based on local variance.
    peak_based_segmentation : function
        Segments the signal based on detected peaks.
    ml_based_segmentation : function
        Segments the signal using a machine learning-based approach with multiple default models.
    custom_segmentation : function
        Allows custom segmentation based on a user-defined function.
    """

    def __init__(self, signal):
        """
        Initialize the SignalSegmentation class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal to be segmented.
        """
        self.signal = signal

    def fixed_size_segmentation(self, segment_size):
        """
        Segment the signal into fixed-size segments.

        Parameters
        ----------
        segment_size : int
            The size of each segment.

        Returns
        -------
        segments : list of numpy.ndarray
            A list of fixed-size segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> seg = SignalSegmentation(signal)
        >>> fixed_segments = seg.fixed_size_segmentation(3)
        >>> print(fixed_segments)
        [array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
        """
        segments = [
            self.signal[i : i + segment_size]
            for i in range(0, len(self.signal), segment_size)
        ]
        return segments

    def adaptive_segmentation(self, adaptive_fn):
        """
        Segment the signal based on adaptive criteria.

        Parameters
        ----------
        adaptive_fn : callable
            A function that defines the adaptive criteria for segmentation. This function should take a segment of the signal as input and return the length of the segment.

        Returns
        -------
        segments : list of numpy.ndarray
            A list of adaptively segmented parts of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> def adaptive_fn(segment):
        ...     return len(segment) // 2  # Example criterion
        >>> seg = SignalSegmentation(signal)
        >>> adaptive_segments = seg.adaptive_segmentation(adaptive_fn)
        >>> print(adaptive_segments)
        [array([1, 2, 3, 4]), array([5, 6, 7, 8, 9])]
        """
        segments = []
        start_idx = 0
        while start_idx < len(self.signal):
            # Limit minimum segment size to avoid unnecessary iterations
            end_idx = adaptive_fn(self.signal[start_idx:])
            end_idx = min(
                len(self.signal) - start_idx, end_idx
            )  # Ensure no index out of range

            if end_idx == 0:
                break  # Avoid infinite loops if adaptive_fn returns 0

            segments.append(self.signal[start_idx : start_idx + end_idx])
            start_idx += end_idx

        return segments

    def threshold_based_segmentation(self, threshold):
        """
        Segment the signal based on a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for segmentation. The signal is segmented whenever it exceeds this threshold.

        Returns
        -------
        segments : list of numpy.ndarray
            A list of segments where the signal exceeds the threshold.

        Examples
        --------
        >>> signal = np.array([1, 2, 5, 2, 8, 1])
        >>> seg = SignalSegmentation(signal)
        >>> threshold_segments = seg.threshold_based_segmentation(4)
        >>> print(threshold_segments)
        [array([5]), array([8])]
        """
        segments = []
        start_idx = None
        for i in range(len(self.signal)):
            if self.signal[i] > threshold:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segments.append(self.signal[start_idx:i])
                    start_idx = None
        if start_idx is not None:
            segments.append(self.signal[start_idx:])
        return segments

    def variance_based_segmentation(self, window_size, variance_threshold):
        """
        Segment the signal based on local variance.

        Parameters
        ----------
        window_size : int
            The size of the window for variance calculation.
        variance_threshold : float
            The threshold of variance to determine segment boundaries.

        Returns
        -------
        segments : list of numpy.ndarray
            A list of segments based on local variance.

        Examples
        --------
        >>> signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 1, 8])
        >>> seg = SignalSegmentation(signal)
        >>> variance_segments = seg.variance_based_segmentation(3, 2.0)
        >>> print(variance_segments)
        [array([1, 2, 2, 2, 5, 6]), array([1, 1, 1, 8])]
        """
        variances = np.array(
            [
                np.var(self.signal[i : i + window_size])
                for i in range(len(self.signal) - window_size)
            ]
        )
        segments = []
        start_idx = 0
        for i in range(1, len(variances)):
            if (
                variances[i] > variance_threshold
                and variances[i - 1] <= variance_threshold
            ):
                segments.append(self.signal[start_idx : i + window_size])
                start_idx = i + window_size
        if start_idx < len(self.signal):
            segments.append(self.signal[start_idx:])
        return segments

    def peak_based_segmentation(self, min_distance=50, height=None):
        """
        Segment the signal based on detected peaks.

        Parameters
        ----------
        min_distance : int, optional
            Minimum distance between consecutive peaks.
        height : float or None, optional
            Minimum height of peaks to be considered.

        Returns
        -------
        segments : list of numpy.ndarray
            A list of segments around detected peaks.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2, 1, 8, 1])
        >>> seg = SignalSegmentation(signal)
        >>> peak_segments = seg.peak_based_segmentation(min_distance=2, height=5)
        >>> print(peak_segments)
        [array([1, 8, 1])]
        """
        peaks = (
            np.where(
                (self.signal[1:-1] > self.signal[:-2])
                & (self.signal[1:-1] > self.signal[2:])
            )[0]
            + 1
        )
        if height is not None:
            peaks = peaks[self.signal[peaks] > height]
        if min_distance > 1:
            peaks = peaks[np.diff(peaks, prepend=0) > min_distance]

        segments = []
        for i in range(len(peaks) - 1):
            segments.append(self.signal[peaks[i] : peaks[i + 1]])
        return segments

    def ml_based_segmentation(self, model="change_detection"):
        """
        Segment the signal using a machine learning-based approach.

        Parameters
        ----------
        model : str, optional
            The name of the default model to use. Options include "change_detection", "kmeans", "gmm",
            "decision_tree", "dtw", "spectral", "autoencoder". Default is "change_detection".

        Returns
        -------
        segments : list of numpy.ndarray
            A list of segments predicted by the model.

        Examples
        --------
        >>> signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 8, 1])
        >>> seg = SignalSegmentation(signal)
        >>> ml_segments = seg.ml_based_segmentation(model="kmeans")
        >>> print(ml_segments)
        """
        if model == "change_detection":
            change_points = (
                np.where(np.abs(np.diff(self.signal)) > np.std(self.signal))[0] + 1
            )
        elif model == "kmeans":
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(self.signal.reshape(-1, 1))
            change_points = np.where(np.diff(labels))[0] + 1
        elif model == "gmm":
            n_components = 3
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(self.signal.reshape(-1, 1))
            hidden_states = gmm.predict(self.signal.reshape(-1, 1))
            change_points = np.where(np.diff(hidden_states))[0] + 1
        elif model == "decision_tree":
            labels = np.zeros(len(self.signal))
            labels[: len(labels) // 2] = 1
            dt = DecisionTreeClassifier()
            dt.fit(self.signal.reshape(-1, 1), labels)
            predictions = dt.predict(self.signal.reshape(-1, 1))
            change_points = np.where(np.diff(predictions))[0] + 1
        elif model == "dtw":
            # Placeholder for a more complex DTW implementation
            change_points = np.array([len(self.signal) // 2])
        elif model == "spectral":
            spectral = SpectralClustering(n_clusters=5, affinity="nearest_neighbors")
            labels = spectral.fit_predict(self.signal.reshape(-1, 1))
            change_points = np.where(np.diff(labels))[0] + 1
        elif model == "autoencoder":
            # Placeholder for autoencoder-based segmentation
            change_points = np.array([len(self.signal) // 2])
        else:
            raise ValueError("Unknown model type specified.")

        segments = [
            self.signal[change_points[i] : change_points[i + 1]]
            for i in range(len(change_points) - 1)
        ]
        return segments

    def custom_segmentation(self, custom_fn):
        """
        Segment the signal based on a custom function.

        Parameters
        ----------
        custom_fn : callable
            A user-defined function that returns segment boundaries. The function should take the entire signal as input and return an array of indices where segments start and end.

        Returns
        -------
        segments : list of numpy.ndarray
            A list of segments based on custom criteria.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> def custom_fn(signal):
        ...     return np.array([0, 3, 6, 9])
        >>> seg = SignalSegmentation(signal)
        >>> custom_segments = seg.custom_segmentation(custom_fn)
        >>> print(custom_segments)
        [array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
        """
        segment_boundaries = custom_fn(self.signal)
        segments = [
            self.signal[segment_boundaries[i] : segment_boundaries[i + 1]]
            for i in range(len(segment_boundaries) - 1)
        ]
        return segments
