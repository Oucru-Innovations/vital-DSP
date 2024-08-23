import numpy as np


class SignalChangeDetection:
    """
    A comprehensive class for detecting changes in physiological signals.

    Methods:
    - zero_crossing_rate: Computes the Zero Crossing Rate (ZCR) of the signal.
    - absolute_difference: Computes the absolute difference between consecutive samples.
    - variance_based_detection: Detects signal changes based on local variance.
    - energy_based_detection: Detects signal changes based on local energy.
    - adaptive_threshold_detection: Detects signal changes using an adaptive threshold.
    - ml_based_change_detection: Detects signal changes using a machine learning-inspired method.
    """

    def __init__(self, signal):
        """
        Initialize the SignalChangeDetection class with the signal.

        Parameters:
        signal (numpy.ndarray): The input physiological signal.
        """
        self.signal = signal

    def zero_crossing_rate(self):
        """
        Compute the Zero Crossing Rate (ZCR) of the signal.

        Returns:
        float: The ZCR of the signal.
        """
        zero_crossings = np.where(np.diff(np.sign(self.signal)))[0]
        zcr = len(zero_crossings) / len(self.signal)
        return zcr

    def absolute_difference(self):
        """
        Compute the absolute difference between consecutive samples.

        Returns:
        numpy.ndarray: The absolute differences of the signal.
        """
        abs_diff = np.abs(np.diff(self.signal))
        return abs_diff

    def variance_based_detection(self, window_size):
        """
        Detect signal changes based on local variance.

        Parameters:
        window_size (int): The size of the window to compute variance.

        Returns:
        numpy.ndarray: The local variance of the signal.
        """
        variances = np.array(
            [
                np.var(self.signal[i : i + window_size])
                for i in range(len(self.signal) - window_size)
            ]
        )
        return variances

    def energy_based_detection(self, window_size):
        """
        Detect signal changes based on local energy.

        Parameters:
        window_size (int): The size of the window to compute energy.

        Returns:
        numpy.ndarray: The local energy of the signal.
        """
        energies = np.array(
            [
                np.sum(self.signal[i : i + window_size] ** 2)
                for i in range(len(self.signal) - window_size)
            ]
        )
        return energies

    def adaptive_threshold_detection(self, threshold_factor=1.5, window_size=10):
        """
        Detect signal changes using an adaptive threshold based on local statistics.

        Parameters:
        threshold_factor (float): Factor to multiply the local standard deviation to determine the threshold.
        window_size (int): The size of the window to compute local statistics.

        Returns:
        numpy.ndarray: Indices of detected signal changes.
        """
        local_means = np.array(
            [
                np.mean(self.signal[i : i + window_size])
                for i in range(len(self.signal) - window_size)
            ]
        )
        local_stds = np.array(
            [
                np.std(self.signal[i : i + window_size])
                for i in range(len(self.signal) - window_size)
            ]
        )
        adaptive_thresholds = local_means + threshold_factor * local_stds
        signal_changes = (
            np.where(
                np.abs(self.signal[window_size:] - local_means) > adaptive_thresholds
            )[0]
            + window_size
        )
        return signal_changes

    def ml_based_change_detection(self, model=None):
        """
        Detect signal changes using a machine learning-inspired method.

        Parameters:
        model (callable or None): A custom model or function for predicting signal changes.

        Returns:
        numpy.ndarray: Predicted change points in the signal.
        """
        if model is None:
            # Example simple model: predict changes based on thresholding the absolute difference
            model = (
                lambda x: np.where(np.abs(np.diff(x)) > np.mean(np.abs(np.diff(x))))[0]
                + 1
            )

        change_points = model(self.signal)
        return change_points
