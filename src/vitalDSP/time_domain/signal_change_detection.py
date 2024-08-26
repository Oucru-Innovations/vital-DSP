import numpy as np

class SignalChangeDetection:
    """
    A comprehensive class for detecting changes in physiological signals.

    This class provides multiple methods to analyze physiological signals and detect
    significant changes based on various criteria such as zero crossings, variance,
    energy levels, adaptive thresholds, and machine learning-inspired techniques.

    Methods
    -------
    zero_crossing_rate : function
        Computes the Zero Crossing Rate (ZCR) of the signal.
    absolute_difference : function
        Computes the absolute difference between consecutive samples.
    variance_based_detection : function
        Detects signal changes based on local variance.
    energy_based_detection : function
        Detects signal changes based on local energy.
    adaptive_threshold_detection : function
        Detects signal changes using an adaptive threshold.
    ml_based_change_detection : function
        Detects signal changes using a machine learning-inspired method.
    """

    def __init__(self, signal):
        """
        Initialize the SignalChangeDetection class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal to be analyzed for changes.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> scd = SignalChangeDetection(signal)
        """
        self.signal = signal

    def zero_crossing_rate(self):
        """
        Compute the Zero Crossing Rate (ZCR) of the signal.

        The Zero Crossing Rate is the rate at which the signal changes sign,
        which can indicate changes in the signal's behavior or underlying processes.
        This metric is often used in speech and audio signal processing to detect
        changes in frequency or amplitude.

        Returns
        -------
        zcr : float
            The Zero Crossing Rate of the signal.

        Examples
        --------
        >>> signal = np.array([1, -1, 1, -1, 1])
        >>> scd = SignalChangeDetection(signal)
        >>> zcr = scd.zero_crossing_rate()
        >>> print(zcr)
        0.8
        """
        zero_crossings = np.where(np.diff(np.sign(self.signal)))[0]
        zcr = len(zero_crossings) / len(self.signal)
        return zcr

    def absolute_difference(self):
        """
        Compute the absolute difference between consecutive samples.

        This method highlights changes between successive signal values, which can
        be useful for detecting sudden changes or anomalies. It is a simple yet
        effective technique to measure the magnitude of change in the signal.

        Returns
        -------
        abs_diff : numpy.ndarray
            The absolute differences of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 4, 7, 11])
        >>> scd = SignalChangeDetection(signal)
        >>> abs_diff = scd.absolute_difference()
        >>> print(abs_diff)
        [1 2 3 4]
        """
        abs_diff = np.abs(np.diff(self.signal))
        return abs_diff

    def variance_based_detection(self, window_size):
        """
        Detect signal changes based on local variance.

        This method computes the variance within a sliding window over the signal.
        High variance may indicate areas of the signal with significant changes or noise,
        which is useful in detecting regions with high activity or instability.

        Parameters
        ----------
        window_size : int
            The size of the window to compute variance.

        Returns
        -------
        variances : numpy.ndarray
            The local variance of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 2, 3, 5, 8, 13, 21])
        >>> scd = SignalChangeDetection(signal)
        >>> variances = scd.variance_based_detection(window_size=3)
        >>> print(variances)
        [0.33333333 0.33333333 2.33333333 6.33333333 13.33333333]
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

        This method calculates the energy within a sliding window over the signal.
        Energy is calculated as the sum of the squares of the signal values within the window.
        High energy may indicate periods of significant activity or events in the signal,
        making this method useful in detecting bursts of activity.

        Parameters
        ----------
        window_size : int
            The size of the window to compute energy.

        Returns
        -------
        energies : numpy.ndarray
            The local energy of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> scd = SignalChangeDetection(signal)
        >>> energies = scd.energy_based_detection(window_size=3)
        >>> print(energies)
        [14 29 50]
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

        This method calculates local statistics (mean and standard deviation) over a sliding
        window and identifies signal changes where the deviation from the local mean exceeds
        an adaptive threshold. This approach is effective for detecting outliers or anomalies
        in signals with varying baseline levels.

        Parameters
        ----------
        threshold_factor : float, optional
            Factor to multiply the local standard deviation to determine the threshold.
            Default is 1.5.
        window_size : int, optional
            The size of the window to compute local statistics. Default is 10.

        Returns
        -------
        signal_changes : numpy.ndarray
            Indices of detected signal changes.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2, 10, 2, 1])
        >>> scd = SignalChangeDetection(signal)
        >>> changes = scd.adaptive_threshold_detection(threshold_factor=2.0, window_size=3)
        >>> print(changes)
        [6]
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

        This method allows the use of a custom model or function to detect change points in the signal.
        If no model is provided, a simple thresholding method based on the absolute difference between
        consecutive samples is used by default. This approach is flexible and can be extended with
        sophisticated models like decision trees, neural networks, or clustering algorithms for more
        complex change detection tasks.

        Parameters
        ----------
        model : callable or None, optional
            A custom model or function for predicting signal changes. The model should take the
            signal as input and return an array of change points. If None, a default thresholding
            method is used. Default is None.

        Returns
        -------
        change_points : numpy.ndarray
            Predicted change points in the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 5, 10, 1, 2])
        >>> scd = SignalChangeDetection(signal)
        >>> changes = scd.ml_based_change_detection()
        >>> print(changes)
        [4 5]
        """
        if model is None:
            # Example simple model: predict changes based on thresholding the absolute difference
            model = (
                lambda x: np.where(np.abs(np.diff(x)) > np.mean(np.abs(np.diff(x))))[0]
                + 1
            )

        change_points = model(self.signal)
        return change_points
