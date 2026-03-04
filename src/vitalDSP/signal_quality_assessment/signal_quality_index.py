"""
Signal Quality Assessment Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- SciPy integration for advanced signal processing
- Signal validation and error handling

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
    >>> signal = np.random.randn(1000)
    >>> processor = SignalQualityIndex(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import zscore, iqr, kurtosis, skew
from vitalDSP.utils.data_processing.validation import SignalValidator


class SignalQualityIndex:
    """
    A class to compute various Signal Quality Index (SQI) metrics for assessing the quality of vital signals.

    This class includes methods to evaluate signal quality based on different characteristics such as amplitude variability, baseline wander, zero-crossing consistency, waveform similarity, entropy, and more. These metrics are useful in ensuring that physiological signals like ECG, PPG, EEG, and respiratory signals are of high quality and reliable for further analysis.
    """

    def __init__(self, signal):
        """
        Initialize the SignalQualityIndex class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to assess for quality.

        Raises
        ------
        ValueError
            If the signal is empty or invalid.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> sqi = SignalQualityIndex(signal)
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        # Validate signal - don't allow empty signals
        SignalValidator.validate_signal(
            signal, min_length=1, allow_empty=False, signal_name="signal"
        )

        self.signal = signal

    def _scale_sqi(self, sqi_values, scale="zscore"):
        """
        Scale the SQI values using the specified scaling method.

        Parameters
        ----------
        sqi_values : numpy.ndarray
            The computed SQI values.
        scale : str
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax').

        Returns
        -------
        scaled_sqi : numpy.ndarray
            Scaled SQI values.
        """
        if scale == "zscore":
            # Use manual z-score calculation to avoid precision loss warnings
            mean_val = np.mean(sqi_values)
            std_val = np.std(sqi_values)
            if (
                std_val > 1e-10
            ):  # Use a small threshold to avoid division by tiny numbers
                return (sqi_values - mean_val) / std_val
            else:
                return sqi_values  # Return original values if all are identical
        elif scale == "iqr":
            median = np.median(sqi_values)
            iqr_val = iqr(sqi_values)
            if iqr_val > 0:
                return (sqi_values - median) / iqr_val
            else:
                return np.zeros_like(sqi_values)
        elif scale == "minmax":
            min_val = np.min(sqi_values)
            max_val = np.max(sqi_values)
            return (sqi_values - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown scale type: {scale}")

    def _process_segments(
        self,
        func,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        reference_waveform=None,
    ):
        """
        Helper method to process the signal in segments using a given function.

        Parameters
        ----------
        func : function
            The SQI function to apply to each segment.
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range').
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax').
        reference_waveform : numpy.ndarray, optional
            A reference waveform to compare against, used in waveform similarity SQI. Default is None.

        Returns
        -------
        segment_sqi : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list
            List of indices of segments classified as normal.
        abnormal_segments : list
            List of indices of segments classified as abnormal.
        """
        num_segments = (len(self.signal) - window_size) // step_size + 1
        segment_sqi = []

        for i in range(num_segments):
            start = i * step_size
            end = start + window_size
            segment = self.signal[start:end]

            if reference_waveform is not None:
                sqi_value = func(segment, reference_waveform)
            else:
                sqi_value = func(segment)

            segment_sqi.append(sqi_value)

        # Scale SQI values
        scaled_sqi = self._scale_sqi(np.array(segment_sqi), scale=scale)

        normal_segments = []
        abnormal_segments = []

        for i, sqi_value in enumerate(scaled_sqi):
            if threshold_type == "below":
                if threshold is not None and sqi_value < threshold:
                    abnormal_segments.append(
                        (i * step_size, i * step_size + window_size)
                    )
                else:
                    normal_segments.append((i * step_size, i * step_size + window_size))
            elif threshold_type == "above":
                if threshold is not None and sqi_value > threshold:
                    abnormal_segments.append(
                        (i * step_size, i * step_size + window_size)
                    )
                else:
                    normal_segments.append((i * step_size, i * step_size + window_size))
            elif threshold_type == "range":
                if threshold is not None and not (
                    threshold[0] <= sqi_value <= threshold[1]
                ):
                    abnormal_segments.append(
                        (i * step_size, i * step_size + window_size)
                    )
                else:
                    normal_segments.append((i * step_size, i * step_size + window_size))

        return scaled_sqi, normal_segments, abnormal_segments

    def amplitude_variability_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the amplitude variability SQI over segments of the signal.

        Parameters
        ----------
        window_size : int
            The size of each segment to compute the SQI.
        step_size : int
            The step size to move the window for each segment.
        threshold : float or tuple, optional
            The threshold to determine normal and abnormal segments.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.
        aggregate : bool, optional
            If True (default), returns the mean SQI value as a single float.
            If False, returns detailed tuple (sqi_values, normal_segments, abnormal_segments).

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments).

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        >>> sqi = SignalQualityIndex(signal)
        >>> # Get single aggregated score (default)
        >>> quality_score = sqi.amplitude_variability_sqi(window_size=5, step_size=2)
        >>> # Get detailed results
        >>> sqi_values, normal_segments, abnormal_segments = sqi.amplitude_variability_sqi(
        ...     window_size=5, step_size=2, threshold=0.8, aggregate=False)
        """

        def compute_sqi(segment):
            variability = np.var(segment)
            max_signal = np.max(segment)
            min_signal = np.min(segment)
            range_signal = max_signal - min_signal
            return variability / (range_signal + 1e-2)

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def baseline_wander_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        moving_avg_window=50,
        aggregate=True,
    ):
        """
        Compute the baseline wander SQI.

        This metric evaluates the amount of baseline wander in the signal, which is unwanted low-frequency noise that can distort the true signal. Baseline wander is particularly important in ECG and PPG signals.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.
        moving_avg_window : int, optional
            Size of the window for calculating the moving average. Default is 50.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.baseline_wander_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            moving_average = np.convolve(
                segment, np.ones(moving_avg_window) / moving_avg_window, mode="valid"
            )
            wander = np.mean(np.abs(segment[: len(moving_average)] - moving_average))
            max_signal = np.max(np.abs(segment))
            if max_signal == 0:
                return 1.0
            sqi_value = 1 - (wander / max_signal)
            return sqi_value

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def zero_crossing_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the zero-crossing SQI.

        This metric assesses the number of zero crossings in the signal, which should be consistent in high-quality signals. Irregular zero crossings can indicate noise or instability in the signal.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([-1, 1, -1, 1, -1, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.zero_crossing_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
            expected_crossings = len(segment) / 2
            # Avoid divide by zero
            if expected_crossings > 0:
                sqi_value = (
                    1 - np.abs(zero_crossings - expected_crossings) / expected_crossings
                )
            else:
                sqi_value = 1.0  # Perfect score for very short segments
            return sqi_value

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def waveform_similarity_sqi(
        self,
        window_size,
        step_size,
        reference_waveform,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        similarity_method="correlation",
        aggregate=True,
    ):
        """
        Compute the waveform similarity SQI using various similarity metrics.

        This metric compares the similarity between consecutive waveforms in the signal. High similarity indicates that the signal is stable and free from significant artifacts.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        reference_waveform : numpy.ndarray
            A reference waveform to compare against.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.
        similarity_method : str, optional
            The method to use for similarity calculation ('correlation', 'custom'). Default is 'correlation'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> reference = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.waveform_similarity_sqi(
        ...     window_size=3, step_size=1, reference_waveform=reference, threshold=0.8)
        """

        def compute_sqi(segment, reference_waveform):
            if similarity_method == "correlation":
                corr_coef, _ = pearsonr(segment, reference_waveform)
                return corr_coef
            elif similarity_method == "custom":
                # Implement any custom similarity function here
                return np.dot(segment, reference_waveform) / (
                    np.linalg.norm(segment) * np.linalg.norm(reference_waveform)
                )
            else:
                raise ValueError(f"Unknown similarity method: {similarity_method}")

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            reference_waveform=reference_waveform,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def signal_entropy_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the signal entropy SQI.

        This metric measures the entropy of the signal, which indicates the complexity or predictability of the signal. Lower entropy generally indicates a more regular and stable signal, while higher entropy suggests more randomness.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 1, 1, 1, 1, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.signal_entropy_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            hist, _ = np.histogram(segment, bins=10)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            probability_distribution = hist
            entropy = -np.sum(
                probability_distribution * np.log2(probability_distribution + 1e-8)
            )
            # Prevent division by zero
            if len(segment) <= 1:
                return 0.0
            normalized_entropy = entropy / np.log2(len(segment))
            return normalized_entropy

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def skewness_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the skewness SQI.

        This metric assesses the asymmetry of the signal distribution, where a high skewness value indicates that the signal is not symmetrically distributed.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.skewness_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return skew(segment)

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def kurtosis_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the kurtosis SQI.

        This metric measures the "tailedness" of the signal distribution. High kurtosis values indicate the presence of outliers or sharp peaks in the signal.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.kurtosis_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return kurtosis(segment)

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def peak_to_peak_amplitude_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the peak-to-peak amplitude SQI.

        This metric assesses the peak-to-peak amplitude of the signal, which is the difference between the maximum and minimum values within a segment.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.peak_to_peak_amplitude_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return np.max(segment) - np.min(segment)

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def snr_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the Signal-to-Noise Ratio (SNR) SQI.

        This metric measures the ratio of the signal power to the noise power, which is an important metric for evaluating signal quality.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.snr_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(segment.astype(float), size=max(3, len(segment) // 10))
            signal_power = np.mean(smoothed ** 2)
            noise_power = np.mean((segment - smoothed) ** 2)
            if noise_power == 0:
                return float('inf')
            snr_value = 10 * np.log10(signal_power / noise_power)
            return snr_value

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def energy_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the energy SQI.

        This metric computes the energy of the signal over a segment, which can be an indicator of the signal's strength or activity.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.energy_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return np.sum(segment**2)

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def heart_rate_variability_sqi(
        self,
        rr_intervals,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the heart rate variability (HRV) SQI.

        This metric assesses the variability in RR intervals of ECG, which should be within a certain range for a healthy signal. Abnormal HRV can indicate issues with heart health or signal quality.

        Parameters
        ----------
        rr_intervals : numpy.ndarray
            Array of RR intervals.
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> rr_intervals = np.array([0.8, 0.9, 1.0, 0.9, 0.8])
        >>> sqi = SignalQualityIndex(rr_intervals)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.heart_rate_variability_sqi(rr_intervals, window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            rmssd = np.sqrt(np.mean(np.diff(segment) ** 2))
            mean_segment = np.mean(segment)
            if mean_segment == 0:
                return 0.0
            normalized_rmssd = rmssd / mean_segment
            return normalized_rmssd

        original_signal = self.signal
        try:
            if rr_intervals is not None:
                self.signal = np.array(rr_intervals)
            sqi_values, normal_segments, abnormal_segments = self._process_segments(
                compute_sqi,
                window_size,
                step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )
        finally:
            self.signal = original_signal

        if aggregate:
            return float(np.mean(sqi_values))
        else:
            return (sqi_values, normal_segments, abnormal_segments)

    def ppg_signal_quality_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the PPG signal quality SQI.

        This metric evaluates the overall quality of PPG signals based on amplitude variability and baseline wander. High-quality PPG signals should have minimal noise and stable amplitude.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> signal = np.array([1, 1.1, 0.9, 1, 1.2, 0.8, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.ppg_signal_quality_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            amplitude_variability = np.var(segment)
            max_signal = np.max(segment)
            min_signal = np.min(segment)
            range_signal = max_signal - min_signal
            return amplitude_variability / (range_signal + 1e-2)

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)

    def eeg_band_power_sqi(
        self,
        band_power,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the EEG band power SQI.

        This metric assesses the consistency of EEG band power, which is important for ensuring signal quality. Stable band power indicates a high-quality EEG signal.

        Parameters
        ----------
        band_power : numpy.ndarray
            Array of band power values for EEG.
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> band_power = np.array([10, 12, 11, 13, 12])
        >>> sqi = SignalQualityIndex(band_power)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.eeg_band_power_sqi(band_power, window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            power_variability = np.var(segment)
            mean_power = np.mean(segment)
            return power_variability / (mean_power + 1e-2)

        original_signal = self.signal
        try:
            if band_power is not None:
                self.signal = np.array(band_power)
            sqi_values, normal_segments, abnormal_segments = self._process_segments(
                compute_sqi,
                window_size,
                step_size,
                threshold=threshold,
                threshold_type=threshold_type,
                scale=scale,
            )
        finally:
            self.signal = original_signal

        if aggregate:
            return float(np.mean(sqi_values))
        else:
            return (sqi_values, normal_segments, abnormal_segments)

    def respiratory_signal_quality_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        aggregate=True,
    ):
        """
        Compute the respiratory signal quality SQI.

        This metric evaluates the quality of respiratory signals, considering factors like consistency in breathing cycles and stability of the amplitude.

        Parameters
        ----------
        window_size : int
            The size of each segment.
        step_size : int
            The step size to slide the window.
        threshold : float or tuple, optional
            Threshold to classify a segment as normal or abnormal.
        threshold_type : str, optional
            The type of thresholding ('below', 'above', 'range'). Default is 'below'.
        scale : str, optional
            The scaling method to apply to SQI values ('zscore', 'iqr', 'minmax'). Default is 'zscore'.

        Returns
        -------
        float or tuple
            If aggregate=True: Mean SQI value as float.
            If aggregate=False: Tuple of (sqi_values, normal_segments, abnormal_segments)
            where sqi_values is array of SQI values for each segment, normal_segments and
            abnormal_segments are lists of (start, end) indices.

        Examples
        --------
        >>> resp_signal = np.array([1, 1.1, 1.0, 1.1, 1.0])
        >>> sqi = SignalQualityIndex(resp_signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.respiratory_signal_quality_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            amplitude_variability = np.var(segment)
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
            expected_crossings = len(segment) / 2
            zero_crossing_consistency = (
                1 - np.abs(zero_crossings - expected_crossings) / expected_crossings
            )
            return amplitude_variability * zero_crossing_consistency

        sqi_values, normal_segments, abnormal_segments = self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

        if aggregate:
            # Return mean SQI value for ease of use
            return float(np.mean(sqi_values))
        else:
            # Return detailed results for advanced analysis
            return (sqi_values, normal_segments, abnormal_segments)
