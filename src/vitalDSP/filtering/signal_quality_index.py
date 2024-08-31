import numpy as np
from scipy.stats import pearsonr
from scipy.stats import zscore, iqr, kurtosis, skew


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

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> sqi = SignalQualityIndex(signal)
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
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
            return zscore(sqi_values)
        elif scale == "iqr":
            median = np.median(sqi_values)
            return (sqi_values - median) / iqr(sqi_values)
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

        Returns
        -------
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.amplitude_variability_sqi(window_size=5, step_size=2, threshold=0.8)
        """

        def compute_sqi(segment):
            variability = np.var(segment)
            max_signal = np.max(segment)
            min_signal = np.min(segment)
            range_signal = max_signal - min_signal
            return variability / (range_signal + 1e-2)

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def baseline_wander_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        moving_avg_window=50,
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

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
            max_signal = np.max(segment)
            sqi_value = 1 - (wander / max_signal)
            return sqi_value

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def zero_crossing_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([-1, 1, -1, 1, -1, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.zero_crossing_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
            expected_crossings = len(segment) / 2
            sqi_value = (
                1 - np.abs(zero_crossings - expected_crossings) / expected_crossings
            )
            return sqi_value

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def waveform_similarity_sqi(
        self,
        window_size,
        step_size,
        reference_waveform,
        threshold=None,
        threshold_type="below",
        scale="zscore",
        similarity_method="correlation",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

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

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            reference_waveform=reference_waveform,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def signal_entropy_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 1, 1, 1, 1, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.signal_entropy_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            probability_distribution = np.histogram(segment, bins=10, density=True)[0]
            entropy = -np.sum(
                probability_distribution * np.log2(probability_distribution + 1e-8)
            )
            normalized_entropy = entropy / np.log2(len(segment))
            return normalized_entropy

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def skewness_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.skewness_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return skew(segment)

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def kurtosis_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.kurtosis_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return kurtosis(segment)

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def peak_to_peak_amplitude_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.peak_to_peak_amplitude_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return np.max(segment) - np.min(segment)

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def snr_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.snr_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            signal_power = np.mean(segment**2)
            noise_power = np.var(segment)
            return 10 * np.log10(signal_power / (noise_power + 1e-8))

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def energy_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.energy_sqi(window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            return np.sum(segment**2)

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def heart_rate_variability_sqi(
        self,
        rr_intervals,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

        Examples
        --------
        >>> rr_intervals = np.array([0.8, 0.9, 1.0, 0.9, 0.8])
        >>> sqi = SignalQualityIndex(rr_intervals)
        >>> sqi_values, normal_segments, abnormal_segments = sqi.heart_rate_variability_sqi(rr_intervals, window_size=3, step_size=1, threshold=0.8)
        """

        def compute_sqi(segment):
            rmssd = np.sqrt(np.mean(np.diff(segment) ** 2))
            normalized_rmssd = rmssd / np.mean(segment)
            return normalized_rmssd

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def ppg_signal_quality_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

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

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def eeg_band_power_sqi(
        self,
        band_power,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

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

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )

    def respiratory_signal_quality_sqi(
        self,
        window_size,
        step_size,
        threshold=None,
        threshold_type="below",
        scale="zscore",
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
        sqi_values : numpy.ndarray
            Array of SQI values for each segment.
        normal_segments : list of tuples
            List of (start, end) indices for normal segments.
        abnormal_segments : list of tuples
            List of (start, end) indices for abnormal segments.

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

        return self._process_segments(
            compute_sqi,
            window_size,
            step_size,
            threshold=threshold,
            threshold_type=threshold_type,
            scale=scale,
        )
