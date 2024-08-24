import numpy as np

class SignalQualityIndex:
    """
    A class to compute various Signal Quality Index (SQI) metrics for assessing the quality of vital signals.

    This class includes methods to evaluate signal quality based on different characteristics such as amplitude variability, baseline wander, zero-crossing consistency, waveform similarity, entropy, and more. These metrics are useful in ensuring that physiological signals like ECG, PPG, EEG, and respiratory signals are of high quality and reliable for further analysis.

    Methods
    -------
    amplitude_variability_sqi : function
        Evaluates the variability in signal amplitude.
    baseline_wander_sqi : function
        Measures baseline wander in ECG/PPG signals.
    zero_crossing_sqi : function
        Assesses the number of zero crossings for signal stability.
    waveform_similarity_sqi : function
        Compares the similarity between consecutive waveforms.
    signal_entropy_sqi : function
        Computes the entropy of the signal, indicating complexity.
    heart_rate_variability_sqi : function
        Assesses heart rate variability in ECG.
    ppg_signal_quality_sqi : function
        Evaluates the overall quality of PPG signals.
    eeg_band_power_sqi : function
        Measures the band power consistency in EEG signals.
    respiratory_signal_quality_sqi : function
        Evaluates the quality of respiratory signals.
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

    def amplitude_variability_sqi(self):
        """
        Compute the amplitude variability SQI.

        This metric assesses the variability in the amplitude of the signal, which should be minimal for high-quality signals. High variability often indicates noise or artifacts in the signal.

        Returns
        -------
        sqi_value : float
            Amplitude variability SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> signal = np.array([1, 1.1, 0.9, 1, 1.2, 0.8, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> print(sqi.amplitude_variability_sqi())
        0.9  # Example output
        """
        variability = np.var(self.signal)
        max_signal = np.max(self.signal)
        min_signal = np.min(self.signal)
        sqi_value = 1 - (variability / (max_signal - min_signal + 1e-8))
        return max(0, min(1, sqi_value))

    def baseline_wander_sqi(self, window_size=50):
        """
        Compute the baseline wander SQI.

        This metric evaluates the amount of baseline wander in the signal, which is unwanted low-frequency noise that can distort the true signal. Baseline wander is particularly important in ECG and PPG signals.

        Parameters
        ----------
        window_size : int, optional
            Size of the window for calculating the moving average, by default 50.

        Returns
        -------
        sqi_value : float
            Baseline wander SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> signal = np.array([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        >>> sqi = SignalQualityIndex(signal)
        >>> print(sqi.baseline_wander_sqi())
        0.85  # Example output
        """
        moving_average = np.convolve(
            self.signal, np.ones(window_size) / window_size, mode="valid"
        )
        wander = np.mean(np.abs(self.signal[: len(moving_average)] - moving_average))
        max_signal = np.max(self.signal)
        sqi_value = 1 - (wander / max_signal)
        return max(0, min(1, sqi_value))

    def zero_crossing_sqi(self):
        """
        Compute the zero-crossing SQI.

        This metric assesses the number of zero crossings in the signal, which should be consistent in high-quality signals. Irregular zero crossings can indicate noise or instability in the signal.

        Returns
        -------
        sqi_value : float
            Zero-crossing SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> signal = np.array([-1, 1, -1, 1, -1, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> print(sqi.zero_crossing_sqi())
        1.0  # Example output
        """
        zero_crossings = np.sum(np.diff(np.sign(self.signal)) != 0)
        expected_crossings = len(self.signal) / 2
        sqi_value = 1 - np.abs(zero_crossings - expected_crossings) / expected_crossings
        return max(0, min(1, sqi_value))

    def waveform_similarity_sqi(self, reference_waveform):
        """
        Compute the waveform similarity SQI.

        This metric compares the similarity between consecutive waveforms in the signal. High similarity indicates that the signal is stable and free from significant artifacts.

        Parameters
        ----------
        reference_waveform : numpy.ndarray
            A reference waveform to compare against.

        Returns
        -------
        sqi_value : float
            Waveform similarity SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1])
        >>> reference = np.array([1, 2, 3, 2, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> print(sqi.waveform_similarity_sqi(reference))
        1.0  # Example output
        """
        correlation = np.corrcoef(self.signal, reference_waveform)[0, 1]
        return max(0, min(1, correlation))

    def signal_entropy_sqi(self):
        """
        Compute the signal entropy SQI.

        This metric measures the entropy of the signal, which indicates the complexity or predictability of the signal. Lower entropy generally indicates a more regular and stable signal, while higher entropy suggests more randomness.

        Returns
        -------
        sqi_value : float
            Signal entropy SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> signal = np.array([1, 1, 1, 1, 1, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> print(sqi.signal_entropy_sqi())
        0.0  # Example output
        """
        probability_distribution = np.histogram(self.signal, bins=10, density=True)[0]
        entropy = -np.sum(
            probability_distribution * np.log2(probability_distribution + 1e-8)
        )
        normalized_entropy = entropy / np.log2(len(self.signal))
        return max(0, min(1, normalized_entropy))

    def heart_rate_variability_sqi(self, rr_intervals):
        """
        Compute the heart rate variability (HRV) SQI.

        This metric assesses the variability in RR intervals of ECG, which should be within a certain range for a healthy signal. Abnormal HRV can indicate issues with heart health or signal quality.

        Parameters
        ----------
        rr_intervals : numpy.ndarray
            Array of RR intervals.

        Returns
        -------
        sqi_value : float
            HRV SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> rr_intervals = np.array([0.8, 0.9, 1.0, 0.9, 0.8])
        >>> sqi = SignalQualityIndex(rr_intervals)
        >>> print(sqi.heart_rate_variability_sqi(rr_intervals))
        0.9  # Example output
        """
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        normalized_rmssd = 1 - (rmssd / np.mean(rr_intervals))
        return max(0, min(1, normalized_rmssd))

    def ppg_signal_quality_sqi(self):
        """
        Compute the PPG signal quality SQI.

        This metric evaluates the overall quality of PPG signals based on amplitude variability and baseline wander. High-quality PPG signals should have minimal noise and stable amplitude.

        Returns
        -------
        sqi_value : float
            PPG SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> signal = np.array([1, 1.1, 0.9, 1, 1.2, 0.8, 1])
        >>> sqi = SignalQualityIndex(signal)
        >>> print(sqi.ppg_signal_quality_sqi())
        0.85  # Example output
        """
        amplitude_variability = self.amplitude_variability_sqi()
        baseline_wander = self.baseline_wander_sqi()
        return (amplitude_variability + baseline_wander) / 2

    def eeg_band_power_sqi(self, band_power):
        """
        Compute the EEG band power SQI.

        This metric assesses the consistency of EEG band power, which is important for ensuring signal quality. Stable band power indicates a high-quality EEG signal.

        Parameters
        ----------
        band_power : numpy.ndarray
            Array of band power values for EEG.

        Returns
        -------
        sqi_value : float
            EEG band power SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> band_power = np.array([10, 12, 11, 13, 12])
        >>> sqi = SignalQualityIndex(band_power)
        >>> print(sqi.eeg_band_power_sqi(band_power))
        0.9  # Example output
        """
        power_variability = np.var(band_power)
        sqi_value = 1 - (power_variability / np.mean(band_power))
        return max(0, min(1, sqi_value))

    def respiratory_signal_quality_sqi(self):
        """
        Compute the respiratory signal quality SQI.

        This metric evaluates the quality of respiratory signals, considering factors like consistency in breathing cycles and stability of the amplitude.

        Returns
        -------
        sqi_value : float
            Respiratory SQI, typically ranging from 0 (poor) to 1 (excellent).

        Examples
        --------
        >>> resp_signal = np.array([1, 1.1, 1.0, 1.1, 1.0])
        >>> sqi = SignalQualityIndex(resp_signal)
        >>> print(sqi.respiratory_signal_quality_sqi())
        0.8  # Example output
        """
        amplitude_variability = self.amplitude_variability_sqi()
        zero_crossing_consistency = self.zero_crossing_sqi()
        return (amplitude_variability + zero_crossing_consistency) / 2
