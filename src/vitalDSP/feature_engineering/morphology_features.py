import numpy as np
from scipy.stats import linregress
from vitalDSP.physiological_features.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import (
    PreprocessConfig,
    preprocess_signal,
)
from vitalDSP.physiological_features.waveform import WaveformMorphology


class PhysiologicalFeatureExtractor:
    """
    A class to extract various physiological features from ECG and PPG signals, such as durations,
    areas, amplitude variability, slope ratios, and dicrotic notch locations.

    Features for PPG:
    - Systolic/diastolic duration, area, amplitude variability
    - Signal skewness, slope, peak trends, and dicrotic notch locations

    Features for ECG:
    - QRS duration, area, T-wave area
    - Amplitude variability, QRS-T ratios, and QRS slope
    - Signal skewness, peak trends

    Methods
    -------
    preprocess_signal(preprocess_config)
        Preprocess the signal by applying bandpass filtering and noise reduction.
    extract_features(signal_type='ECG', preprocess_config=None)
        Extract all features (morphology, volume, amplitude variability, dicrotic notch) for ECG or PPG signals.
    """

    def __init__(self, signal, fs=1000):
        """
        Initialize the PhysiologicalFeatureExtractor class.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal (ECG or PPG).
        fs : int, optional
            Sampling frequency in Hz. Default is 1000.
        """
        self.signal = np.asarray(signal, dtype=np.float64)
        self.fs = fs

    def detect_troughs(self, peaks):
        """
        Detect troughs (valleys) in the signal based on the given peaks.

        Parameters
        ----------
        peaks : numpy.ndarray
            The indices of detected peaks.

        Returns
        -------
        troughs : numpy.ndarray
            The indices of detected troughs.

        Examples
        --------
        >>> peaks = np.array([100, 200, 300])
        >>> troughs = extractor.detect_troughs(peaks)
        >>> print(troughs)
        """
        troughs = []
        for i in range(len(peaks) - 1):
            # Find the local minimum between two consecutive peaks
            segment = self.signal[peaks[i] : peaks[i + 1]]
            trough = np.argmin(segment) + peaks[i]
            troughs.append(trough)
        return np.array(troughs)

    def compute_peak_trend(self, peaks):
        """
        Compute the trend slope of peak amplitudes over time.

        Parameters
        ----------
        peaks : numpy.ndarray
            The set of peaks.

        Returns
        -------
        trend_slope : float
            The slope of the peak amplitude trend over time.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> peaks = np.array([100, 200, 300])
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> trend_slope = extractor.compute_peak_trend(peaks)
        >>> print(trend_slope)
        """
        amplitudes = self.signal[peaks]
        if len(peaks) > 1:
            slope, _, _, _, _ = linregress(peaks, amplitudes)
            return slope
        return 0.0

    def compute_amplitude_variability(self, peaks):
        """
        Compute the variability of the amplitudes at the given peak locations.

        Parameters
        ----------
        peaks : numpy.ndarray
            The set of peaks.

        Returns
        -------
        variability : float
            The amplitude variability (standard deviation of the peak amplitudes).

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> peaks = np.array([100, 200, 300])
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> variability = extractor.compute_amplitude_variability(peaks)
        >>> print(variability)
        """
        amplitudes = self.signal[peaks]
        return np.std(amplitudes) if len(amplitudes) > 1 else 0.0

    def get_preprocess_signal(self, preprocess_config):
        """
        Preprocess the signal by applying bandpass filtering and noise reduction.

        Parameters
        ----------
        preprocess_config : PreprocessConfig
            Configuration for both signal filtering and artifact removal.

        Returns
        -------
        clean_signal : numpy.ndarray
            The preprocessed signal, cleaned of noise and artifacts.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
        >>> preprocess_config = PreprocessConfig()
        >>> extractor = PhysiologicalFeatureExtractor(signal, fs=1000)
        >>> preprocessed_signal = extractor.preprocess_signal(preprocess_config)
        >>> print(preprocessed_signal)
        """
        return preprocess_signal(
            signal=self.signal,
            sampling_rate=self.fs,
            filter_type=preprocess_config.filter_type,
            lowcut=preprocess_config.lowcut,
            highcut=preprocess_config.highcut,
            order=preprocess_config.order,
            noise_reduction_method=preprocess_config.noise_reduction_method,
            wavelet_name=preprocess_config.wavelet_name,
            level=preprocess_config.level,
            window_length=preprocess_config.window_length,
            polyorder=preprocess_config.polyorder,
            kernel_size=preprocess_config.kernel_size,
            sigma=preprocess_config.sigma,
            respiratory_mode=preprocess_config.respiratory_mode,
        )

    def extract_features(self, signal_type="ECG", preprocess_config=None, mode="peak"):
        """
        Extract all physiological features from the signal for either ECG or PPG.

        Parameters
        ----------
        signal_type : str, optional
            The type of signal ("ECG" or "PPG"). Default is "ECG".
        preprocess_config : PreprocessConfig, optional
            The configuration object for signal preprocessing. If None, default settings are used.
        mode : str, optional
            The type of area computation method ("peak" or "trough"). Default is "peak".

        Returns
        -------
        features : dict
            A dictionary containing the extracted features, such as durations, areas, amplitude variability,
            slopes, skewness, peak trends, and dicrotic notch locations.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> preprocess_config = PreprocessConfig()
        >>> extractor = PhysiologicalFeatureExtractor(signal, fs=1000)
        >>> features = extractor.extract_features(signal_type="ECG", preprocess_config=preprocess_config)
        >>> print(features)
        """
        if preprocess_config is None:
            preprocess_config = PreprocessConfig()

        # Preprocess the signal
        clean_signal = self.get_preprocess_signal(preprocess_config)

        # Initialize the morphology class
        morphology = WaveformMorphology(
            clean_signal, fs=self.fs, signal_type=signal_type
        )

        if signal_type == "PPG":
            peaks = PeakDetection(
                clean_signal, method="ppg_first_derivative"
            ).detect_peaks()
            if mode == "peak":
                systolic_area = morphology.compute_volume(
                    peaks[:-1], peaks[1:], mode=mode
                )
                diastolic_area = morphology.compute_volume(
                    peaks[1:], peaks[2:], mode=mode
                )
                systolic_duration = morphology.compute_duration(
                    peaks[:-1], peaks[1:], mode=mode
                )
                diastolic_duration = morphology.compute_duration(
                    peaks[1:], peaks[2:], mode=mode
                )
            else:
                troughs = self.detect_troughs(peaks)
                systolic_area = morphology.compute_volume(peaks, troughs, mode=mode)
                diastolic_area = morphology.compute_volume(peaks, troughs, mode=mode)
                systolic_duration = morphology.compute_duration(
                    peaks, troughs, mode=mode
                )
                diastolic_duration = morphology.compute_duration(
                    peaks, troughs, mode=mode
                )

            dicrotic_notch_locs = morphology.compute_ppg_dicrotic_notch()

            # systolic_area = morphology.compute_volume_sequence(peaks[:-1], peaks[1:])
            # diastolic_area = morphology.compute_volume_sequence(peaks[1:], peaks[2:])

            systolic_variability = self.compute_amplitude_variability(peaks[:-1])
            diastolic_variability = self.compute_amplitude_variability(peaks[1:])
            signal_skewness = morphology.compute_skewness()
            peak_trend_slope = self.compute_peak_trend(peaks)

            return {
                "systolic_duration": systolic_duration.value(),
                "diastolic_duration": diastolic_duration,
                "systolic_area": systolic_area,
                "diastolic_area": diastolic_area,
                "signal_skewness": signal_skewness,
                "peak_trend_slope": peak_trend_slope,
                "systolic_amplitude_variability": systolic_variability,
                "diastolic_amplitude_variability": diastolic_variability,
                "dicrotic_notch_locs": dicrotic_notch_locs,
            }

        elif signal_type == "ECG":
            r_peaks = PeakDetection(clean_signal, method="ecg_r_peak").detect_peaks()
            qrs_duration = morphology.compute_qrs_duration()
            qrs_area = morphology.compute_volume(r_peaks[:-1], r_peaks[1:], mode="peak")
            t_wave_area = morphology.compute_volume(
                r_peaks[1:], r_peaks[2:], mode="peak"
            )
            qrs_amplitude = np.mean(clean_signal[r_peaks])
            qrs_slope = morphology.compute_slope(r_peaks[:-1], r_peaks[1:], mode="peak")
            qrs_t_ratio = qrs_area / t_wave_area if t_wave_area != 0 else 0
            signal_skewness = morphology.compute_skewness()
            peak_trend_slope = self.compute_peak_trend(r_peaks)
            amplitude_variability = self.compute_amplitude_variability(r_peaks)

            return {
                "qrs_duration": qrs_duration,
                "qrs_area": qrs_area,
                "t_wave_area": t_wave_area,
                "qrs_amplitude": qrs_amplitude,
                "qrs_slope": qrs_slope,
                "qrs_t_ratio": qrs_t_ratio,
                "signal_skewness": signal_skewness,
                "peak_trend_slope": peak_trend_slope,
                "r_peak_amplitude_variability": amplitude_variability,
            }

        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")
