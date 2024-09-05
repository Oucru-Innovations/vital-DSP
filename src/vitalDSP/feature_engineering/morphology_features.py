import numpy as np
from scipy.stats import skew, linregress
from scipy.integrate import simps
from vitalDSP.physiological_features.peak_detection import PeakDetection
from vitalDSP.filtering.signal_filtering import SignalFiltering
from vitalDSP.preprocess.noise_reduction import (
    wavelet_denoising,
    savgol_denoising,
    median_denoising,
    gaussian_denoising
)

class PreprocessConfig:
    """
    A configuration class for signal preprocessing, including filtering and noise reduction settings.

    Attributes
    ----------
    filter_type : str
        The type of filtering ('bandpass', 'butterworth', 'chebyshev', 'elliptic', or 'ignore').
    filter_config : dict
        Dictionary of filter parameters (e.g., cutoff frequencies, order).
    noise_reduction_method : str
        The method for noise reduction ('wavelet', 'savgol', 'median', 'gaussian', 'moving_average', or 'ignore').
    noise_reduction_config : dict
        Configuration parameters for the noise reduction method.
    respiratory_mode : bool
        Whether to apply respiratory signal-specific filtering.
    """

    def __init__(
        self,
        filter_type="bandpass",
        filter_config=None,
        noise_reduction_method="wavelet",
        noise_reduction_config=None,
        respiratory_mode=False,
    ):
        if filter_config is None:
            filter_config = {"lowcut": 0.1, "highcut": 0.5, "order": 4}
        if noise_reduction_config is None:
            noise_reduction_config = {"wavelet_name": "haar", "level": 1}
        self.filter_type = filter_type
        self.filter_config = filter_config
        self.noise_reduction_method = noise_reduction_method
        self.noise_reduction_config = noise_reduction_config
        self.respiratory_mode = respiratory_mode


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
    compute_duration(peaks1, peaks2)
        Compute the mean duration between two sets of peaks (e.g., systolic and diastolic peaks).
    compute_volume(peaks1, peaks2)
        Compute the area under the curve between two sets of peaks.
    compute_amplitude_variability(peaks)
        Compute the variability of the amplitudes at the given peak locations.
    compute_slope(peaks1, peaks2)
        Compute the mean slope between two sets of peaks.
    detect_dicrotic_notch()
        Detect the dicrotic notch in the signal based on its second derivative.
    compute_skewness()
        Compute the skewness of the signal.
    compute_peak_trend(peaks)
        Compute the trend slope of peak amplitudes over time.
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

    def preprocess_signal(self, preprocess_config):
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
        # Apply signal filtering
        signal_filter = SignalFiltering(self.signal)

        if preprocess_config.filter_type == "bandpass":
            filtered_signal = signal_filter.bandpass(
                preprocess_config.filter_config["lowcut"],
                preprocess_config.filter_config["highcut"],
                fs=self.fs,
                order=preprocess_config.filter_config["order"],
                filter_type=preprocess_config.filter_type
            )
        elif preprocess_config.filter_type == "butterworth":
            filtered_signal = signal_filter.butterworth(
                cutoff=preprocess_config.filter_config["highcut"],
                fs=self.fs,
                order=preprocess_config.filter_config["order"],
                btype="low"
            )
        else:
            raise ValueError(f"Unsupported filter type: {preprocess_config.filter_type}")

        # Ensure the filtered signal is still float64 and clip any extreme values
        filtered_signal = np.clip(np.asarray(filtered_signal, dtype=np.float64), -1e10, 1e10)

        # Apply noise reduction
        if preprocess_config.noise_reduction_method == "wavelet":
            clean_signal = wavelet_denoising(filtered_signal, wavelet_name="haar", level=1)
        elif preprocess_config.noise_reduction_method == "savgol":
            clean_signal = savgol_denoising(filtered_signal, window_length=5, polyorder=2)
        elif preprocess_config.noise_reduction_method == "median":
            clean_signal = median_denoising(filtered_signal, kernel_size=3)
        elif preprocess_config.noise_reduction_method == "gaussian":
            clean_signal = gaussian_denoising(filtered_signal, sigma=1.0)
        else:
            raise ValueError(f"Unsupported noise reduction method: {preprocess_config.noise_reduction_method}")

        return clean_signal

    def compute_duration(self, peaks1, peaks2):
        """
        Compute the mean duration between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks (e.g., systolic peaks).
        peaks2 : numpy.ndarray
            The second set of peaks (e.g., diastolic peaks).

        Returns
        -------
        duration : float
            The mean duration between the two sets of peaks in seconds.

        Examples
        --------
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(np.random.randn(1000))
        >>> duration = extractor.compute_duration(peaks1, peaks2)
        >>> print(duration)
        """
        durations = [(p2 - p1) / self.fs for p1, p2 in zip(peaks1, peaks2) if p1 < p2]
        return np.mean(durations) if durations else 0.0

    def compute_volume(self, peaks1, peaks2):
        """
        Compute the area under the curve between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks (e.g., systolic peaks).
        peaks2 : numpy.ndarray
            The second set of peaks (e.g., diastolic peaks).

        Returns
        -------
        volume : float
            The mean area between the two sets of peaks.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> volume = extractor.compute_volume(peaks1, peaks2)
        >>> print(volume)
        """
        areas = [simps(self.signal[p1:p2]) for p1, p2 in zip(peaks1, peaks2) if p1 < p2]
        return np.mean(areas) if areas else 0.0

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

    def compute_slope(self, peaks1, peaks2):
        """
        Compute the mean slope between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks.
        peaks2 : numpy.ndarray
            The second set of peaks.

        Returns
        -------
        slope : float
            The mean slope between the two sets of peaks.

        Examples
        --------
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(np.random.randn(1000))
        >>> slope = extractor.compute_slope(peaks1, peaks2)
        >>> print(slope)
        """
        slopes = [
            (self.signal[p2] - self.signal[p1]) / (p2 - p1)
            for p1, p2 in zip(peaks1, peaks2)
            if p1 < p2
        ]
        return np.mean(slopes) if slopes else 0.0

    def detect_dicrotic_notch(self):
        """
        Detect the dicrotic notch in the PPG signal.

        Returns
        -------
        dicrotic_notch_locs : numpy.ndarray
            The locations of the detected dicrotic notches.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> dicrotic_notches = extractor.detect_dicrotic_notch()
        >>> print(dicrotic_notches)
        """
        second_derivative = np.diff(self.signal, n=2)
        notch_locs = PeakDetection(
            second_derivative, method="rel_extrema"
        ).detect_peaks()
        return notch_locs

    def compute_skewness(self):
        """
        Compute the skewness of the signal.

        Returns
        -------
        skewness : float
            The skewness of the signal.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> skewness = extractor.compute_skewness()
        >>> print(skewness)
        """
        return skew(self.signal)

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

    def extract_features(self, signal_type="ECG", preprocess_config=None):
        """
        Extract all physiological features from the signal for either ECG or PPG.

        Parameters
        ----------
        signal_type : str, optional
            The type of signal ("ECG" or "PPG"). Default is "ECG".
        preprocess_config : PreprocessConfig, optional
            The configuration object for signal preprocessing. If None, default settings are used.

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
        clean_signal = self.preprocess_signal(preprocess_config)

        if signal_type == "PPG":
            peaks = PeakDetection(
                clean_signal, method="ppg_first_derivative"
            ).detect_peaks()
            dicrotic_notch_locs = self.detect_dicrotic_notch()
            systolic_duration = self.compute_duration(peaks[:-1], peaks[1:])
            diastolic_duration = self.compute_duration(peaks[1:], peaks[2:])
            systolic_area = self.compute_volume(peaks[:-1], peaks[1:])
            diastolic_area = self.compute_volume(peaks[1:], peaks[2:])
            systolic_variability = self.compute_amplitude_variability(peaks[:-1])
            diastolic_variability = self.compute_amplitude_variability(peaks[1:])
            signal_skewness = self.compute_skewness()
            peak_trend_slope = self.compute_peak_trend(peaks)

            return {
                "systolic_duration": systolic_duration,
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
            qrs_duration = self.compute_duration(r_peaks[:-1], r_peaks[1:])
            qrs_area = self.compute_volume(r_peaks[:-1], r_peaks[1:])
            t_wave_area = self.compute_volume(r_peaks[1:], r_peaks[2:])
            qrs_amplitude = np.mean(self.signal[r_peaks])
            qrs_slope = self.compute_slope(r_peaks[:-1], r_peaks[1:])
            qrs_t_ratio = qrs_area / t_wave_area if t_wave_area != 0 else 0
            signal_skewness = self.compute_skewness()
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
