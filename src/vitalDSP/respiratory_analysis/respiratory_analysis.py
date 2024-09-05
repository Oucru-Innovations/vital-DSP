from vitalDSP.preprocess.preprocess import preprocess_signal
from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import (
    peak_detection_rr,
)
from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import (
    frequency_domain_rr,
)
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np


class PreprocessConfig:
    """
    Configuration class for signal preprocessing, which includes filtering and noise reduction parameters.

    Attributes
    ----------
    filter_type : str
        The type of filtering to apply ('bandpass', 'butterworth', 'chebyshev', 'elliptic').
    noise_reduction_method : str
        The noise reduction method to apply ('wavelet', 'savgol', 'median', 'gaussian', 'moving_average').
    lowcut : float
        The lower cutoff frequency for filtering.
    highcut : float
        The upper cutoff frequency for filtering.
    order : int
        The order of the filter.
    wavelet_name : str
        The name of the wavelet to use for wavelet-based noise reduction.
    level : int
        The level of wavelet decomposition.
    window_length : int
        The window length for Savitzky-Golay filtering.
    polyorder : int
        The polynomial order for Savitzky-Golay filtering.
    kernel_size : int
        The kernel size for median filtering.
    sigma : float
        The standard deviation for Gaussian filtering.
    respiratory_mode: bool
        Apply the preprocessing function specifically for respiratory signals (e.g., PPG or ECG-derived respiration).
    repreprocess: bool
        Re preprocessing function
    """

    def __init__(
        self,
        filter_type="bandpass",
        noise_reduction_method="wavelet",
        lowcut=0.1,
        highcut=0.5,
        order=4,
        wavelet_name="haar",
        level=1,
        window_length=5,
        polyorder=2,
        kernel_size=3,
        sigma=1.0,
        respiratory_mode=True,
        repreprocess=False,
    ):
        self.filter_type = filter_type
        self.noise_reduction_method = noise_reduction_method
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.wavelet_name = wavelet_name
        self.level = level
        self.window_length = window_length
        self.polyorder = polyorder
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.respiratory_mode = respiratory_mode
        self.repreprocess = repreprocess


class RespiratoryAnalysis:
    """
    A class for analyzing respiratory patterns in physiological signals (e.g., PPG, ECG),
    with built-in preprocessing, filtering, and noise reduction options.

    Attributes
    ----------
    signal : numpy.ndarray
        The raw PPG or ECG signal to analyze.
    fs : int
        The sampling frequency of the signal in Hz.
    """

    def __init__(self, signal, fs=256):
        """
        Initializes the RespiratoryAnalysis object.

        Parameters
        ----------
        signal : numpy.ndarray
            The raw PPG or ECG signal to analyze.
        fs : int, optional
            The sampling frequency of the signal in Hz. Default is 1000 Hz.
        """
        self.signal = signal
        self.fs = fs

    def compute_respiratory_rate(
        self,
        method="peaks",
        correction_method=None,
        min_breath_duration=0.5,
        max_breath_duration=6,
        preprocess_config=None,
    ):
        """
        Computes the respiratory rate from the signal after applying preprocessing,
        using peak detection or zero-crossing methods.

        Parameters
        ----------
        method : str, optional
            Method used for breath detection ('peaks', 'zero_crossing'). Default is 'peaks'.
        correction_method : str, optional
            Method for correcting false detections ('interpolation', 'adaptive_threshold'). Default is None.
        min_breath_duration : float, optional
            Minimum breath duration in seconds. Default is 0.5s (30 breaths/min).
        max_breath_duration : float, optional
            Maximum breath duration in seconds. Default is 6s (10 breaths/min).
        preprocess_config : PreprocessConfig, optional
            Configuration for signal preprocessing (filtering and noise reduction). Default is None, and a default PreprocessConfig will be used.

        Returns
        -------
        float
            The estimated respiratory rate in breaths per minute.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)
        >>> ra = RespiratoryAnalysis(signal, fs=1000)
        >>> preprocess_config = PreprocessConfig(filter_type='bandpass', noise_reduction_method='wavelet')
        >>> respiratory_rate = ra.compute_respiratory_rate(method='peaks', preprocess_config=preprocess_config)
        >>> print(f"Respiratory Rate: {respiratory_rate} breaths per minute")
        """
        if preprocess_config is None:
            preprocess_config = PreprocessConfig()

        # Preprocess the signal
        preprocessed_signal = preprocess_signal(
            signal=self.signal,
            sampling_rate=self.fs,
            filter_type=preprocess_config.filter_type,
            noise_reduction_method=preprocess_config.noise_reduction_method,
            lowcut=preprocess_config.lowcut,
            highcut=preprocess_config.highcut,
            order=preprocess_config.order,
            wavelet_name=preprocess_config.wavelet_name,
            level=preprocess_config.level,
            window_length=preprocess_config.window_length,
            polyorder=preprocess_config.polyorder,
            kernel_size=preprocess_config.kernel_size,
            sigma=preprocess_config.sigma,
            respiratory_mode=preprocess_config.respiratory_mode,
            repreprocess=preprocess_config.repreprocess,
        )

        # Detect breaths using the chosen method
        if method == "peaks":
            breath_intervals = self._detect_breaths_by_peaks(
                preprocessed_signal, min_breath_duration, max_breath_duration
            )
        elif method == "zero_crossing":
            breath_intervals = self._detect_breaths_by_zero_crossing(
                preprocessed_signal, min_breath_duration, max_breath_duration
            )
        elif method == "time_domain":
            respiratory_rate = time_domain_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=preprocess_config.repreprocess,
                filter_type=preprocess_config.filter_type,
                lowcut=preprocess_config.lowcut,
                highcut=preprocess_config.highcut,
            )
            return respiratory_rate
        elif method == "frequency_domain":
            respiratory_rate = frequency_domain_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=preprocess_config.repreprocess,
                filter_type=preprocess_config.filter_type,
                lowcut=preprocess_config.lowcut,
                highcut=preprocess_config.highcut,
            )
            return respiratory_rate
        elif method == "fft_based":
            respiratory_rate = fft_based_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=preprocess_config.repreprocess,
                filter_type=preprocess_config.filter_type,
                lowcut=preprocess_config.lowcut,
                highcut=preprocess_config.highcut,
            )
            return respiratory_rate
        elif method == "counting":
            respiratory_rate = peak_detection_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=preprocess_config.repreprocess,
                filter_type=preprocess_config.filter_type,
                lowcut=preprocess_config.lowcut,
                highcut=preprocess_config.highcut,
            )
            return respiratory_rate
        else:
            raise ValueError("Invalid method. Choose 'peaks' or 'zero_crossing'.")

        # Apply correction method if specified
        if correction_method == "interpolation":
            breath_intervals = self._correct_by_interpolation(breath_intervals)
        elif correction_method == "adaptive_threshold":
            breath_intervals = self._correct_by_adaptive_threshold(breath_intervals)

        # Compute respiratory rate from breath intervals
        respiratory_rate = (
            60 / np.mean(breath_intervals) if len(breath_intervals) > 0 else 0
        )
        return respiratory_rate

    def _detect_breaths_by_peaks(
        self, preprocessed_signal, min_breath_duration, max_breath_duration
    ):
        """
        Detects breaths by finding peaks in the preprocessed signal.

        Parameters
        ----------
        preprocessed_signal : numpy.ndarray
            The preprocessed signal.
        min_breath_duration : float
            Minimum breath duration in seconds.
        max_breath_duration : float
            Maximum breath duration in seconds.

        Returns
        -------
        np.ndarray
            The intervals between breaths in seconds.
        """
        min_distance = int(min_breath_duration * self.fs)
        peaks, _ = find_peaks(preprocessed_signal, distance=min_distance)
        breath_intervals = np.diff(peaks) / self.fs
        return breath_intervals[
            (breath_intervals > min_breath_duration)
            & (breath_intervals < max_breath_duration)
        ]

    def _detect_breaths_by_zero_crossing(
        self, preprocessed_signal, min_breath_duration, max_breath_duration
    ):
        """
        Detects breaths by finding zero-crossings in the preprocessed signal.

        Parameters
        ----------
        preprocessed_signal : numpy.ndarray
            The preprocessed signal.
        min_breath_duration : float
            Minimum breath duration in seconds.
        max_breath_duration : float
            Maximum breath duration in seconds.

        Returns
        -------
        np.ndarray
            The intervals between breaths in seconds.
        """
        derivative_signal = np.diff(preprocessed_signal)
        zero_crossings = np.where(np.diff(np.sign(derivative_signal)))[0]
        breath_intervals = np.diff(zero_crossings) / self.fs
        return breath_intervals[
            (breath_intervals > min_breath_duration)
            & (breath_intervals < max_breath_duration)
        ]

    def _correct_by_interpolation(self, intervals):
        """
        Corrects false breath detections using linear interpolation.

        Parameters
        ----------
        intervals : np.ndarray
            The intervals between breaths.

        Returns
        -------
        np.ndarray
            The corrected intervals after interpolation.
        """
        outliers = np.abs(intervals - np.median(intervals)) > 1.5 * np.std(intervals)
        if np.any(outliers):
            x = np.arange(len(intervals))
            valid_points = x[~outliers]
            valid_intervals = intervals[~outliers]
            f_interp = interp1d(
                valid_points, valid_intervals, kind="linear", fill_value="extrapolate"
            )
            intervals[outliers] = f_interp(x[outliers])
        return intervals

    def _correct_by_adaptive_threshold(self, intervals, threshold=150):
        """
        Corrects false breath detections using an adaptive threshold method.

        Parameters
        ----------
        intervals : np.ndarray
            The intervals between breaths.
        threshold : int, optional
            Threshold for detecting false intervals. Default is 150 ms.

        Returns
        -------
        np.ndarray
            The corrected intervals after applying the adaptive threshold.
        """
        mean_interval = np.mean(intervals)
        valid_intervals = intervals[
            (intervals > mean_interval - threshold / 1000)
            & (intervals < mean_interval + threshold / 1000)
        ]
        return valid_intervals
