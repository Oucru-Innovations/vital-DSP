from vitalDSP.preprocess.preprocess_operations import preprocess_signal
from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import (
    peak_detection_rr,
)
from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import (
    frequency_domain_rr,
)
from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np


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

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
    >>> 
    >>> # Example 1: Basic respiratory rate estimation
    >>> ppg_signal = np.random.randn(2000)  # Simulated PPG signal
    >>> resp_analysis = RespiratoryAnalysis(ppg_signal, fs=128)
    >>> rr_result = resp_analysis.compute_respiratory_rate(method="counting")
    >>> print(f"Respiratory rate: {rr_result['respiratory_rate']:.2f} breaths/min")
    >>> 
    >>> # Example 2: FFT-based respiratory rate estimation
    >>> rr_fft = resp_analysis.compute_respiratory_rate(method="fft_based")
    >>> print(f"FFT-based RR: {rr_fft['respiratory_rate']:.2f} breaths/min")
    >>> 
    >>> # Example 3: Respiratory rate with preprocessing
    >>> from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
    >>> config = PreprocessConfig(
    ...     filter_type="bandpass",
    ...     lowcut=0.1,
    ...     highcut=2.0,
    ...     noise_reduction_method="wavelet"
    ... )
    >>> rr_preprocessed = resp_analysis.compute_respiratory_rate(
    ...     method="peaks",
    ...     preprocess_config=config
    ... )
    >>> print(f"Preprocessed RR: {rr_preprocessed['respiratory_rate']:.2f} breaths/min")
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
        method="counting",
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
            Method used for breath detection. Options: 'peaks', 'zero_crossing', 'time_domain', 'frequency_domain', 'fft_based', 'counting'. Default is 'counting'.
        correction_method : str, optional
            Method for correcting false detections. Options: 'interpolation', 'adaptive_threshold'. Default is None.
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
