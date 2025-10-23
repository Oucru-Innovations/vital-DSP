"""
Respiratory Analysis Module for Physiological Signal Processing

This module provides comprehensive respiratory analysis capabilities for physiological
signals including PPG, ECG, and other vital signs. It implements multiple methods
for respiratory rate estimation including time-domain counting, frequency-domain
analysis, and advanced signal processing techniques.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Multiple respiratory rate estimation methods
- Time-domain peak counting and interval analysis
- Frequency-domain FFT-based analysis
- Advanced preprocessing and filtering options
- Noise reduction and artifact handling
- Comprehensive respiratory pattern analysis

Examples:
--------
Basic respiratory rate estimation:
    >>> import numpy as np
    >>> from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
    >>> ppg_signal = np.random.randn(2000)  # Simulated PPG signal
    >>> resp_analysis = RespiratoryAnalysis(ppg_signal, fs=128)
    >>> rr_result = resp_analysis.compute_respiratory_rate(method="counting")
    >>> print(f"Respiratory rate: {rr_result['respiratory_rate']:.2f} breaths/min")

FFT-based analysis:
    >>> rr_fft = resp_analysis.compute_respiratory_rate(method="fft_based")
    >>> print(f"FFT-based RR: {rr_fft['respiratory_rate']:.2f} breaths/min")

With preprocessing:
    >>> from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
    >>> config = PreprocessConfig(filter_type="bandpass", lowcut=0.1, highcut=2.0)
    >>> rr_preprocessed = resp_analysis.compute_respiratory_rate(method="counting", preprocess_config=config)
    >>> print(f"Preprocessed RR: {rr_preprocessed['respiratory_rate']:.2f} breaths/min")
"""

import numpy as np
import warnings
from scipy.signal import find_peaks
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
from scipy.interpolate import interp1d


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
        # Validate sampling frequency
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")

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

        # Validate signal
        if len(self.signal) == 0:
            raise ValueError("Signal cannot be empty")

        if len(self.signal) < 10:
            import warnings

            warnings.warn(
                f"Signal length ({len(self.signal)}) is very short for reliable respiratory analysis. Results may be inaccurate."
            )

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
                preprocess=None,  # Signal already preprocessed, don't preprocess again
            )
            return respiratory_rate
        elif method == "frequency_domain":
            respiratory_rate = frequency_domain_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=None,  # Signal already preprocessed, don't preprocess again
            )
            return respiratory_rate
        elif method == "fft_based":
            respiratory_rate = fft_based_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=None,  # Signal already preprocessed, don't preprocess again
            )
            return respiratory_rate
        elif method == "counting":
            respiratory_rate = peak_detection_rr(
                preprocessed_signal,
                sampling_rate=self.fs,
                preprocess=None,  # Signal already preprocessed, don't preprocess again
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
        # Input validation
        if len(preprocessed_signal) == 0:
            return np.array([])
        if self.fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Validate duration parameters
        if min_breath_duration <= 0 or max_breath_duration <= 0:
            raise ValueError("Breath durations must be positive")
        if min_breath_duration >= max_breath_duration:
            raise ValueError("Minimum breath duration must be less than maximum")

        min_distance = int(min_breath_duration * self.fs)

        # Ensure minimum distance is reasonable
        if min_distance >= len(preprocessed_signal) // 2:
            warnings.warn("Minimum breath duration too large for signal length")
            return np.array([])

        try:
            peaks, _ = find_peaks(preprocessed_signal, distance=min_distance)
        except Exception as e:
            warnings.warn(f"Peak detection failed: {e}")
            return np.array([])

        # Check if we have enough peaks
        if len(peaks) < 2:
            warnings.warn("Insufficient peaks found for breath interval calculation")
            return np.array([])

        breath_intervals = np.diff(peaks) / self.fs
        valid_intervals = breath_intervals[
            (breath_intervals > min_breath_duration)
            & (breath_intervals < max_breath_duration)
        ]

        # Additional validation: check for reasonable intervals
        if len(valid_intervals) == 0:
            warnings.warn(
                "No valid breath intervals found within specified duration range"
            )

        return valid_intervals

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

    def compute_respiratory_rate_ensemble(self, preprocess_config=None, methods=None):
        """
        Compute respiratory rate using multiple methods and return consensus estimate.

        This method runs multiple RR estimation algorithms in parallel and combines
        their results using median consensus, providing a more robust estimate than
        any single method. It also returns quality metrics including standard deviation
        and confidence scores.

        Parameters
        ----------
        preprocess_config : PreprocessConfig, optional
            Configuration for signal preprocessing. If None, uses default configuration.
        methods : list of str, optional
            List of methods to use for ensemble estimation.
            Default: ['counting', 'fft_based', 'frequency_domain', 'time_domain']
            Available methods:
            - 'counting': Peak detection with interval analysis (peak_detection_rr)
            - 'fft_based': Fast Fourier Transform based estimation
            - 'frequency_domain': Welch PSD based estimation
            - 'time_domain': Autocorrelation based estimation
            - 'peaks': Peak detection (old method, uses _detect_breaths_by_peaks)
            - 'zero_crossing': Zero-crossing detection

        Returns
        -------
        dict
            Dictionary containing:
            - 'respiratory_rate': float, consensus RR estimate (median of valid methods)
            - 'individual_estimates': dict, per-method estimates (method_name: rr_value)
            - 'std': float, standard deviation across methods (measure of agreement)
            - 'confidence': float, confidence score (0-1 scale, higher = more agreement)
            - 'n_methods': int, number of methods that returned valid estimates
            - 'quality': str, quality rating ('high', 'medium', 'low')

        Examples
        --------
        >>> import numpy as np
        >>> from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
        >>> signal = np.sin(2 * np.pi * 0.25 * np.arange(0, 60, 1/128))  # 15 BPM signal
        >>> resp = RespiratoryAnalysis(signal, fs=128)
        >>> result = resp.compute_respiratory_rate_ensemble()
        >>> print(f"RR: {result['respiratory_rate']:.1f} BPM")
        >>> print(f"Confidence: {result['confidence']:.2f}")
        >>> print(f"Individual estimates: {result['individual_estimates']}")

        >>> # Use specific methods only
        >>> result = resp.compute_respiratory_rate_ensemble(methods=['fft_based', 'frequency_domain'])

        Notes
        -----
        The ensemble approach provides several advantages:
        1. More robust to method-specific failures
        2. Automatic outlier rejection (methods with implausible results)
        3. Quality/confidence metrics for result interpretation
        4. Better performance on noisy or challenging signals

        Confidence scoring:
        - confidence > 0.8: High agreement (std < 1 BPM) - Very reliable
        - confidence 0.5-0.8: Medium agreement (std 1-3 BPM) - Reliable
        - confidence < 0.5: Low agreement (std > 3 BPM) - Use with caution

        If fewer than 2 methods return valid estimates, confidence is automatically
        set to 0.3 (low) as consensus requires multiple opinions.

        References
        ----------
        .. [1] Charlton, P.H., et al. (2018). Breathing rate estimation from the
               electrocardiogram and photoplethysmogram: A review. IEEE Reviews in
               Biomedical Engineering, 11, 2-20.
        """
        if methods is None:
            methods = ["counting", "fft_based", "frequency_domain", "time_domain"]

        estimates = {}
        valid_estimates = []

        for method in methods:
            try:
                result = self.compute_respiratory_rate(
                    method=method, preprocess_config=preprocess_config
                )

                # Handle different return types (float vs dict)
                if isinstance(result, dict):
                    rr = result.get("respiratory_rate", 0)
                else:
                    rr = result

                # Only accept physiologically plausible estimates (6-40 BPM)
                if 6 <= rr <= 40:
                    estimates[method] = float(rr)
                    valid_estimates.append(rr)
                else:
                    estimates[method] = None
                    warnings.warn(
                        f"Method '{method}' returned implausible RR ({rr:.1f} BPM), "
                        f"excluded from ensemble"
                    )
            except Exception as e:
                estimates[method] = None
                warnings.warn(f"Method '{method}' failed with error: {e}")

        # Handle case with insufficient valid estimates
        if len(valid_estimates) == 0:
            return {
                "respiratory_rate": 0.0,
                "individual_estimates": estimates,
                "std": 0.0,
                "confidence": 0.0,
                "n_methods": 0,
                "quality": "failed",
            }

        if len(valid_estimates) == 1:
            return {
                "respiratory_rate": float(valid_estimates[0]),
                "individual_estimates": estimates,
                "std": 0.0,
                "confidence": 0.3,  # Low confidence with single method
                "n_methods": 1,
                "quality": "low",
            }

        # Use median for consensus (robust to outliers)
        consensus_rr = np.median(valid_estimates)
        std = np.std(valid_estimates)
        mean_rr = np.mean(valid_estimates)

        # Confidence based on agreement between methods
        # High confidence if std < 2 BPM, low confidence if std > 5 BPM
        if std < 1:
            confidence = 1.0
        elif std < 2:
            confidence = 0.9
        elif std < 3:
            confidence = 0.7
        elif std < 5:
            confidence = 0.5
        else:
            confidence = max(0, 0.5 - (std - 5) / 10)

        # Adjust confidence based on number of methods
        # More methods → higher confidence
        if len(valid_estimates) >= 4:
            confidence = min(1.0, confidence * 1.1)
        elif len(valid_estimates) == 2:
            confidence = confidence * 0.9

        # Determine quality rating
        if confidence >= 0.8 and len(valid_estimates) >= 3:
            quality = "high"
        elif confidence >= 0.5 or len(valid_estimates) >= 2:
            quality = "medium"
        else:
            quality = "low"

        return {
            "respiratory_rate": float(consensus_rr),
            "mean_rate": float(mean_rr),
            "individual_estimates": estimates,
            "std": float(std),
            "confidence": float(confidence),
            "n_methods": len(valid_estimates),
            "quality": quality,
            "method": "ensemble_median",
        }
