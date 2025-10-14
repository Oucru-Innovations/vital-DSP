# src/vitalDSP/utils/adaptive_parameters.py
"""
Adaptive parameter adjustment utilities for vitalDSP functions.

This module provides intelligent parameter adjustment based on signal
characteristics, ensuring optimal performance across different signal
types and conditions.
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks


@dataclass
class SignalCharacteristics:
    """Signal characteristics data class."""

    length: int
    sampling_rate: float
    mean: float
    std: float
    skewness: float
    kurtosis: float
    dynamic_range: float
    signal_to_noise_ratio: float
    dominant_frequency: float
    spectral_centroid: float
    zero_crossing_rate: float
    peak_count: int
    is_stationary: bool
    noise_level: str  # 'low', 'medium', 'high'
    signal_type: str  # 'ecg', 'ppg', 'eeg', 'respiratory', 'unknown'


class AdaptiveParameterAdjuster:
    """
    Adaptive parameter adjustment based on signal characteristics.

    This class analyzes signal characteristics and automatically adjusts
    parameters for optimal performance across different signal types.
    """

    def __init__(self):
        """Initialize the adaptive parameter adjuster."""
        self.signal_characteristics = None
        self.parameter_cache = {}

    def analyze_signal(
        self, signal: np.ndarray, fs: float = 1.0
    ) -> SignalCharacteristics:
        """
        Analyze signal characteristics for adaptive parameter adjustment.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        fs : float, optional
            Sampling frequency (default: 1.0)

        Returns
        -------
        SignalCharacteristics
            Analyzed signal characteristics
        """
        signal = np.asarray(signal)

        # Basic statistics
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        skewness = stats.skew(signal)
        kurtosis = stats.kurtosis(signal)
        dynamic_range = np.max(signal) - np.min(signal)

        # Signal-to-noise ratio estimation
        signal_power = np.mean(signal**2)
        noise_estimate = np.std(np.diff(signal))
        snr = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))

        # Frequency domain analysis
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / fs)
        magnitude = np.abs(fft_signal)

        # Dominant frequency
        positive_freqs = freqs[: len(freqs) // 2]
        positive_magnitude = magnitude[: len(magnitude) // 2]
        dominant_freq_idx = np.argmax(positive_magnitude[1:]) + 1  # Skip DC component
        dominant_frequency = positive_freqs[dominant_freq_idx]

        # Spectral centroid
        spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(
            positive_magnitude
        )

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        zero_crossing_rate = zero_crossings / len(signal)

        # Peak count
        peaks, _ = find_peaks(signal, height=np.mean(signal))
        peak_count = len(peaks)

        # Stationarity test (simplified)
        n_segments = min(10, len(signal) // 100)
        if n_segments > 1:
            segment_size = len(signal) // n_segments
            segment_means = [
                np.mean(signal[i : i + segment_size])
                for i in range(0, len(signal), segment_size)
            ]
            is_stationary = np.std(segment_means) < std_val * 0.1
        else:
            is_stationary = True

        # Noise level classification
        if snr > 20:
            noise_level = "low"
        elif snr > 10:
            noise_level = "medium"
        else:
            noise_level = "high"

        # Signal type classification (simplified)
        signal_type = self._classify_signal_type(
            signal, fs, dominant_frequency, peak_count
        )

        characteristics = SignalCharacteristics(
            length=len(signal),
            sampling_rate=fs,
            mean=mean_val,
            std=std_val,
            skewness=skewness,
            kurtosis=kurtosis,
            dynamic_range=dynamic_range,
            signal_to_noise_ratio=snr,
            dominant_frequency=dominant_frequency,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            peak_count=peak_count,
            is_stationary=is_stationary,
            noise_level=noise_level,
            signal_type=signal_type,
        )

        self.signal_characteristics = characteristics
        return characteristics

    def _classify_signal_type(
        self, signal: np.ndarray, fs: float, dominant_freq: float, peak_count: int
    ) -> str:
        """Classify signal type based on characteristics."""
        # ECG characteristics: ~1-2 Hz dominant frequency, regular peaks
        if 0.5 <= dominant_freq <= 3.0 and peak_count > len(signal) * 0.01:
            return "ecg"

        # PPG characteristics: ~1-2 Hz dominant frequency, regular peaks
        elif 0.5 <= dominant_freq <= 3.0 and peak_count > len(signal) * 0.005:
            return "ppg"

        # Respiratory characteristics: ~0.1-0.5 Hz dominant frequency
        elif 0.05 <= dominant_freq <= 0.8:
            return "respiratory"

        # EEG characteristics: higher frequency content
        elif dominant_freq > 5.0:
            return "eeg"

        else:
            return "unknown"

    def adjust_filter_parameters(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust filter parameters based on signal characteristics.

        Parameters
        ----------
        base_params : dict
            Base filter parameters

        Returns
        -------
        dict
            Adjusted filter parameters
        """
        if self.signal_characteristics is None:
            return base_params

        char = self.signal_characteristics
        adjusted_params = base_params.copy()

        # Adjust filter order based on signal length and noise level
        if "order" in adjusted_params:
            base_order = adjusted_params["order"]

            if char.length < 1000:
                # Reduce order for short signals
                adjusted_params["order"] = max(2, base_order // 2)
            elif char.noise_level == "high":
                # Increase order for noisy signals
                adjusted_params["order"] = min(10, base_order + 2)
            elif char.noise_level == "low":
                # Reduce order for clean signals
                adjusted_params["order"] = max(2, base_order - 1)

        # Adjust cutoff frequency based on signal characteristics
        if "cutoff" in adjusted_params and "fs" in adjusted_params:
            base_cutoff = adjusted_params["cutoff"]
            fs = adjusted_params["fs"]
            nyquist = fs / 2

            # Adjust based on dominant frequency
            if char.dominant_frequency > 0:
                # Set cutoff to 2-3 times dominant frequency
                suggested_cutoff = char.dominant_frequency * 2.5
                adjusted_params["cutoff"] = min(suggested_cutoff, nyquist * 0.9)

            # Adjust based on signal type
            if char.signal_type == "ecg":
                # ECG typically needs 0.5-40 Hz bandpass
                adjusted_params["cutoff"] = min(40, adjusted_params["cutoff"])
            elif char.signal_type == "ppg":
                # PPG typically needs 0.5-10 Hz bandpass
                adjusted_params["cutoff"] = min(10, adjusted_params["cutoff"])
            elif char.signal_type == "respiratory":
                # Respiratory typically needs 0.1-2 Hz bandpass
                adjusted_params["cutoff"] = min(2, adjusted_params["cutoff"])

        # Adjust window size for moving average filters
        if "window_size" in adjusted_params:
            base_window_size = adjusted_params["window_size"]

            if char.length < 1000:
                # Reduce window size for short signals
                adjusted_params["window_size"] = max(3, base_window_size // 2)
            elif char.noise_level == "high":
                # Increase window size for noisy signals
                adjusted_params["window_size"] = min(
                    char.length // 10, base_window_size * 2
                )
            elif char.noise_level == "low":
                # Reduce window size for clean signals
                adjusted_params["window_size"] = max(3, base_window_size // 2)

        return adjusted_params

    def adjust_analysis_parameters(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust analysis parameters based on signal characteristics.

        Parameters
        ----------
        base_params : dict
            Base analysis parameters

        Returns
        -------
        dict
            Adjusted analysis parameters
        """
        if self.signal_characteristics is None:
            return base_params

        char = self.signal_characteristics
        adjusted_params = base_params.copy()

        # Adjust window size for windowed analysis
        if "window_size" in adjusted_params:
            base_window_size = adjusted_params["window_size"]

            # Adjust based on signal length
            if char.length < 1000:
                adjusted_params["window_size"] = max(10, char.length // 10)
            elif char.length > 10000:
                adjusted_params["window_size"] = min(
                    base_window_size * 2, char.length // 5
                )

            # Adjust based on signal type
            if char.signal_type == "ecg":
                # ECG needs longer windows for reliable analysis
                adjusted_params["window_size"] = max(
                    adjusted_params["window_size"], 1000
                )
            elif char.signal_type == "respiratory":
                # Respiratory signals need very long windows
                adjusted_params["window_size"] = max(
                    adjusted_params["window_size"], 2000
                )

        # Adjust step size for sliding window analysis
        if "step_size" in adjusted_params:
            base_step_size = adjusted_params["step_size"]

            if char.length < 1000:
                # Smaller step size for short signals
                adjusted_params["step_size"] = max(1, base_step_size // 2)
            elif char.is_stationary:
                # Larger step size for stationary signals
                adjusted_params["step_size"] = min(
                    base_step_size * 2, adjusted_params.get("window_size", 100) // 2
                )

        # Adjust threshold parameters
        if "threshold" in adjusted_params:
            base_threshold = adjusted_params["threshold"]

            # Adjust based on signal characteristics
            if char.noise_level == "high":
                # Increase threshold for noisy signals
                adjusted_params["threshold"] = base_threshold * 1.5
            elif char.noise_level == "low":
                # Decrease threshold for clean signals
                adjusted_params["threshold"] = base_threshold * 0.7

        return adjusted_params

    def adjust_feature_extraction_parameters(
        self, base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust feature extraction parameters based on signal characteristics.

        Parameters
        ----------
        base_params : dict
            Base feature extraction parameters

        Returns
        -------
        dict
            Adjusted feature extraction parameters
        """
        if self.signal_characteristics is None:
            return base_params

        char = self.signal_characteristics
        adjusted_params = base_params.copy()

        # Adjust DFA parameters
        if "order" in adjusted_params and "dfa" in str(base_params.get("method", "")):
            base_order = adjusted_params["order"]

            if char.length < 1000:
                # Reduce order for short signals
                adjusted_params["order"] = 1
            elif char.noise_level == "high":
                # Increase order for noisy signals
                adjusted_params["order"] = min(3, base_order + 1)

        # Adjust entropy parameters
        if "m" in adjusted_params:  # Sample entropy parameter
            base_m = adjusted_params["m"]

            if char.length < 1000:
                # Reduce m for short signals
                adjusted_params["m"] = max(1, base_m - 1)
            elif char.noise_level == "high":
                # Increase m for noisy signals
                adjusted_params["m"] = min(3, base_m + 1)

        # Adjust recurrence analysis parameters
        if "threshold" in adjusted_params and "recurrence" in str(
            base_params.get("method", "")
        ):
            base_threshold = adjusted_params["threshold"]

            if char.noise_level == "high":
                # Increase threshold for noisy signals
                adjusted_params["threshold"] = base_threshold * 1.5
            elif char.noise_level == "low":
                # Decrease threshold for clean signals
                adjusted_params["threshold"] = base_threshold * 0.7

        return adjusted_params

    def adjust_respiratory_parameters(
        self, base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust respiratory analysis parameters based on signal characteristics.

        Parameters
        ----------
        base_params : dict
            Base respiratory analysis parameters

        Returns
        -------
        dict
            Adjusted respiratory analysis parameters
        """
        if self.signal_characteristics is None:
            return base_params

        char = self.signal_characteristics
        adjusted_params = base_params.copy()

        # Adjust breath duration parameters
        if "min_breath_duration" in adjusted_params:
            base_min_duration = adjusted_params["min_breath_duration"]

            # Adjust based on dominant frequency
            if char.dominant_frequency > 0:
                suggested_min_duration = 1.0 / char.dominant_frequency * 0.5
                adjusted_params["min_breath_duration"] = max(
                    0.5, suggested_min_duration
                )

        if "max_breath_duration" in adjusted_params:
            base_max_duration = adjusted_params["max_breath_duration"]

            # Adjust based on dominant frequency
            if char.dominant_frequency > 0:
                suggested_max_duration = 1.0 / char.dominant_frequency * 3.0
                adjusted_params["max_breath_duration"] = min(
                    10.0, suggested_max_duration
                )

        # Adjust peak detection parameters
        if "distance" in adjusted_params:
            base_distance = adjusted_params["distance"]

            # Adjust based on sampling rate and expected frequency
            if char.signal_type == "respiratory":
                expected_period = (
                    1.0 / char.dominant_frequency
                    if char.dominant_frequency > 0
                    else 4.0
                )
                suggested_distance = int(expected_period * char.sampling_rate * 0.3)
                adjusted_params["distance"] = max(10, suggested_distance)

        return adjusted_params

    def get_optimal_parameters(
        self, operation_type: str, base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimal parameters for a specific operation type.

        Parameters
        ----------
        operation_type : str
            Type of operation ('filtering', 'analysis', 'feature_extraction', 'respiratory')
        base_params : dict
            Base parameters

        Returns
        -------
        dict
            Optimized parameters
        """
        if operation_type == "filtering":
            return self.adjust_filter_parameters(base_params)
        elif operation_type == "analysis":
            return self.adjust_analysis_parameters(base_params)
        elif operation_type == "feature_extraction":
            return self.adjust_feature_extraction_parameters(base_params)
        elif operation_type == "respiratory":
            return self.adjust_respiratory_parameters(base_params)
        else:
            return base_params

    def get_signal_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for signal processing based on characteristics.

        Returns
        -------
        dict
            Processing recommendations
        """
        if self.signal_characteristics is None:
            return {"message": "No signal characteristics available"}

        char = self.signal_characteristics
        recommendations = {
            "signal_type": char.signal_type,
            "noise_level": char.noise_level,
            "recommended_filters": [],
            "recommended_analysis_methods": [],
            "parameter_adjustments": {},
        }

        # Filter recommendations
        if char.noise_level == "high":
            recommendations["recommended_filters"].extend(
                ["moving_average", "gaussian", "butterworth"]
            )
        elif char.noise_level == "medium":
            recommendations["recommended_filters"].extend(["butterworth", "chebyshev"])
        else:
            recommendations["recommended_filters"].extend(["butterworth", "elliptic"])

        # Analysis method recommendations
        if char.signal_type == "ecg":
            recommendations["recommended_analysis_methods"].extend(
                ["hrv_analysis", "peak_detection", "morphology_analysis"]
            )
        elif char.signal_type == "ppg":
            recommendations["recommended_analysis_methods"].extend(
                ["peak_detection", "respiratory_rate", "quality_assessment"]
            )
        elif char.signal_type == "respiratory":
            recommendations["recommended_analysis_methods"].extend(
                ["respiratory_rate", "breath_detection", "frequency_analysis"]
            )

        # Parameter adjustments
        if char.length < 1000:
            recommendations["parameter_adjustments"]["reduce_complexity"] = True
            recommendations["parameter_adjustments"]["use_simple_methods"] = True

        if char.noise_level == "high":
            recommendations["parameter_adjustments"]["increase_filter_order"] = True
            recommendations["parameter_adjustments"]["use_robust_methods"] = True

        return recommendations


# Global adaptive parameter adjuster instance
_global_adjuster = AdaptiveParameterAdjuster()


def analyze_signal_characteristics(
    signal: np.ndarray, fs: float = 1.0
) -> SignalCharacteristics:
    """
    Analyze signal characteristics using global adjuster.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float, optional
        Sampling frequency (default: 1.0)

    Returns
    -------
    SignalCharacteristics
        Analyzed signal characteristics
    """
    return _global_adjuster.analyze_signal(signal, fs)


def get_optimal_parameters(
    operation_type: str, base_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get optimal parameters using global adjuster.

    Parameters
    ----------
    operation_type : str
        Type of operation
    base_params : dict
        Base parameters

    Returns
    -------
    dict
        Optimized parameters
    """
    return _global_adjuster.get_optimal_parameters(operation_type, base_params)


def get_signal_recommendations() -> Dict[str, Any]:
    """
    Get signal processing recommendations using global adjuster.

    Returns
    -------
    dict
        Processing recommendations
    """
    return _global_adjuster.get_signal_recommendations()


# Convenience functions for specific operations
def optimize_filtering_parameters(
    signal: np.ndarray, fs: float, base_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize filtering parameters for a signal."""
    _global_adjuster.analyze_signal(signal, fs)
    return _global_adjuster.adjust_filter_parameters(base_params)


def optimize_analysis_parameters(
    signal: np.ndarray, fs: float, base_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize analysis parameters for a signal."""
    _global_adjuster.analyze_signal(signal, fs)
    return _global_adjuster.adjust_analysis_parameters(base_params)


def optimize_feature_extraction_parameters(
    signal: np.ndarray, fs: float, base_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize feature extraction parameters for a signal."""
    _global_adjuster.analyze_signal(signal, fs)
    return _global_adjuster.adjust_feature_extraction_parameters(base_params)


def optimize_respiratory_parameters(
    signal: np.ndarray, fs: float, base_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize respiratory analysis parameters for a signal."""
    _global_adjuster.analyze_signal(signal, fs)
    return _global_adjuster.adjust_respiratory_parameters(base_params)
