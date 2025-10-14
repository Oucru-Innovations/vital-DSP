# src/vitalDSP/utils/error_recovery.py
"""
Error recovery mechanisms for vitalDSP functions.

This module provides fallback strategies and error recovery mechanisms
to ensure robust signal processing operations even when primary methods fail.
"""

import numpy as np
import warnings
from typing import Union, Callable, Optional, Tuple
from ..data_processing.validation import SignalValidator


class ErrorRecovery:
    """
    Error recovery mechanisms for vitalDSP functions.

    This class provides static methods for implementing fallback strategies
    and graceful degradation when primary signal processing methods fail.
    """

    @staticmethod
    def with_fallback_methods(
        primary_method: Callable, fallback_methods: list, *args, **kwargs
    ) -> any:
        """
        Execute primary method with automatic fallback to alternative methods.

        Parameters
        ----------
        primary_method : callable
            Primary method to try first
        fallback_methods : list
            List of fallback methods to try if primary fails
        *args : tuple
            Arguments to pass to methods
        **kwargs : dict
            Keyword arguments to pass to methods

        Returns
        -------
        any
            Result from first successful method

        Raises
        ------
        RuntimeError
            If all methods fail
        """
        methods_to_try = [primary_method] + fallback_methods

        for i, method in enumerate(methods_to_try):
            try:
                result = method(*args, **kwargs)

                # Validate result
                if result is not None:
                    # Check if result is valid (not all NaN)
                    is_valid = True
                    if isinstance(result, dict):
                        # For dictionaries, check if any values are not NaN
                        for value in result.values():
                            if isinstance(value, np.ndarray):
                                if np.all(np.isnan(value)):
                                    is_valid = False
                                    break
                            elif isinstance(value, (int, float)) and np.isnan(value):
                                is_valid = False
                                break
                    elif isinstance(result, np.ndarray):
                        is_valid = not np.all(np.isnan(result))
                    elif isinstance(result, (int, float)):
                        is_valid = not np.isnan(result)

                    if is_valid:
                        if i > 0:  # Used fallback method
                            warnings.warn(
                                f"Primary method failed, used fallback method {i}"
                            )
                        return result

            except Exception as e:
                if i == len(methods_to_try) - 1:  # Last method
                    raise RuntimeError(f"All methods failed. Last error: {e}")
                warnings.warn(f"Method {i} failed: {e}. Trying next method.")
                continue

        raise RuntimeError("All methods failed")

    @staticmethod
    def respiratory_rate_with_fallback(
        signal: np.ndarray, fs: float, method: str = "counting", **kwargs
    ) -> float:
        """
        Compute respiratory rate with automatic fallback methods.

        Parameters
        ----------
        signal : np.ndarray
            Input respiratory signal
        fs : float
            Sampling frequency
        method : str, optional
            Primary method to use (default: "counting")
        **kwargs : dict
            Additional parameters for methods

        Returns
        -------
        float
            Estimated respiratory rate
        """
        from vitalDSP.respiratory_analysis.respiratory_analysis import (
            RespiratoryAnalysis,
        )

        # Define fallback order based on robustness
        fallback_order = [
            "counting",
            "peaks",
            "zero_crossing",
            "time_domain",
            "frequency_domain",
            "fft_based",
        ]

        # Remove primary method from fallback list
        if method in fallback_order:
            fallback_order.remove(method)

        ra = RespiratoryAnalysis(signal, fs)

        def try_method(method_name):
            return ra.compute_respiratory_rate(method=method_name, **kwargs)

        fallback_methods = [lambda: try_method(m) for m in fallback_order]

        return ErrorRecovery.with_fallback_methods(
            lambda: try_method(method), fallback_methods
        )

    @staticmethod
    def filtering_with_fallback(
        signal: np.ndarray, filter_type: str = "butterworth", **kwargs
    ) -> np.ndarray:
        """
        Apply filtering with automatic fallback to simpler methods.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        filter_type : str, optional
            Primary filter type (default: "butterworth")
        **kwargs : dict
            Filter parameters

        Returns
        -------
        np.ndarray
            Filtered signal
        """
        from vitalDSP.filtering.signal_filtering import SignalFiltering

        # Define fallback order (from complex to simple)
        fallback_order = [
            "butterworth",
            "chebyshev",
            "elliptic",
            "moving_average",
            "gaussian",
        ]

        if filter_type in fallback_order:
            fallback_order.remove(filter_type)

        sf = SignalFiltering(signal)

        def try_filter(filter_name):
            if filter_name == "butterworth":
                return sf.butterworth(**kwargs)
            elif filter_name == "chebyshev":
                return sf.chebyshev(**kwargs)
            elif filter_name == "elliptic":
                return sf.elliptic(**kwargs)
            elif filter_name == "moving_average":
                return sf.moving_average(kwargs.get("window_size", 5))
            elif filter_name == "gaussian":
                return sf.gaussian(kwargs.get("sigma", 1.0))

        fallback_methods = [lambda: try_filter(f) for f in fallback_order]

        return ErrorRecovery.with_fallback_methods(
            lambda: try_filter(filter_type), fallback_methods
        )

    @staticmethod
    def feature_extraction_with_fallback(
        signal: np.ndarray, feature_type: str = "time_domain", **kwargs
    ) -> dict:
        """
        Extract features with automatic fallback to simpler methods.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        feature_type : str, optional
            Primary feature type (default: "time_domain")
        **kwargs : dict
            Feature extraction parameters

        Returns
        -------
        dict
            Extracted features
        """
        from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
        from vitalDSP.physiological_features.frequency_domain import (
            FrequencyDomainFeatures,
        )

        # Define fallback order
        fallback_order = ["time_domain", "frequency_domain", "basic_stats"]

        if feature_type in fallback_order:
            fallback_order.remove(feature_type)

        def try_time_domain():
            tdf = TimeDomainFeatures(signal)
            return {
                "sdnn": tdf.compute_sdnn(),
                "rmssd": tdf.compute_rmssd(),
                "mean_nn": tdf.compute_mean_nn(),
                "median_nn": tdf.compute_median_nn(),
            }

        def try_frequency_domain():
            fdf = FrequencyDomainFeatures(signal, kwargs.get("fs", 1.0))
            return {
                "lf": fdf.compute_lf(),
                "hf": fdf.compute_hf(),
                "lf_hf_ratio": fdf.compute_lf_hf_ratio(),
            }

        def try_basic_stats():
            return {
                "mean": np.mean(signal),
                "std": np.std(signal),
                "min": np.min(signal),
                "max": np.max(signal),
                "range": np.max(signal) - np.min(signal),
            }

        method_map = {
            "time_domain": try_time_domain,
            "frequency_domain": try_frequency_domain,
            "basic_stats": try_basic_stats,
        }

        fallback_methods = [method_map[f] for f in fallback_order if f in method_map]

        # If feature_type is invalid, use fallback methods directly
        if feature_type not in method_map:
            return ErrorRecovery.with_fallback_methods(
                None, fallback_methods  # No primary method
            )

        return ErrorRecovery.with_fallback_methods(
            method_map[feature_type], fallback_methods
        )

    @staticmethod
    def transform_with_fallback(
        signal: np.ndarray, transform_type: str = "fft", **kwargs
    ) -> np.ndarray:
        """
        Apply transformation with automatic fallback methods.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        transform_type : str, optional
            Primary transform type (default: "fft")
        **kwargs : dict
            Transform parameters

        Returns
        -------
        np.ndarray
            Transformed signal
        """
        from vitalDSP.transforms.fourier_transform import FourierTransform
        from vitalDSP.transforms.wavelet_transform import WaveletTransform

        # Define fallback order
        fallback_order = ["fft", "dwt", "hilbert", "identity"]

        if transform_type in fallback_order:
            fallback_order.remove(transform_type)

        def try_fft():
            ft = FourierTransform(signal)
            return ft.compute_dft()

        def try_dwt():
            wt = WaveletTransform(signal)
            return wt.perform_wavelet_transform(kwargs.get("level", 1))

        def try_hilbert():
            from vitalDSP.transforms.hilbert_transform import HilbertTransform

            ht = HilbertTransform(signal)
            return ht.compute_hilbert()

        def try_identity():
            return signal.copy()

        method_map = {
            "fft": try_fft,
            "dwt": try_dwt,
            "hilbert": try_hilbert,
            "identity": try_identity,
        }

        fallback_methods = [method_map[f] for f in fallback_order if f in method_map]

        return ErrorRecovery.with_fallback_methods(
            method_map[transform_type], fallback_methods
        )

    @staticmethod
    def quality_assessment_with_fallback(
        signal: np.ndarray, assessment_type: str = "snr", **kwargs
    ) -> float:
        """
        Perform quality assessment with automatic fallback methods.

        Parameters
        ----------
        signal : np.ndarray
            Input signal
        assessment_type : str, optional
            Primary assessment type (default: "snr")
        **kwargs : dict
            Assessment parameters

        Returns
        -------
        float
            Quality metric value
        """
        from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality

        # Define fallback order
        fallback_order = ["snr", "psnr", "mse", "variance"]

        if assessment_type in fallback_order:
            fallback_order.remove(assessment_type)

        def try_snr():
            sq = SignalQuality(signal, signal)  # Assuming no reference
            return sq.snr()

        def try_psnr():
            sq = SignalQuality(signal, signal)
            return sq.psnr()

        def try_mse():
            sq = SignalQuality(signal, signal)
            return sq.mse()

        def try_variance():
            return np.var(signal)

        method_map = {
            "snr": try_snr,
            "psnr": try_psnr,
            "mse": try_mse,
            "variance": try_variance,
        }

        fallback_methods = [method_map[f] for f in fallback_order if f in method_map]

        return ErrorRecovery.with_fallback_methods(
            method_map[assessment_type], fallback_methods
        )


def robust_signal_processing(func):
    """
    Decorator for automatic error recovery in signal processing functions.

    This decorator wraps signal processing functions with automatic
    error recovery and fallback mechanisms.

    Parameters
    ----------
    func : callable
        Function to wrap with error recovery

    Returns
    -------
    callable
        Wrapped function with error recovery
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Primary method failed: {e}. Attempting recovery.")

            # Try to recover by simplifying parameters
            try:
                # Reduce complexity of parameters
                simplified_kwargs = kwargs.copy()

                # Reduce filter order if present
                if "order" in simplified_kwargs and simplified_kwargs["order"] > 2:
                    simplified_kwargs["order"] = 2

                # Reduce window size if present
                if (
                    "window_size" in simplified_kwargs
                    and simplified_kwargs["window_size"] > 10
                ):
                    simplified_kwargs["window_size"] = 10

                # Reduce iterations if present
                if (
                    "iterations" in simplified_kwargs
                    and simplified_kwargs["iterations"] > 1
                ):
                    simplified_kwargs["iterations"] = 1

                return func(*args, **simplified_kwargs)

            except Exception as e2:
                warnings.warn(f"Recovery attempt failed: {e2}")
                raise e  # Re-raise original exception

    return wrapper


# Convenience functions for common error recovery scenarios
def safe_respiratory_rate(signal: np.ndarray, fs: float, **kwargs) -> float:
    """Safely compute respiratory rate with automatic fallback."""
    return ErrorRecovery.respiratory_rate_with_fallback(signal, fs, **kwargs)


def safe_filtering(
    signal: np.ndarray, filter_type: str = "butterworth", **kwargs
) -> np.ndarray:
    """Safely apply filtering with automatic fallback."""
    return ErrorRecovery.filtering_with_fallback(signal, filter_type, **kwargs)


def safe_feature_extraction(
    signal: np.ndarray, feature_type: str = "time_domain", **kwargs
) -> dict:
    """Safely extract features with automatic fallback."""
    return ErrorRecovery.feature_extraction_with_fallback(
        signal, feature_type, **kwargs
    )
