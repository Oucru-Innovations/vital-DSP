"""
Utility Functions Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Signal validation and error handling

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.data_processing.validation import Validation
    >>> signal = np.random.randn(1000)
    >>> processor = Validation(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

# src/vitalDSP/utils/validation.py
"""
Comprehensive input validation utilities for vitalDSP functions.

This module provides centralized validation functions to ensure robust
signal processing operations and prevent runtime errors.
"""

import numpy as np
import warnings
from typing import Union, Optional, Tuple


class SignalValidator:
    """
    Comprehensive signal validation utilities for vitalDSP functions.

    This class provides static methods for validating various aspects
    of input signals including length, data types, numerical properties,
    and signal characteristics.
    """

    @staticmethod
    def validate_signal(
        signal: Union[np.ndarray, list],
        min_length: int = 1,
        allow_nan: bool = False,
        allow_inf: bool = False,
        allow_empty: bool = False,
        signal_name: str = "signal",
    ) -> np.ndarray:
        """
        Comprehensive signal validation with multiple checks.

        Parameters
        ----------
        signal : array-like
            Input signal to validate
        min_length : int, optional
            Minimum required signal length (default: 1)
        allow_nan : bool, optional
            Whether to allow NaN values (default: False)
        allow_inf : bool, optional
            Whether to allow infinite values (default: False)
        allow_empty : bool, optional
            Whether to allow empty signals (default: False)
        signal_name : str, optional
            Name of signal for error messages (default: "signal")

        Returns
        -------
        np.ndarray
            Validated signal as numpy array

        Raises
        ------
        ValueError
            If signal fails validation checks
        TypeError
            If signal is not array-like
        """
        # Convert to numpy array
        if not isinstance(signal, np.ndarray):
            try:
                signal = np.array(signal)
            except Exception as e:
                raise TypeError(f"{signal_name} must be array-like: {e}")

        # Check if empty
        if len(signal) == 0:
            if not allow_empty:
                raise ValueError(f"{signal_name} cannot be empty")
            return signal

        # Check minimum length
        if len(signal) < min_length:
            raise ValueError(
                f"{signal_name} length {len(signal)} < minimum required {min_length}"
            )

        # Check for NaN values
        if not allow_nan and np.any(np.isnan(signal)):
            nan_count = np.sum(np.isnan(signal))
            raise ValueError(f"{signal_name} contains {nan_count} NaN values")

        # Check for infinite values
        if not allow_inf and np.any(np.isinf(signal)):
            inf_count = np.sum(np.isinf(signal))
            raise ValueError(f"{signal_name} contains {inf_count} infinite values")

        return signal

    @staticmethod
    def validate_signal_pair(
        signal1: Union[np.ndarray, list],
        signal2: Union[np.ndarray, list],
        require_same_length: bool = True,
        signal1_name: str = "signal1",
        signal2_name: str = "signal2",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate a pair of signals for compatibility.

        Parameters
        ----------
        signal1 : array-like
            First signal
        signal2 : array-like
            Second signal
        require_same_length : bool, optional
            Whether signals must have same length (default: True)
        signal1_name : str, optional
            Name of first signal for error messages
        signal2_name : str, optional
            Name of second signal for error messages

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Validated signal pair
        """
        sig1 = SignalValidator.validate_signal(signal1, signal_name=signal1_name)
        sig2 = SignalValidator.validate_signal(signal2, signal_name=signal2_name)

        if require_same_length and len(sig1) != len(sig2):
            raise ValueError(
                f"{signal1_name} length {len(sig1)} != {signal2_name} length {len(sig2)}"
            )

        return sig1, sig2

    @staticmethod
    def validate_frequency_parameters(
        cutoff: Union[float, list], fs: float, filter_type: str = "lowpass"
    ) -> Tuple[Union[float, list], float]:
        """
        Validate frequency parameters for filtering operations.

        Parameters
        ----------
        cutoff : float or list
            Cutoff frequency(ies)
        fs : float
            Sampling frequency
        filter_type : str, optional
            Type of filter for validation (default: "lowpass")

        Returns
        -------
        Tuple[Union[float, list], float]
            Validated cutoff and sampling frequency

        Raises
        ------
        ValueError
            If frequency parameters are invalid
        """
        # Validate sampling frequency
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {fs}")

        # Validate cutoff frequency
        if isinstance(cutoff, (list, tuple, np.ndarray)):
            cutoff = np.array(cutoff)
            if np.any(cutoff <= 0):
                raise ValueError("All cutoff frequencies must be positive")
            if np.any(cutoff >= fs / 2):
                raise ValueError(
                    f"Cutoff frequencies must be less than Nyquist frequency ({fs/2})"
                )

            if filter_type == "bandpass" and len(cutoff) != 2:
                raise ValueError(
                    "Bandpass filter requires exactly 2 cutoff frequencies"
                )
            if filter_type == "bandstop" and len(cutoff) != 2:
                raise ValueError(
                    "Bandstop filter requires exactly 2 cutoff frequencies"
                )
        else:
            if cutoff <= 0:
                raise ValueError(f"Cutoff frequency must be positive, got {cutoff}")
            if cutoff >= fs / 2:
                raise ValueError(
                    f"Cutoff frequency must be less than Nyquist frequency ({fs/2})"
                )

        return cutoff, fs

    @staticmethod
    def validate_filter_order(order: int, max_order: int = 20) -> int:
        """
        Validate filter order parameter.

        Parameters
        ----------
        order : int
            Filter order
        max_order : int, optional
            Maximum allowed order (default: 20)

        Returns
        -------
        int
            Validated filter order

        Raises
        ------
        ValueError
            If filter order is invalid
        """
        if not isinstance(order, int):
            raise TypeError(f"Filter order must be integer, got {type(order)}")

        if order <= 0:
            raise ValueError(f"Filter order must be positive, got {order}")

        if order > max_order:
            raise ValueError(f"Filter order {order} exceeds maximum {max_order}")

        return order

    @staticmethod
    def validate_window_parameters(
        window_size: int, signal_length: int, min_window_size: int = 2
    ) -> int:
        """
        Validate window size parameters.

        Parameters
        ----------
        window_size : int
            Window size
        signal_length : int
            Signal length
        min_window_size : int, optional
            Minimum window size (default: 2)

        Returns
        -------
        int
            Validated window size

        Raises
        ------
        ValueError
            If window parameters are invalid
        """
        if not isinstance(window_size, int):
            raise TypeError(f"Window size must be integer, got {type(window_size)}")

        if window_size < min_window_size:
            raise ValueError(f"Window size {window_size} < minimum {min_window_size}")

        if window_size > signal_length:
            raise ValueError(
                f"Window size {window_size} > signal length {signal_length}"
            )

        return window_size

    @staticmethod
    def validate_nn_intervals(nn_intervals: Union[np.ndarray, list]) -> np.ndarray:
        """
        Validate NN intervals for HRV analysis.

        Parameters
        ----------
        nn_intervals : array-like
            NN intervals in milliseconds

        Returns
        -------
        np.ndarray
            Validated NN intervals

        Raises
        ------
        ValueError
            If NN intervals are invalid
        """
        nn_intervals = SignalValidator.validate_signal(
            nn_intervals, min_length=2, allow_empty=False, signal_name="NN intervals"
        )

        # Check for reasonable physiological values
        if np.any(nn_intervals <= 0):
            raise ValueError("NN intervals must be positive")

        if np.any(nn_intervals > 3000):  # > 3 seconds is unusual
            warnings.warn(
                "Some NN intervals exceed 3000ms, which may indicate measurement errors"
            )

        if np.any(nn_intervals < 300):  # < 0.3 seconds is unusual
            warnings.warn(
                "Some NN intervals are less than 300ms, which may indicate measurement errors"
            )

        return nn_intervals

    @staticmethod
    def validate_threshold(threshold: float, threshold_type: str = "absolute") -> float:
        """
        Validate threshold parameters.

        Parameters
        ----------
        threshold : float
            Threshold value
        threshold_type : str, optional
            Type of threshold for validation (default: "absolute")

        Returns
        -------
        float
            Validated threshold

        Raises
        ------
        ValueError
            If threshold is invalid
        """
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"Threshold must be numeric, got {type(threshold)}")

        if threshold_type == "absolute" and threshold < 0:
            raise ValueError(
                f"Absolute threshold must be non-negative, got {threshold}"
            )

        if threshold_type == "relative" and (threshold < 0 or threshold > 1):
            raise ValueError(
                f"Relative threshold must be between 0 and 1, got {threshold}"
            )

        return float(threshold)


def validate_signal_input(func):
    """
    Decorator for automatic signal validation.

    This decorator can be applied to functions that process signals
    to automatically validate input parameters.

    Parameters
    ----------
    func : callable
        Function to decorate

    Returns
    -------
    callable
        Decorated function with validation
    """

    def wrapper(*args, **kwargs):
        # Extract signal parameter (usually first argument after self)
        if len(args) > 1:
            signal = args[1]  # Assuming signal is second argument
            args = list(args)
            args[1] = SignalValidator.validate_signal(signal)
            args = tuple(args)

        return func(*args, **kwargs)

    return wrapper


# Convenience functions for common validations
def validate_signal_length(
    signal: Union[np.ndarray, list], min_length: int = 1
) -> np.ndarray:
    """Quick signal length validation."""
    return SignalValidator.validate_signal(signal, min_length=min_length)


def validate_frequency_range(cutoff: float, fs: float) -> Tuple[float, float]:
    """Quick frequency range validation."""
    return SignalValidator.validate_frequency_parameters(cutoff, fs)


def validate_positive_parameter(value: float, param_name: str) -> float:
    """Quick positive parameter validation."""
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")
    return value
