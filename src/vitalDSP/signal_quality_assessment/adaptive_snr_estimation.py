"""
Signal Quality Assessment Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.adaptive_snr_estimation import AdaptiveSnrEstimation
    >>> signal = np.random.randn(1000)
    >>> processor = AdaptiveSnrEstimation(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np


def sliding_window_snr(signal, window_size=100, step_size=50):
    """
    Estimate SNR adaptively using a sliding window approach.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    window_size : int, optional (default=100)
        The size of the sliding window.
    step_size : int, optional (default=50)
        The step size for moving the window.

    Returns
    -------
    snr_estimates : numpy.ndarray
        Array of SNR estimates for each window.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    >>> snr_estimates = sliding_window_snr(signal)
    >>> print(snr_estimates)
    """
    snr_estimates = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        window_signal = signal[i : i + window_size]
        signal_power = np.mean(window_signal**2)
        noise_power = np.var(window_signal)

        if noise_power == 0:  # Handle division by zero
            snr = float("inf")
        else:
            snr = 10 * np.log10(signal_power / noise_power)

        snr_estimates.append(snr)
    return np.array(snr_estimates)


def adaptive_threshold_snr(signal, threshold=0.5):
    """
    Estimate SNR adaptively by applying a threshold to segment the signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    threshold : float, optional (default=0.5)
        The amplitude threshold for detecting noise segments.

    Returns
    -------
    snr_estimate : float
        The estimated SNR.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    >>> snr_estimate = adaptive_threshold_snr(signal, threshold=0.3)
    >>> print(snr_estimate)
    """
    noise_segments = signal[np.abs(signal) < threshold]
    signal_segments = signal[np.abs(signal) >= threshold]

    if len(signal_segments) == 0 or len(noise_segments) == 0:
        # Avoid division by zero and return a sentinel value
        return float("inf") if len(signal_segments) == 0 else -float("inf")

    signal_power = np.mean(signal_segments**2)
    noise_power = np.mean(noise_segments**2)

    if noise_power == 0:
        return float("inf")  # Infinite SNR due to no noise

    if signal_power == 0:
        return -float("inf")  # No signal means undefined SNR (set to -inf)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def recursive_snr_estimation(signal, alpha=0.9):
    """
    Estimate SNR recursively using an exponential moving average.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    alpha : float, optional (default=0.9)
        Smoothing factor for the exponential moving average.

    Returns
    -------
    snr_estimates : numpy.ndarray
        Array of SNR estimates for each point in the signal.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    >>> snr_estimates = recursive_snr_estimation(signal)
    >>> print(snr_estimates)
    """
    snr_estimates = []
    avg_signal_power = 0
    avg_noise_power = 0

    mean_signal = np.mean(signal)

    for x in signal:
        avg_signal_power = alpha * avg_signal_power + (1 - alpha) * x**2
        avg_noise_power = alpha * avg_noise_power + (1 - alpha) * (x - mean_signal) ** 2

        if avg_noise_power == 0:  # Handle division by zero
            snr = float("inf")
        else:
            snr = 10 * np.log10(avg_signal_power / avg_noise_power)

        snr_estimates.append(snr)

    return np.array(snr_estimates)
