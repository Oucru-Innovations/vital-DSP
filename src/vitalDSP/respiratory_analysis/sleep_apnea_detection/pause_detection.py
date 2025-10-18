"""
Respiratory Analysis Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Configurable parameters and settings
- Pattern and anomaly detection

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.sleep_apnea_detection.pause_detection import PauseDetection
    >>> signal = np.random.randn(1000)
    >>> processor = PauseDetection(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


from vitalDSP.utils.config_utilities.common import find_peaks
from vitalDSP.preprocess.preprocess_operations import preprocess_signal


def detect_apnea_pauses(
    signal, sampling_rate, min_pause_duration=10, preprocess=None, **preprocess_kwargs
):
    """
    Detect sleep apnea events based on pauses in the respiratory signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    min_pause_duration : int, optional (default=10)
        Minimum duration (in seconds) of a pause to be considered an apnea event.
    preprocess : str, optional
        The preprocessing method to apply before detection (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    apnea_events : list of tuple
        List of apnea events, each represented as a tuple (start_time, end_time) in seconds.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 60, 0.01))
    >>> apnea_events = detect_apnea_pauses(signal, sampling_rate=100, min_pause_duration=10)
    >>> print(apnea_events)
    """
    if preprocess:
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )

    # Detect peaks in the respiratory signal
    peaks = find_peaks(
        signal, distance=sampling_rate
    )  # Ensure at least 1 second between peaks

    apnea_events = []
    for i in range(1, len(peaks)):
        pause_duration = (peaks[i] - peaks[i - 1]) / sampling_rate
        if pause_duration >= min_pause_duration:
            apnea_events.append(
                (peaks[i - 1] / sampling_rate, peaks[i] / sampling_rate)
            )

    return apnea_events
