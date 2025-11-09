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

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.estimate_rr.time_domain_rr import TimeDomainRr
    >>> signal = np.random.randn(1000)
    >>> processor = TimeDomainRr(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
import warnings
import logging
from scipy.signal import find_peaks
from vitalDSP.preprocess.preprocess_operations import preprocess_signal

logger = logging.getLogger(__name__)


def time_domain_rr(signal, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Estimate respiratory rate using time-domain autocorrelation method.

    This method computes the autocorrelation of the respiratory signal and finds
    the dominant periodicity within the physiological respiratory range (1.5-10 seconds
    corresponding to 40-6 breaths/min).

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = time_domain_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)

    Notes
    -----
    This implementation fixes the critical bug in the original version that used
    np.diff() followed by np.argmax(), which finds maximum slope instead of maximum peak.
    The corrected version uses scipy.signal.find_peaks() to properly identify
    autocorrelation peaks within the valid respiratory range.

    References
    ----------
    .. [1] Box, G.E., Jenkins, G.M. and Reinsel, G.C., 1994. Time series analysis:
           forecasting and control. Prentice Hall.
    """
    logger.debug("=" * 80)
    logger.debug("TIME DOMAIN (AUTOCORRELATION) RR - Starting estimation")
    logger.debug(
        f"Input signal: {len(signal)} samples at {sampling_rate} Hz ({len(signal)/sampling_rate:.2f}s duration)"
    )
    logger.debug(
        f"Signal statistics: range=[{np.min(signal):.4f}, {np.max(signal):.4f}], mean={np.mean(signal):.4f}, std={np.std(signal):.4f}"
    )

    # Apply preprocessing if specified
    if preprocess:
        logger.debug(f"Preprocessing: {preprocess} with params {preprocess_kwargs}")
        original_signal = signal.copy()
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )
        logger.debug(
            f"After preprocessing: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}"
        )

    # Validate signal
    if len(signal) < 10:
        warnings.warn("Signal too short for reliable autocorrelation analysis")
        return 0.0

    # Normalize the signal (zero mean, unit variance)
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    logger.debug(f"Normalizing signal: mean={signal_mean:.4f}, std={signal_std:.4f}")

    if signal_std < 1e-10:
        logger.warning("Signal has zero or near-zero variance")
        warnings.warn("Signal has zero or near-zero variance")
        return 0.0

    signal = (signal - signal_mean) / signal_std

    # Compute the autocorrelation of the signal
    logger.debug("Computing autocorrelation...")
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags
    logger.debug(f"Autocorrelation computed: {len(autocorr)} lag points")

    # Normalize autocorrelation to [0, 1] range
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
        logger.debug(f"Autocorrelation normalized: max={np.max(autocorr):.4f}")
    else:
        logger.warning("Invalid autocorrelation computation")
        warnings.warn("Invalid autocorrelation computation")
        return 0.0

    # Define valid lag range for respiratory signals
    # Normal breathing: 6-40 BPM → 1.5-10 seconds period
    min_lag = int(1.5 * sampling_rate)  # 40 BPM max
    max_lag = int(10 * sampling_rate)  # 6 BPM min

    logger.debug(
        f"Respiratory lag range: {min_lag}-{max_lag} samples ({min_lag/sampling_rate:.2f}-{max_lag/sampling_rate:.2f}s)"
    )
    logger.debug(f"Corresponding to: 40-6 BPM")

    # Ensure we have valid search range
    if min_lag >= len(autocorr):
        logger.warning(
            f"Signal too short ({len(signal)/sampling_rate:.1f}s) for minimum respiratory period (1.5s)"
        )
        warnings.warn(
            f"Signal too short ({len(signal)/sampling_rate:.1f}s) for minimum respiratory period (1.5s)"
        )
        return 0.0

    max_lag = min(max_lag, len(autocorr) - 1)

    if min_lag >= max_lag:
        logger.warning("Invalid lag range for autocorrelation search")
        warnings.warn("Invalid lag range for autocorrelation search")
        return 0.0

    # Only search for peaks within valid respiratory range
    search_range = autocorr[min_lag:max_lag]
    logger.debug(f"Search range: {len(search_range)} lag points")
    logger.debug(
        f"Search range autocorr values: min={np.min(search_range):.4f}, max={np.max(search_range):.4f}, mean={np.mean(search_range):.4f}"
    )

    # Find peaks in autocorrelation with minimum prominence
    logger.debug("Finding peaks in autocorrelation with prominence >= 0.1...")
    peaks, properties = find_peaks(search_range, prominence=0.1)

    if len(peaks) == 0:
        # No clear peaks found - use global maximum in search range
        logger.warning("No clear autocorrelation peaks found with prominence >= 0.1")
        logger.debug("Falling back to global maximum in search range")
        warnings.warn("No clear autocorrelation peak found, using global maximum")
        peak_lag = np.argmax(search_range) + min_lag
        peak_value = search_range[np.argmax(search_range)]
        logger.debug(
            f"Global max: lag={peak_lag} samples ({peak_lag/sampling_rate:.3f}s), value={peak_value:.4f}"
        )
    else:
        # Take the strongest peak (highest prominence)
        logger.debug(f"Found {len(peaks)} autocorrelation peaks")
        logger.debug("Top 5 peaks by prominence:")
        sorted_indices = np.argsort(properties["prominences"])[::-1][:5]
        for i, idx in enumerate(sorted_indices):
            if idx < len(peaks):
                peak_pos = peaks[idx]
                actual_lag = peak_pos + min_lag
                period = actual_lag / sampling_rate
                rr_candidate = 60 / period
                prominence = properties["prominences"][idx]
                autocorr_value = search_range[peak_pos]
                logger.debug(
                    f"  {i+1}. Lag={actual_lag} samples ({period:.3f}s) → {rr_candidate:.1f} BPM, prominence={prominence:.4f}, autocorr={autocorr_value:.4f}"
                )

        strongest_peak_idx = peaks[np.argmax(properties["prominences"])]
        peak_lag = strongest_peak_idx + min_lag
        logger.debug(
            f"Selected strongest peak: lag={peak_lag} samples ({peak_lag/sampling_rate:.3f}s)"
        )

    # Convert lag to respiratory period (in seconds)
    rr_interval = peak_lag / sampling_rate

    # Calculate respiratory rate
    rr = 60 / rr_interval

    logger.debug(f"Respiratory interval: {rr_interval:.3f}s")
    logger.debug(f"Calculated RR: {rr:.1f} BPM")

    # Validate result is within physiological range
    if rr < 6 or rr > 40:
        logger.warning(
            f"RR estimate ({rr:.1f} BPM) outside physiological range (6-40 BPM)"
        )
        warnings.warn(
            f"Estimated RR ({rr:.1f} BPM) outside physiological range (6-40 BPM). "
            f"Result may be unreliable."
        )
    else:
        logger.debug(f"✓ RR estimate within physiological range")

    logger.debug(f"✓ FINAL RR ESTIMATE: {rr:.1f} BPM")
    logger.debug("=" * 80)

    return float(rr)
