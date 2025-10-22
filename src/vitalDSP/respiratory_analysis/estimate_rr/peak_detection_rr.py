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
- SciPy integration for advanced signal processing
- Configurable parameters and settings
- Pattern and anomaly detection

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.estimate_rr.peak_detection_rr import PeakDetectionRr
    >>> signal = np.random.randn(1000)
    >>> processor = PeakDetectionRr(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
import warnings
import logging
from vitalDSP.utils.config_utilities.common import find_peaks
from vitalDSP.preprocess.preprocess_operations import preprocess_signal

# Set up logger for this module
logger = logging.getLogger(__name__)


def peak_detection_rr(
    signal,
    sampling_rate,
    preprocess=None,
    min_peak_distance=1.5,
    height=None,
    threshold=None,
    prominence=None,
    width=None,
    **preprocess_kwargs,
):
    """
    Estimate respiratory rate using peak detection with interval-based analysis.

    This method detects breathing peaks in the respiratory signal and calculates
    the rate based on the median interval between peaks, which is more robust than
    simple peak counting. Peaks are validated against physiological constraints
    (1.5-10 seconds between breaths).

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    min_peak_distance : float, optional (default=1.5)
        Minimum distance between peaks in seconds. Default 1.5s corresponds to 40 BPM max.
        Previous default of 0.5s was too permissive (allowed 120 BPM).
    height : float or None, optional
        Minimum height required for a peak. If None, uses adaptive threshold.
    threshold : float or None, optional
        Minimum difference between a peak and its neighboring points.
    prominence : float or None, optional
        Minimum prominence of peaks. If None, uses adaptive prominence (0.3 * signal std).
    width : int or None, optional
        Minimum width required for a peak, measured as the number of samples.
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = peak_detection_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)

    >>> # For faster breathing (exercise), reduce min_peak_distance
    >>> rr = peak_detection_rr(signal, sampling_rate=100, min_peak_distance=1.0)

    Notes
    -----
    This implementation fixes critical bugs in the original version:
    1. Changed from simple peak counting (num_peaks / duration) to interval-based analysis
    2. Increased min_peak_distance default from 0.5s to 1.5s (40 BPM max instead of 120 BPM)
    3. Added adaptive prominence threshold if not specified
    4. Validates intervals against physiological range (1.5-10 seconds)
    5. Uses median interval for robustness against outliers
    6. Adds quality metrics (coefficient of variation) for breathing regularity

    The interval-based approach is more accurate because:
    - Handles irregular breathing patterns
    - Robust to missed or false peaks
    - Provides quality metrics for result confidence

    References
    ----------
    .. [1] Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
           IEEE transactions on biomedical engineering, (3), 230-236.
    """
    logger.debug("=" * 80)
    logger.debug("PEAK DETECTION RR - Starting estimation")
    logger.debug(f"Input signal length: {len(signal)} samples")
    logger.debug(f"Sampling rate: {sampling_rate} Hz")
    logger.debug(f"Signal duration: {len(signal)/sampling_rate:.2f} seconds")
    logger.debug(f"Signal range: [{np.min(signal):.4f}, {np.max(signal):.4f}]")
    logger.debug(f"Signal mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}")
    logger.debug(f"Min peak distance: {min_peak_distance}s")
    logger.debug(f"Prominence: {prominence if prominence is not None else 'auto'}")

    # Apply preprocessing if specified
    if preprocess:
        logger.debug(f"Applying preprocessing: {preprocess}")
        signal_before = signal.copy()
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )
        logger.debug(
            f"After preprocessing - mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}"
        )
        logger.debug(
            f"Preprocessing changed signal by: {np.mean(np.abs(signal - signal_before)):.4f}"
        )

    # Validate signal
    if len(signal) < 10:
        logger.warning("Signal too short for reliable peak detection")
        warnings.warn("Signal too short for reliable peak detection")
        return 0.0

    # Set adaptive prominence if not provided (critical fix)
    if prominence is None:
        signal_std = np.std(signal)
        if signal_std > 1e-10:
            prominence = 0.3 * signal_std
            logger.debug(f"Auto-computed prominence: {prominence:.4f} (0.3 * std)")
        else:
            logger.error("Signal has near-zero variance - cannot detect peaks")
            warnings.warn("Signal has near-zero variance")
            return 0.0

    # Convert minimum peak distance to samples (default 1.5s = 40 BPM max)
    distance = int(min_peak_distance * sampling_rate)
    logger.debug(
        f"Minimum peak distance in samples: {distance} ({distance/sampling_rate:.2f}s)"
    )

    # Ensure distance doesn't exceed signal length
    if distance >= len(signal) // 2:
        warnings.warn(
            f"min_peak_distance ({min_peak_distance}s) too large for signal length "
            f"({len(signal)/sampling_rate:.1f}s)"
        )
        return 0.0

    # Detect peaks
    peaks = find_peaks(
        signal,
        height=height,
        distance=distance,
        threshold=threshold,
        prominence=prominence,
        width=width,
    )

    # Check if we have sufficient peaks
    logger.debug(f"Detected {len(peaks)} peaks")
    if len(peaks) < 2:
        logger.warning(f"Insufficient peaks detected ({len(peaks)}) - need at least 2")
        warnings.warn(
            f"Insufficient peaks detected ({len(peaks)}). "
            f"Signal may be too short, noisy, or parameters too restrictive."
        )
        return 0.0

    # Log peak locations (first 10)
    peak_indices = np.array(peaks)
    if len(peak_indices) <= 10:
        logger.debug(f"Peak locations (samples): {peak_indices}")
    else:
        logger.debug(f"First 10 peaks (samples): {peak_indices[:10]}")

    # Calculate inter-breath intervals (in seconds)
    breath_intervals = np.diff(peak_indices) / sampling_rate
    logger.debug(f"Calculated {len(breath_intervals)} inter-breath intervals")
    logger.debug(
        f"Interval range: [{np.min(breath_intervals):.3f}, {np.max(breath_intervals):.3f}] seconds"
    )
    logger.debug(
        f"Interval mean: {np.mean(breath_intervals):.3f}s, median: {np.median(breath_intervals):.3f}s"
    )

    # Filter intervals to physiological range (1.5-10 seconds)
    # 1.5s = 40 BPM max, 10s = 6 BPM min
    valid_mask = (breath_intervals >= 1.5) & (breath_intervals <= 10)
    valid_intervals = breath_intervals[valid_mask]
    invalid_intervals = breath_intervals[~valid_mask]

    logger.debug(
        f"Valid intervals (1.5-10s): {len(valid_intervals)}/{len(breath_intervals)}"
    )
    if len(invalid_intervals) > 0:
        logger.warning(
            f"Invalid intervals: {invalid_intervals[:10] if len(invalid_intervals) > 10 else invalid_intervals}"
        )

    if len(valid_intervals) == 0:
        logger.error(
            "No valid breath intervals found within physiological range (1.5-10s)"
        )
        logger.error(f"All intervals out of range: {breath_intervals}")
        warnings.warn(
            "No valid breath intervals found within physiological range (1.5-10s). "
            "All detected intervals are outside normal breathing range."
        )
        return 0.0

    # Use median for robustness against outliers (critical fix)
    median_interval = np.median(valid_intervals)
    mean_interval = np.mean(valid_intervals)
    rr_median = 60 / median_interval
    rr_mean = 60 / mean_interval

    logger.debug(f"Valid interval statistics:")
    logger.debug(f"  Median: {median_interval:.3f}s → RR = {rr_median:.1f} BPM")
    logger.debug(f"  Mean: {mean_interval:.3f}s → RR = {rr_mean:.1f} BPM")
    logger.debug(f"  Std: {np.std(valid_intervals):.3f}s")

    rr = rr_median

    # Quality assessment: check coefficient of variation
    std_interval = np.std(valid_intervals)
    cv = std_interval / mean_interval if mean_interval > 0 else float("inf")
    logger.debug(f"Coefficient of variation: {cv:.3f}")

    if cv > 0.5:
        logger.warning(f"High breathing variability (CV={cv:.2f})")
        warnings.warn(
            f"High breathing variability detected (CV={cv:.2f}). "
            f"Result may be less reliable. Consider checking for irregular breathing "
            f"or signal quality issues."
        )

    # Additional validation: warn if many intervals were rejected
    rejection_rate = 1 - (len(valid_intervals) / len(breath_intervals))
    if rejection_rate > 0.3:
        logger.warning(
            f"High rejection rate: {rejection_rate*100:.1f}% ({len(breath_intervals) - len(valid_intervals)}/{len(breath_intervals)})"
        )
        warnings.warn(
            f"High interval rejection rate ({rejection_rate*100:.1f}%). "
            f"{len(breath_intervals) - len(valid_intervals)} of {len(breath_intervals)} "
            f"intervals outside physiological range."
        )

    # Final sanity check
    if rr < 6 or rr > 40:
        logger.warning(f"RESULT OUTSIDE NORMAL RANGE: {rr:.1f} BPM")
        warnings.warn(f"Estimated RR ({rr:.1f} BPM) outside normal range (6-40 BPM)")
    else:
        logger.debug(f"✓ FINAL RR ESTIMATE: {rr:.1f} BPM (within normal range)")

    logger.debug("=" * 80)
    return float(rr)
