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
    >>> from vitalDSP.estimate_rr.fft_based_rr import FftBasedRr
    >>> signal = np.random.randn(1000)
    >>> processor = FftBasedRr(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
import warnings
import logging
from vitalDSP.preprocess.preprocess_operations import preprocess_signal

# Set up logger for this module
logger = logging.getLogger(__name__)


def fft_based_rr(
    signal,
    sampling_rate,
    preprocess=None,
    freq_min=0.1,
    freq_max=0.5,
    **preprocess_kwargs,
):
    """
    Estimate respiratory rate using the FFT (Fast Fourier Transform) method
    with respiratory band filtering.

    This method computes the FFT power spectrum and identifies the dominant
    frequency within the physiological respiratory range (0.1-0.5 Hz or 6-30 BPM).
    The respiratory band filtering prevents false detection of cardiac frequencies
    or high-frequency noise artifacts.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    freq_min : float, optional (default=0.1)
        Minimum respiratory frequency in Hz (6 BPM). Can be adjusted for different
        populations (e.g., 0.05 Hz for very slow breathing).
    freq_max : float, optional (default=0.5)
        Maximum respiratory frequency in Hz (30 BPM). Can be increased to 0.67 Hz
        (40 BPM) for tachypnea or exercise conditions.
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = fft_based_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)

    >>> # For exercise/tachypnea (up to 40 BPM)
    >>> rr = fft_based_rr(signal, sampling_rate=100, freq_max=0.67)

    Notes
    -----
    This implementation fixes the critical bug in the original version that searched
    the entire frequency spectrum, often picking cardiac frequencies (1-2 Hz / 60-120 BPM)
    instead of respiratory frequencies. The corrected version restricts the search to
    the physiological respiratory band.

    The method includes SNR validation to warn if the detected peak has low prominence,
    which may indicate poor signal quality or unreliable estimation.

    References
    ----------
    .. [1] Charlton, P.H., et al. (2018). Breathing rate estimation from the
           electrocardiogram and photoplethysmogram: A review. IEEE Reviews in
           Biomedical Engineering, 11, 2-20.
    """
    logger.debug("=" * 80)
    logger.debug("FFT-BASED RR - Starting estimation")
    logger.debug(f"Input signal length: {len(signal)} samples")
    logger.debug(f"Sampling rate: {sampling_rate} Hz")
    logger.debug(f"Signal duration: {len(signal)/sampling_rate:.2f} seconds")
    logger.debug(f"Signal range: [{np.min(signal):.4f}, {np.max(signal):.4f}]")
    logger.debug(f"Signal mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}")
    logger.debug(
        f"Respiratory frequency range: {freq_min}-{freq_max} Hz ({freq_min*60:.0f}-{freq_max*60:.0f} BPM)"
    )

    if preprocess:
        logger.debug(f"Applying preprocessing: {preprocess}")
        signal_before = signal.copy()
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )
        logger.debug(
            f"After preprocessing - mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}"
        )

    # Validate signal
    if len(signal) < 10:
        logger.error("Signal too short for reliable FFT analysis")
        warnings.warn("Signal too short for reliable FFT analysis")
        return 0.0

    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / sampling_rate)
    logger.debug(f"FFT computed - {len(freqs)} frequency bins")
    logger.debug(f"Frequency resolution: {freqs[1]-freqs[0]:.4f} Hz")

    # Filter to respiratory frequency range ONLY (critical fix)
    respiratory_mask = (freqs >= freq_min) & (freqs <= freq_max)
    resp_freqs = freqs[respiratory_mask]
    resp_fft = np.abs(fft_result[respiratory_mask])

    logger.debug(f"Frequencies in respiratory band: {len(resp_fft)}")

    if len(resp_fft) == 0:
        logger.error(f"No frequencies in respiratory range ({freq_min}-{freq_max} Hz)")
        raise ValueError(
            f"No frequencies found in respiratory range ({freq_min}-{freq_max} Hz). "
            f"Signal may be too short or sampling rate too low. "
            f"Minimum signal length: {int(1/freq_min * sampling_rate)} samples."
        )

    # Find peak in respiratory band
    peak_idx = np.argmax(resp_fft)
    peak_freq = resp_freqs[peak_idx]
    peak_power = resp_fft[peak_idx]

    logger.debug(f"Peak found at frequency: {peak_freq:.4f} Hz")
    logger.debug(f"Peak power: {peak_power:.4e}")

    # Show top 5 peaks for comparison
    sorted_indices = np.argsort(resp_fft)[::-1][:5]
    logger.debug("Top 5 frequency peaks in respiratory band:")
    for i, idx in enumerate(sorted_indices):
        freq = resp_freqs[idx]
        power = resp_fft[idx]
        bpm = freq * 60
        logger.debug(f"  {i+1}. {freq:.4f} Hz ({bpm:.1f} BPM) - Power: {power:.4e}")

    # Validate peak prominence (SNR check)
    # Compare peak power to median power (noise floor estimate)
    noise_floor = np.median(resp_fft)
    snr = peak_power / (noise_floor + 1e-10)

    logger.debug(f"Noise floor (median): {noise_floor:.4e}")
    logger.debug(f"SNR: {snr:.2f}")

    if snr < 2.0:
        logger.warning(f"Low SNR ({snr:.2f}) - result may be unreliable")
        warnings.warn(
            f"Low SNR ({snr:.2f}) in FFT peak detection. "
            f"Peak power: {peak_power:.2e}, Noise floor: {noise_floor:.2e}. "
            f"Result may be unreliable due to poor signal quality."
        )

    # Convert frequency to BPM
    rr = float(peak_freq * 60)

    logger.debug(f"Computed RR: {rr:.1f} BPM from frequency {peak_freq:.4f} Hz")

    # Sanity check - validate result is within expected range
    expected_min = freq_min * 60
    expected_max = freq_max * 60

    if rr < expected_min or rr > expected_max:
        logger.warning(
            f"RR ({rr:.1f} BPM) outside expected range ({expected_min:.0f}-{expected_max:.0f} BPM)"
        )
        warnings.warn(
            f"Estimated RR ({rr:.1f} BPM) outside expected range "
            f"({expected_min:.0f}-{expected_max:.0f} BPM)"
        )
    else:
        logger.debug(f"✓ FINAL RR ESTIMATE: {rr:.1f} BPM (within expected range)")

    logger.debug("=" * 80)
    return rr
