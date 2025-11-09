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

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.estimate_rr.frequency_domain_rr import FrequencyDomainRr
    >>> signal = np.random.randn(1000)
    >>> processor = FrequencyDomainRr(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
import warnings
import logging
from scipy.signal import welch
from vitalDSP.preprocess.preprocess_operations import preprocess_signal

# Set up logger for this module
logger = logging.getLogger(__name__)


def frequency_domain_rr(
    signal,
    sampling_rate,
    preprocess=None,
    nperseg=None,
    freq_min=0.1,
    freq_max=0.5,
    **preprocess_kwargs,
):
    """
    Estimate respiratory rate using frequency-domain Welch method with
    respiratory band filtering.

    This method computes the power spectral density (PSD) using Welch's method,
    which provides better noise reduction than standard FFT through averaging of
    overlapping segments. The search is restricted to the physiological respiratory
    frequency range (0.1-0.5 Hz or 6-30 BPM).

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    nperseg : int, optional
        Length of each segment for Welch method. If None, automatically computed to
        achieve ~0.05 Hz frequency resolution (3 BPM discrimination). Larger nperseg
        provides better frequency resolution but requires longer signals.
    freq_min : float, optional (default=0.1)
        Minimum respiratory frequency in Hz (6 BPM).
    freq_max : float, optional (default=0.5)
        Maximum respiratory frequency in Hz (30 BPM). Can be increased to 0.67 Hz
        (40 BPM) for tachypnea or exercise.
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = frequency_domain_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)

    >>> # Specify custom nperseg for better frequency resolution
    >>> rr = frequency_domain_rr(signal, sampling_rate=100, nperseg=512)

    Notes
    -----
    This implementation fixes two critical bugs in the original version:
    1. Missing respiratory band filtering (searched entire spectrum, could pick cardiac)
    2. Inappropriate nperseg default (256 samples → only 0.5 Hz resolution at 128 Hz sampling)

    The corrected version:
    - Automatically sets nperseg to achieve 0.05 Hz resolution (3 BPM discrimination)
    - Restricts search to respiratory frequency band
    - Includes SNR validation for quality assessment
    - Provides comprehensive warnings for edge cases

    Frequency Resolution:
        Δf = sampling_rate / nperseg

    For respiratory analysis, we need Δf ≤ 0.05 Hz to discriminate between different
    breathing rates (e.g., 12 BPM vs 15 BPM = 0.2 Hz vs 0.25 Hz).

    References
    ----------
    .. [1] Welch, P. (1967). The use of fast Fourier transform for the estimation
           of power spectra: a method based on time averaging over short, modified
           periodograms. IEEE Transactions on audio and electroacoustics, 15(2), 70-73.
    .. [2] Charlton, P.H., et al. (2018). Breathing rate estimation from the
           electrocardiogram and photoplethysmogram: A review. IEEE Reviews in
           Biomedical Engineering, 11, 2-20.
    """
    logger.debug("=" * 80)
    logger.debug("FREQUENCY-DOMAIN RR (Welch PSD) - Starting estimation")
    logger.debug(
        f"Input signal: {len(signal)} samples, {sampling_rate} Hz, {len(signal)/sampling_rate:.2f}s"
    )
    logger.debug(f"Signal stats: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    logger.debug(
        f"Respiratory range: {freq_min}-{freq_max} Hz ({freq_min*60:.0f}-{freq_max*60:.0f} BPM)"
    )

    # Apply preprocessing if specified
    if preprocess:
        logger.debug(f"Preprocessing: {preprocess}")
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )

    # Validate signal
    if len(signal) < 10:
        logger.error("Signal too short")
        warnings.warn("Signal too short for reliable Welch PSD analysis")
        return 0.0

    # Set appropriate nperseg for respiratory frequency resolution
    # Target: 0.05 Hz resolution (3 BPM discrimination capability)
    # Frequency resolution = sampling_rate / nperseg
    # So: nperseg = sampling_rate / 0.05
    if nperseg is None:
        target_resolution = 0.05  # Hz
        nperseg = int(sampling_rate / target_resolution)

        # Limit nperseg to signal length
        nperseg = min(nperseg, len(signal))

        # Ensure nperseg is at least 256 for stable estimation
        nperseg = max(256, nperseg)

        # Make nperseg even for efficiency
        if nperseg % 2 != 0:
            nperseg += 1

    # Compute the power spectral density using the Welch method
    logger.debug(f"Computing Welch PSD with nperseg={nperseg}")
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 0
    logger.debug(
        f"PSD computed: {len(freqs)} frequency bins, resolution={freq_resolution:.4f} Hz"
    )

    # Filter to respiratory frequency range ONLY (critical fix)
    respiratory_mask = (freqs >= freq_min) & (freqs <= freq_max)
    resp_freqs = freqs[respiratory_mask]
    resp_psd = psd[respiratory_mask]

    logger.debug(f"Frequencies in respiratory band: {len(resp_psd)}")

    if len(resp_psd) == 0:
        logger.error(f"No frequencies in range {freq_min}-{freq_max} Hz")
        raise ValueError(
            f"No frequencies found in respiratory range ({freq_min}-{freq_max} Hz). "
            f"Signal may be too short, sampling rate too low, or nperseg too large. "
            f"Frequency resolution: {freq_resolution:.4f} Hz. "
            f"Minimum signal length: {int(1/freq_min * sampling_rate)} samples."
        )

    # Find peak in respiratory band
    peak_idx = np.argmax(resp_psd)
    peak_freq = resp_freqs[peak_idx]
    peak_power = resp_psd[peak_idx]

    logger.debug(f"Peak: {peak_freq:.4f} Hz, Power: {peak_power:.4e}")

    # Show top 5 peaks
    sorted_indices = np.argsort(resp_psd)[::-1][:5]
    logger.debug("Top 5 PSD peaks:")
    for i, idx in enumerate(sorted_indices):
        freq = resp_freqs[idx]
        power = resp_psd[idx]
        logger.debug(f"  {i+1}. {freq:.4f} Hz ({freq*60:.1f} BPM) - Power: {power:.4e}")

    # SNR validation
    noise_floor = np.median(resp_psd)
    snr = peak_power / (noise_floor + 1e-10)
    logger.debug(f"SNR: {snr:.2f} (noise floor: {noise_floor:.4e})")

    if snr < 3.0:
        logger.warning(f"Low SNR: {snr:.2f}")
        warnings.warn(
            f"Low SNR ({snr:.2f}) in Welch PSD peak detection. "
            f"Peak power: {peak_power:.2e}, Noise floor: {noise_floor:.2e}. "
            f"Result may be unreliable due to poor signal quality or high variability."
        )

    # Convert frequency to BPM
    rr = float(peak_freq * 60)
    logger.debug(f"Computed RR: {rr:.1f} BPM from {peak_freq:.4f} Hz")

    # Validation - check if result is within expected range
    expected_min = freq_min * 60
    expected_max = freq_max * 60

    if rr < expected_min or rr > expected_max:
        logger.warning(
            f"RR outside range: {rr:.1f} not in [{expected_min:.0f}, {expected_max:.0f}]"
        )
        warnings.warn(
            f"Estimated RR ({rr:.1f} BPM) outside expected range "
            f"({expected_min:.0f}-{expected_max:.0f} BPM)"
        )
    else:
        logger.debug(f"✓ FINAL RR ESTIMATE: {rr:.1f} BPM")

    # Additional diagnostic info
    if freq_resolution > 0.1:
        logger.warning(f"Coarse frequency resolution: {freq_resolution:.3f} Hz")
        warnings.warn(
            f"Frequency resolution ({freq_resolution:.3f} Hz) may be too coarse "
            f"for accurate RR discrimination. Consider longer signal or smaller nperseg."
        )

    logger.debug("=" * 80)
    return rr
