"""
Signal processing utilities for PPG analysis.

This module provides core signal processing functions including:
- Safe type conversion with fallback values
- IIR filter design for various filter families
- Signal processing chains (detrend, filter, notch, invert)
- Rate estimation from power spectral density
- Signal-to-noise ratio estimation
- Automatic decimation for display optimization
- Cross-correlation analysis
"""

import numpy as np
from scipy.signal import (
    coherence,
    find_peaks,
    iirfilter,
    iirnotch,
    sosfiltfilt,
    spectrogram,
    tf2sos,
    welch,
)


def safe_float(x, fallback):
    """
    Safely convert input to float with fallback value.

    This function attempts to convert the input to a float, returning
    the fallback value if conversion fails. Useful for handling
    user inputs that may be invalid or None.

    Args:
        x: Input value to convert to float
        fallback (float): Value to return if conversion fails

    Returns:
        float: Converted value or fallback
    """
    try:
        return float(x)
    except Exception:
        return fallback


def safe_int(x, fallback):
    """
    Safely convert input to integer with fallback value.

    This function attempts to convert the input to an integer, returning
    the fallback value if conversion fails. Useful for handling
    user inputs that may be invalid or None.

    Args:
        x: Input value to convert to integer
        fallback (int): Value to return if conversion fails

    Returns:
        int: Converted value or fallback
    """
    try:
        return int(x)
    except Exception:
        return fallback


def design_base_filter(fs, family, resp_type, low_hz, high_hz, order, rp, rs):
    """
    Design IIR filter with specified parameters.

    This function creates digital filters using various IIR filter families
    (Butterworth, Chebyshev, Elliptic) with configurable response types
    (lowpass, highpass, bandpass, bandstop).

    Args:
        fs (float): Sampling frequency in Hz
        family (str): Filter family ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel')
        resp_type (str): Filter response type ('lowpass', 'highpass', 'bandpass', 'bandstop')
        low_hz (float): Lower cutoff frequency in Hz
        high_hz (float): Upper cutoff frequency in Hz
        order (int): Filter order
        rp (float): Passband ripple in dB (for Chebyshev and Elliptic)
        rs (float): Stopband attenuation in dB (for Chebyshev and Elliptic)

    Returns:
        numpy.ndarray: Second-order sections (SOS) filter coefficients

    Note:
        - Frequencies are automatically constrained to prevent aliasing
        - For bandpass/bandstop, frequencies are automatically sorted
        - Filter parameters (rp, rs) are only used for relevant filter families
    """
    fs = float(fs)
    nyq = fs / 2.0  # Nyquist frequency

    # Validate and constrain cutoff frequencies
    low_hz = max(1e-6, float(low_hz))
    high_hz = max(1e-6, float(high_hz))

    # Set up frequency specifications based on response type
    if resp_type in ("lowpass", "highpass"):
        Wn = high_hz if resp_type == "lowpass" else low_hz
        Wn = min(Wn, nyq * 0.999)  # Prevent aliasing
    else:
        # For bandpass/bandstop, sort frequencies and constrain
        lo, hi = sorted([low_hz, high_hz])
        hi = min(hi, nyq * 0.999)  # Prevent aliasing
        lo = max(1e-6, min(lo, hi - 1e-6))  # Ensure valid band
        Wn = [lo, hi]

    # Set up filter design parameters
    kwargs = dict(ftype=family, fs=fs, output="sos")
    if family == "cheby1":
        kwargs["rp"] = rp
    elif family == "cheby2":
        kwargs["rs"] = rs
    elif family == "ellip":
        kwargs["rp"] = rp
        kwargs["rs"] = rs

    return iirfilter(order, Wn, btype=resp_type, **kwargs)


def apply_chain(
    x,
    fs,
    base_sos=None,
    notch_enable=False,
    notch_hz=50.0,
    notch_q=30.0,
    detrend_mean=False,
    invert=False,
):
    """
    Apply signal processing chain: detrend → filter → notch → invert.

    This function applies a sequence of signal processing operations in order:
    1. Detrending (remove mean if enabled)
    2. Base filtering (if SOS coefficients provided)
    3. Notch filtering (if enabled)
    4. Signal inversion (if enabled)

    Args:
        x (array-like): Input signal
        fs (float): Sampling frequency in Hz
        base_sos (numpy.ndarray, optional): Base filter SOS coefficients
        notch_enable (bool): Whether to enable notch filter
        notch_hz (float): Notch filter center frequency in Hz
        notch_q (float): Notch filter quality factor
        detrend_mean (bool): Whether to remove signal mean
        invert (bool): Whether to invert the signal

    Returns:
        numpy.ndarray: Processed signal

    Note:
        - All operations are applied in sequence
        - Notch filter is only applied if notch_enable is True
        - Base filter is only applied if base_sos is provided
    """
    y = np.asarray(x, dtype=float)

    # Step 1: Detrend (remove mean if requested)
    if detrend_mean:
        y = y - np.mean(y)

    # Step 2: Apply base filter if provided
    if base_sos is not None:
        y = sosfiltfilt(base_sos, y)

    # Step 3: Apply notch filter if enabled
    if notch_enable and notch_hz > 0:
        b, a = iirnotch(w0=notch_hz, Q=notch_q, fs=fs)
        sos_notch = tf2sos(b, a)
        y = sosfiltfilt(sos_notch, y)

    # Step 4: Invert signal if requested
    if invert:
        y = -y

    return y


def estimate_rates_psd(sig, fs, band_tuple):
    """
    Estimate rate from PSD peak in specified frequency band.

    This function calculates the power spectral density of a signal and
    finds the frequency with maximum power within a specified band.
    Useful for estimating heart rate, respiratory rate, etc.

    Args:
        sig (array-like): Input signal
        fs (float): Sampling frequency in Hz
        band_tuple (tuple): (low_freq, high_freq) frequency band in Hz

    Returns:
        float or None: Estimated rate in units per minute, or None if estimation fails

    Note:
        - Uses Welch's method for PSD estimation
        - Automatically adjusts window size based on signal length
        - Returns rate in units per minute (60 * frequency)
    """
    # Calculate power spectral density using Welch's method
    f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 2048))
    lo, hi = band_tuple

    # Find frequencies within the specified band
    mask = (f >= lo) & (f <= hi)

    if not np.any(mask):
        return None

    # Extract power values within the band
    f_band = f[mask]
    P_band = Pxx[mask]

    # Check for valid power values
    if len(P_band) == 0 or np.all(np.isnan(P_band)):
        return None

    # Find frequency with maximum power
    f_peak = f_band[np.nanargmax(P_band)]
    return 60.0 * f_peak  # Convert to per minute


def quick_snr(sig):
    """
    Quick signal-to-noise ratio estimation.

    This function provides a simple estimate of SNR based on the
    peak-to-peak amplitude of the signal. This is a rough approximation
    suitable for quick quality assessment.

    Args:
        sig (array-like): Input signal

    Returns:
        float or None: Estimated SNR, or None if signal is too short

    Note:
        - Uses peak-to-peak amplitude as a simple SNR proxy
        - Returns None for signals with less than 2 samples
        - This is a simplified SNR estimate, not a rigorous calculation
    """
    if len(sig) < 2:
        return None

    # Calculate peak-to-peak amplitude as SNR proxy
    ptp = float(np.ptp(sig))
    std = float(np.std(sig))

    if std == 0:
        return None

    return ptp / (6 * std)


def auto_decimation(n, decim_user, traces, cap):
    """Auto-decimate signal for display."""
    target_per = max(1, cap // max(traces, 1))
    d = max(1, int(decim_user))

    if n // d > target_per:
        d = int(np.ceil(n / target_per))

    return max(1, d)


def cross_correlation_lag(sig1, sig2, max_lag=None):
    """
    Compute cross-correlation between two signals and find the lag.

    Args:
        sig1 (np.ndarray): First signal
        sig2 (np.ndarray): Second signal
        max_lag (int, optional): Maximum lag to compute. Defaults to None.

    Returns:
        tuple: (lags, correlation, max_corr_lag)
    """
    if max_lag is None:
        max_lag = min(len(sig1), len(sig2)) // 4

    correlation = np.correlate(sig1, sig2, mode="full")
    lags = np.arange(-len(sig2) + 1, len(sig1))

    # Find the lag with maximum correlation
    max_corr_idx = np.argmax(np.abs(correlation))
    max_corr_lag = lags[max_corr_idx]

    return lags, correlation, max_corr_lag


def analyze_waveform(signal, fs, window_s=5.0, annotations=None):
    """
    Analyze waveform characteristics including peaks, valleys, and zero crossings.

    Args:
        signal (np.ndarray): Input signal
        fs (float): Sampling frequency in Hz
        window_s (float): Analysis window size in seconds
        annotations (list): List of annotation types to include ('peaks', 'valleys', 'zero_crossings')

    Returns:
        dict: Dictionary containing waveform analysis results
    """
    if annotations is None:
        annotations = ["peaks"]

    # Calculate window size in samples
    window_samples = int(window_s * fs)

    # Ensure window doesn't exceed signal length
    if window_samples > len(signal):
        window_samples = len(signal)

    # Analyze the first window
    signal_window = signal[:window_samples]
    t_window = np.arange(len(signal_window)) / fs

    results = {
        "time": t_window,
        "signal": signal_window,
        "fs": fs,
        "window_s": window_s,
        "window_samples": window_samples,
    }

    # Peak detection
    if "peaks" in annotations:
        peaks, properties = find_peaks(signal_window, prominence=0.1 * np.std(signal_window))
        results["peaks"] = {
            "indices": peaks,
            "times": peaks / fs,
            "values": signal_window[peaks],
            "prominences": properties.get("prominences", []),
        }

    # Valley detection (negative peaks)
    if "valleys" in annotations:
        valleys, properties = find_peaks(-signal_window, prominence=0.1 * np.std(signal_window))
        results["valleys"] = {
            "indices": valleys,
            "times": valleys / fs,
            "values": signal_window[valleys],
            "prominences": properties.get("prominences", []),
        }

    # Zero crossings
    if "zero_crossings" in annotations:
        zero_crossings = np.where(np.diff(np.signbit(signal_window)))[0]
        results["zero_crossings"] = {"indices": zero_crossings, "times": zero_crossings / fs}

    # Basic statistics
    results["statistics"] = {
        "mean": np.mean(signal_window),
        "std": np.std(signal_window),
        "min": np.min(signal_window),
        "max": np.max(signal_window),
        "rms": np.sqrt(np.mean(signal_window**2)),
        "peak_to_peak": np.max(signal_window) - np.min(signal_window),
        "crest_factor": (
            np.max(np.abs(signal_window)) / np.sqrt(np.mean(signal_window**2))
            if np.mean(signal_window**2) > 0
            else 0
        ),
    }

    return results


def compute_waveform_features(signal, fs):
    """
    Compute comprehensive waveform features for PPG analysis.

    Args:
        signal (np.ndarray): Input PPG signal
        fs (float): Sampling frequency in Hz

    Returns:
        dict: Dictionary containing waveform features
    """
    try:
        # Validate input
        if signal is None or len(signal) == 0:
            print("WARNING: Empty or None signal passed to compute_waveform_features")
            return {}

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print("WARNING: Signal contains NaN or Inf values")
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Basic statistical features
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms_val = np.sqrt(np.mean(signal**2))

        # Peak-to-peak amplitude
        peak_to_peak = np.max(signal) - np.min(signal)

        # Crest factor (peak amplitude / RMS)
        crest_factor = np.max(np.abs(signal)) / rms_val if rms_val > 0 else 0

        # Shape factor (RMS / mean absolute value)
        shape_factor = rms_val / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0

        # Impulse factor (peak amplitude / mean absolute value)
        impulse_factor = (
            np.max(np.abs(signal)) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
        )

        # Margin factor (peak amplitude / mean absolute value of signal above mean)
        above_mean = signal[signal > mean_val]
        margin_factor = (
            np.max(signal) / np.mean(above_mean)
            if len(above_mean) > 0 and np.mean(above_mean) > 0
            else 0
        )

        # Peak detection for timing features
        peaks, _ = find_peaks(signal, prominence=0.1 * std_val)

        if len(peaks) > 1:
            # Inter-peak intervals
            peak_intervals = np.diff(peaks) / fs
            mean_interval = np.mean(peak_intervals)
            std_interval = np.std(peak_intervals)

            # Heart rate variability (if applicable)
            hrv = 60.0 / mean_interval if mean_interval > 0 else 0
        else:
            mean_interval = std_interval = hrv = 0

        features = {
            "statistical": {
                "mean": mean_val,
                "std": std_val,
                "rms": rms_val,
                "peak_to_peak": peak_to_peak,
                "crest_factor": crest_factor,
                "shape_factor": shape_factor,
                "impulse_factor": impulse_factor,
                "margin_factor": margin_factor,
            },
            "timing": {
                "num_peaks": len(peaks),
                "mean_peak_interval": mean_interval,
                "std_peak_interval": std_interval,
                "estimated_hr": hrv,
            },
            "signal_quality": {
                "snr_estimate": 20 * np.log10(np.max(signal) / std_val) if std_val > 0 else 0,
                "dynamic_range": np.log10(peak_to_peak / std_val) if std_val > 0 else 0,
            },
        }

        return features

    except Exception as e:
        print(f"ERROR in compute_waveform_features: {e}")
        return {}
