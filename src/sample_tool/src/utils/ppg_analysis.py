"""
PPG-specific analysis utilities.

This module provides functions for analyzing photoplethysmogram (PPG) signals,
including peak detection, heart rate calculation, SpO2 estimation, and
signal processing utilities.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import coherence, find_peaks, spectrogram, welch

from ..config.logging_config import (
    get_logger,
    log_analysis_step,
    log_computation_complete,
    log_computation_progress,
    log_computation_start,
    log_data_validation,
)
from .exceptions import (
    InsufficientDataError,
    InvalidParameterError,
    PeakDetectionError,
    PPGError,
    SignalProcessingError,
    ValidationError,
)
from .validation import (
    validate_heart_rate_range,
    validate_numeric_array,
    validate_sampling_frequency,
)

# Get logger for this module
logger = get_logger(__name__)


def robust_absorbance(x: np.ndarray) -> np.ndarray:
    """
    Calculate robust absorbance using Beer-Lambert law approximation.

    Uses median as reference intensity (I0) and guards against zero values.

    Args:
        x: Input signal array

    Returns:
        Absorbance values: A = -log(I / I0)

    Raises:
        ValidationError: If input array is invalid
    """
    log_computation_start("robust absorbance calculation", input_shape=x.shape, input_dtype=x.dtype)

    try:
        log_data_validation("input array", x.shape, min_length=1, allow_nan=False, allow_inf=False)
        validate_numeric_array(x, min_length=1, allow_nan=False, allow_inf=False)

        x = np.asarray(x, dtype=float)
        I0 = np.median(x) if np.median(x) > 0 else np.maximum(np.mean(x), 1.0)

        log_computation_progress(
            "robust absorbance calculation", "calculated reference intensity", I0=I0
        )

        result = -np.log(np.clip(x / I0, 1e-9, None))

        log_computation_complete(
            "robust absorbance calculation",
            f"calculated absorbance for {len(x)} samples",
            output_shape=result.shape,
            output_range=(np.min(result), np.max(result)),
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate absorbance: {e}")
        raise PPGError(f"Failed to calculate absorbance: {e}") from e


def beats_from_peaks(
    sig_ac: np.ndarray, fs: float, t_peaks: np.ndarray, guard: float = 0.15
) -> List[Tuple[int, int, int]]:
    """
    Return list of (idx_start, idx_peak, idx_end) for each beat.

    Uses midpoints between peaks to segment beats and applies guard time
    to trim edges by a fraction of local inter-beat interval.

    Args:
        sig_ac: AC component of the signal
        fs: Sampling frequency in Hz
        t_peaks: Peak time indices in seconds
        guard: Fraction of local IBI to trim from edges

    Returns:
        List of tuples: (start_idx, peak_idx, end_idx) for each beat

    Raises:
        ValidationError: If input parameters are invalid
        InsufficientDataError: If insufficient peaks for beat segmentation
    """
    log_computation_start(
        "beat segmentation from peaks",
        signal_length=len(sig_ac),
        fs=fs,
        num_peaks=len(t_peaks),
        guard=guard,
    )

    try:
        log_data_validation("AC signal", sig_ac.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)
        log_data_validation("peak times", t_peaks.shape, min_length=1)

        validate_numeric_array(sig_ac, min_length=1)
        validate_sampling_frequency(fs)
        validate_numeric_array(t_peaks, min_length=1)

        if not (0 < guard < 0.5):
            raise InvalidParameterError(
                "Guard fraction must be between 0 and 0.5", field="guard", value=guard
            )

        if len(t_peaks) < 2:
            raise InsufficientDataError("Need at least 2 peaks for beat segmentation")

        log_computation_progress("beat segmentation", "validated input parameters")

        # Convert time peaks to sample indices
        p = (t_peaks * fs).astype(int)
        ibis = np.diff(p)  # Inter-beat intervals

        log_computation_progress(
            "beat segmentation",
            f"calculated {len(ibis)} inter-beat intervals",
            ibi_range=(np.min(ibis), np.max(ibis)),
        )

        # Calculate midpoints between peaks
        mids = p[:-1] + ibis // 2

        # Calculate start and end indices with guard time
        starts = np.r_[max(0, p[0] - int(guard * ibis[0])), mids]
        ends = np.r_[mids, p[-1] + int(guard * ibis[-1])]

        log_computation_progress("beat segmentation", "calculated beat boundaries with guard time")

        # Create beat segments
        beats = []
        n = len(sig_ac)

        for s, pk, e in zip(starts, p, ends):
            s = int(np.clip(s, 0, n - 2))
            e = int(np.clip(e, s + 1, n - 1))
            beats.append((s, pk, e))

        log_computation_complete(
            "beat segmentation",
            f"created {len(beats)} beat segments",
            num_beats=len(beats),
            signal_length=n,
        )

        return beats

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to segment beats from peaks: {e}")
        raise PPGError(f"Failed to segment beats from peaks: {e}") from e


def beat_ac_dc(
    signal_raw: np.ndarray,
    signal_ac: np.ndarray,
    beats: List[Tuple[int, int, int]],
    fs: float,
    dc_win_s: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate per-beat AC amplitude and DC component.

    AC amplitude is peak-to-trough within each beat.
    DC component is median around the peak within specified window.

    Args:
        signal_raw: Raw signal values
        signal_ac: AC component of the signal
        beats: List of beat segments (start, peak, end)
        fs: Sampling frequency in Hz
        dc_win_s: Window size for DC calculation in seconds

    Returns:
        Tuple of (time_mid, ac_amplitude, dc_component) arrays

    Raises:
        ValidationError: If input parameters are invalid
        PPGError: For other processing errors
    """
    log_computation_start("beat AC/DC calculation", num_beats=len(beats), fs=fs, dc_window=dc_win_s)

    try:
        log_data_validation("raw signal", signal_raw.shape, min_length=1)
        log_data_validation("AC signal", signal_ac.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)

        validate_numeric_array(signal_raw, min_length=1)
        validate_numeric_array(signal_ac, min_length=1)
        validate_sampling_frequency(fs)

        if dc_win_s <= 0:
            raise InvalidParameterError(
                "DC window size must be positive", field="dc_win_s", value=dc_win_s
            )

        if len(signal_raw) != len(signal_ac):
            raise ValidationError("Raw and AC signals must have same length")

        log_computation_progress("beat AC/DC calculation", "validated input parameters")

        ac, dc, t_mid = [], [], []
        n = len(signal_raw)
        half = int(dc_win_s * fs / 2)

        log_computation_progress("beat AC/DC calculation", f"processing {len(beats)} beats")

        for i, (s, pk, e) in enumerate(beats):
            # Validate beat indices
            if not (0 <= s < pk < e < n):
                continue

            seg = signal_ac[s : e + 1]
            if len(seg) < 3:
                continue

            # Calculate AC amplitude (peak-to-trough)
            a = float(np.max(seg) - np.min(seg))

            # Calculate DC as median around peak
            lo = max(0, pk - half)
            hi = min(n, pk + half)
            d = float(np.median(signal_raw[lo:hi])) if hi > lo else float(np.median(signal_raw))

            ac.append(a)
            dc.append(d)
            t_mid.append((s + e) / (2 * fs))

            if i % 10 == 0:  # Log progress every 10 beats
                log_computation_progress(
                    "beat AC/DC calculation", f"processed {i+1}/{len(beats)} beats"
                )

        result = (np.array(t_mid), np.array(ac), np.array(dc))

        log_computation_complete(
            "beat AC/DC calculation",
            f"calculated AC/DC for {len(ac)} valid beats",
            num_valid_beats=len(ac),
            ac_range=(np.min(ac), np.max(ac)),
            dc_range=(np.min(dc), np.max(dc)),
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate beat AC/DC: {e}")
        raise PPGError(f"Failed to calculate beat AC/DC: {e}") from e


def r_series_spo2(
    red_raw: np.ndarray,
    ir_raw: np.ndarray,
    red_ac: np.ndarray,
    ir_ac: np.ndarray,
    t_peaks: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute beat-by-beat R ratio and SpO2.

    Uses IR peaks to segment beats and calculates R ratio and SpO2
    for each beat using the Beer-Lambert relationship.

    Args:
        red_raw: Raw RED channel signal
        ir_raw: Raw IR channel signal
        red_ac: AC component of RED channel
        ir_ac: AC component of IR channel
        t_peaks: Peak time indices in seconds
        fs: Sampling frequency in Hz

    Returns:
        Tuple of (time_beats, R_ratios, SpO2_values) arrays

    Raises:
        ValidationError: If input parameters are invalid
        PPGError: For other processing errors
    """
    log_computation_start(
        "R-series and SpO2 calculation", num_peaks=len(t_peaks), fs=fs, signal_length=len(red_raw)
    )

    try:
        log_data_validation("red raw signal", red_raw.shape, min_length=1)
        log_data_validation("IR raw signal", ir_raw.shape, min_length=1)
        log_data_validation("red AC signal", red_ac.shape, min_length=1)
        log_data_validation("IR AC signal", ir_ac.shape, min_length=1)
        log_data_validation("peak times", t_peaks.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)

        # Validate inputs
        for signal, name in [
            (red_raw, "red_raw"),
            (ir_raw, "ir_raw"),
            (red_ac, "red_ac"),
            (ir_ac, "ir_ac"),
        ]:
            validate_numeric_array(signal, min_length=1)

        validate_sampling_frequency(fs)
        validate_numeric_array(t_peaks, min_length=1)

        # Check signal lengths
        signals = [red_raw, ir_raw, red_ac, ir_ac]
        if len(set(len(s) for s in signals)) != 1:
            raise ValidationError("All signals must have the same length")

        log_computation_progress("R-series and SpO2 calculation", "validated input parameters")

        # Get beat segments using IR peaks
        log_analysis_step("beat segmentation", "using IR peaks for beat detection")
        beats = beats_from_peaks(ir_ac, fs, t_peaks)
        if not beats:
            log_computation_complete("R-series and SpO2 calculation", "no valid beats found")
            return np.array([]), np.array([]), np.array([])

        log_computation_progress("R-series and SpO2 calculation", f"segmented {len(beats)} beats")

        # Calculate AC/DC for both channels
        log_analysis_step("AC/DC calculation", "calculating per-beat AC and DC components")
        t_mid, ac_r, dc_r = beat_ac_dc(red_raw, red_ac, beats, fs)
        _, ac_i, dc_i = beat_ac_dc(ir_raw, ir_ac, beats, fs)

        log_computation_progress(
            "R-series and SpO2 calculation", "calculated AC/DC components for both channels"
        )

        # Filter valid beats (positive AC and DC values)
        ok = (ac_r > 0) & (ac_i > 0) & (dc_r > 0) & (dc_i > 0)
        if not np.any(ok):
            log_computation_complete(
                "R-series and SpO2 calculation", "no valid beats after filtering"
            )
            return t_mid, np.array([]), np.array([])

        log_computation_progress(
            "R-series and SpO2 calculation", f"filtered to {np.sum(ok)} valid beats"
        )

        # Calculate R ratio and SpO2
        log_analysis_step(
            "R-ratio calculation", "computing R ratios using Beer-Lambert relationship"
        )
        R = ((ac_r / dc_r) / (ac_i / dc_i))[ok]
        tB = t_mid[ok]

        # Empirical SpO2 formula (calibrated for specific sensor)
        log_analysis_step("SpO2 estimation", "applying empirical calibration formula")
        spo2 = -45.06 * R**2 + 30.354 * R + 94.845

        log_computation_complete(
            "R-series and SpO2 calculation",
            f"calculated R-ratios and SpO2 for {len(R)} beats",
            num_valid_beats=len(R),
            r_range=(np.min(R), np.max(R)),
            spo2_range=(np.min(spo2), np.max(spo2)),
        )

        return tB, R, spo2

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate R series and SpO2: {e}")
        raise PPGError(f"Failed to calculate R series and SpO2: {e}") from e


def avg_beat(
    signal: np.ndarray, t_peaks: np.ndarray, fs: float, width_s: float = 1.2, out_len: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ensemble average of beats around peaks.

    Args:
        signal: Input signal
        t_peaks: Peak time indices in seconds
        fs: Sampling frequency in Hz
        width_s: Width of beat window in seconds
        out_len: Output length for resampling

    Returns:
        Tuple of (time_relative, mean_signal, std_signal) arrays

    Raises:
        ValidationError: If input parameters are invalid
        PPGError: For other processing errors
    """
    log_computation_start(
        "ensemble average beat calculation",
        num_peaks=len(t_peaks),
        fs=fs,
        width=width_s,
        output_length=out_len,
    )

    try:
        log_data_validation("input signal", signal.shape, min_length=1)
        log_data_validation("peak times", t_peaks.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)

        validate_numeric_array(signal, min_length=1)
        validate_sampling_frequency(fs)
        validate_numeric_array(t_peaks, min_length=1)

        if width_s <= 0:
            raise InvalidParameterError(
                "Beat width must be positive", field="width_s", value=width_s
            )

        if out_len <= 0:
            raise InvalidParameterError(
                "Output length must be positive", field="out_len", value=out_len
            )

        if len(t_peaks) < 2:
            raise InsufficientDataError("Need at least 2 peaks for ensemble average")

        log_computation_progress("ensemble average beat calculation", "validated input parameters")

        half = int(width_s * fs / 2)
        p = (t_peaks * fs).astype(int)
        segs = []

        log_computation_progress(
            "ensemble average beat calculation", f"extracting {len(p)} beat segments"
        )

        # Extract beat segments
        for i, pk in enumerate(p):
            s = max(0, pk - half)
            e = min(len(signal), pk + half)
            seg = signal[s:e]

            if len(seg) < 10:  # Minimum segment length
                continue

            segs.append(seg)

            if i % 10 == 0:  # Log progress every 10 segments
                log_computation_progress(
                    "ensemble average beat calculation", f"extracted {i+1}/{len(p)} segments"
                )

        if not segs:
            log_computation_complete("ensemble average beat calculation", "no valid segments found")
            return np.array([]), np.array([]), np.array([])

        log_computation_progress(
            "ensemble average beat calculation", f"extracted {len(segs)} valid segments"
        )

        # Resample segments to common length
        log_analysis_step("segment resampling", "interpolating segments to common length")

        resampled = []
        for i, seg in enumerate(segs):
            if len(seg) < 2:
                continue

            # Create interpolation function
            x_old = np.linspace(0, 1, len(seg))
            x_new = np.linspace(0, 1, out_len)

            try:
                f = interp1d(
                    x_old, seg, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                resampled.append(f(x_new))
            except Exception:
                continue

            if i % 10 == 0:  # Log progress every 10 segments
                log_computation_progress(
                    "ensemble average beat calculation", f"resampled {i+1}/{len(segs)} segments"
                )

        if not resampled:
            log_computation_complete(
                "ensemble average beat calculation", "no segments could be resampled"
            )
            return np.array([]), np.array([]), np.array([])

        log_computation_progress(
            "ensemble average beat calculation",
            f"resampled {len(resampled)} segments to {out_len} points",
        )

        # Calculate ensemble statistics
        log_analysis_step("ensemble statistics", "calculating mean and standard deviation")
        resampled = np.array(resampled)
        mean_sig = np.mean(resampled, axis=0)
        std_sig = np.std(resampled, axis=0)
        t_rel = np.linspace(-width_s / 2, width_s / 2, out_len)

        result = (t_rel, mean_sig, std_sig)

        log_computation_complete(
            "ensemble average beat calculation",
            f"calculated ensemble average from {len(resampled)} beats",
            num_beats=len(resampled),
            output_length=out_len,
            mean_range=(np.min(mean_sig), np.max(mean_sig)),
            std_range=(np.min(std_sig), np.max(std_sig)),
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate average beat: {e}")
        raise PPGError(f"Failed to calculate average beat: {e}") from e


def find_ppg_peaks(
    signal: np.ndarray,
    fs: float,
    hr_min: float = 40,
    hr_max: float = 180,
    prominence_factor: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """
    Find peaks in PPG signal with heart rate constraints.

    Args:
        signal: Input PPG signal
        fs: Sampling frequency in Hz
        hr_min: Minimum heart rate in BPM
        hr_max: Maximum heart rate in BPM
        prominence_factor: Factor for peak prominence calculation

    Returns:
        Tuple of (peak_indices, peak_properties)

    Raises:
        ValidationError: If input parameters are invalid
        PeakDetectionError: If peak detection fails
    """
    log_computation_start(
        "PPG peak detection",
        signal_length=len(signal),
        fs=fs,
        hr_min=hr_min,
        hr_max=hr_max,
        prominence_factor=prominence_factor,
    )

    try:
        log_data_validation("input signal", signal.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)
        log_data_validation("heart rate range", hr_min=hr_min, hr_max=hr_max)

        # Validate inputs
        validate_numeric_array(signal, min_length=1)
        validate_sampling_frequency(fs)
        validate_heart_rate_range(hr_min, hr_max)

        if not (0.1 <= prominence_factor <= 2.0):
            raise InvalidParameterError(
                "Prominence factor must be between 0.1 and 2.0",
                field="prominence_factor",
                value=prominence_factor,
            )

        log_computation_progress("PPG peak detection", "validated input parameters")

        # Calculate minimum distance between peaks based on heart rate
        min_distance = int(fs * 60 / hr_max)  # Maximum heart rate = minimum distance

        # Calculate prominence threshold
        prominence_threshold = prominence_factor * np.std(signal)

        log_computation_progress(
            "PPG peak detection",
            "calculated detection parameters",
            min_distance=min_distance,
            prominence_threshold=prominence_threshold,
        )

        # Find peaks
        log_analysis_step("peak detection", "applying scipy.find_peaks with constraints")
        peaks, properties = find_peaks(
            signal, distance=min_distance, prominence=prominence_threshold, height=np.mean(signal)
        )

        if len(peaks) == 0:
            raise PeakDetectionError("No peaks found with current parameters")

        log_computation_progress("PPG peak detection", f"found {len(peaks)} initial peaks")

        # Filter peaks by maximum heart rate constraint
        if len(peaks) > 1:
            log_analysis_step("peak filtering", "filtering peaks by heart rate constraints")
            peak_times = peaks / fs
            ibis = np.diff(peak_times)
            max_ibi = 60 / hr_min  # Minimum heart rate = maximum IBI

            # Keep peaks that don't violate minimum heart rate
            valid_peaks = [peaks[0]]  # Always keep first peak
            for i, peak in enumerate(peaks[1:], 1):
                if ibis[i - 1] <= max_ibi:
                    valid_peaks.append(peak)
                else:
                    break

            peaks = np.array(valid_peaks)
            # Update properties for valid peaks
            properties = {k: v[: len(peaks)] for k, v in properties.items()}

            log_computation_progress("PPG peak detection", f"filtered to {len(peaks)} valid peaks")

        result = (peaks, properties)

        log_computation_complete(
            "PPG peak detection",
            f"detected {len(peaks)} valid peaks",
            num_peaks=len(peaks),
            signal_length=len(signal),
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to detect PPG peaks: {e}")
        raise PPGError(f"Failed to detect PPG peaks: {e}") from e


def calculate_heart_rate(peaks: np.ndarray, fs: float, method: str = "mean") -> float:
    """
    Calculate heart rate from peak intervals.

    Args:
        peaks: Peak indices
        fs: Sampling frequency in Hz
        method: Method for heart rate calculation ('mean', 'median', 'mode')

    Returns:
        Heart rate in BPM

    Raises:
        ValidationError: If input parameters are invalid
        HeartRateCalculationError: If heart rate calculation fails
    """
    log_computation_start("heart rate calculation", num_peaks=len(peaks), fs=fs, method=method)

    try:
        log_data_validation("peak indices", peaks.shape, min_length=2)
        log_data_validation("sampling frequency", fs=fs)
        log_data_validation("calculation method", method=method)

        # Validate inputs
        validate_numeric_array(peaks, min_length=2)
        validate_sampling_frequency(fs)

        if method not in ["mean", "median", "mode"]:
            raise InvalidParameterError(
                "Method must be 'mean', 'median', or 'mode'", field="method", value=method
            )

        log_computation_progress("heart rate calculation", "validated input parameters")

        # Calculate inter-beat intervals
        log_analysis_step("IBI calculation", "computing inter-beat intervals from peaks")
        peak_times = peaks / fs
        ibis = np.diff(peak_times)

        log_computation_progress(
            "heart rate calculation",
            f"calculated {len(ibis)} inter-beat intervals",
            ibi_range=(np.min(ibis), np.max(ibis)),
        )

        # Convert to heart rate
        hr_values = 60 / ibis

        log_computation_progress(
            "heart rate calculation",
            "converted to heart rate values",
            hr_range=(np.min(hr_values), np.max(hr_values)),
        )

        # Calculate final heart rate based on method
        log_analysis_step("final heart rate calculation", f"applying {method} method")
        if method == "mean":
            hr = np.mean(hr_values)
        elif method == "median":
            hr = np.median(hr_values)
        else:  # mode
            from scipy.stats import mode

            hr = float(mode(hr_values, keepdims=False)[0])

        log_computation_complete(
            "heart rate calculation",
            f"calculated {method} heart rate",
            heart_rate=hr,
            method=method,
            num_intervals=len(ibis),
        )

        return hr

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate heart rate: {e}")
        raise PPGError(f"Failed to calculate heart rate: {e}") from e


def ms_coherence(red_ac: np.ndarray, ir_ac: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate magnitude-squared coherence between red and IR AC signals.

    Args:
        red_ac: Red channel AC component
        ir_ac: IR channel AC component
        fs: Sampling frequency in Hz

    Returns:
        Tuple of (frequencies, coherence_values)

    Raises:
        ValidationError: If input parameters are invalid
        PPGError: If coherence calculation fails
    """
    log_computation_start(
        "magnitude-squared coherence calculation",
        red_length=len(red_ac),
        ir_length=len(ir_ac),
        fs=fs,
    )

    try:
        log_data_validation("red AC signal", red_ac.shape, min_length=1)
        log_data_validation("IR AC signal", ir_ac.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)

        # Validate inputs
        validate_numeric_array(red_ac, min_length=1)
        validate_numeric_array(ir_ac, min_length=1)
        validate_sampling_frequency(fs)

        # Ensure both signals have the same length
        min_length = min(len(red_ac), len(ir_ac))
        if min_length < 1:
            raise InsufficientDataError("Signals must have at least 1 sample")

        log_computation_progress(
            "magnitude-squared coherence calculation",
            f"using minimum length of {min_length} samples",
        )

        # Calculate coherence
        log_analysis_step("coherence calculation", "applying scipy.signal.coherence")
        f, C = coherence(
            red_ac[:min_length], ir_ac[:min_length], fs=fs, nperseg=min(min_length, 2048)
        )

        log_computation_complete(
            "magnitude-squared coherence calculation",
            f"calculated coherence for {len(f)} frequency bins",
            num_freq_bins=len(f),
            freq_range=(np.min(f), np.max(f)),
            coherence_range=(np.min(C), np.max(C)),
        )

        return f, C

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate coherence: {e}")
        raise PPGError(f"Failed to calculate coherence: {e}") from e


def estimate_spo2(
    red_raw: np.ndarray, ir_raw: np.ndarray, red_ac: np.ndarray, ir_ac: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate SpO2 using the Beer-Lambert law approximation.

    Args:
        red_raw: Raw red channel signal
        ir_raw: Raw IR channel signal
        red_ac: Red channel AC component
        ir_ac: IR channel AC component

    Returns:
        Tuple of (SpO2_percentage, R_ratio, perfusion_index) or (None, None, None) if calculation fails

    Raises:
        ValidationError: If input parameters are invalid
        PPGError: If SpO2 estimation fails
    """
    log_computation_start("SpO2 estimation", red_length=len(red_raw), ir_length=len(ir_raw))

    try:
        log_data_validation("red raw signal", red_raw.shape, min_length=1)
        log_data_validation("IR raw signal", ir_raw.shape, min_length=1)
        log_data_validation("red AC signal", red_ac.shape, min_length=1)
        log_data_validation("IR AC signal", ir_ac.shape, min_length=1)

        # Validate inputs
        validate_numeric_array(red_raw, min_length=1)
        validate_numeric_array(ir_raw, min_length=1)
        validate_numeric_array(red_ac, min_length=1)
        validate_numeric_array(ir_ac, min_length=1)

        log_computation_progress("SpO2 estimation", "validated input parameters")

        # Calculate DC components (mean values)
        log_analysis_step("DC component calculation", "calculating mean values for DC components")
        dc_red = float(np.mean(red_raw))
        dc_ir = float(np.mean(ir_raw))

        log_computation_progress(
            "SpO2 estimation", "calculated DC components", dc_red=dc_red, dc_ir=dc_ir
        )

        # Calculate AC amplitudes (peak-to-peak / 2)
        log_analysis_step("AC amplitude calculation", "calculating peak-to-peak amplitudes")
        ac_red_amp = float(np.ptp(red_ac) / 2.0)
        ac_ir_amp = float(np.ptp(ir_ac) / 2.0)

        log_computation_progress(
            "SpO2 estimation",
            "calculated AC amplitudes",
            ac_red_amp=ac_red_amp,
            ac_ir_amp=ac_ir_amp,
        )

        # Check for valid values
        if min(dc_red, dc_ir, ac_red_amp, ac_ir_amp) <= 0:
            log_computation_complete(
                "SpO2 estimation", "calculation failed - invalid values detected"
            )
            return None, None, None

        # Calculate R ratio
        log_analysis_step(
            "R-ratio calculation", "computing R ratio using Beer-Lambert relationship"
        )
        R = (ac_red_amp / dc_red) / (ac_ir_amp / dc_ir)

        log_computation_progress("SpO2 estimation", "calculated R ratio", R_ratio=R)

        # Estimate SpO2 using empirical formula
        log_analysis_step("SpO2 calculation", "applying empirical calibration formula")
        spo2 = -45.06 * R**2 + 30.354 * R + 94.845

        # Calculate perfusion index (PI) via IR channel
        log_analysis_step("perfusion index calculation", "calculating PI from IR channel")
        PI = 100.0 * (ac_ir_amp / dc_ir)

        result = (spo2, R, PI)

        log_computation_complete(
            "SpO2 estimation",
            f"estimated SpO2: {spo2:.1f}%, R: {R:.3f}, PI: {PI:.1f}%",
            spo2=spo2,
            r_ratio=R,
            perfusion_index=PI,
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to estimate SpO2: {e}")
        raise PPGError(f"Failed to estimate SpO2: {e}") from e


def sdppg(x: np.ndarray) -> np.ndarray:
    """
    Calculate second derivative of PPG signal.

    Args:
        x: Input signal array

    Returns:
        Second derivative of the signal

    Raises:
        ValidationError: If input array is invalid
        PPGError: If second derivative calculation fails
    """
    log_computation_start("second derivative calculation", input_shape=x.shape, input_dtype=x.dtype)

    try:
        log_data_validation("input signal", x.shape, min_length=3)

        # Validate input
        validate_numeric_array(x, min_length=3)  # Need at least 3 points for second derivative

        log_computation_progress("second derivative calculation", "validated input parameters")

        # Calculate second derivative using gradient
        log_analysis_step("derivative calculation", "applying numpy.gradient twice")
        result = np.gradient(np.gradient(x))

        log_computation_complete(
            "second derivative calculation",
            f"calculated second derivative for {len(x)} samples",
            output_shape=result.shape,
            output_range=(np.min(result), np.max(result)),
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to calculate second derivative: {e}")
        raise PPGError(f"Failed to calculate second derivative: {e}") from e


def compute_hr_trend(
    ac_signal: np.ndarray, fs: float, hr_min: int = 40, hr_max: int = 180, prom_factor: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Beat detection on AC signal; returns peak times (s), ibi array (s), HR rolling over windows (s, bpm).

    Args:
        ac_signal: AC component of the PPG signal
        fs: Sampling frequency in Hz
        hr_min: Minimum heart rate in BPM
        hr_max: Maximum heart rate in BPM
        prom_factor: Peak prominence factor (peaks prominence = factor * std(AC))

    Returns:
        Tuple containing:
        - t_peaks: Peak times in seconds
        - ibis: Inter-beat intervals in seconds
        - (hr_t, hr_bpm): Heart rate time points and values (filtered by HR range)

    Raises:
        ValidationError: If input parameters are invalid
        PPGError: If signal processing fails
    """
    log_computation_start(
        "heart rate trend calculation",
        signal_length=len(ac_signal),
        fs=fs,
        hr_min=hr_min,
        hr_max=hr_max,
        prom_factor=prom_factor,
    )

    try:
        log_data_validation("AC signal", ac_signal.shape, min_length=1)
        log_data_validation("sampling frequency", fs=fs)
        log_data_validation("heart rate range", hr_min=hr_min, hr_max=hr_max)

        # Validate inputs
        validate_numeric_array(ac_signal, min_length=1)
        validate_sampling_frequency(fs)

        if len(ac_signal) < fs:  # too short
            log_computation_complete(
                "heart rate trend calculation", "signal too short for analysis"
            )
            return np.array([]), np.array([]), (np.array([]), np.array([]))

        log_computation_progress("heart rate trend calculation", "validated input parameters")

        # Calculate detection parameters
        min_dist = int(max(1, fs * 60.0 / hr_max))
        prom = float(prom_factor) * float(np.std(ac_signal))

        log_computation_progress(
            "heart rate trend calculation",
            "calculated detection parameters",
            min_distance=min_dist,
            prominence_threshold=prom,
        )

        # Find peaks
        log_analysis_step("peak detection", "applying scipy.find_peaks with heart rate constraints")
        peaks, _ = find_peaks(ac_signal, distance=min_dist, prominence=max(1e-12, prom))

        if len(peaks) < 2:
            log_computation_complete("heart rate trend calculation", "insufficient peaks found")
            return np.array([]), np.array([]), (np.array([]), np.array([]))

        log_computation_progress("heart rate trend calculation", f"found {len(peaks)} peaks")

        # Calculate time-based metrics
        log_analysis_step(
            "time calculation", "converting peaks to time domain and calculating IBIs"
        )
        t_peaks = peaks / fs
        ibis = np.diff(t_peaks)  # seconds

        log_computation_progress(
            "heart rate trend calculation",
            f"calculated {len(ibis)} inter-beat intervals",
            ibi_range=(np.min(ibis), np.max(ibis)),
        )

        # HR over time: midpoint between consecutive peaks
        log_analysis_step(
            "heart rate calculation", "calculating heart rate at midpoints between peaks"
        )
        hr_t = (t_peaks[:-1] + t_peaks[1:]) / 2.0
        hr_bpm = 60.0 / np.clip(ibis, 1e-6, None)

        log_computation_progress(
            "heart rate trend calculation",
            "calculated heart rate values",
            hr_range=(np.min(hr_bpm), np.max(hr_bpm)),
        )

        # mask outside HR band
        log_analysis_step("heart rate filtering", "filtering heart rates within specified range")
        hr_mask = (hr_bpm >= hr_min) & (hr_bpm <= hr_max)
        filtered_hr_t = hr_t[hr_mask]
        filtered_hr_bpm = hr_bpm[hr_mask]

        log_computation_progress(
            "heart rate trend calculation", f"filtered to {np.sum(hr_mask)} valid heart rate values"
        )

        result = (t_peaks, ibis, (filtered_hr_t, filtered_hr_bpm))

        log_computation_complete(
            "heart rate trend calculation",
            f"calculated HR trend from {len(peaks)} peaks, {np.sum(hr_mask)} valid HR values",
            num_peaks=len(peaks),
            num_valid_hr=len(filtered_hr_bpm),
            hr_range=(np.min(filtered_hr_bpm), np.max(filtered_hr_bpm)),
        )

        return result

    except Exception as e:
        if isinstance(e, PPGError):
            raise
        logger.error(f"Failed to compute HR trend: {e}")
        raise PPGError(f"Failed to compute HR trend: {e}") from e
