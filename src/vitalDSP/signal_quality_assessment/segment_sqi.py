"""Per-segment SQI computation.

Slices a signal into fixed-duration segments and computes a small set
of raw (unscaled) SQI values for each.  Returns a DataFrame whose rows
correspond to segments — the input format expected by
:func:`vitalDSP.signal_quality_assessment.segment_classifier.classify_segments`.

We deliberately call the scalar SQI primitives (``scipy.stats.kurtosis``,
``scipy.stats.skew``, ``scipy.stats.entropy``, in-line SNR math) on each
segment, rather than going through :class:`SignalQualityIndex.<method>`.
The class's ``_process_segments`` helper always z-scores values across
segments, which would defeat the rule-dict thresholds (which live in
absolute units).

Available SQIs (v1):

================= =======================================================
Column            What it measures
================= =======================================================
``kurtosis_sqi``  Tailedness of the segment amplitude distribution.
``skewness_sqi``  Asymmetry of the segment amplitude distribution.
``entropy_sqi``   Shannon entropy of a histogram of the segment values.
``snr_sqi``       Crude per-segment SNR (dB) — signal power / robust noise.
``zero_crossing_sqi``  Zero-crossing rate (count / N).
``perfusion_sqi`` (AC peak-to-peak) / (DC mean), scaled to ‰; PPG-leaning.
``peak_to_peak_amplitude_sqi`` Raw peak-to-peak amplitude.
================= =======================================================

The full catalogue can grow later; v1 covers what's calibrated in the
bundled rule dicts.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

logger = logging.getLogger(__name__)


# Names of SQIs we know how to compute.  Order matches what the
# bundled rule dicts contain.
AVAILABLE_SQIS: Tuple[str, ...] = (
    "kurtosis_sqi",
    "skewness_sqi",
    "entropy_sqi",
    "snr_sqi",
    "zero_crossing_sqi",
    "perfusion_sqi",
    "peak_to_peak_amplitude_sqi",
)


# Default SQI subset per signal type.  Order matters because RuleSet
# short-circuits on the first reject — cheap+discriminative SQIs first.
DEFAULT_SEGMENT_SQIS = {
    "PPG": (
        "perfusion_sqi",
        "kurtosis_sqi",
        "skewness_sqi",
        "snr_sqi",
        "entropy_sqi",
    ),
    "ECG": (
        "kurtosis_sqi",
        "snr_sqi",
        "skewness_sqi",
        "entropy_sqi",
        "zero_crossing_sqi",
    ),
}


# ---------------------------------------------------------------------------
# Scalar SQI primitives
# ---------------------------------------------------------------------------


def _kurtosis_sqi(segment: np.ndarray) -> float:
    if segment.size < 3:
        return float("nan")
    return float(_kurtosis(segment, fisher=False, bias=False))


def _skewness_sqi(segment: np.ndarray) -> float:
    if segment.size < 3:
        return float("nan")
    return float(_skew(segment, bias=False))


def _entropy_sqi(segment: np.ndarray, n_bins: int = 16) -> float:
    """Shannon entropy of the segment's histogram, in nats."""
    if segment.size < 4:
        return float("nan")
    finite = segment[np.isfinite(segment)]
    if finite.size < 4:
        return float("nan")
    hist, _ = np.histogram(finite, bins=n_bins)
    p = hist.astype(float)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p /= total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _snr_sqi(segment: np.ndarray) -> float:
    """Crude per-segment SNR (dB).

    Uses ``20 * log10(rms / mad)`` where MAD is the median absolute
    deviation as a robust noise floor proxy.  Not as principled as a
    spectral SNR but cheap and consistent across short windows.
    """
    if segment.size < 4:
        return float("nan")
    finite = segment[np.isfinite(segment)]
    if finite.size < 4:
        return float("nan")
    centered = finite - np.median(finite)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    mad = float(np.median(np.abs(centered)))
    if mad < 1e-12 or rms < 1e-12:
        return float("nan")
    return float(20.0 * np.log10(rms / mad))


def _zero_crossing_sqi(segment: np.ndarray) -> float:
    """Zero-crossings per sample (count / N).  Mean-centred first."""
    if segment.size < 2:
        return float("nan")
    centered = segment - np.mean(segment)
    crossings = int(np.sum(np.diff(np.sign(centered)) != 0))
    return float(crossings / segment.size)


def _perfusion_sqi(segment: np.ndarray) -> float:
    """(AC peak-to-peak) / (|DC mean|), times 1000.

    Classic PPG perfusion index.  Defined for any signal but most
    meaningful when the DC component is non-zero (i.e. raw photoplethysmogram
    rather than mean-removed).  Returns NaN when DC is ~0.
    """
    if segment.size < 2:
        return float("nan")
    finite = segment[np.isfinite(segment)]
    if finite.size < 2:
        return float("nan")
    ac = float(np.ptp(finite))
    dc = float(np.mean(finite))
    if abs(dc) < 1e-9:
        return float("nan")
    return float(1000.0 * ac / abs(dc))


def _peak_to_peak_amplitude_sqi(segment: np.ndarray) -> float:
    if segment.size < 2:
        return float("nan")
    finite = segment[np.isfinite(segment)]
    if finite.size < 2:
        return float("nan")
    return float(np.ptp(finite))


_SQI_FUNCS = {
    "kurtosis_sqi": _kurtosis_sqi,
    "skewness_sqi": _skewness_sqi,
    "entropy_sqi": _entropy_sqi,
    "snr_sqi": _snr_sqi,
    "zero_crossing_sqi": _zero_crossing_sqi,
    "perfusion_sqi": _perfusion_sqi,
    "peak_to_peak_amplitude_sqi": _peak_to_peak_amplitude_sqi,
}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _segment_indices(
    n_samples: int,
    window_samples: int,
    step_samples: int,
) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` for a fixed-window sweep."""
    if window_samples <= 0 or n_samples < window_samples:
        return []
    out = []
    start = 0
    while start + window_samples <= n_samples:
        out.append((start, start + window_samples))
        start += step_samples
    return out


def compute_segment_sqis(
    signal: Sequence[float],
    sampling_freq: float,
    *,
    segment_seconds: float = 30.0,
    overlap_pct: float = 0.0,
    sqi_names: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Compute per-segment SQI values.

    Parameters
    ----------
    signal
        1-D array-like of samples.
    sampling_freq
        Sampling frequency in Hz.  Must be positive.
    segment_seconds
        Length of each segment in seconds.  Default 30 s (matches
        vital-sqi's pipeline default).
    overlap_pct
        Inter-segment overlap as a fraction in ``[0, 0.95]``.  ``0.0``
        gives non-overlapping segments; ``0.5`` gives 50% overlap.
    sqi_names
        Iterable of SQI names to compute.  Defaults to all in
        :data:`AVAILABLE_SQIS`.  Unknown names are ignored with a warning.

    Returns
    -------
    (sqi_df, milestones)
        ``sqi_df`` — DataFrame with one row per segment and one column per
        SQI; row index is the segment ordinal starting at 0.
        ``milestones`` — list of ``{start_idx, end_idx, t_start, t_end}``
        dicts.  Empty when ``signal`` is too short for one segment.
    """
    arr = np.asarray(signal, dtype=float).ravel()
    if sampling_freq is None or sampling_freq <= 0:
        raise ValueError(f"sampling_freq must be positive; got {sampling_freq!r}.")
    if not (0.0 <= overlap_pct < 0.95):
        raise ValueError(f"overlap_pct must be in [0, 0.95); got {overlap_pct!r}.")

    window_samples = max(1, int(round(segment_seconds * sampling_freq)))
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_pct))))

    if sqi_names is None:
        sqi_names = AVAILABLE_SQIS
    else:
        unknown = [n for n in sqi_names if n not in _SQI_FUNCS]
        if unknown:
            logger.warning("Ignoring unknown SQI names: %s", unknown)
        sqi_names = [n for n in sqi_names if n in _SQI_FUNCS]

    spans = _segment_indices(arr.size, window_samples, step_samples)
    if not spans:
        return (
            pd.DataFrame(columns=list(sqi_names)),
            [],
        )

    rows: List[dict] = []
    milestones: List[dict] = []
    for seg_idx, (s, e) in enumerate(spans):
        segment = arr[s:e]
        row = {name: _SQI_FUNCS[name](segment) for name in sqi_names}
        rows.append(row)
        milestones.append({
            "start_idx": int(s),
            "end_idx": int(e),
            "t_start": float(s / sampling_freq),
            "t_end": float(e / sampling_freq),
        })

    return pd.DataFrame(rows, columns=list(sqi_names)), milestones
