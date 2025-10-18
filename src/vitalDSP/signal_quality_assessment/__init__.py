"""
Signal Quality Assessment Module for Physiological Signal Processing

This module provides comprehensive signal quality assessment capabilities for
physiological signals including ECG, PPG, EEG, and other vital signs. It
implements various quality metrics, artifact detection, and signal enhancement
methods for ensuring reliable signal analysis.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Signal-to-Noise Ratio (SNR) computation
- Peak Signal-to-Noise Ratio (PSNR) calculation
- Mean Square Error (MSE) assessment
- Adaptive SNR estimation
- Artifact detection and removal
- Blind source separation
- Multi-modal artifact detection
- Signal quality indexing
- Comprehensive quality metrics

Examples:
--------
Basic signal quality assessment:
    >>> from vitalDSP.signal_quality_assessment import SignalQuality
    >>> sq = SignalQuality(original_signal, processed_signal)
    >>> snr_db = sq.snr()
    >>> psnr_db = sq.psnr()

Artifact detection:
    >>> from vitalDSP.signal_quality_assessment import ArtifactDetectionRemoval
    >>> adr = ArtifactDetectionRemoval(signal, fs=250)
    >>> artifacts = adr.detect_artifacts()

Adaptive SNR estimation:
    >>> from vitalDSP.signal_quality_assessment import AdaptiveSNREstimation
    >>> asnr = AdaptiveSNREstimation(signal, fs=250)
    >>> snr_adaptive = asnr.estimate_snr()
"""

from .signal_quality import SignalQuality
from .signal_quality_index import SignalQualityIndex

__all__ = [
    "SignalQuality",
    "SignalQualityIndex",
]
