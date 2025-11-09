"""
Respiratory Rate Estimation Module

This module provides multiple algorithms for estimating respiratory rate from physiological signals.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- FFT-based respiratory rate estimation
- Time-domain respiratory analysis
- Frequency-domain methods
- Peak detection algorithms

Examples:
--------
FFT-based estimation:
    >>> from vitalDSP.respiratory_analysis.estimate_rr import fft_based_rr
    >>> rr = fft_based_rr(signal, fs=250)
"""

from .fft_based_rr import fft_based_rr
from .peak_detection_rr import peak_detection_rr
from .time_domain_rr import time_domain_rr
from .frequency_domain_rr import frequency_domain_rr

__all__ = [
    "fft_based_rr",
    "peak_detection_rr",
    "time_domain_rr",
    "frequency_domain_rr",
]
