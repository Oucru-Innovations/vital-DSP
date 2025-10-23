"""
Respiratory Analysis Module for Physiological Signal Processing

This module provides comprehensive respiratory analysis capabilities for
physiological signals including PPG, ECG, and other vital signs. It implements
multiple methods for respiratory rate estimation, sleep apnea detection, and
multimodal respiratory analysis.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Multiple respiratory rate estimation methods
- Time-domain and frequency-domain analysis
- FFT-based respiratory rate computation
- Peak detection algorithms for respiration
- Sleep apnea detection and analysis
- Multimodal respiratory-cardiac fusion
- PPG-ECG respiratory fusion
- Comprehensive respiratory pattern analysis

Examples:
--------
Basic respiratory analysis:
    >>> from vitalDSP.respiratory_analysis import RespiratoryAnalysis
    >>> resp_analyzer = RespiratoryAnalysis(signal, fs=250)
    >>> rr_rate = resp_analyzer.compute_respiratory_rate()

FFT-based respiratory rate:
    >>> from vitalDSP.respiratory_analysis.estimate_rr import fft_based_rr
    >>> rr_fft = fft_based_rr(signal, fs=250)

Sleep apnea detection:
    >>> from vitalDSP.respiratory_analysis.sleep_apnea_detection import pause_detection
    >>> apnea_events = pause_detection(signal, fs=250)
"""

from .respiratory_analysis import RespiratoryAnalysis

__all__ = ["RespiratoryAnalysis"]
