"""
Signal Preprocessing Module for Physiological Signal Processing

This module provides comprehensive signal preprocessing capabilities including noise reduction, filtering, and signal conditioning.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Noise reduction algorithms
- Signal filtering and conditioning
- Preprocessing configuration
- Signal validation

Examples:
--------
Basic preprocessing:
    >>> from vitalDSP.preprocess import PreprocessConfig, preprocess_signal
    >>> config = PreprocessConfig(filter_type='bandpass')
    >>> processed = preprocess_signal(signal, fs=250, config=config)
"""

