"""
Data Processing Utilities Module

This module provides data processing utilities including validation, synthesis, and data manipulation.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Signal validation
- Data synthesis
- Data loading utilities
- Processing helpers

Examples:
--------
Data validation:
    >>> from vitalDSP.utils.data_processing import SignalValidator
    >>> validator = SignalValidator()
"""

from .data_loader import (
    DataLoader,
    StreamDataLoader,
    DataFormat,
    SignalType,
    load_signal,
    load_multi_channel,
    load_oucru_csv,
)
from .validation import (
    SignalValidator,
    validate_signal_input,
    validate_signal_length,
    validate_frequency_range,
    validate_positive_parameter,
)
from .synthesize_data import (
    generate_sinusoidal,
    generate_square_wave,
    generate_noisy_signal,
    generate_ecg_signal,
    generate_resp_signal,
    generate_synthetic_ppg,
)

"""
Data Processing Module
This module contains utilities for data loading, validation, and synthesis.
Components:
- data_loader: Comprehensive data loader for multiple formats
- validation: Data validation utilities
- synthesize_data: Data synthesis and generation utilities
"""
__all__ = [
    # Data Loader
    "DataLoader",
    "StreamDataLoader",
    "DataFormat",
    "SignalType",
    "load_signal",
    "load_multi_channel",
    "load_oucru_csv",
    # Validation
    "SignalValidator",
    "validate_signal_input",
    "validate_signal_length",
    "validate_frequency_range",
    "validate_positive_parameter",
    # Data Synthesis
    "generate_sinusoidal",
    "generate_square_wave",
    "generate_noisy_signal",
    "generate_ecg_signal",
    "generate_resp_signal",
    "generate_synthetic_ppg",
]
