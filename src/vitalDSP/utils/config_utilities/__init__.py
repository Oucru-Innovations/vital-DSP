"""
Configuration Utilities Module

This module provides configuration management and adaptive parameter optimization for physiological signal processing.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Dynamic configuration management
- Adaptive parameter optimization
- Error recovery mechanisms
- Common utility functions

Examples:
--------
Configuration management:
    >>> from vitalDSP.utils.config_utilities import get_config
    >>> config = get_config('filtering')
"""

from .dynamic_config import (
    DynamicConfig,
    Environment,
    SystemResources,
    DataLoaderConfig,
    QualityScreenerConfig,
    ParallelPipelineConfig,
    get_config,
    set_config,
    reset_config,
)
from .common import (
    find_peaks,
    filtfilt,
    argrelextrema,
    deprecated,
    pearsonr,
    coherence,
    grangercausalitytests,
)
from .error_recovery import (
    ErrorRecovery,
    robust_signal_processing,
    safe_respiratory_rate,
    safe_filtering,
    safe_feature_extraction,
)
from .adaptive_parameters import (
    SignalCharacteristics,
    AdaptiveParameterAdjuster,
    analyze_signal_characteristics,
    get_optimal_parameters,
)

"""
Configuration & Utilities Module
This module contains configuration management and common utilities.
Components:
- dynamic_config: Dynamic configuration system
- common: Common utility functions
- error_recovery: Error recovery and fallback mechanisms
- adaptive_parameters: Adaptive parameter adjustment utilities
"""
__all__ = [
    # Dynamic Config
    "DynamicConfig",
    "Environment",
    "SystemResources",
    "DataLoaderConfig",
    "QualityScreenerConfig",
    "ParallelPipelineConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Common
    "find_peaks",
    "filtfilt",
    "argrelextrema",
    "deprecated",
    "pearsonr",
    "coherence",
    "grangercausalitytests",
    # Error Recovery
    "ErrorRecovery",
    "robust_signal_processing",
    "safe_respiratory_rate",
    "safe_filtering",
    "safe_feature_extraction",
    # Adaptive Parameters
    "SignalCharacteristics",
    "AdaptiveParameterAdjuster",
    "analyze_signal_characteristics",
    "get_optimal_parameters",
]
