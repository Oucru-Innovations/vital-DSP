"""
Configuration & Utilities Module

This module contains configuration management and common utilities.

Components:
- dynamic_config: Dynamic configuration system
- common: Common utility functions
- error_recovery: Error recovery and fallback mechanisms
- adaptive_parameters: Adaptive parameter adjustment utilities
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
