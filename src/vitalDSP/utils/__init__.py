"""
Utilities Module for Physiological Signal Processing

This module provides comprehensive utility functions for physiological signal processing including configuration, data processing, and core infrastructure.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Configuration management
- Data processing utilities
- Core infrastructure
- Signal processing utilities
- Quality and performance monitoring
- Warning configuration

Examples:
--------
Basic utilities:
    >>> from vitalDSP.utils import configure_warnings
    >>> configure_warnings()
"""

# Import from submodules for backward compatibility
from .signal_processing import (
    PeakDetection,
    StandardScaler,
    Wavelet,
    ConvolutionKernels,
    LossFunctions,
    AttentionWeights,
    z_score_normalization,
    min_max_normalization,
    linear_interpolation,
    spline_interpolation,
    mean_imputation,
)

from .data_processing import (
    DataLoader,
    StreamDataLoader,
    DataFormat,
    SignalType,
    load_signal,
    load_multi_channel,
    load_oucru_csv,
    generate_sinusoidal,
    generate_ecg_signal,
    generate_resp_signal,
    generate_synthetic_ppg,
)

from .config_utilities import (
    find_peaks,
    filtfilt,
    argrelextrema,
    deprecated,
    DynamicConfig,
    get_config,
)

from .core_infrastructure import (
    ChunkedDataLoader,
    MemoryMappedLoader,
    ProgressInfo,
    CancellationToken,
    select_optimal_loader,
    OptimizedChunkedDataLoader,
    OptimizedMemoryMappedLoader,
    ParallelPipeline,
    OptimizedParallelPipeline,
    QualityScreener,
    OptimizedQualityScreener,
)

from .quality_performance import PerformanceMonitor

__all__ = [
    # Signal Processing
    "PeakDetection",
    "linear_interpolation",
    "spline_interpolation",
    "mean_imputation",
    "z_score_normalization",
    "min_max_normalization",
    "StandardScaler",
    "Wavelet",
    "ConvolutionKernels",
    "LossFunctions",
    "AttentionWeights",
    # Data Processing
    "DataLoader",
    "StreamDataLoader",
    "DataFormat",
    "SignalType",
    "load_signal",
    "load_multi_channel",
    "load_oucru_csv",
    "generate_sinusoidal",
    "generate_ecg_signal",
    "generate_resp_signal",
    "generate_synthetic_ppg",
    # Config Utilities
    "find_peaks",
    "filtfilt",
    "argrelextrema",
    "deprecated",
    "DynamicConfig",
    "get_config",
    # Core Infrastructure
    "ChunkedDataLoader",
    "MemoryMappedLoader",
    "ProgressInfo",
    "CancellationToken",
    "select_optimal_loader",
    "OptimizedChunkedDataLoader",
    "OptimizedMemoryMappedLoader",
    "ParallelPipeline",
    "OptimizedParallelPipeline",
    "QualityScreener",
    "OptimizedQualityScreener",
    # Quality Performance
    "PerformanceMonitor",
]