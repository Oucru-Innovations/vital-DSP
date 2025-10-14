"""
vitalDSP Utilities Module

Provides utility functions for signal processing organized into logical modules:

- core_infrastructure: Core infrastructure components for large-scale processing
- data_processing: Data loading, validation, and synthesis utilities
- signal_processing: Signal processing, transforms, and features
- quality_performance: Quality assessment and performance monitoring
- config_utilities: Configuration management and common utilities
"""

# Import from submodules for backward compatibility
from .signal_processing import (
    PeakDetection,
    # linear_interpolation,
    # spline_interpolation,
    mean_imputation,
    # z_score_normalization,
    # min_max_normalization,
    StandardScaler,
    Wavelet,
    ConvolutionKernels,
    LossFunctions,
    AttentionWeights,
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

# Legacy imports for backward compatibility
from .signal_processing.normalization import (
    z_score_normalization,
    min_max_normalization,
)
from .signal_processing.interpolations import (
    linear_interpolation,
    spline_interpolation,
    mean_imputation,
)
from .signal_processing.scaler import StandardScaler

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
