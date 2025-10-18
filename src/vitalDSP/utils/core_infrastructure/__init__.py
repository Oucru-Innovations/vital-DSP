"""
Core Infrastructure Module

This module provides core infrastructure components for large-scale physiological signal processing.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Processing pipelines
- Memory management
- Data loaders
- Quality screening
- Error recovery
- Parallel processing

Examples:
--------
Core pipeline:
    >>> from vitalDSP.utils.core_infrastructure import ProcessingPipeline
    >>> pipeline = ProcessingPipeline()
"""

# Import main classes for easy access
from .data_loaders import (
    ChunkedDataLoader,
    MemoryMappedLoader,
    ProgressInfo,
    CancellationToken,
    select_optimal_loader,
)

from .optimized_data_loaders import (
    OptimizedChunkedDataLoader,
    OptimizedMemoryMappedLoader,
    ProgressInfo,
    CancellationToken,
    select_optimal_loader as select_optimal_loader_optimized,
)

from .parallel_pipeline import (
    ParallelPipeline,
    PipelineConfig,
    WorkerPoolManager,
    ResultAggregator,
    ProcessingTask,
    ProcessingResult,
    ProcessingStrategy,
)

from .optimized_parallel_pipeline import (
    OptimizedParallelPipeline,
    OptimizedPipelineConfig,
    OptimizedWorkerPoolManager,
    OptimizedResultAggregator,
    ProcessingTask,
    ProcessingResult,
    ProcessingStrategy,
)

from .quality_screener import (
    QualityScreener,
    QualityLevel,
    QualityMetrics,
    ScreeningResult,
)

from .optimized_quality_screener import (
    OptimizedQualityScreener,
    QualityLevel,
    QualityMetrics,
    ScreeningResult,
)

# Phase 2 Components
from .processing_pipeline import (
    StandardProcessingPipeline,
    ProcessingStage,
    ProcessingCache,
    CheckpointManager,
    ProcessingResult,
    ProcessingCheckpoint,
)

from .memory_manager import (
    MemoryManager,
    DataTypeOptimizer,
    MemoryProfiler,
    MemoryStrategy,
    MemoryInfo,
    ProcessingMemoryProfile,
)

from .error_recovery import (
    ErrorHandler,
    ErrorRecoveryManager,
    RobustProcessingPipeline,
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    RecoveryResult,
    error_handler,
)

# Dynamic Configuration Manager
from ..config_utilities.dynamic_config import DynamicConfigManager

__all__ = [
    # Data Loaders
    "ChunkedDataLoader",
    "MemoryMappedLoader",
    "ProgressInfo",
    "CancellationToken",
    "select_optimal_loader",
    # Optimized Data Loaders
    "OptimizedChunkedDataLoader",
    "OptimizedMemoryMappedLoader",
    "ProgressInfo",
    "CancellationToken",
    "select_optimal_loader_optimized",
    # Parallel Pipeline
    "ParallelPipeline",
    "PipelineConfig",
    "WorkerPoolManager",
    "ResultAggregator",
    "ProcessingTask",
    "ProcessingResult",
    "ProcessingStrategy",
    # Optimized Parallel Pipeline
    "OptimizedParallelPipeline",
    "OptimizedPipelineConfig",
    "OptimizedWorkerPoolManager",
    "OptimizedResultAggregator",
    "ProcessingTask",
    "ProcessingResult",
    "ProcessingStrategy",
    # Quality Screener
    "QualityScreener",
    "QualityLevel",
    "QualityMetrics",
    "ScreeningResult",
    # Optimized Quality Screener
    "OptimizedQualityScreener",
    "QualityLevel",
    "QualityMetrics",
    "ScreeningResult",
    # Phase 2: Processing Pipeline
    "StandardProcessingPipeline",
    "ProcessingStage",
    "ProcessingCache",
    "CheckpointManager",
    "ProcessingResult",
    "ProcessingCheckpoint",
    # Phase 2: Memory Management
    "MemoryManager",
    "DataTypeOptimizer",
    "MemoryProfiler",
    "MemoryStrategy",
    "MemoryInfo",
    "ProcessingMemoryProfile",
    # Phase 2: Error Recovery
    "ErrorHandler",
    "ErrorRecoveryManager",
    "RobustProcessingPipeline",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorInfo",
    "RecoveryResult",
    "error_handler",
    # Dynamic Configuration
    "DynamicConfigManager",
]