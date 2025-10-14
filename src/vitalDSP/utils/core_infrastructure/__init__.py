"""
Core Infrastructure Module

This module contains the core infrastructure components for large-scale
physiological signal processing, including data loaders, parallel processing,
quality screening, processing pipeline, memory management, and error recovery.

Phase 1 Components:
- data_loaders: Advanced data loading with chunking and memory mapping
- optimized_data_loaders: Optimized versions with dynamic configuration
- parallel_pipeline: Parallel processing pipeline
- optimized_parallel_pipeline: Optimized parallel processing
- quality_screener: 3-stage quality screening system
- optimized_quality_screener: Optimized quality screening

Phase 2 Components:
- processing_pipeline: Standard 8-stage processing pipeline with checkpointing
- memory_manager: Intelligent memory management and data type optimization
- error_recovery: Robust error handling and recovery system
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
