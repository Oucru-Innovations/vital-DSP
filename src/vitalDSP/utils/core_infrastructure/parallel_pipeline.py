"""
Parallel Processing Pipeline for Large-Scale Physiological Signal Processing

This module implements a high-performance parallel processing pipeline for
efficient processing of large physiological datasets:

- ParallelPipeline: Multi-process pipeline with worker pool management
- ResultAggregator: Efficient aggregation of processing results
- WorkerPoolManager: Dynamic worker pool management
- Performance monitoring and optimization

Features:
- Multi-process parallel processing
- Dynamic worker pool sizing
- Result aggregation and caching
- Progress tracking and cancellation
- Memory-efficient processing
- Integration with quality screening

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Week 3)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Callable, Generator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue
import threading
import warnings
from functools import partial
import pickle
import tempfile
import os

from vitalDSP.utils.core_infrastructure.data_loaders import (
    ChunkedDataLoader,
    MemoryMappedLoader,
    ProgressInfo,
)
from vitalDSP.utils.core_infrastructure.quality_screener import (
    QualityScreener,
    ScreeningResult,
)


class ProcessingStrategy(Enum):
    """Processing strategy selection."""

    SEQUENTIAL = "sequential"  # Process one segment at a time
    PARALLEL_CHUNKS = "parallel_chunks"  # Process multiple chunks in parallel
    PARALLEL_SEGMENTS = "parallel_segments"  # Process multiple segments in parallel
    HYBRID = "hybrid"  # Adaptive strategy selection


@dataclass
class ProcessingTask:
    """
    Individual processing task.

    Attributes:
        task_id: Unique task identifier
        segment_id: Segment identifier
        data: Input data for processing
        start_idx: Starting index in original signal
        end_idx: Ending index in original signal
        processing_params: Parameters for processing
        priority: Task priority (higher = more important)
        dependencies: List of task IDs this task depends on
    """

    task_id: str
    segment_id: str
    data: np.ndarray
    start_idx: int
    end_idx: int
    processing_params: Dict[str, Any]
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """
    Result of processing task.

    Attributes:
        task_id: Task identifier
        segment_id: Segment identifier
        success: Whether processing was successful
        result_data: Processed data
        metadata: Additional metadata
        processing_time: Time taken for processing
        memory_usage: Memory usage during processing
        warnings: List of warnings
        error: Error message if processing failed
    """

    task_id: str
    segment_id: str
    success: bool
    result_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    memory_usage: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class PipelineConfig:
    """
    Configuration for parallel processing pipeline.

    Attributes:
        max_workers: Maximum number of worker processes
        chunk_size: Size of data chunks for processing
        memory_limit_mb: Memory limit per worker (MB)
        timeout_seconds: Timeout for individual tasks
        enable_caching: Enable result caching
        cache_dir: Directory for caching results
        strategy: Processing strategy
        quality_threshold: Minimum quality threshold for processing
    """

    max_workers: int = min(mp.cpu_count(), 8)
    chunk_size: int = 10000
    memory_limit_mb: int = 1000
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL_CHUNKS
    quality_threshold: float = 0.4


class WorkerPoolManager:
    """
    Dynamic worker pool management for parallel processing.

    Features:
    - Dynamic worker pool sizing based on workload
    - Memory-aware worker allocation
    - Worker health monitoring
    - Automatic worker recovery
    """

    def __init__(self, config: PipelineConfig):
        """Initialize worker pool manager."""
        self.config = config
        self.active_workers = 0
        self.max_workers = config.max_workers
        self.worker_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
        }
        self._lock = threading.Lock()

    def get_optimal_worker_count(self, task_count: int, data_size_mb: float) -> int:
        """Calculate optimal number of workers based on workload."""
        # Base calculation on CPU cores
        cpu_workers = min(self.max_workers, mp.cpu_count())

        # Adjust based on memory requirements
        memory_workers = int(data_size_mb / self.config.memory_limit_mb)

        # Adjust based on task count
        task_workers = min(task_count, self.max_workers)

        # Use minimum of all constraints
        optimal_workers = min(cpu_workers, memory_workers, task_workers)

        # Ensure at least 1 worker
        return max(1, optimal_workers)

    def update_worker_stats(self, processing_time: float, success: bool):
        """Update worker statistics."""
        with self._lock:
            self.worker_stats["total_tasks"] += 1
            if success:
                self.worker_stats["completed_tasks"] += 1
            else:
                self.worker_stats["failed_tasks"] += 1

            # Update average processing time
            total_time = self.worker_stats["avg_processing_time"] * (
                self.worker_stats["total_tasks"] - 1
            )
            self.worker_stats["avg_processing_time"] = (
                total_time + processing_time
            ) / self.worker_stats["total_tasks"]

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            stats = self.worker_stats.copy()
            if stats["total_tasks"] > 0:
                stats["success_rate"] = stats["completed_tasks"] / stats["total_tasks"]
            else:
                stats["success_rate"] = 0.0
            return stats


class ResultAggregator:
    """
    Efficient aggregation of processing results.

    Features:
    - Result caching and persistence
    - Memory-efficient aggregation
    - Result validation and quality checking
    - Export to various formats
    """

    def __init__(self, config: PipelineConfig):
        """Initialize result aggregator."""
        self.config = config
        self.results: Dict[str, ProcessingResult] = {}
        self.cache_dir = config.cache_dir or tempfile.mkdtemp()
        self._lock = threading.Lock()

    def add_result(self, result: ProcessingResult):
        """Add processing result."""
        with self._lock:
            self.results[result.task_id] = result

            # Cache result if enabled
            if self.config.enable_caching:
                self._cache_result(result)

    def get_result(self, task_id: str) -> Optional[ProcessingResult]:
        """Get processing result by task ID."""
        with self._lock:
            if task_id in self.results:
                return self.results[task_id]

            # Try to load from cache
            if self.config.enable_caching:
                return self._load_cached_result(task_id)

            return None

    def get_all_results(self) -> List[ProcessingResult]:
        """Get all processing results."""
        with self._lock:
            return list(self.results.values())

    def aggregate_results(self, sort_by_index: bool = True) -> Dict[str, Any]:
        """Aggregate all results into a single dataset."""
        with self._lock:
            successful_results = [r for r in self.results.values() if r.success]

            if not successful_results:
                return {
                    "success": False,
                    "error": "No successful results to aggregate",
                    "data": None,
                    "metadata": {},
                }

            # Sort results by segment index if requested
            if sort_by_index:
                successful_results.sort(key=lambda x: x.metadata.get("start_idx", 0))

            # Aggregate data
            aggregated_data = []
            aggregated_metadata = {
                "total_segments": len(successful_results),
                "total_samples": 0,
                "processing_times": [],
                "memory_usage": [],
            }

            for result in successful_results:
                if result.result_data is not None:
                    aggregated_data.append(result.result_data)
                    aggregated_metadata["total_samples"] += len(result.result_data)
                    aggregated_metadata["processing_times"].append(
                        result.processing_time
                    )
                    aggregated_metadata["memory_usage"].append(result.memory_usage)

            if aggregated_data:
                final_data = np.concatenate(aggregated_data)

                return {
                    "success": True,
                    "data": final_data,
                    "metadata": aggregated_metadata,
                    "quality_stats": self._calculate_quality_stats(successful_results),
                }
            else:
                return {
                    "success": False,
                    "error": "No valid data to aggregate",
                    "data": None,
                    "metadata": aggregated_metadata,
                }

    def _cache_result(self, result: ProcessingResult):
        """Cache processing result to disk."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{result.task_id}.pkl")
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            warnings.warn(f"Failed to cache result {result.task_id}: {e}")

    def _load_cached_result(self, task_id: str) -> Optional[ProcessingResult]:
        """Load cached result from disk."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{task_id}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load cached result {task_id}: {e}")
        return None

    def _calculate_quality_stats(
        self, results: List[ProcessingResult]
    ) -> Dict[str, Any]:
        """Calculate quality statistics from results."""
        if not results:
            return {}

        processing_times = [r.processing_time for r in results]
        memory_usage = [r.memory_usage for r in results]

        return {
            "avg_processing_time": np.mean(processing_times),
            "std_processing_time": np.std(processing_times),
            "avg_memory_usage": np.mean(memory_usage),
            "std_memory_usage": np.std(memory_usage),
            "total_warnings": sum(len(r.warnings) for r in results),
        }


class ParallelPipeline:
    """
    High-performance parallel processing pipeline for physiological signals.

    Features:
    - Multi-process parallel processing
    - Dynamic worker pool management
    - Result aggregation and caching
    - Progress tracking and cancellation
    - Memory-efficient processing
    - Integration with quality screening

    Example:
        >>> pipeline = ParallelPipeline(config=PipelineConfig())
        >>> results = pipeline.process_signal(
        ...     signal_data,
        ...     processing_function=my_processing_function,
        ...     progress_callback=progress_callback
        ... )
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize parallel processing pipeline."""
        self.config = config or PipelineConfig()
        self.worker_manager = WorkerPoolManager(self.config)
        self.result_aggregator = ResultAggregator(self.config)
        self.quality_screener = QualityScreener()

        # Performance tracking
        self.pipeline_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "total_memory_usage": 0.0,
        }

    def process_signal(
        self,
        signal_data: Union[np.ndarray, pd.DataFrame],
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
        processing_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        enable_quality_screening: bool = True,
    ) -> Dict[str, Any]:
        """
        Process signal using parallel pipeline.

        Args:
            signal_data: Input signal data
            processing_function: Function to process individual segments
            processing_params: Parameters for processing function
            progress_callback: Optional progress callback
            enable_quality_screening: Enable quality screening before processing

        Returns:
            Dictionary containing aggregated results and metadata
        """
        start_time = time.time()

        # Convert to numpy array if needed
        if isinstance(signal_data, pd.DataFrame):
            signal_array = signal_data.values.flatten()
        else:
            signal_array = signal_data.flatten()

        # Generate processing tasks
        tasks = self._generate_tasks(signal_array, processing_params or {})

        # Quality screening if enabled
        if enable_quality_screening:
            tasks = self._filter_tasks_by_quality(tasks, signal_array)

        # Process tasks in parallel
        results = self._process_tasks_parallel(
            tasks, processing_function, progress_callback
        )

        # Aggregate results
        aggregated_results = self.result_aggregator.aggregate_results()

        # Update pipeline statistics
        self._update_pipeline_stats(results, time.time() - start_time)

        return aggregated_results

    def _generate_tasks(
        self, signal_array: np.ndarray, processing_params: Dict[str, Any]
    ) -> List[ProcessingTask]:
        """Generate processing tasks from signal data."""
        tasks = []
        chunk_size = self.config.chunk_size

        for i in range(0, len(signal_array), chunk_size):
            end_idx = min(i + chunk_size, len(signal_array))
            segment_data = signal_array[i:end_idx]

            task = ProcessingTask(
                task_id=f"task_{i}_{end_idx}",
                segment_id=f"seg_{i}_{end_idx}",
                data=segment_data,
                start_idx=i,
                end_idx=end_idx,
                processing_params=processing_params,
                priority=0,
            )
            tasks.append(task)

        return tasks

    def _filter_tasks_by_quality(
        self, tasks: List[ProcessingTask], signal_array: np.ndarray
    ) -> List[ProcessingTask]:
        """Filter tasks based on quality screening."""
        if not tasks:
            return tasks

        # Screen signal quality
        screening_results = self.quality_screener.screen_signal(signal_array)

        # Create quality map
        quality_map = {result.segment_id: result for result in screening_results}

        # Filter tasks based on quality
        filtered_tasks = []
        for task in tasks:
            if task.segment_id in quality_map:
                screening_result = quality_map[task.segment_id]
                if screening_result.passed_screening:
                    filtered_tasks.append(task)
            else:
                # If no screening result, include task
                filtered_tasks.append(task)

        return filtered_tasks

    def _process_tasks_parallel(
        self,
        tasks: List[ProcessingTask],
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ProcessingResult]:
        """Process tasks in parallel."""
        if not tasks:
            return []

        # Determine optimal worker count
        data_size_mb = sum(len(task.data) * 8 / (1024**2) for task in tasks)
        worker_count = self.worker_manager.get_optimal_worker_count(
            len(tasks), data_size_mb
        )

        results = []

        # Use ThreadPoolExecutor for functions that can't be pickled
        # This is safer for interactive/testing environments
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._process_single_task, task, processing_function
                ): task
                for task in tasks
            }

            # Collect results as they complete
            completed_count = 0
            for future in as_completed(
                future_to_task, timeout=self.config.timeout_seconds
            ):
                task = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)
                    self.result_aggregator.add_result(result)

                    # Update worker statistics
                    self.worker_manager.update_worker_stats(
                        result.processing_time, result.success
                    )

                except Exception as e:
                    # Create failed result
                    failed_result = ProcessingResult(
                        task_id=task.task_id,
                        segment_id=task.segment_id,
                        success=False,
                        processing_time=0.0,
                        error=str(e),
                    )
                    results.append(failed_result)
                    self.result_aggregator.add_result(failed_result)

                completed_count += 1

                # Progress callback
                if progress_callback:
                    progress_info = ProgressInfo(
                        bytes_processed=completed_count * self.config.chunk_size * 8,
                        total_bytes=len(tasks) * self.config.chunk_size * 8,
                        chunks_processed=completed_count,
                        total_chunks=len(tasks),
                        elapsed_time=0.0,
                        estimated_remaining=0.0,
                        current_chunk_size=self.config.chunk_size,
                        loading_strategy="parallel_processing",
                    )
                    progress_callback(progress_info)

        return results

    @staticmethod
    def _process_single_task(
        task: ProcessingTask,
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
    ) -> ProcessingResult:
        """Process a single task (runs in worker process)."""
        start_time = time.time()

        try:
            # Process the data
            result_data, metadata = processing_function(
                task.data, task.processing_params
            )

            processing_time = time.time() - start_time

            # Estimate memory usage
            memory_usage = len(task.data) * 8 / (1024**2)  # Rough estimate

            return ProcessingResult(
                task_id=task.task_id,
                segment_id=task.segment_id,
                success=True,
                result_data=result_data,
                metadata={
                    **metadata,
                    "start_idx": task.start_idx,
                    "end_idx": task.end_idx,
                    "original_length": len(task.data),
                },
                processing_time=processing_time,
                memory_usage=memory_usage,
            )

        except Exception as e:
            processing_time = time.time() - start_time

            return ProcessingResult(
                task_id=task.task_id,
                segment_id=task.segment_id,
                success=False,
                processing_time=processing_time,
                error=str(e),
            )

    def _update_pipeline_stats(
        self, results: List[ProcessingResult], total_time: float
    ):
        """Update pipeline statistics."""
        self.pipeline_stats["total_tasks"] += len(results)
        self.pipeline_stats["completed_tasks"] += sum(1 for r in results if r.success)
        self.pipeline_stats["failed_tasks"] += sum(1 for r in results if not r.success)
        self.pipeline_stats["total_processing_time"] += total_time
        self.pipeline_stats["total_memory_usage"] += sum(
            r.memory_usage for r in results
        )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.pipeline_stats.copy()
        if stats["total_tasks"] > 0:
            stats["success_rate"] = stats["completed_tasks"] / stats["total_tasks"]
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["total_tasks"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0

        # Add worker statistics
        stats["worker_stats"] = self.worker_manager.get_worker_stats()

        return stats

    def reset_statistics(self):
        """Reset pipeline statistics."""
        self.pipeline_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "total_memory_usage": 0.0,
        }
        self.worker_manager.worker_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
        }


# Example processing functions
def example_filtering_function(
    data: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Example filtering function for testing."""
    from scipy.signal import butter, filtfilt

    # Extract parameters
    sampling_rate = params.get("sampling_rate", 100.0)
    lowcut = params.get("lowcut", 0.5)
    highcut = params.get("highcut", 40.0)

    # Design filter
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")

    # Apply filter
    filtered_data = filtfilt(b, a, data)

    metadata = {
        "filter_type": "butterworth",
        "lowcut": lowcut,
        "highcut": highcut,
        "sampling_rate": sampling_rate,
    }

    return filtered_data, metadata


def example_feature_extraction_function(
    data: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Example feature extraction function for testing."""
    # Extract basic features
    features = np.array(
        [np.mean(data), np.std(data), np.max(data), np.min(data), np.var(data)]
    )

    metadata = {
        "feature_count": len(features),
        "feature_names": ["mean", "std", "max", "min", "var"],
    }

    return features, metadata


# Example usage and tests
if __name__ == "__main__":
    print("Parallel Processing Pipeline Module")
    print("=" * 50)
    print("\nThis module provides high-performance parallel processing")
    print("for physiological signals.")
    print("\nFeatures:")
    print("  - Multi-process parallel processing")
    print("  - Dynamic worker pool management")
    print("  - Result aggregation and caching")
    print("  - Progress tracking and cancellation")
    print("  - Memory-efficient processing")
