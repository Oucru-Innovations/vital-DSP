"""
Optimized Parallel Processing Pipeline for Large-Scale Physiological Signal Processing

This module implements an optimized high-performance parallel processing pipeline
with dynamic configuration and advanced performance optimization.

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Optimized)
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
import gc
import psutil
from collections import deque

from vitalDSP.utils.config_utilities.dynamic_config import (
    get_config,
    DynamicConfig,
    Environment,
)
from vitalDSP.utils.core_infrastructure.optimized_data_loaders import ProgressInfo
from vitalDSP.utils.core_infrastructure.optimized_quality_screener import (
    OptimizedQualityScreener,
    ScreeningResult,
)


class ProcessingStrategy(Enum):
    """Processing strategy selection."""

    SEQUENTIAL = "sequential"
    PARALLEL_CHUNKS = "parallel_chunks"
    PARALLEL_SEGMENTS = "parallel_segments"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingTask:
    """
    Individual processing task with enhanced metadata.
    """

    task_id: str
    segment_id: str
    data: np.ndarray
    start_idx: int
    end_idx: int
    processing_params: Dict[str, Any]
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: float = 1.0
    memory_estimate_mb: float = 0.0


@dataclass
class ProcessingResult:
    """
    Result of processing task with enhanced metadata.
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
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizedPipelineConfig:
    """
    Enhanced configuration for optimized parallel processing pipeline.
    """

    # Worker management
    max_workers_factor: float = 0.75
    max_workers_cap: int = 16
    min_workers: int = 1

    # Memory management
    memory_limit_factor: float = 0.1
    min_memory_limit_mb: int = 100
    max_memory_limit_mb: int = 2000

    # Task management
    default_chunk_size: int = 10000
    min_chunk_size: int = 1000
    max_chunk_size: int = 100000

    # Timeout settings
    default_timeout_seconds: int = 300
    min_timeout_seconds: int = 30
    max_timeout_seconds: int = 3600

    # Caching
    enable_caching_by_default: bool = True
    cache_compression_level: int = 6
    cache_max_size_mb: int = 1000

    # Quality thresholds
    default_quality_threshold: float = 0.4
    min_quality_threshold: float = 0.0
    max_quality_threshold: float = 1.0

    # Performance optimization
    enable_memory_monitoring: bool = True
    enable_cpu_monitoring: bool = True
    memory_cleanup_interval: int = 10  # Cleanup every N tasks
    adaptive_worker_scaling: bool = True
    load_balancing: bool = True

    # Performance monitoring
    performance_monitoring: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_monitoring": True,
            "metrics_history_limit": 1000,
            "performance_report_interval": 60.0,
            "memory_usage_threshold_mb": 1000.0,
            "cpu_usage_threshold_percent": 80.0,
            "execution_time_threshold_seconds": 30.0,
        }
    )


class OptimizedWorkerPoolManager:
    """
    Enhanced worker pool management with adaptive scaling and monitoring.
    """

    def __init__(self, config: OptimizedPipelineConfig, system_config: DynamicConfig):
        """Initialize optimized worker pool manager."""
        self.config = config
        self.system_config = system_config
        self.active_workers = 0
        self.worker_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
            "avg_memory_usage": 0.0,
            "worker_efficiency": 0.0,
        }
        self._lock = threading.Lock()
        self._performance_history = deque(maxlen=100)
        self._memory_history = deque(maxlen=100)

    def get_optimal_worker_count(
        self, task_count: int, data_size_mb: float, complexity_estimate: float = 1.0
    ) -> int:
        """Calculate optimal number of workers with adaptive scaling."""
        # Base calculation on CPU cores
        cpu_workers = int(
            self.system_config.system_resources.cpu_count
            * self.config.max_workers_factor
        )
        cpu_workers = min(cpu_workers, self.config.max_workers_cap)

        # Adjust based on memory requirements
        memory_per_worker_mb = self.system_config.system_resources.memory_per_worker_mb
        memory_workers = (
            int(data_size_mb / memory_per_worker_mb)
            if memory_per_worker_mb > 0
            else cpu_workers
        )

        # Adjust based on task count
        task_workers = min(task_count, cpu_workers)

        # Adjust based on complexity
        complexity_factor = min(complexity_estimate, 2.0)  # Cap at 2x
        complexity_workers = int(cpu_workers / complexity_factor)

        # Use minimum of all constraints
        optimal_workers = min(
            cpu_workers, memory_workers, task_workers, complexity_workers
        )

        # Ensure minimum workers
        optimal_workers = max(optimal_workers, self.config.min_workers)

        # Adaptive scaling based on historical performance
        if self.config.adaptive_worker_scaling and len(self._performance_history) > 5:
            avg_efficiency = np.mean(list(self._performance_history)[-5:])
            if avg_efficiency < 0.7:  # Low efficiency, reduce workers
                optimal_workers = max(optimal_workers // 2, self.config.min_workers)
            elif avg_efficiency > 0.9:  # High efficiency, can increase workers
                optimal_workers = min(optimal_workers * 2, self.config.max_workers_cap)

        return optimal_workers

    def update_worker_stats(
        self,
        processing_time: float,
        success: bool,
        memory_usage: float = 0.0,
        complexity: float = 1.0,
    ):
        """Update worker statistics with enhanced metrics."""
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

            # Update average memory usage
            if memory_usage > 0:
                total_memory = self.worker_stats["avg_memory_usage"] * (
                    self.worker_stats["total_tasks"] - 1
                )
                self.worker_stats["avg_memory_usage"] = (
                    total_memory + memory_usage
                ) / self.worker_stats["total_tasks"]

            # Calculate worker efficiency
            expected_time = complexity * 1.0  # Base time for complexity 1.0
            efficiency = expected_time / processing_time if processing_time > 0 else 0.0
            self._performance_history.append(efficiency)

            if memory_usage > 0:
                self._memory_history.append(memory_usage)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get enhanced worker statistics."""
        with self._lock:
            stats = self.worker_stats.copy()
            if stats["total_tasks"] > 0:
                stats["success_rate"] = stats["completed_tasks"] / stats["total_tasks"]
                stats["worker_efficiency"] = (
                    np.mean(list(self._performance_history))
                    if self._performance_history
                    else 0.0
                )
                stats["avg_memory_usage"] = (
                    np.mean(list(self._memory_history)) if self._memory_history else 0.0
                )
            else:
                stats["success_rate"] = 0.0
                stats["worker_efficiency"] = 0.0
            return stats


class OptimizedResultAggregator:
    """
    Enhanced result aggregator with advanced caching and optimization.
    """

    def __init__(self, config: OptimizedPipelineConfig):
        """Initialize optimized result aggregator."""
        self.config = config
        self.results: Dict[str, ProcessingResult] = {}
        self.cache_dir = tempfile.mkdtemp()
        self._lock = threading.Lock()
        self._cache_stats = {"hits": 0, "misses": 0, "size_mb": 0.0}

    def add_result(self, result: ProcessingResult):
        """Add processing result with enhanced caching."""
        with self._lock:
            self.results[result.task_id] = result

            # Enhanced caching with compression
            if self.config.enable_caching_by_default:
                self._cache_result_optimized(result)

    def get_result(self, task_id: str) -> Optional[ProcessingResult]:
        """Get processing result with cache optimization."""
        with self._lock:
            if task_id in self.results:
                return self.results[task_id]

            # Try to load from cache
            if self.config.enable_caching_by_default:
                cached_result = self._load_cached_result_optimized(task_id)
                if cached_result:
                    self._cache_stats["hits"] += 1
                    return cached_result
                else:
                    self._cache_stats["misses"] += 1

            return None

    def get_all_results(self) -> List[ProcessingResult]:
        """Get all processing results."""
        with self._lock:
            return list(self.results.values())

    def aggregate_results(self, sort_by_index: bool = True) -> Dict[str, Any]:
        """Aggregate all results with enhanced optimization."""
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

            # Optimized aggregation
            aggregated_data = []
            aggregated_metadata = {
                "total_segments": len(successful_results),
                "total_samples": 0,
                "processing_times": [],
                "memory_usage": [],
                "performance_metrics": {},
            }

            # Process results in batches to manage memory
            batch_size = 100
            for i in range(0, len(successful_results), batch_size):
                batch = successful_results[i : i + batch_size]

                for result in batch:
                    if result.result_data is not None:
                        aggregated_data.append(result.result_data)
                        aggregated_metadata["total_samples"] += len(result.result_data)
                        aggregated_metadata["processing_times"].append(
                            result.processing_time
                        )
                        aggregated_metadata["memory_usage"].append(result.memory_usage)

                # Memory cleanup
                if i % (batch_size * 5) == 0:
                    gc.collect()

            if aggregated_data:
                final_data = np.concatenate(aggregated_data)

                # Calculate performance metrics
                aggregated_metadata["performance_metrics"] = (
                    self._calculate_performance_metrics(successful_results)
                )

                return {
                    "success": True,
                    "data": final_data,
                    "metadata": aggregated_metadata,
                    "quality_stats": self._calculate_quality_stats(successful_results),
                    "cache_stats": self._cache_stats.copy(),
                }
            else:
                return {
                    "success": False,
                    "error": "No valid data to aggregate",
                    "data": None,
                    "metadata": aggregated_metadata,
                }

    def _cache_result_optimized(self, result: ProcessingResult):
        """Optimized result caching with compression."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{result.task_id}.pkl")

            # Compress large results
            if result.result_data is not None and len(result.result_data) > 10000:
                import gzip

                with gzip.open(
                    cache_file, "wb", compresslevel=self.config.cache_compression_level
                ) as f:
                    pickle.dump(result, f)
            else:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)

            # Update cache size
            file_size = os.path.getsize(cache_file)
            self._cache_stats["size_mb"] += file_size / (1024 * 1024)

        except Exception as e:
            warnings.warn(f"Failed to cache result {result.task_id}: {e}")

    def _load_cached_result_optimized(self, task_id: str) -> Optional[ProcessingResult]:
        """Optimized cached result loading."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{task_id}.pkl")
            if os.path.exists(cache_file):
                # Try compressed first, then uncompressed
                try:
                    import gzip

                    with gzip.open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load cached result {task_id}: {e}")
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load cached result {task_id}: {e}")
        return None

    def _calculate_performance_metrics(
        self, results: List[ProcessingResult]
    ) -> Dict[str, Any]:
        """Calculate enhanced performance metrics."""
        if not results:
            return {}

        processing_times = [r.processing_time for r in results]
        memory_usage = [r.memory_usage for r in results]

        # Calculate additional metrics
        total_processing_time = sum(processing_times)
        avg_processing_time = np.mean(processing_times)
        std_processing_time = np.std(processing_times)

        return {
            "total_processing_time": total_processing_time,
            "avg_processing_time": avg_processing_time,
            "std_processing_time": std_processing_time,
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "avg_memory_usage": np.mean(memory_usage),
            "std_memory_usage": np.std(memory_usage),
            "total_warnings": sum(len(r.warnings) for r in results),
            "throughput_samples_per_second": (
                sum(len(r.result_data) for r in results if r.result_data is not None)
                / total_processing_time
                if total_processing_time > 0
                else 0
            ),
        }

    def _calculate_quality_stats(
        self, results: List[ProcessingResult]
    ) -> Dict[str, Any]:
        """Calculate quality statistics from results."""
        if not results:
            return {}

        # Extract quality metrics if available
        quality_scores = []
        for result in results:
            if "quality_score" in result.metadata:
                quality_scores.append(result.metadata["quality_score"])

        if quality_scores:
            return {
                "avg_quality_score": np.mean(quality_scores),
                "std_quality_score": np.std(quality_scores),
                "min_quality_score": min(quality_scores),
                "max_quality_score": max(quality_scores),
            }

        return {}


class OptimizedParallelPipeline:
    """
    Optimized high-performance parallel processing pipeline.
    """

    def __init__(
        self,
        config: Optional[OptimizedPipelineConfig] = None,
        system_config: Optional[DynamicConfig] = None,
    ):
        """Initialize optimized parallel processing pipeline."""
        self.system_config = system_config or get_config()
        self.config = config or OptimizedPipelineConfig()
        self.worker_manager = OptimizedWorkerPoolManager(
            self.config, self.system_config
        )
        self.result_aggregator = OptimizedResultAggregator(self.config)
        self.quality_screener = OptimizedQualityScreener(config=self.system_config)

        # Performance tracking
        self.pipeline_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "total_memory_usage": 0.0,
            "avg_throughput": 0.0,
            "efficiency_score": 0.0,
        }

        # Memory monitoring
        self._memory_monitor = None
        if self.config.enable_memory_monitoring:
            self._start_memory_monitoring()

    def _start_memory_monitoring(self):
        """Start memory monitoring thread."""

        def monitor_memory():
            while True:
                try:
                    memory_usage = psutil.virtual_memory().percent
                    if (
                        memory_usage
                        > self.config.performance_monitoring[
                            "memory_usage_threshold_mb"
                        ]
                    ):
                        warnings.warn(f"High memory usage detected: {memory_usage}%")
                        gc.collect()  # Force garbage collection
                    time.sleep(5)  # Check every 5 seconds
                except Exception:
                    break

        self._memory_monitor = threading.Thread(target=monitor_memory, daemon=True)
        self._memory_monitor.start()

    def process_signal(
        self,
        signal_data: Union[np.ndarray, pd.DataFrame],
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
        processing_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        enable_quality_screening: bool = True,
        strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,
    ) -> Dict[str, Any]:
        """
        Process signal using optimized parallel pipeline.
        """
        start_time = time.time()

        # Convert to numpy array if needed
        if isinstance(signal_data, pd.DataFrame):
            signal_array = signal_data.values.flatten()
        else:
            signal_array = signal_data.flatten()

        # Generate processing tasks with complexity estimation
        tasks = self._generate_optimized_tasks(signal_array, processing_params or {})

        # Quality screening if enabled
        if enable_quality_screening:
            tasks = self._filter_tasks_by_quality_optimized(tasks, signal_array)

        # Select optimal strategy
        if strategy == ProcessingStrategy.ADAPTIVE:
            strategy = self._select_optimal_strategy(tasks, signal_array)

        # Process tasks using selected strategy
        results = self._process_tasks_optimized(
            tasks, processing_function, strategy, progress_callback
        )

        # Aggregate results
        aggregated_results = self.result_aggregator.aggregate_results()

        # Update pipeline statistics
        self._update_pipeline_stats(results, time.time() - start_time)

        return aggregated_results

    def _generate_optimized_tasks(
        self, signal_array: np.ndarray, processing_params: Dict[str, Any]
    ) -> List[ProcessingTask]:
        """Generate optimized processing tasks with complexity estimation."""
        tasks = []

        # Determine optimal chunk size
        chunk_size = self.system_config.get_optimal_chunk_size(
            file_size_mb=len(signal_array) * 8 / (1024**2)
        )

        for i in range(0, len(signal_array), chunk_size):
            end_idx = min(i + chunk_size, len(signal_array))
            segment_data = signal_array[i:end_idx]

            # Estimate complexity based on signal characteristics
            complexity = self._estimate_task_complexity(segment_data, processing_params)

            # Estimate memory usage
            memory_estimate = len(segment_data) * 8 / (1024**2)  # Rough estimate

            task = ProcessingTask(
                task_id=f"task_{i}_{end_idx}",
                segment_id=f"seg_{i}_{end_idx}",
                data=segment_data,
                start_idx=i,
                end_idx=end_idx,
                processing_params=processing_params,
                priority=0,
                estimated_complexity=complexity,
                memory_estimate_mb=memory_estimate,
            )
            tasks.append(task)

        return tasks

    def _estimate_task_complexity(
        self, signal: np.ndarray, params: Dict[str, Any]
    ) -> float:
        """Estimate task complexity based on signal characteristics."""
        # Base complexity
        complexity = 1.0

        # Adjust based on signal length
        complexity *= (len(signal) / 10000) ** 0.5  # Square root scaling

        # Adjust based on signal variance (more complex signals)
        signal_var = np.var(signal)
        complexity *= 1 + signal_var

        # Adjust based on processing parameters
        if "filter_order" in params:
            complexity *= 1 + params["filter_order"] * 0.1

        if "window_size" in params:
            complexity *= 1 + params["window_size"] / 1000

        return min(complexity, 5.0)  # Cap at 5x complexity

    def _filter_tasks_by_quality_optimized(
        self, tasks: List[ProcessingTask], signal_array: np.ndarray
    ) -> List[ProcessingTask]:
        """Filter tasks based on optimized quality screening."""
        if not tasks:
            return tasks

        # Use optimized quality screener
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

    def _select_optimal_strategy(
        self, tasks: List[ProcessingTask], signal_array: np.ndarray
    ) -> ProcessingStrategy:
        """Select optimal processing strategy based on workload characteristics."""
        task_count = len(tasks)
        data_size_mb = len(signal_array) * 8 / (1024**2)
        avg_complexity = np.mean([task.estimated_complexity for task in tasks])

        # Decision logic
        if task_count < 2:
            return ProcessingStrategy.SEQUENTIAL
        elif data_size_mb < 100:  # Small dataset
            return ProcessingStrategy.PARALLEL_CHUNKS
        elif avg_complexity > 2.0:  # High complexity
            return ProcessingStrategy.PARALLEL_SEGMENTS
        else:
            return ProcessingStrategy.PARALLEL_CHUNKS

    def _process_tasks_optimized(
        self,
        tasks: List[ProcessingTask],
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
        strategy: ProcessingStrategy,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ProcessingResult]:
        """Process tasks using optimized strategy."""
        if not tasks:
            return []

        # Calculate optimal worker count
        data_size_mb = sum(task.memory_estimate_mb for task in tasks)
        avg_complexity = np.mean([task.estimated_complexity for task in tasks])
        worker_count = self.worker_manager.get_optimal_worker_count(
            len(tasks), data_size_mb, avg_complexity
        )

        if strategy == ProcessingStrategy.SEQUENTIAL:
            return self._process_tasks_sequential(
                tasks, processing_function, progress_callback
            )
        else:
            return self._process_tasks_parallel(
                tasks, processing_function, worker_count, progress_callback
            )

    def _process_tasks_sequential(
        self,
        tasks: List[ProcessingTask],
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ProcessingResult]:
        """Process tasks sequentially with optimization."""
        results = []
        last_progress_update = time.time()

        for i, task in enumerate(tasks):
            result = self._process_single_task_optimized(task, processing_function)
            results.append(result)
            self.result_aggregator.add_result(result)

            # Update worker statistics
            self.worker_manager.update_worker_stats(
                result.processing_time,
                result.success,
                result.memory_usage,
                task.estimated_complexity,
            )

            # Progress callback with throttling
            current_time = time.time()
            if (
                progress_callback
                and current_time - last_progress_update
                >= self.system_config.data_loader.progress_update_interval
            ):

                progress_info = ProgressInfo(
                    bytes_processed=i * len(task.data) * 8,
                    total_bytes=len(tasks) * len(task.data) * 8,
                    chunks_processed=i,
                    total_chunks=len(tasks),
                    elapsed_time=current_time - last_progress_update,
                    estimated_remaining=0.0,
                    current_chunk_size=len(task.data),
                    loading_strategy="sequential_processing",
                )
                progress_callback(progress_info)
                last_progress_update = current_time

            # Memory cleanup
            if i % self.config.memory_cleanup_interval == 0:
                gc.collect()

        return results

    def _process_tasks_parallel(
        self,
        tasks: List[ProcessingTask],
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
        worker_count: int,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ProcessingResult]:
        """Process tasks in parallel with optimization."""
        results = []

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._process_single_task_optimized, task, processing_function
                ): task
                for task in tasks
            }

            # Collect results as they complete
            completed_count = 0
            last_progress_update = time.time()

            for future in as_completed(
                future_to_task, timeout=self.config.default_timeout_seconds
            ):
                task = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)
                    self.result_aggregator.add_result(result)

                    # Update worker statistics
                    self.worker_manager.update_worker_stats(
                        result.processing_time,
                        result.success,
                        result.memory_usage,
                        task.estimated_complexity,
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

                # Progress callback with throttling
                current_time = time.time()
                if (
                    progress_callback
                    and current_time - last_progress_update
                    >= self.system_config.data_loader.progress_update_interval
                ):

                    progress_info = ProgressInfo(
                        bytes_processed=completed_count * len(task.data) * 8,
                        total_bytes=len(tasks) * len(task.data) * 8,
                        chunks_processed=completed_count,
                        total_chunks=len(tasks),
                        elapsed_time=current_time - last_progress_update,
                        estimated_remaining=0.0,
                        current_chunk_size=len(task.data),
                        loading_strategy="parallel_processing",
                    )
                    progress_callback(progress_info)
                    last_progress_update = current_time

                # Memory cleanup
                if completed_count % self.config.memory_cleanup_interval == 0:
                    gc.collect()

        return results

    @staticmethod
    def _process_single_task_optimized(
        task: ProcessingTask,
        processing_function: Callable[
            [np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]
        ],
    ) -> ProcessingResult:
        """Process a single task with optimization."""
        start_time = time.time()

        try:
            # Process the data
            result_data, metadata = processing_function(
                task.data, task.processing_params
            )

            processing_time = time.time() - start_time

            # Estimate memory usage
            memory_usage = task.memory_estimate_mb

            # Performance metrics
            performance_metrics = {
                "complexity": task.estimated_complexity,
                "throughput_samples_per_second": (
                    len(task.data) / processing_time if processing_time > 0 else 0
                ),
                "memory_efficiency": (
                    len(task.data) / (memory_usage * 1024 * 1024)
                    if memory_usage > 0
                    else 0
                ),
            }

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
                    "complexity": task.estimated_complexity,
                },
                processing_time=processing_time,
                memory_usage=memory_usage,
                performance_metrics=performance_metrics,
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
        """Update pipeline statistics with enhanced metrics."""
        self.pipeline_stats["total_tasks"] += len(results)
        self.pipeline_stats["completed_tasks"] += sum(1 for r in results if r.success)
        self.pipeline_stats["failed_tasks"] += sum(1 for r in results if not r.success)
        self.pipeline_stats["total_processing_time"] += total_time
        self.pipeline_stats["total_memory_usage"] += sum(
            r.memory_usage for r in results
        )

        # Calculate throughput
        total_samples = sum(
            len(r.result_data) for r in results if r.result_data is not None
        )
        if total_time > 0:
            self.pipeline_stats["avg_throughput"] = total_samples / total_time

        # Calculate efficiency score
        if self.pipeline_stats["total_tasks"] > 0:
            success_rate = (
                self.pipeline_stats["completed_tasks"]
                / self.pipeline_stats["total_tasks"]
            )
            avg_processing_time = (
                self.pipeline_stats["total_processing_time"]
                / self.pipeline_stats["total_tasks"]
            )
            self.pipeline_stats["efficiency_score"] = success_rate * (
                1.0 / (1.0 + avg_processing_time)
            )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get enhanced pipeline statistics."""
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
            "avg_throughput": 0.0,
            "efficiency_score": 0.0,
        }
        self.worker_manager.worker_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
            "avg_memory_usage": 0.0,
            "worker_efficiency": 0.0,
        }


# Backward compatibility aliases
ParallelPipeline = OptimizedParallelPipeline
PipelineConfig = OptimizedPipelineConfig
WorkerPoolManager = OptimizedWorkerPoolManager
ResultAggregator = OptimizedResultAggregator

# Example usage and tests
if __name__ == "__main__":
    print("Optimized Parallel Processing Pipeline Module")
    print("=" * 50)
    print("\nThis module provides optimized parallel processing")
    print("with dynamic configuration and advanced optimization.")
    print("\nFeatures:")
    print("  - Adaptive worker scaling")
    print("  - Enhanced performance monitoring")
    print("  - Optimized memory management")
    print("  - Advanced caching system")
