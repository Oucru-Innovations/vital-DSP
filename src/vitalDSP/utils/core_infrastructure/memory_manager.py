"""
Memory Management and Data Type Optimization for Large Data Processing

This module implements intelligent memory management strategies and data type
optimization to handle large physiological datasets efficiently.

Author: vitalDSP Development Team
Date: October 12, 2025
Version: 1.0.0
"""
"""
Utility Functions Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Parallel processing capabilities
- Performance optimization

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.core_infrastructure.memory_manager import MemoryManager
    >>> signal = np.random.randn(1000)
    >>> processor = MemoryManager(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""



import os
import psutil
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time
from pathlib import Path
import gc

from ..config_utilities.dynamic_config import DynamicConfigManager

# Configure logging
logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Enumeration of memory management strategies."""

    CONSERVATIVE = "conservative"  # Use minimal memory
    BALANCED = "balanced"  # Balance memory and performance
    AGGRESSIVE = "aggressive"  # Use maximum available memory


@dataclass
class MemoryInfo:
    """Data class for memory information."""

    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float


@dataclass
class ProcessingMemoryProfile:
    """Data class for processing memory profile."""

    operation: str
    input_size_mb: float
    output_size_mb: float
    peak_memory_mb: float
    memory_efficiency: float
    processing_time: float


class DataTypeOptimizer:
    """
    Optimizes data types to reduce memory footprint while maintaining precision.
    """

    def __init__(self, config_manager: Optional[DynamicConfigManager] = None):
        """
        Initialize data type optimizer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or DynamicConfigManager()

        # Precision requirements for different signal types
        self.precision_requirements = {
            "ecg": {"min_precision": "float32", "recommended_precision": "float32"},
            "ppg": {"min_precision": "float32", "recommended_precision": "float32"},
            "eeg": {"min_precision": "float32", "recommended_precision": "float32"},
            "resp": {"min_precision": "float32", "recommended_precision": "float32"},
            "generic": {"min_precision": "float32", "recommended_precision": "float32"},
        }

    def optimize_signal(
        self,
        signal: np.ndarray,
        signal_type: str = "generic",
        target_precision: Optional[str] = None,
    ) -> np.ndarray:
        """
        Optimize signal data type based on signal characteristics.

        Args:
            signal: Input signal array
            signal_type: Type of signal (ECG, PPG, EEG, etc.)
            target_precision: Optional target precision override

        Returns:
            Optimized signal array
        """
        if target_precision is None:
            target_precision = self._determine_optimal_precision(signal, signal_type)

        current_dtype = signal.dtype
        target_dtype = np.dtype(target_precision)

        if current_dtype == target_dtype:
            logger.debug(f"Signal already in optimal precision: {target_precision}")
            return signal

        # Check if precision loss is acceptable
        if not self._is_precision_loss_acceptable(signal, current_dtype, target_dtype):
            logger.warning(
                f"Precision loss may be significant for {signal_type}, keeping {current_dtype}"
            )
            return signal

        # Convert to target precision
        optimized_signal = signal.astype(target_dtype)

        # Verify conversion didn't introduce artifacts
        if self._verify_conversion_quality(signal, optimized_signal):
            logger.info(
                f"Optimized signal from {current_dtype} to {target_precision}, "
                f"saved {self._calculate_memory_savings(signal, optimized_signal):.1f}% memory"
            )
            return optimized_signal
        else:
            logger.warning(
                f"Conversion verification failed, reverting to {current_dtype}"
            )
            return signal

    def optimize_features(
        self, features: Dict[str, Any], signal_type: str = "generic"
    ) -> Dict[str, Any]:
        """
        Optimize feature dictionary data types.

        Args:
            features: Feature dictionary
            signal_type: Type of signal

        Returns:
            Optimized feature dictionary
        """
        optimized_features = {}

        for key, value in features.items():
            if isinstance(value, np.ndarray):
                optimized_features[key] = self.optimize_signal(value, signal_type)
            elif isinstance(value, (int, float)):
                # Most features don't need float64 precision
                if isinstance(value, float):
                    optimized_features[key] = np.float32(value)
                else:
                    optimized_features[key] = value
            elif isinstance(value, list):
                # Optimize list of numbers
                if all(isinstance(x, (int, float)) for x in value):
                    optimized_features[key] = np.array(value, dtype=np.float32)
                else:
                    optimized_features[key] = value
            else:
                optimized_features[key] = value

        return optimized_features

    def _determine_optimal_precision(self, signal: np.ndarray, signal_type: str) -> str:
        """
        Determine optimal precision based on signal characteristics.

        Args:
            signal: Input signal
            signal_type: Type of signal

        Returns:
            Optimal precision string
        """
        # Get precision requirements for signal type
        requirements = self.precision_requirements.get(
            signal_type.lower(), self.precision_requirements["generic"]
        )

        # Analyze signal characteristics
        signal_range = np.ptp(signal)
        signal_std = np.std(signal)

        # Determine precision based on signal characteristics
        if signal_range < 1.0 and signal_std < 0.1:
            # Small range and low noise - can use float16
            return "float16"
        elif signal_range < 100.0 and signal_std < 10.0:
            # Medium range - float32 is sufficient
            return "float32"
        else:
            # Large range or high noise - use float64
            return "float64"

    def _is_precision_loss_acceptable(
        self, signal: np.ndarray, current_dtype: np.dtype, target_dtype: np.dtype
    ) -> bool:
        """
        Check if precision loss is acceptable.

        Args:
            signal: Input signal
            current_dtype: Current data type
            target_dtype: Target data type

        Returns:
            True if precision loss is acceptable
        """
        # Calculate precision loss
        current_precision = np.finfo(current_dtype).precision
        target_precision = np.finfo(target_dtype).precision

        precision_loss = current_precision - target_precision

        # Acceptable precision loss threshold
        acceptable_loss = self.config.get("memory.optimization.max_precision_loss", 2.0)

        return precision_loss <= acceptable_loss

    def _verify_conversion_quality(
        self, original: np.ndarray, converted: np.ndarray
    ) -> bool:
        """
        Verify that conversion didn't introduce significant artifacts.

        Args:
            original: Original signal
            converted: Converted signal

        Returns:
            True if conversion quality is acceptable
        """
        # Calculate conversion error
        error = np.abs(original - converted)
        relative_error = error / (np.abs(original) + 1e-10)

        # Check maximum relative error
        max_relative_error = np.max(relative_error)
        max_acceptable_error = self.config.get(
            "memory.optimization.max_conversion_error", 0.01
        )

        return max_relative_error <= max_acceptable_error

    def _calculate_memory_savings(
        self, original: np.ndarray, optimized: np.ndarray
    ) -> float:
        """
        Calculate memory savings percentage.

        Args:
            original: Original signal
            optimized: Optimized signal

        Returns:
            Memory savings percentage
        """
        original_size = original.nbytes
        optimized_size = optimized.nbytes

        if original_size == 0:
            return 0.0

        savings = (original_size - optimized_size) / original_size * 100
        return max(0.0, savings)


class MemoryManager:
    """
    Intelligent memory management for large data processing.
    Monitors and manages memory usage during processing operations.
    """

    def __init__(
        self,
        config_manager: Optional[DynamicConfigManager] = None,
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
    ):
        """
        Initialize memory manager.

        Args:
            config_manager: Configuration manager instance
            strategy: Memory management strategy
        """
        self.config = config_manager or DynamicConfigManager()
        self.strategy = strategy
        self.data_type_optimizer = DataTypeOptimizer(self.config)

        # Memory monitoring
        self.memory_history = []
        self.processing_profiles = []
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Memory limits based on strategy
        self.memory_limits = self._get_memory_limits()

        logger.info(f"Memory manager initialized with {strategy.value} strategy")

    def _get_memory_limits(self) -> Dict[str, float]:
        """Get memory limits based on strategy."""
        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        if self.strategy == MemoryStrategy.CONSERVATIVE:
            max_memory_percent = 0.5
            chunk_memory_percent = 0.05
        elif self.strategy == MemoryStrategy.BALANCED:
            max_memory_percent = 0.7
            chunk_memory_percent = 0.1
        else:  # AGGRESSIVE
            max_memory_percent = 0.9
            chunk_memory_percent = 0.2

        return {
            "max_memory_percent": max_memory_percent,
            "chunk_memory_percent": chunk_memory_percent,
            "max_memory_gb": total_memory_gb * max_memory_percent,
            "chunk_memory_gb": total_memory_gb * chunk_memory_percent,
        }

    def get_memory_info(self) -> MemoryInfo:
        """
        Get current memory information.

        Returns:
            Memory information
        """
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return MemoryInfo(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            memory_percent=memory.percent,
            swap_total_gb=swap.total / (1024**3),
            swap_used_gb=swap.used / (1024**3),
            swap_percent=swap.percent,
        )

    def can_process_in_memory(self, data_size_mb: float, operations: List[str]) -> bool:
        """
        Check if data can be processed entirely in memory.

        Args:
            data_size_mb: Size of data in MB
            operations: List of operations to perform

        Returns:
            True if data can be processed in memory
        """
        # Estimate memory requirement for operations
        memory_multiplier = self._estimate_memory_multiplier(operations)
        required_memory_mb = data_size_mb * memory_multiplier

        available_memory_mb = self.memory_limits["max_memory_gb"] * 1024

        return required_memory_mb <= available_memory_mb

    def recommend_chunk_size(self, data_size_mb: float, operations: List[str]) -> int:
        """
        Recommend optimal chunk size for processing.

        Args:
            data_size_mb: Size of data in MB
            operations: List of operations to perform

        Returns:
            Recommended chunk size in samples
        """
        if self.can_process_in_memory(data_size_mb, operations):
            return int(data_size_mb * 1024**2 / 8)  # Convert to samples (float64)

        # Calculate chunk size based on available memory
        memory_multiplier = self._estimate_memory_multiplier(operations)
        available_memory_mb = self.memory_limits["chunk_memory_gb"] * 1024

        chunk_size_mb = available_memory_mb / memory_multiplier
        chunk_size_samples = int(chunk_size_mb * 1024**2 / 8)  # Convert to samples

        # Apply minimum and maximum constraints
        min_chunk_size = self.config.get("memory.chunk_size.min_samples", 10000)
        max_chunk_size = self.config.get("memory.chunk_size.max_samples", 10000000)

        return max(min_chunk_size, min(chunk_size_samples, max_chunk_size))

    def _estimate_memory_multiplier(self, operations: List[str]) -> float:
        """
        Estimate memory multiplier for operations.

        Args:
            operations: List of operations

        Returns:
            Memory multiplier
        """
        multipliers = {
            "load": 2.0,  # Loading into memory
            "filter": 1.5,  # Filtering creates temp arrays
            "fft": 3.0,  # FFT needs extra workspace
            "features": 0.5,  # Feature extraction minimal
            "quality": 1.0,  # Quality assessment moderate
            "preprocessing": 2.0,  # Preprocessing creates copies
            "segmentation": 1.2,  # Segmentation minimal overhead
            "aggregation": 0.8,  # Aggregation reduces memory
        }

        total_multiplier = sum(multipliers.get(op, 1.0) for op in operations)
        return total_multiplier

    def start_memory_monitoring(self, interval: float = 1.0) -> None:
        """
        Start memory monitoring in background thread.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_active:
            logger.warning("Memory monitoring already active")
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"Memory monitoring started with {interval}s interval")

    def stop_memory_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        logger.info("Memory monitoring stopped")

    def _memory_monitor_loop(self, interval: float) -> None:
        """Memory monitoring loop."""
        while self._monitoring_active:
            try:
                memory_info = self.get_memory_info()

                with self._lock:
                    self.memory_history.append(
                        {"timestamp": time.time(), "memory_info": memory_info}
                    )

                    # Keep only last 1000 entries
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]

                # Check for memory warnings
                if memory_info.memory_percent > 90:
                    logger.warning(
                        f"High memory usage: {memory_info.memory_percent:.1f}%"
                    )
                elif memory_info.memory_percent > 80:
                    logger.info(f"Memory usage: {memory_info.memory_percent:.1f}%")

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)

    def profile_operation(
        self,
        operation: str,
        data_size_mb: float,
        processing_func: Callable,
        *args,
        **kwargs,
    ) -> ProcessingMemoryProfile:
        """
        Profile memory usage of an operation.

        Args:
            operation: Operation name
            data_size_mb: Input data size in MB
            processing_func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Memory profile
        """
        # Get initial memory state
        initial_memory = self.get_memory_info()
        start_time = time.time()

        # Run garbage collection
        gc.collect()

        # Execute operation
        try:
            result = processing_func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Operation {operation} failed: {e}")
            result = None
            success = False

        # Get final memory state
        final_memory = self.get_memory_info()
        end_time = time.time()

        # Calculate memory profile
        peak_memory_mb = (
            max(initial_memory.used_memory_gb, final_memory.used_memory_gb) * 1024
        )
        memory_efficiency = data_size_mb / max(peak_memory_mb, 1.0)

        # Estimate output size
        output_size_mb = self._estimate_output_size(result)

        profile = ProcessingMemoryProfile(
            operation=operation,
            input_size_mb=data_size_mb,
            output_size_mb=output_size_mb,
            peak_memory_mb=peak_memory_mb,
            memory_efficiency=memory_efficiency,
            processing_time=end_time - start_time,
        )

        with self._lock:
            self.processing_profiles.append(profile)

        logger.info(
            f"Operation {operation} profiled: "
            f"peak memory {peak_memory_mb:.1f}MB, "
            f"efficiency {memory_efficiency:.2f}, "
            f"time {profile.processing_time:.2f}s"
        )

        return profile

    def _estimate_output_size(self, result: Any) -> float:
        """
        Estimate output size in MB.

        Args:
            result: Processing result

        Returns:
            Estimated size in MB
        """
        if isinstance(result, np.ndarray):
            return result.nbytes / (1024**2)
        elif isinstance(result, dict):
            total_size = 0
            for value in result.values():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes
                elif isinstance(value, (int, float)):
                    total_size += 8  # Assume 8 bytes per number
            return total_size / (1024**2)
        else:
            return 0.0

    def optimize_data_types(self, data: Any, signal_type: str = "generic") -> Any:
        """
        Optimize data types for memory efficiency.

        Args:
            data: Data to optimize
            signal_type: Type of signal

        Returns:
            Optimized data
        """
        if isinstance(data, np.ndarray):
            return self.data_type_optimizer.optimize_signal(data, signal_type)
        elif isinstance(data, dict):
            return self.data_type_optimizer.optimize_features(data, signal_type)
        else:
            return data

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Memory statistics
        """
        current_memory = self.get_memory_info()

        # Calculate memory trends
        if len(self.memory_history) > 1:
            memory_trend = self._calculate_memory_trend()
        else:
            memory_trend = {"trend": "stable", "change_percent": 0.0}

        # Calculate processing efficiency
        if self.processing_profiles:
            avg_efficiency = np.mean(
                [p.memory_efficiency for p in self.processing_profiles]
            )
            avg_processing_time = np.mean(
                [p.processing_time for p in self.processing_profiles]
            )
        else:
            avg_efficiency = 0.0
            avg_processing_time = 0.0

        return {
            "current_memory": {
                "total_gb": current_memory.total_memory_gb,
                "available_gb": current_memory.available_memory_gb,
                "used_gb": current_memory.used_memory_gb,
                "percent": current_memory.memory_percent,
                "swap_percent": current_memory.swap_percent,
            },
            "memory_limits": self.memory_limits,
            "memory_trend": memory_trend,
            "processing_efficiency": {
                "average_efficiency": avg_efficiency,
                "average_processing_time": avg_processing_time,
                "total_operations_profiled": len(self.processing_profiles),
            },
            "monitoring": {
                "active": self._monitoring_active,
                "history_length": len(self.memory_history),
            },
        }

    def _calculate_memory_trend(self) -> Dict[str, Any]:
        """Calculate memory usage trend."""
        if len(self.memory_history) < 2:
            return {"trend": "stable", "change_percent": 0.0}

        recent_memory = self.memory_history[-10:]  # Last 10 measurements
        older_memory = (
            self.memory_history[-20:-10] if len(self.memory_history) >= 20 else []
        )

        if not older_memory:
            return {"trend": "stable", "change_percent": 0.0}

        recent_avg = np.mean([m["memory_info"].memory_percent for m in recent_memory])
        older_avg = np.mean([m["memory_info"].memory_percent for m in older_memory])

        change_percent = recent_avg - older_avg

        if change_percent > 5:
            trend = "increasing"
        elif change_percent < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change_percent": change_percent,
            "recent_average": recent_avg,
            "older_average": older_avg,
        }

    def get_memory_warnings(self) -> List[str]:
        """
        Get memory-related warnings.

        Returns:
            List of warning messages
        """
        warnings = []
        current_memory = self.get_memory_info()

        # High memory usage warning
        if current_memory.memory_percent > 90:
            warnings.append(
                f"Critical memory usage: {current_memory.memory_percent:.1f}%"
            )
        elif current_memory.memory_percent > 80:
            warnings.append(f"High memory usage: {current_memory.memory_percent:.1f}%")

        # Swap usage warning
        if current_memory.swap_percent > 50:
            warnings.append(f"High swap usage: {current_memory.swap_percent:.1f}%")

        # Memory trend warning
        memory_stats = self.get_memory_statistics()
        trend = memory_stats["memory_trend"]

        if trend["trend"] == "increasing" and trend["change_percent"] > 10:
            warnings.append(
                f"Memory usage increasing rapidly: +{trend['change_percent']:.1f}%"
            )

        return warnings

    def cleanup_memory(self) -> None:
        """Perform comprehensive memory cleanup."""
        logger.info("Performing memory cleanup")

        # Force garbage collection
        self.force_garbage_collection()

        # Clear processing profiles if too many
        with self._lock:
            if len(self.processing_profiles) > 100:
                self.processing_profiles = self.processing_profiles[-50:]

            if len(self.memory_history) > 500:
                self.memory_history = self.memory_history[-250:]

        logger.info("Memory cleanup completed")


class MemoryProfiler:
    """
    Advanced memory profiling tools for optimization analysis.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize memory profiler.

        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager
        self.profiles = []

    def profile_pipeline(
        self, pipeline_func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Profile entire processing pipeline.

        Args:
            pipeline_func: Pipeline function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Comprehensive profiling results
        """
        logger.info("Starting pipeline memory profiling")

        # Start memory monitoring
        self.memory_manager.start_memory_monitoring(interval=0.5)

        try:
            # Profile the pipeline
            result = self.memory_manager.profile_operation(
                "pipeline", 0, pipeline_func, *args, **kwargs
            )

            # Get detailed memory statistics
            memory_stats = self.memory_manager.get_memory_statistics()

            # Analyze memory patterns
            memory_patterns = self._analyze_memory_patterns()

            profiling_results = {
                "pipeline_profile": result,
                "memory_statistics": memory_stats,
                "memory_patterns": memory_patterns,
                "optimization_recommendations": self._generate_optimization_recommendations(
                    result, memory_stats, memory_patterns
                ),
            }

            logger.info("Pipeline memory profiling completed")
            return profiling_results

        finally:
            self.memory_manager.stop_memory_monitoring()

    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if len(self.memory_manager.memory_history) < 2:
            return {"pattern": "insufficient_data"}

        memory_percentages = [
            m["memory_info"].memory_percent for m in self.memory_manager.memory_history
        ]

        # Calculate patterns
        memory_variance = np.var(memory_percentages)
        memory_range = np.ptp(memory_percentages)
        memory_trend = self._calculate_trend(memory_percentages)

        return {
            "variance": memory_variance,
            "range": memory_range,
            "trend": memory_trend,
            "stability": "stable" if memory_variance < 10 else "unstable",
            "peak_usage": np.max(memory_percentages),
            "average_usage": np.mean(memory_percentages),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 1:
            return "increasing"
        elif slope < -1:
            return "decreasing"
        else:
            return "stable"

    def _generate_optimization_recommendations(
        self,
        profile: ProcessingMemoryProfile,
        memory_stats: Dict[str, Any],
        patterns: Dict[str, Any],
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Memory efficiency recommendations
        if profile.memory_efficiency < 0.5:
            recommendations.append(
                "Low memory efficiency - consider chunked processing"
            )

        if memory_stats["current_memory"]["percent"] > 80:
            recommendations.append(
                "High memory usage - consider data type optimization"
            )

        if patterns["stability"] == "unstable":
            recommendations.append("Unstable memory usage - check for memory leaks")

        if memory_stats["memory_trend"]["trend"] == "increasing":
            recommendations.append(
                "Memory usage increasing - consider garbage collection"
            )

        # Processing time recommendations
        if profile.processing_time > 60:
            recommendations.append(
                "Long processing time - consider parallel processing"
            )

        return recommendations

    def generate_optimization_report(self, profiling_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive optimization report.

        Args:
            profiling_results: Profiling results

        Returns:
            Optimization report
        """
        report = []
        report.append("=== Memory Optimization Report ===")
        report.append("")

        # Pipeline profile
        profile = profiling_results["pipeline_profile"]
        report.append("Pipeline Profile:")
        report.append(f"  Operation: {profile.operation}")
        report.append(f"  Input Size: {profile.input_size_mb:.1f} MB")
        report.append(f"  Output Size: {profile.output_size_mb:.1f} MB")
        report.append(f"  Peak Memory: {profile.peak_memory_mb:.1f} MB")
        report.append(f"  Memory Efficiency: {profile.memory_efficiency:.2f}")
        report.append(f"  Processing Time: {profile.processing_time:.2f} seconds")
        report.append("")

        # Memory statistics
        memory_stats = profiling_results["memory_statistics"]
        report.append("Memory Statistics:")
        report.append(
            f"  Total Memory: {memory_stats['current_memory']['total_gb']:.1f} GB"
        )
        report.append(
            f"  Available Memory: {memory_stats['current_memory']['available_gb']:.1f} GB"
        )
        report.append(
            f"  Used Memory: {memory_stats['current_memory']['used_gb']:.1f} GB"
        )
        report.append(
            f"  Memory Usage: {memory_stats['current_memory']['percent']:.1f}%"
        )
        report.append(
            f"  Swap Usage: {memory_stats['current_memory']['swap_percent']:.1f}%"
        )
        report.append("")

        # Memory patterns
        patterns = profiling_results["memory_patterns"]
        report.append("Memory Patterns:")
        report.append(f"  Stability: {patterns['stability']}")
        report.append(f"  Variance: {patterns['variance']:.2f}")
        report.append(f"  Range: {patterns['range']:.1f}%")
        report.append(f"  Trend: {patterns['trend']}")
        report.append(f"  Peak Usage: {patterns['peak_usage']:.1f}%")
        report.append(f"  Average Usage: {patterns['average_usage']:.1f}%")
        report.append("")

        # Optimization recommendations
        recommendations = profiling_results["optimization_recommendations"]
        report.append("Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            report.append(f"  {i}. {rec}")
        report.append("")

        return "\n".join(report)
