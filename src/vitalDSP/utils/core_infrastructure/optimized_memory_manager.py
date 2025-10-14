"""
Optimized Memory Management and Data Type Optimization for Large Data Processing

This module implements the optimized version of intelligent memory management strategies
and data type optimization with dynamic configuration, adaptive resource management,
and performance optimization based on Phase 1 optimization patterns.

Author: vitalDSP Development Team
Date: October 12, 2025
Version: 2.0.0 (Optimized)
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
import warnings

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


class OptimizedDataTypeOptimizer:
    """
    Optimized data type optimizer with signal-type awareness and precision analysis.
    """

    def __init__(self, config_manager: DynamicConfigManager):
        """
        Initialize optimized data type optimizer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager

        # Dynamic precision requirements for different signal types
        self.precision_requirements = self._load_precision_requirements()

        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "memory_savings_mb": 0.0,
            "precision_loss_incidents": 0,
            "optimization_time": 0.0,
        }

    def _load_precision_requirements(self) -> Dict[str, Dict[str, str]]:
        """Load precision requirements from configuration."""
        return {
            "ecg": {
                "min_precision": self.config.get(
                    "data_types.ecg.min_precision", "float32"
                ),
                "recommended_precision": self.config.get(
                    "data_types.ecg.recommended_precision", "float32"
                ),
                "max_precision_loss": self.config.get(
                    "data_types.ecg.max_precision_loss", 2.0
                ),
            },
            "ppg": {
                "min_precision": self.config.get(
                    "data_types.ppg.min_precision", "float32"
                ),
                "recommended_precision": self.config.get(
                    "data_types.ppg.recommended_precision", "float32"
                ),
                "max_precision_loss": self.config.get(
                    "data_types.ppg.max_precision_loss", 2.0
                ),
            },
            "eeg": {
                "min_precision": self.config.get(
                    "data_types.eeg.min_precision", "float32"
                ),
                "recommended_precision": self.config.get(
                    "data_types.eeg.recommended_precision", "float32"
                ),
                "max_precision_loss": self.config.get(
                    "data_types.eeg.max_precision_loss", 2.0
                ),
            },
            "resp": {
                "min_precision": self.config.get(
                    "data_types.resp.min_precision", "float32"
                ),
                "recommended_precision": self.config.get(
                    "data_types.resp.recommended_precision", "float32"
                ),
                "max_precision_loss": self.config.get(
                    "data_types.resp.max_precision_loss", 2.0
                ),
            },
            "generic": {
                "min_precision": self.config.get(
                    "data_types.generic.min_precision", "float32"
                ),
                "recommended_precision": self.config.get(
                    "data_types.generic.recommended_precision", "float32"
                ),
                "max_precision_loss": self.config.get(
                    "data_types.generic.max_precision_loss", 2.0
                ),
            },
        }

    def optimize_signal(
        self,
        signal: np.ndarray,
        signal_type: str = "generic",
        target_precision: Optional[str] = None,
    ) -> np.ndarray:
        """
        Optimize signal data type with enhanced analysis.

        Args:
            signal: Input signal array
            signal_type: Type of signal (ECG, PPG, EEG, etc.)
            target_precision: Optional target precision override

        Returns:
            Optimized signal array
        """
        start_time = time.time()

        if target_precision is None:
            target_precision = self._determine_optimal_precision_enhanced(
                signal, signal_type
            )

        current_dtype = signal.dtype
        target_dtype = np.dtype(target_precision)

        if current_dtype == target_dtype:
            logger.debug(f"Signal already in optimal precision: {target_precision}")
            return signal

        # Enhanced precision loss analysis
        precision_loss_acceptable = self._is_precision_loss_acceptable_enhanced(
            signal, current_dtype, target_dtype, signal_type
        )

        if not precision_loss_acceptable:
            logger.warning(
                f"Precision loss may be significant for {signal_type}, keeping {current_dtype}"
            )
            self.optimization_stats["precision_loss_incidents"] += 1
            return signal

        # Convert to target precision
        optimized_signal = signal.astype(target_dtype)

        # Enhanced conversion verification
        if self._verify_conversion_quality_enhanced(
            signal, optimized_signal, signal_type
        ):
            memory_savings = self._calculate_memory_savings(signal, optimized_signal)

            self.optimization_stats["total_optimizations"] += 1
            self.optimization_stats["memory_savings_mb"] += memory_savings / (1024**2)
            self.optimization_stats["optimization_time"] += time.time() - start_time

            logger.info(
                f"Optimized signal from {current_dtype} to {target_precision}, "
                f"saved {memory_savings / (1024**2):.1f}MB memory"
            )
            return optimized_signal
        else:
            logger.warning(
                f"Conversion verification failed, reverting to {current_dtype}"
            )
            self.optimization_stats["precision_loss_incidents"] += 1
            return signal

    def _determine_optimal_precision_enhanced(
        self, signal: np.ndarray, signal_type: str
    ) -> str:
        """
        Enhanced precision determination with signal analysis.

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

        # Enhanced signal analysis
        signal_analysis = self._analyze_signal_characteristics(signal)

        # Determine precision based on analysis
        if signal_analysis["range"] < 1.0 and signal_analysis["noise_level"] < 0.1:
            # Small range and low noise - can use float16
            return "float16"
        elif signal_analysis["range"] < 100.0 and signal_analysis["noise_level"] < 10.0:
            # Medium range - float32 is sufficient
            return "float32"
        elif signal_analysis["dynamic_range"] > 1000:
            # High dynamic range - use float64
            return "float64"
        else:
            # Default to recommended precision
            return requirements["recommended_precision"]

    def _analyze_signal_characteristics(self, signal: np.ndarray) -> Dict[str, float]:
        """Analyze signal characteristics for precision determination."""
        return {
            "range": np.ptp(signal),
            "std": np.std(signal),
            "noise_level": self._estimate_noise_level(signal),
            "dynamic_range": np.max(np.abs(signal)) / max(np.std(signal), 1e-10),
            "entropy": self._calculate_signal_entropy(signal),
        }

    def _estimate_noise_level(self, signal: np.ndarray) -> float:
        """Estimate noise level in signal."""
        # Use high-frequency content as noise estimate
        if len(signal) > 100:
            diff_signal = np.diff(signal)
            noise_level = np.std(diff_signal) / np.std(signal)
            return min(noise_level, 1.0)
        return 0.0

    def _calculate_signal_entropy(self, signal: np.ndarray, bins: int = 50) -> float:
        """Calculate signal entropy."""
        try:
            hist, _ = np.histogram(signal, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        except Exception as e:
            logger.error(f"Error calculating signal entropy: {str(e)}")
            return 0.0

    def _is_precision_loss_acceptable_enhanced(
        self,
        signal: np.ndarray,
        current_dtype: np.dtype,
        target_dtype: np.dtype,
        signal_type: str,
    ) -> bool:
        """
        Enhanced precision loss assessment.

        Args:
            signal: Input signal
            current_dtype: Current data type
            target_dtype: Target data type
            signal_type: Type of signal

        Returns:
            True if precision loss is acceptable
        """
        # Get signal-specific precision loss threshold
        requirements = self.precision_requirements.get(
            signal_type.lower(), self.precision_requirements["generic"]
        )
        max_precision_loss = requirements["max_precision_loss"]

        # Calculate precision loss
        current_precision = np.finfo(current_dtype).precision
        target_precision = np.finfo(target_dtype).precision
        precision_loss = current_precision - target_precision

        # Additional checks for signal characteristics
        signal_range = np.ptp(signal)
        signal_mean = np.mean(np.abs(signal))

        # Adjust threshold based on signal characteristics
        if signal_range > 1000:
            max_precision_loss *= 0.5  # Be more conservative for large ranges
        if signal_mean < 0.01:
            max_precision_loss *= 0.7  # Be more conservative for small signals

        return precision_loss <= max_precision_loss

    def _verify_conversion_quality_enhanced(
        self, original: np.ndarray, converted: np.ndarray, signal_type: str
    ) -> bool:
        """
        Enhanced conversion quality verification.

        Args:
            original: Original signal
            converted: Converted signal
            signal_type: Type of signal

        Returns:
            True if conversion quality is acceptable
        """
        # Calculate conversion error
        error = np.abs(original - converted)
        relative_error = error / (np.abs(original) + 1e-10)

        # Signal-specific error thresholds
        max_error_thresholds = {
            "ecg": 0.01,
            "ppg": 0.01,
            "eeg": 0.005,
            "resp": 0.02,
            "generic": 0.01,
        }

        max_acceptable_error = max_error_thresholds.get(signal_type.lower(), 0.01)

        # Check maximum relative error
        max_relative_error = np.max(relative_error)

        # Additional checks for signal-specific characteristics
        if signal_type.lower() == "ecg":
            # Check if R-peaks are preserved
            return self._verify_ecg_peak_preservation(original, converted)
        elif signal_type.lower() == "ppg":
            # Check if pulse peaks are preserved
            return self._verify_ppg_peak_preservation(original, converted)

        return max_relative_error <= max_acceptable_error

    def _verify_ecg_peak_preservation(
        self, original: np.ndarray, converted: np.ndarray
    ) -> bool:
        """Verify ECG peak preservation after conversion."""
        try:
            from scipy.signal import find_peaks

            # Find peaks in both signals
            orig_peaks, _ = find_peaks(
                original, height=np.mean(original) + np.std(original)
            )
            conv_peaks, _ = find_peaks(
                converted, height=np.mean(converted) + np.std(converted)
            )

            # Check if peak count is similar
            peak_count_ratio = len(conv_peaks) / max(len(orig_peaks), 1)
            return 0.8 <= peak_count_ratio <= 1.2

        except ImportError:
            # Fall back to simple correlation check
            correlation = np.corrcoef(original, converted)[0, 1]
            return correlation > 0.99

    def _verify_ppg_peak_preservation(
        self, original: np.ndarray, converted: np.ndarray
    ) -> bool:
        """Verify PPG peak preservation after conversion."""
        # Similar to ECG but with PPG-specific thresholds
        correlation = np.corrcoef(original, converted)[0, 1]
        return correlation > 0.98

    def optimize_features(
        self, features: Dict[str, Any], signal_type: str = "generic"
    ) -> Dict[str, Any]:
        """
        Optimize feature dictionary with enhanced analysis.

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
                # Enhanced precision optimization for features
                optimized_features[key] = self._optimize_feature_precision(
                    value, key, signal_type
                )
            elif isinstance(value, list):
                # Optimize list of numbers
                if all(isinstance(x, (int, float)) for x in value):
                    optimized_features[key] = np.array(
                        value, dtype=self._get_optimal_array_dtype(value)
                    )
                else:
                    optimized_features[key] = value
            else:
                optimized_features[key] = value

        return optimized_features

    def _optimize_feature_precision(
        self, value: float, key: str, signal_type: str
    ) -> Union[float, np.float32, np.float64]:
        """Optimize precision for individual feature values."""
        # Determine optimal precision based on feature type and value magnitude
        if abs(value) < 1e-6:
            return np.float64(value)  # Keep high precision for very small values
        elif abs(value) < 1e3:
            return np.float32(value)  # Use float32 for moderate values
        else:
            return np.float64(value)  # Keep float64 for large values

    def _get_optimal_array_dtype(self, values: List[Union[int, float]]) -> np.dtype:
        """Determine optimal dtype for array of values."""
        values_array = np.array(values)

        if np.all(values_array == values_array.astype(np.int32)):
            return np.int32
        elif np.all(values_array == values_array.astype(np.float32)):
            return np.float32
        else:
            return np.float64

    def _calculate_memory_savings(
        self, original: np.ndarray, optimized: np.ndarray
    ) -> float:
        """
        Calculate memory savings from optimization.

        Args:
            original: Original signal
            optimized: Optimized signal

        Returns:
            Memory savings in bytes
        """
        original_size = original.nbytes
        optimized_size = optimized.nbytes

        if original_size == 0:
            return 0.0

        savings = original_size - optimized_size
        return max(0.0, savings)

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_optimizations": self.optimization_stats["total_optimizations"],
            "memory_savings_mb": self.optimization_stats["memory_savings_mb"],
            "precision_loss_incidents": self.optimization_stats[
                "precision_loss_incidents"
            ],
            "average_optimization_time": (
                self.optimization_stats["optimization_time"]
                / max(self.optimization_stats["total_optimizations"], 1)
            ),
            "success_rate": (
                (
                    self.optimization_stats["total_optimizations"]
                    - self.optimization_stats["precision_loss_incidents"]
                )
                / max(self.optimization_stats["total_optimizations"], 1)
            ),
        }


class OptimizedMemoryManager:
    """
    Optimized intelligent memory management with adaptive strategies and performance optimization.
    """

    def __init__(
        self,
        config_manager: DynamicConfigManager,
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
    ):
        """
        Initialize optimized memory manager.

        Args:
            config_manager: Configuration manager instance
            strategy: Memory management strategy
        """
        self.config = config_manager
        self.strategy = strategy
        self.data_type_optimizer = OptimizedDataTypeOptimizer(self.config)

        # Dynamic memory monitoring
        self.memory_history = []
        self.processing_profiles = []
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Adaptive memory limits based on strategy and system
        self.memory_limits = self._get_adaptive_memory_limits()

        # Performance tracking
        self.performance_stats = {
            "total_monitoring_time": 0.0,
            "memory_warnings_issued": 0,
            "optimizations_applied": 0,
            "cleanup_operations": 0,
        }

        logger.info(
            f"Optimized memory manager initialized with {strategy.value} strategy"
        )

    def _get_adaptive_memory_limits(self) -> Dict[str, float]:
        """Get adaptive memory limits based on strategy and system resources."""
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()

        # Base limits from configuration
        base_limits = {
            "conservative": {
                "max_memory_percent": self.config.get(
                    "memory.conservative.max_memory_percent", 0.5
                ),
                "chunk_memory_percent": self.config.get(
                    "memory.conservative.chunk_memory_percent", 0.05
                ),
            },
            "balanced": {
                "max_memory_percent": self.config.get(
                    "memory.balanced.max_memory_percent", 0.7
                ),
                "chunk_memory_percent": self.config.get(
                    "memory.balanced.chunk_memory_percent", 0.1
                ),
            },
            "aggressive": {
                "max_memory_percent": self.config.get(
                    "memory.aggressive.max_memory_percent", 0.9
                ),
                "chunk_memory_percent": self.config.get(
                    "memory.aggressive.chunk_memory_percent", 0.2
                ),
            },
        }

        strategy_limits = base_limits[self.strategy.value]

        # Adjust based on system resources
        if total_memory_gb > 32:  # High-memory system
            strategy_limits["max_memory_percent"] *= 1.1
        elif total_memory_gb < 8:  # Low-memory system
            strategy_limits["max_memory_percent"] *= 0.8

        if cpu_count > 8:  # High-CPU system
            strategy_limits["chunk_memory_percent"] *= 1.2
        elif cpu_count < 4:  # Low-CPU system
            strategy_limits["chunk_memory_percent"] *= 0.8

        return {
            "max_memory_percent": strategy_limits["max_memory_percent"],
            "chunk_memory_percent": strategy_limits["chunk_memory_percent"],
            "max_memory_gb": total_memory_gb * strategy_limits["max_memory_percent"],
            "chunk_memory_gb": total_memory_gb
            * strategy_limits["chunk_memory_percent"],
            "total_memory_gb": total_memory_gb,
            "cpu_count": cpu_count,
        }

    def get_memory_info(self) -> MemoryInfo:
        """
        Get current memory information with enhanced details.

        Returns:
            Enhanced memory information
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
        Enhanced memory capability assessment.

        Args:
            data_size_mb: Size of data in MB
            operations: List of operations

        Returns:
            True if data can be processed in memory
        """
        # Enhanced memory requirement estimation
        memory_multiplier = self._estimate_memory_multiplier_enhanced(operations)
        required_memory_mb = data_size_mb * memory_multiplier

        # Consider available memory and safety margin
        available_memory_mb = self.memory_limits["max_memory_gb"] * 1024
        safety_margin = self.config.get("memory.safety_margin_percent", 10)
        usable_memory_mb = available_memory_mb * (1 - safety_margin / 100)

        return required_memory_mb <= usable_memory_mb

    def _estimate_memory_multiplier_enhanced(self, operations: List[str]) -> float:
        """
        Enhanced memory multiplier estimation.

        Args:
            operations: List of operations

        Returns:
            Memory multiplier
        """
        # Base multipliers from configuration
        base_multipliers = {
            "load": self.config.get("memory.multipliers.load", 2.0),
            "filter": self.config.get("memory.multipliers.filter", 1.5),
            "fft": self.config.get("memory.multipliers.fft", 3.0),
            "features": self.config.get("memory.multipliers.features", 0.5),
            "quality": self.config.get("memory.multipliers.quality", 1.0),
            "preprocessing": self.config.get("memory.multipliers.preprocessing", 2.0),
            "segmentation": self.config.get("memory.multipliers.segmentation", 1.2),
            "aggregation": self.config.get("memory.multipliers.aggregation", 0.8),
        }

        total_multiplier = sum(base_multipliers.get(op, 1.0) for op in operations)

        # Adjust based on system resources
        if self.memory_limits["total_memory_gb"] > 16:
            total_multiplier *= 0.9  # More efficient on high-memory systems
        elif self.memory_limits["total_memory_gb"] < 4:
            total_multiplier *= 1.2  # Less efficient on low-memory systems

        return total_multiplier

    def recommend_chunk_size(self, data_size_mb: float, operations: List[str]) -> int:
        """
        Enhanced chunk size recommendation.

        Args:
            data_size_mb: Size of data in MB
            operations: List of operations

        Returns:
            Recommended chunk size in samples
        """
        if self.can_process_in_memory(data_size_mb, operations):
            return int(data_size_mb * 1024**2 / 8)  # Convert to samples (float64)

        # Enhanced chunk size calculation
        memory_multiplier = self._estimate_memory_multiplier_enhanced(operations)
        available_memory_mb = self.memory_limits["chunk_memory_gb"] * 1024

        # Consider operation complexity
        complexity_factor = self._calculate_operation_complexity(operations)
        chunk_size_mb = available_memory_mb / (memory_multiplier * complexity_factor)
        chunk_size_samples = int(chunk_size_mb * 1024**2 / 8)

        # Apply dynamic constraints
        min_chunk_size = self.config.get("memory.chunk_size.min_samples", 10000)
        max_chunk_size = self.config.get("memory.chunk_size.max_samples", 10000000)

        # Adjust based on data size
        if data_size_mb > 1000:  # Large dataset
            max_chunk_size = min(max_chunk_size, int(data_size_mb * 1024**2 / 8 * 0.1))
        elif data_size_mb < 10:  # Small dataset
            min_chunk_size = min(min_chunk_size, int(data_size_mb * 1024**2 / 8))

        return max(min_chunk_size, min(chunk_size_samples, max_chunk_size))

    def _calculate_operation_complexity(self, operations: List[str]) -> float:
        """Calculate operation complexity factor."""
        complexity_factors = {
            "load": 1.0,
            "filter": 1.2,
            "fft": 1.5,
            "features": 1.1,
            "quality": 1.3,
            "preprocessing": 1.4,
            "segmentation": 1.0,
            "aggregation": 0.9,
        }

        return max(complexity_factors.get(op, 1.0) for op in operations)

    def start_memory_monitoring(self, interval: Optional[float] = None) -> None:
        """
        Start optimized memory monitoring.

        Args:
            interval: Monitoring interval in seconds (auto-calculated if None)
        """
        if self._monitoring_active:
            logger.warning("Memory monitoring already active")
            return

        if interval is None:
            # Adaptive monitoring interval based on system load
            interval = self._calculate_adaptive_monitoring_interval()

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._optimized_memory_monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"Optimized memory monitoring started with {interval}s interval")

    def _calculate_adaptive_monitoring_interval(self) -> float:
        """Calculate adaptive monitoring interval based on system state."""
        base_interval = self.config.get("memory.monitoring.base_interval", 1.0)

        # Adjust based on memory usage
        memory_info = self.get_memory_info()
        if memory_info.memory_percent > 80:
            return base_interval * 0.5  # Monitor more frequently
        elif memory_info.memory_percent < 30:
            return base_interval * 2.0  # Monitor less frequently
        else:
            return base_interval

    def _optimized_memory_monitor_loop(self, interval: float) -> None:
        """Optimized memory monitoring loop."""
        start_time = time.time()

        while self._monitoring_active:
            try:
                memory_info = self.get_memory_info()

                with self._lock:
                    self.memory_history.append(
                        {"timestamp": time.time(), "memory_info": memory_info}
                    )

                    # Adaptive history management
                    self._manage_memory_history()

                # Enhanced memory warnings
                self._check_memory_warnings(memory_info)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)

        self.performance_stats["total_monitoring_time"] += time.time() - start_time

    def _manage_memory_history(self) -> None:
        """Manage memory history with adaptive limits."""
        max_history_entries = self.config.get(
            "memory.monitoring.max_history_entries", 1000
        )

        # Scale based on available memory
        memory_gb = self.memory_limits["total_memory_gb"]
        if memory_gb > 16:
            max_history_entries = int(max_history_entries * 1.5)
        elif memory_gb < 4:
            max_history_entries = int(max_history_entries * 0.5)

        if len(self.memory_history) > max_history_entries:
            # Keep most recent entries
            self.memory_history = self.memory_history[-max_history_entries:]

    def _check_memory_warnings(self, memory_info: MemoryInfo) -> None:
        """Check for memory warnings with enhanced thresholds."""
        warning_thresholds = {
            "critical": self.config.get("memory.warnings.critical_percent", 95),
            "high": self.config.get("memory.warnings.high_percent", 85),
            "medium": self.config.get("memory.warnings.medium_percent", 75),
        }

        if memory_info.memory_percent > warning_thresholds["critical"]:
            logger.critical(f"CRITICAL memory usage: {memory_info.memory_percent:.1f}%")
            self.performance_stats["memory_warnings_issued"] += 1
            self._trigger_emergency_cleanup()
        elif memory_info.memory_percent > warning_thresholds["high"]:
            logger.warning(f"HIGH memory usage: {memory_info.memory_percent:.1f}%")
            self.performance_stats["memory_warnings_issued"] += 1
            self._trigger_aggressive_cleanup()
        elif memory_info.memory_percent > warning_thresholds["medium"]:
            logger.info(f"Medium memory usage: {memory_info.memory_percent:.1f}%")
            self._trigger_standard_cleanup()

    def _trigger_emergency_cleanup(self) -> None:
        """Trigger emergency memory cleanup."""
        logger.warning("Triggering emergency memory cleanup")
        self.force_garbage_collection()
        self.cleanup_memory()
        self.performance_stats["cleanup_operations"] += 1

    def _trigger_aggressive_cleanup(self) -> None:
        """Trigger aggressive memory cleanup."""
        logger.info("Triggering aggressive memory cleanup")
        self.force_garbage_collection()
        self.performance_stats["cleanup_operations"] += 1

    def _trigger_standard_cleanup(self) -> None:
        """Trigger standard memory cleanup."""
        self.force_garbage_collection()

    def profile_operation(
        self,
        operation: str,
        data_size_mb: float,
        processing_func: Callable,
        *args,
        **kwargs,
    ) -> ProcessingMemoryProfile:
        """
        Enhanced operation profiling with optimization.

        Args:
            operation: Operation name
            data_size_mb: Input data size in MB
            processing_func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Enhanced memory profile
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

        # Calculate enhanced memory profile
        peak_memory_mb = (
            max(initial_memory.used_memory_gb, final_memory.used_memory_gb) * 1024
        )
        memory_efficiency = data_size_mb / max(peak_memory_mb, 1.0)

        # Estimate output size
        output_size_mb = self._estimate_output_size_enhanced(result)

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
            f"Enhanced operation {operation} profiled: "
            f"peak memory {peak_memory_mb:.1f}MB, "
            f"efficiency {memory_efficiency:.2f}, "
            f"time {profile.processing_time:.2f}s"
        )

        return profile

    def _estimate_output_size_enhanced(self, result: Any) -> float:
        """
        Enhanced output size estimation.

        Args:
            result: Processing result

        Returns:
            Estimated size in MB
        """
        if isinstance(result, np.ndarray):
            return result.nbytes / (1024**2)
        elif isinstance(result, dict):
            total_size = 0
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes
                elif isinstance(value, (int, float)):
                    total_size += 8
                elif isinstance(value, str):
                    total_size += len(value.encode("utf-8"))
                elif isinstance(value, list):
                    total_size += len(str(value).encode("utf-8"))
            return total_size / (1024**2)
        else:
            return len(str(result).encode("utf-8")) / (1024**2)

    def optimize_data_types(self, data: Any, signal_type: str = "generic") -> Any:
        """
        Optimize data types with enhanced analysis.

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
        """Force enhanced garbage collection."""
        collected = gc.collect()
        logger.debug(f"Enhanced garbage collection freed {collected} objects")

    def cleanup_memory(self) -> None:
        """Perform comprehensive memory cleanup."""
        logger.info("Performing comprehensive memory cleanup")

        # Force garbage collection
        self.force_garbage_collection()

        # Clean up processing profiles
        with self._lock:
            max_profiles = self.config.get("memory.max_profiles", 100)
            if len(self.processing_profiles) > max_profiles:
                self.processing_profiles = self.processing_profiles[-max_profiles:]

        self.performance_stats["cleanup_operations"] += 1
        logger.info("Comprehensive memory cleanup completed")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Enhanced memory statistics
        """
        current_memory = self.get_memory_info()

        # Calculate memory trends
        if len(self.memory_history) > 1:
            memory_trend = self._calculate_memory_trend_enhanced()
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
            "performance_stats": self.performance_stats,
            "optimization_stats": self.data_type_optimizer.get_optimization_statistics(),
        }

    def _calculate_memory_trend_enhanced(self) -> Dict[str, Any]:
        """Calculate enhanced memory usage trend."""
        if len(self.memory_history) < 2:
            return {"trend": "stable", "change_percent": 0.0}

        # Use more recent data for trend analysis
        recent_window = min(20, len(self.memory_history))
        recent_memory = self.memory_history[-recent_window:]
        older_memory = (
            self.memory_history[-recent_window * 2 : -recent_window]
            if len(self.memory_history) >= recent_window * 2
            else []
        )

        if not older_memory:
            return {"trend": "stable", "change_percent": 0.0}

        recent_avg = np.mean([m["memory_info"].memory_percent for m in recent_memory])
        older_avg = np.mean([m["memory_info"].memory_percent for m in older_memory])

        change_percent = recent_avg - older_avg

        # Enhanced trend classification
        if change_percent > 10:
            trend = "rapidly_increasing"
        elif change_percent > 5:
            trend = "increasing"
        elif change_percent < -10:
            trend = "rapidly_decreasing"
        elif change_percent < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change_percent": change_percent,
            "recent_average": recent_avg,
            "older_average": older_avg,
            "trend_strength": abs(change_percent),
        }

    def get_memory_warnings(self) -> List[str]:
        """
        Get enhanced memory-related warnings.

        Returns:
            List of warning messages
        """
        warnings_content = []
        current_memory = self.get_memory_info()

        # Enhanced warning thresholds
        warning_thresholds = {
            "critical": self.config.get("memory.warnings.critical_percent", 95),
            "high": self.config.get("memory.warnings.high_percent", 85),
            "medium": self.config.get("memory.warnings.medium_percent", 75),
        }

        # Memory usage warnings
        if current_memory.memory_percent > warning_thresholds["critical"]:
            warnings_content.append(
                f"CRITICAL memory usage: {current_memory.memory_percent:.1f}%"
            )
        elif current_memory.memory_percent > warning_thresholds["high"]:
            warnings_content.append(
                f"HIGH memory usage: {current_memory.memory_percent:.1f}%"
            )
        elif current_memory.memory_percent > warning_thresholds["medium"]:
            warnings_content.append(
                f"Medium memory usage: {current_memory.memory_percent:.1f}%"
            )

        # Swap usage warnings
        swap_threshold = self.config.get("memory.warnings.swap_percent", 50)
        if current_memory.swap_percent > swap_threshold:
            warnings_content.append(
                f"High swap usage: {current_memory.swap_percent:.1f}%"
            )

        # Memory trend warnings
        memory_stats = self.get_memory_statistics()
        trend = memory_stats["memory_trend"]

        if (
            trend["trend"] in ["rapidly_increasing", "increasing"]
            and trend["change_percent"] > 5
        ):
            warnings_content.append(
                f"Memory usage {trend['trend']}: +{trend['change_percent']:.1f}%"
            )

        return warnings_content
