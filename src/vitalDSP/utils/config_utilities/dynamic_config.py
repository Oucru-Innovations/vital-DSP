"""
Dynamic Configuration System for VitalDSP Core Infrastructure

This module provides a centralized configuration system for all VitalDSP components,
eliminating hard-coded values and enabling dynamic optimization.

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Optimized)
"""

import os
import json
from pathlib import Path

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import multiprocessing as mp
import numpy as np


class Environment(Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SystemResources:
    """System resource information."""

    cpu_count: int = field(default_factory=lambda: mp.cpu_count())
    memory_gb: float = field(
        default_factory=lambda: psutil.virtual_memory().total / (1024**3)
    )
    available_memory_gb: float = field(
        default_factory=lambda: psutil.virtual_memory().available / (1024**3)
    )
    disk_space_gb: float = field(
        default_factory=lambda: psutil.disk_usage("/").free / (1024**3)
    )

    @property
    def optimal_worker_count(self) -> int:
        """Calculate optimal worker count based on system resources."""
        # Use 75% of CPU cores, but cap at 16 for memory efficiency
        return min(int(self.cpu_count * 0.75), 16)

    @property
    def memory_per_worker_mb(self) -> int:
        """Calculate memory allocation per worker."""
        # Allocate 10% of available memory per worker, minimum 100MB
        memory_per_worker = (
            self.available_memory_gb * 1024 * 0.1
        ) / self.optimal_worker_count
        return max(int(memory_per_worker), 100)


@dataclass
class DataLoaderConfig:
    """Configuration for data loaders."""

    # File size thresholds (MB)
    small_file_threshold: float = 100.0
    medium_file_threshold: float = 2000.0

    # Chunk sizing parameters
    memory_usage_ratio: float = 0.10  # Use 10% of available memory per chunk
    min_chunk_size: int = 10000
    max_chunk_size: int = 10000000
    chunk_scaling_factor: float = 2.0
    chunk_scaling_divisor: int = 4

    # Memory mapping parameters
    memory_map_threshold_mb: float = 2000.0
    memory_map_supported_formats: list = field(
        default_factory=lambda: [".npy", ".dat", ".bin"]
    )

    # Progress tracking
    progress_update_interval: float = 0.1  # Update every 100ms
    progress_callback_threshold: int = 1000  # Minimum samples for progress callbacks

    # Performance optimization
    bytes_per_sample_overhead: float = 1.5  # 50% overhead for DataFrames
    overlap_samples_default: int = 0
    sampling_rate_alignment_seconds: float = (
        10.0  # Align chunks to 10-second boundaries
    )


@dataclass
class QualityScreenerConfig:
    """Configuration for quality screener."""

    # Default parameters
    default_sampling_rate: float = 100.0
    default_segment_duration: float = 10.0
    default_overlap_ratio: float = 0.1

    # Quality thresholds by signal type
    quality_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "generic": {
                "snr_min_db": 10.0,
                "artifact_max_ratio": 0.3,
                "baseline_max_drift": 0.5,
                "peak_detection_min_rate": 0.7,
                "frequency_score_min": 0.6,
                "temporal_consistency_min": 0.5,
                "overall_quality_min": 0.4,
            },
            "ecg": {
                "snr_min_db": 15.0,
                "artifact_max_ratio": 0.2,
                "baseline_max_drift": 0.5,
                "peak_detection_min_rate": 0.8,
                "frequency_score_min": 0.6,
                "temporal_consistency_min": 0.5,
                "overall_quality_min": 0.4,
            },
            "ppg": {
                "snr_min_db": 12.0,
                "artifact_max_ratio": 0.25,
                "baseline_max_drift": 0.3,
                "peak_detection_min_rate": 0.7,
                "frequency_score_min": 0.6,
                "temporal_consistency_min": 0.5,
                "overall_quality_min": 0.4,
            },
            "eeg": {
                "snr_min_db": 8.0,
                "artifact_max_ratio": 0.4,
                "baseline_max_drift": 0.6,
                "peak_detection_min_rate": 0.6,
                "frequency_score_min": 0.5,
                "temporal_consistency_min": 0.4,
                "overall_quality_min": 0.3,
            },
        }
    )

    # Quality level thresholds
    quality_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "excellent": 0.8,
            "good": 0.6,
            "fair": 0.4,
            "poor": 0.2,
            "unusable": 0.0,
        }
    )

    # Signal-specific parameters
    signal_parameters: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "ecg": {
                "max_heart_rate_bpm": 150,
                "min_heart_rate_bpm": 40,
                "expected_heart_rate_bpm": 72,
                "peak_detection_distance_factor": 0.4,
            },
            "ppg": {
                "max_pulse_rate_bpm": 120,
                "min_pulse_rate_bpm": 40,
                "expected_pulse_rate_bpm": 60,
                "peak_detection_distance_factor": 0.5,
            },
            "eeg": {
                "frequency_bands": {
                    "delta": (0.5, 4.0),
                    "theta": (4.0, 8.0),
                    "alpha": (8.0, 13.0),
                    "beta": (13.0, 30.0),
                    "gamma": (30.0, 100.0),
                }
            },
        }
    )

    # Statistical screening parameters
    statistical_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "outlier_sigma_multiplier": 3.0,
            "outlier_max_ratio": 0.1,
            "constant_max_ratio": 0.5,
            "jump_sigma_multiplier": 3.0,
            "jump_max_ratio": 0.05,
        }
    )

    # Frequency analysis parameters
    frequency_analysis: Dict[str, Any] = field(
        default_factory=lambda: {
            "spectral_centroid_normalization_factor": 4.0,
            "window_size_factor": 0.1,  # 10% of signal length
            "min_window_size": 100,
            "autocorr_lag": 1,
        }
    )

    # Performance optimization
    parallel_processing: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_by_default": True,
            "min_segments_for_parallel": 2,
            "max_workers_factor": 0.75,  # 75% of CPU cores
            "max_workers_cap": 16,
        }
    )


@dataclass
class ParallelPipelineConfig:
    """Configuration for parallel processing pipeline."""

    # Worker management
    max_workers_factor: float = 0.75  # 75% of CPU cores
    max_workers_cap: int = 16
    min_workers: int = 1

    # Memory management
    memory_limit_factor: float = 0.1  # 10% of available memory per worker
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

    # Performance monitoring
    performance_monitoring: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_monitoring": True,
            "metrics_history_limit": 1000,
            "performance_report_interval": 60.0,  # seconds
            "memory_usage_threshold_mb": 1000.0,
            "cpu_usage_threshold_percent": 80.0,
            "execution_time_threshold_seconds": 30.0,
        }
    )


@dataclass
class DynamicConfig:
    """Main configuration class for VitalDSP components."""

    environment: Environment = Environment.DEVELOPMENT
    system_resources: SystemResources = field(default_factory=SystemResources)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    quality_screener: QualityScreenerConfig = field(
        default_factory=QualityScreenerConfig
    )
    parallel_pipeline: ParallelPipelineConfig = field(
        default_factory=ParallelPipelineConfig
    )

    # Global settings
    debug_mode: bool = False
    log_level: str = "INFO"
    config_file_path: Optional[str] = None

    def __post_init__(self):
        """Post-initialization setup."""
        self._optimize_for_environment()
        self._validate_configuration()

    def _optimize_for_environment(self):
        """Optimize configuration based on environment."""
        if self.environment == Environment.PRODUCTION:
            # Production optimizations
            self.parallel_pipeline.max_workers_cap = min(
                self.system_resources.cpu_count, 8
            )
            self.data_loader.memory_usage_ratio = 0.05  # More conservative memory usage
            self.quality_screener.parallel_processing["max_workers_factor"] = 0.5
            self.debug_mode = False

        elif self.environment == Environment.DEVELOPMENT:
            # Development optimizations
            self.parallel_pipeline.max_workers_cap = min(
                self.system_resources.cpu_count, 4
            )
            self.data_loader.memory_usage_ratio = 0.15  # More aggressive memory usage
            self.debug_mode = True

        elif self.environment == Environment.TESTING:
            # Testing optimizations
            self.parallel_pipeline.max_workers_cap = 2
            self.data_loader.memory_usage_ratio = 0.20
            self.quality_screener.parallel_processing["enable_by_default"] = False

    def _validate_configuration(self):
        """Validate configuration parameters."""
        # Validate memory limits
        if not hasattr(self.parallel_pipeline, "memory_limit_mb"):
            # Calculate memory limit based on system resources
            memory_per_worker = self.system_resources.memory_per_worker_mb
            self.parallel_pipeline.memory_limit_mb = memory_per_worker

        if (
            self.parallel_pipeline.memory_limit_mb
            < self.parallel_pipeline.min_memory_limit_mb
        ):
            self.parallel_pipeline.memory_limit_mb = (
                self.parallel_pipeline.min_memory_limit_mb
            )

        if (
            self.parallel_pipeline.memory_limit_mb
            > self.parallel_pipeline.max_memory_limit_mb
        ):
            self.parallel_pipeline.memory_limit_mb = (
                self.parallel_pipeline.max_memory_limit_mb
            )

        # Validate chunk sizes
        if self.data_loader.min_chunk_size > self.data_loader.max_chunk_size:
            self.data_loader.min_chunk_size, self.data_loader.max_chunk_size = (
                self.data_loader.max_chunk_size,
                self.data_loader.min_chunk_size,
            )

        # Validate quality thresholds
        for signal_type, thresholds in self.quality_screener.quality_thresholds.items():
            for threshold_name, threshold_value in thresholds.items():
                if not (0.0 <= threshold_value <= 1.0):
                    if threshold_name == "snr_min_db":
                        continue  # SNR can be any positive value
                    else:
                        raise ValueError(
                            f"Invalid threshold {threshold_name} for {signal_type}: {threshold_value}"
                        )

    def get_optimal_chunk_size(
        self, file_size_mb: float, sampling_rate: float = 100.0
    ) -> int:
        """Calculate optimal chunk size based on file size and system resources."""
        # Base calculation on available memory
        available_memory_mb = self.system_resources.available_memory_gb * 1024
        target_chunk_mb = available_memory_mb * self.data_loader.memory_usage_ratio

        # Estimate samples per MB (assuming float64 + overhead)
        bytes_per_sample = 8 * self.data_loader.bytes_per_sample_overhead
        samples_per_mb = (1024 * 1024) / bytes_per_sample

        # Calculate chunk size in samples
        chunk_size_samples = int(target_chunk_mb * samples_per_mb)

        # Scale by CPU cores
        scaling_factor = min(
            self.system_resources.cpu_count / self.data_loader.chunk_scaling_divisor,
            self.data_loader.chunk_scaling_factor,
        )
        chunk_size_samples = int(chunk_size_samples * scaling_factor)

        # Apply bounds
        chunk_size_samples = max(
            self.data_loader.min_chunk_size,
            min(chunk_size_samples, self.data_loader.max_chunk_size),
        )

        # Align to sampling rate boundaries if provided
        if sampling_rate > 0:
            seconds_per_chunk = chunk_size_samples / sampling_rate
            aligned_seconds = (
                round(
                    seconds_per_chunk / self.data_loader.sampling_rate_alignment_seconds
                )
                * self.data_loader.sampling_rate_alignment_seconds
            )
            chunk_size_samples = int(aligned_seconds * sampling_rate)

        return chunk_size_samples

    def get_optimal_worker_count(self, task_count: int, data_size_mb: float) -> int:
        """Calculate optimal worker count based on workload and system resources."""
        # Base calculation on CPU cores
        cpu_workers = int(
            self.system_resources.cpu_count * self.parallel_pipeline.max_workers_factor
        )
        cpu_workers = min(cpu_workers, self.parallel_pipeline.max_workers_cap)

        # Adjust based on memory requirements
        memory_per_worker_mb = self.system_resources.memory_per_worker_mb
        memory_workers = (
            int(data_size_mb / memory_per_worker_mb)
            if memory_per_worker_mb > 0
            else cpu_workers
        )

        # Adjust based on task count
        task_workers = min(task_count, cpu_workers)

        # Use minimum of all constraints
        optimal_workers = min(cpu_workers, memory_workers, task_workers)

        # Ensure minimum workers
        return max(optimal_workers, self.parallel_pipeline.min_workers)

    def save_config(self, file_path: Union[str, Path]):
        """Save configuration to file."""
        config_dict = {
            "environment": self.environment.value,
            "system_resources": {
                "cpu_count": self.system_resources.cpu_count,
                "memory_gb": self.system_resources.memory_gb,
                "available_memory_gb": self.system_resources.available_memory_gb,
                "disk_space_gb": self.system_resources.disk_space_gb,
            },
            "data_loader": {
                "small_file_threshold": self.data_loader.small_file_threshold,
                "medium_file_threshold": self.data_loader.medium_file_threshold,
                "memory_usage_ratio": self.data_loader.memory_usage_ratio,
                "min_chunk_size": self.data_loader.min_chunk_size,
                "max_chunk_size": self.data_loader.max_chunk_size,
                "chunk_scaling_factor": self.data_loader.chunk_scaling_factor,
                "chunk_scaling_divisor": self.data_loader.chunk_scaling_divisor,
                "memory_map_threshold_mb": self.data_loader.memory_map_threshold_mb,
                "memory_map_supported_formats": self.data_loader.memory_map_supported_formats,
                "progress_update_interval": self.data_loader.progress_update_interval,
                "progress_callback_threshold": self.data_loader.progress_callback_threshold,
                "bytes_per_sample_overhead": self.data_loader.bytes_per_sample_overhead,
                "overlap_samples_default": self.data_loader.overlap_samples_default,
                "sampling_rate_alignment_seconds": self.data_loader.sampling_rate_alignment_seconds,
            },
            "quality_screener": {
                "default_sampling_rate": self.quality_screener.default_sampling_rate,
                "default_segment_duration": self.quality_screener.default_segment_duration,
                "default_overlap_ratio": self.quality_screener.default_overlap_ratio,
                "quality_thresholds": self.quality_screener.quality_thresholds,
                "quality_levels": self.quality_screener.quality_levels,
                "signal_parameters": self.quality_screener.signal_parameters,
                "statistical_thresholds": self.quality_screener.statistical_thresholds,
                "frequency_analysis": self.quality_screener.frequency_analysis,
                "parallel_processing": self.quality_screener.parallel_processing,
            },
            "parallel_pipeline": {
                "max_workers_factor": self.parallel_pipeline.max_workers_factor,
                "max_workers_cap": self.parallel_pipeline.max_workers_cap,
                "min_workers": self.parallel_pipeline.min_workers,
                "memory_limit_factor": self.parallel_pipeline.memory_limit_factor,
                "min_memory_limit_mb": self.parallel_pipeline.min_memory_limit_mb,
                "max_memory_limit_mb": self.parallel_pipeline.max_memory_limit_mb,
                "default_chunk_size": self.parallel_pipeline.default_chunk_size,
                "min_chunk_size": self.parallel_pipeline.min_chunk_size,
                "max_chunk_size": self.parallel_pipeline.max_chunk_size,
                "default_timeout_seconds": self.parallel_pipeline.default_timeout_seconds,
                "min_timeout_seconds": self.parallel_pipeline.min_timeout_seconds,
                "max_timeout_seconds": self.parallel_pipeline.max_timeout_seconds,
                "enable_caching_by_default": self.parallel_pipeline.enable_caching_by_default,
                "cache_compression_level": self.parallel_pipeline.cache_compression_level,
                "cache_max_size_mb": self.parallel_pipeline.cache_max_size_mb,
                "default_quality_threshold": self.parallel_pipeline.default_quality_threshold,
                "min_quality_threshold": self.parallel_pipeline.min_quality_threshold,
                "max_quality_threshold": self.parallel_pipeline.max_quality_threshold,
                "performance_monitoring": self.parallel_pipeline.performance_monitoring,
            },
            "global_settings": {
                "debug_mode": self.debug_mode,
                "log_level": self.log_level,
            },
        }

        file_path = Path(file_path)
        if (
            file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml"
        ) and YAML_AVAILABLE:
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(
        cls, file_path: Union[str, Path], environment: Optional[Environment] = None
    ) -> "DynamicConfig":
        """Load configuration from file."""
        file_path = Path(file_path)

        if (
            file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml"
        ) and YAML_AVAILABLE:
            with open(file_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, "r") as f:
                config_dict = json.load(f)

        # Create configuration object
        config = cls()

        # Set environment
        if environment:
            config.environment = environment
        elif "environment" in config_dict:
            config.environment = Environment(config_dict["environment"])

        # Load system resources
        if "system_resources" in config_dict:
            sr = config_dict["system_resources"]
            config.system_resources = SystemResources(
                cpu_count=sr.get("cpu_count", mp.cpu_count()),
                memory_gb=sr.get(
                    "memory_gb", psutil.virtual_memory().total / (1024**3)
                ),
                available_memory_gb=sr.get(
                    "available_memory_gb", psutil.virtual_memory().available / (1024**3)
                ),
                disk_space_gb=sr.get(
                    "disk_space_gb", psutil.disk_usage("/").free / (1024**3)
                ),
            )

        # Load data loader config
        if "data_loader" in config_dict:
            dl = config_dict["data_loader"]
            config.data_loader = DataLoaderConfig(
                small_file_threshold=dl.get("small_file_threshold", 100.0),
                medium_file_threshold=dl.get("medium_file_threshold", 2000.0),
                memory_usage_ratio=dl.get("memory_usage_ratio", 0.10),
                min_chunk_size=dl.get("min_chunk_size", 10000),
                max_chunk_size=dl.get("max_chunk_size", 10000000),
                chunk_scaling_factor=dl.get("chunk_scaling_factor", 2.0),
                chunk_scaling_divisor=dl.get("chunk_scaling_divisor", 4),
                memory_map_threshold_mb=dl.get("memory_map_threshold_mb", 2000.0),
                memory_map_supported_formats=dl.get(
                    "memory_map_supported_formats", [".npy", ".dat", ".bin"]
                ),
                progress_update_interval=dl.get("progress_update_interval", 0.1),
                progress_callback_threshold=dl.get("progress_callback_threshold", 1000),
                bytes_per_sample_overhead=dl.get("bytes_per_sample_overhead", 1.5),
                overlap_samples_default=dl.get("overlap_samples_default", 0),
                sampling_rate_alignment_seconds=dl.get(
                    "sampling_rate_alignment_seconds", 10.0
                ),
            )

        # Load quality screener config
        if "quality_screener" in config_dict:
            qs = config_dict["quality_screener"]
            config.quality_screener = QualityScreenerConfig(
                default_sampling_rate=qs.get("default_sampling_rate", 100.0),
                default_segment_duration=qs.get("default_segment_duration", 10.0),
                default_overlap_ratio=qs.get("default_overlap_ratio", 0.1),
                quality_thresholds=qs.get("quality_thresholds", {}),
                quality_levels=qs.get("quality_levels", {}),
                signal_parameters=qs.get("signal_parameters", {}),
                statistical_thresholds=qs.get("statistical_thresholds", {}),
                frequency_analysis=qs.get("frequency_analysis", {}),
                parallel_processing=qs.get("parallel_processing", {}),
            )

        # Load parallel pipeline config
        if "parallel_pipeline" in config_dict:
            pp = config_dict["parallel_pipeline"]
            config.parallel_pipeline = ParallelPipelineConfig(
                max_workers_factor=pp.get("max_workers_factor", 0.75),
                max_workers_cap=pp.get("max_workers_cap", 16),
                min_workers=pp.get("min_workers", 1),
                memory_limit_factor=pp.get("memory_limit_factor", 0.1),
                min_memory_limit_mb=pp.get("min_memory_limit_mb", 100),
                max_memory_limit_mb=pp.get("max_memory_limit_mb", 2000),
                default_chunk_size=pp.get("default_chunk_size", 10000),
                min_chunk_size=pp.get("min_chunk_size", 1000),
                max_chunk_size=pp.get("max_chunk_size", 100000),
                default_timeout_seconds=pp.get("default_timeout_seconds", 300),
                min_timeout_seconds=pp.get("min_timeout_seconds", 30),
                max_timeout_seconds=pp.get("max_timeout_seconds", 3600),
                enable_caching_by_default=pp.get("enable_caching_by_default", True),
                cache_compression_level=pp.get("cache_compression_level", 6),
                cache_max_size_mb=pp.get("cache_max_size_mb", 1000),
                default_quality_threshold=pp.get("default_quality_threshold", 0.4),
                min_quality_threshold=pp.get("min_quality_threshold", 0.0),
                max_quality_threshold=pp.get("max_quality_threshold", 1.0),
                performance_monitoring=pp.get("performance_monitoring", {}),
            )

        # Load global settings
        if "global_settings" in config_dict:
            gs = config_dict["global_settings"]
            config.debug_mode = gs.get("debug_mode", False)
            config.log_level = gs.get("log_level", "INFO")

        config.config_file_path = str(file_path)
        config._optimize_for_environment()
        config._validate_configuration()

        return config

    @classmethod
    def from_environment(
        cls, environment: Environment = Environment.DEVELOPMENT
    ) -> "DynamicConfig":
        """Create configuration from environment variables."""
        config = cls(environment=environment)

        # Override with environment variables if present
        env_vars = {
            "VITALDSP_DEBUG": os.getenv("VITALDSP_DEBUG", "").lower() == "true",
            "VITALDSP_LOG_LEVEL": os.getenv("VITALDSP_LOG_LEVEL", "INFO"),
            "VITALDSP_MAX_WORKERS": int(os.getenv("VITALDSP_MAX_WORKERS", "0")) or None,
            "VITALDSP_MEMORY_LIMIT_MB": int(os.getenv("VITALDSP_MEMORY_LIMIT_MB", "0"))
            or None,
            "VITALDSP_CHUNK_SIZE": int(os.getenv("VITALDSP_CHUNK_SIZE", "0")) or None,
            "VITALDSP_QUALITY_THRESHOLD": float(
                os.getenv("VITALDSP_QUALITY_THRESHOLD", "0")
            )
            or None,
        }

        if env_vars["VITALDSP_DEBUG"]:
            config.debug_mode = True

        if env_vars["VITALDSP_LOG_LEVEL"]:
            config.log_level = env_vars["VITALDSP_LOG_LEVEL"]

        if env_vars["VITALDSP_MAX_WORKERS"]:
            config.parallel_pipeline.max_workers_cap = env_vars["VITALDSP_MAX_WORKERS"]

        if env_vars["VITALDSP_MEMORY_LIMIT_MB"]:
            config.parallel_pipeline.max_memory_limit_mb = env_vars[
                "VITALDSP_MEMORY_LIMIT_MB"
            ]

        if env_vars["VITALDSP_CHUNK_SIZE"]:
            config.parallel_pipeline.default_chunk_size = env_vars[
                "VITALDSP_CHUNK_SIZE"
            ]

        if env_vars["VITALDSP_QUALITY_THRESHOLD"]:
            config.parallel_pipeline.default_quality_threshold = env_vars[
                "VITALDSP_QUALITY_THRESHOLD"
            ]

        config._optimize_for_environment()
        config._validate_configuration()

        return config


# Global configuration instance
_config_instance: Optional[DynamicConfig] = None


def get_config() -> DynamicConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DynamicConfig.from_environment()
    return _config_instance


def set_config(config: DynamicConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config


def reset_config():
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None


# Example usage and tests
if __name__ == "__main__":
    print("Dynamic Configuration System")
    print("=" * 50)
    print("\nThis module provides centralized configuration")
    print("for all VitalDSP components.")
    print("\nFeatures:")
    print("  - Environment-based optimization")
    print("  - Dynamic parameter calculation")
    print("  - Configuration persistence")
    print("  - Environment variable support")

    # Example usage
    config = DynamicConfig.from_environment(Environment.DEVELOPMENT)
    print(f"\nOptimal chunk size for 1GB file: {config.get_optimal_chunk_size(1000)}")
    print(
        f"Optimal worker count for 100 tasks: {config.get_optimal_worker_count(100, 500)}"
    )


# Create DynamicConfigManager alias for backward compatibility
class DynamicConfigManager:
    """Dynamic Configuration Manager for backward compatibility."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self._config = DynamicConfig.from_environment()
        if config_file:
            self._config = DynamicConfig.load_config(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        # Check user preferences first
        if hasattr(self, "_user_preferences") and key in self._user_preferences:
            return self._user_preferences[key]

        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default

    def set_user_preference(self, key: str, value: Any) -> None:
        """Set user preference."""
        # For now, just store in a simple dict
        if not hasattr(self, "_user_preferences"):
            self._user_preferences = {}
        self._user_preferences[key] = value

    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "environment": self._config.environment.value,
            "cpu_count": self._config.system_resources.cpu_count,
            "memory_gb": self._config.system_resources.memory_gb,
            "user_preferences": getattr(self, "_user_preferences", {}),
        }
