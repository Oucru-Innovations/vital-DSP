"""
Optimized Advanced Data Loaders for Large-Scale Physiological Signal Processing

This module implements high-performance data loading strategies with dynamic
configuration and performance optimization.

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Optimized)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Callable, Generator, Dict, Any, List
import psutil
import time
from dataclasses import dataclass
from enum import Enum
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

from vitalDSP.utils.config_utilities.dynamic_config import (
    get_config,
    DynamicConfig,
    Environment,
)


class LoadingStrategy(Enum):
    """Loading strategy selection based on file size."""

    STANDARD = "standard"  # < threshold: Load entirely into memory
    CHUNKED = "chunked"  # threshold-2GB: Chunked loading
    MEMORY_MAPPED = "memory_mapped"  # > 2GB: Memory-mapped access


@dataclass
class ProgressInfo:
    """
    Progress information for data loading operations.
    """

    bytes_processed: int
    total_bytes: int
    chunks_processed: int
    total_chunks: int
    elapsed_time: float
    estimated_remaining: float
    current_chunk_size: int
    loading_strategy: str
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.bytes_processed / self.total_bytes) * 100

    @property
    def loading_speed_mbps(self) -> float:
        """Calculate loading speed in MB/s."""
        if self.elapsed_time == 0:
            return 0.0
        mb_processed = self.bytes_processed / (1024**2)
        return mb_processed / self.elapsed_time


class CancellationToken:
    """
    Token for cancelling long-running data loading operations.
    """

    def __init__(self):
        self._is_cancelled = False
        self._cancel_message = None
        self._lock = threading.Lock()

    def cancel(self, message: Optional[str] = None):
        """Cancel the operation."""
        with self._lock:
            self._is_cancelled = True
            self._cancel_message = message or "Operation cancelled by user"

    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        with self._lock:
            return self._is_cancelled

    def throw_if_cancelled(self):
        """Raise exception if operation is cancelled."""
        if self.is_cancelled():
            raise InterruptedError(self._cancel_message)

    def reset(self):
        """Reset cancellation state."""
        with self._lock:
            self._is_cancelled = False
            self._cancel_message = None


class OptimizedChunkedDataLoader:
    """
    Optimized chunked data loader with dynamic configuration.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        chunk_size: Union[int, str] = "auto",
        file_format: str = "csv",
        sampling_rate: Optional[float] = None,
        overlap_samples: int = None,
        cancellation_token: Optional[CancellationToken] = None,
        config: Optional[DynamicConfig] = None,
        **kwargs,
    ):
        """
        Initialize OptimizedChunkedDataLoader.
        """
        self.file_path = Path(file_path)
        self.file_format = file_format.lower()
        self.sampling_rate = sampling_rate
        self.cancellation_token = cancellation_token or CancellationToken()
        self.kwargs = kwargs
        self.config = config or get_config()

        # Set overlap samples from config if not provided
        self.overlap_samples = (
            overlap_samples
            if overlap_samples is not None
            else self.config.data_loader.overlap_samples_default
        )

        # Get file size
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.file_size = self.file_path.stat().st_size

        # Determine optimal chunk size
        if chunk_size == "auto":
            self.chunk_size = self._determine_optimal_chunk_size()
        else:
            self.chunk_size = chunk_size

        # Metadata
        self.metadata = {
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "chunk_size": self.chunk_size,
            "overlap_samples": self.overlap_samples,
            "config_used": self.config.environment.value,
        }

        # Performance tracking
        self._performance_stats = {
            "total_chunks_processed": 0,
            "total_bytes_processed": 0,
            "total_processing_time": 0.0,
            "avg_chunk_processing_time": 0.0,
            "peak_memory_usage_mb": 0.0,
        }

    def _determine_optimal_chunk_size(self) -> int:
        """
        Dynamically determine optimal chunk size based on configuration.
        """
        return self.config.get_optimal_chunk_size(
            file_size_mb=self.file_size / (1024**2),
            sampling_rate=self.sampling_rate
            or self.config.quality_screener.default_sampling_rate,
        )

    def load_chunks(
        self,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        start_chunk: int = 0,
        max_chunks: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load data in chunks with optimized progress tracking.
        """
        start_time = time.time()
        bytes_processed = 0
        chunks_processed = 0
        last_progress_update = start_time

        # Estimate total chunks
        estimated_total_chunks = self._estimate_total_chunks()

        try:
            if self.file_format == "csv":
                reader = pd.read_csv(
                    self.file_path, chunksize=self.chunk_size, **self.kwargs
                )
            elif self.file_format == "parquet":
                reader = self._parquet_chunked_reader()
            elif self.file_format == "hdf5":
                reader = self._hdf5_chunked_reader()
            else:
                raise ValueError(
                    f"Unsupported file format for chunking: {self.file_format}"
                )

            # Process chunks
            prev_chunk = None
            for chunk_idx, chunk in enumerate(reader):
                # Check cancellation
                self.cancellation_token.throw_if_cancelled()

                # Skip to start_chunk if resuming
                if chunk_idx < start_chunk:
                    continue

                # Handle overlap
                if self.overlap_samples > 0 and prev_chunk is not None:
                    overlap_data = prev_chunk.tail(self.overlap_samples)
                    chunk = pd.concat([overlap_data, chunk], ignore_index=True)

                # Update progress with throttling
                current_time = time.time()
                if (
                    progress_callback
                    and current_time - last_progress_update
                    >= self.config.data_loader.progress_update_interval
                ):

                    chunk_bytes = chunk.memory_usage(deep=True).sum()
                    bytes_processed += chunk_bytes
                    chunks_processed += 1

                    elapsed = current_time - start_time
                    remaining = self._estimate_remaining_time(
                        bytes_processed, self.file_size, elapsed
                    )

                    # Get system metrics
                    memory_usage = psutil.virtual_memory().percent
                    cpu_usage = psutil.cpu_percent()

                    progress_info = ProgressInfo(
                        bytes_processed=bytes_processed,
                        total_bytes=self.file_size,
                        chunks_processed=chunks_processed,
                        total_chunks=estimated_total_chunks,
                        elapsed_time=elapsed,
                        estimated_remaining=remaining,
                        current_chunk_size=len(chunk),
                        loading_strategy="optimized_chunked",
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_usage,
                    )
                    progress_callback(progress_info)
                    last_progress_update = current_time

                # Yield chunk
                yield chunk
                prev_chunk = chunk

                # Update performance stats
                self._update_performance_stats(
                    len(chunk), chunk.memory_usage(deep=True).sum()
                )

                # Check max_chunks limit
                if max_chunks and chunks_processed >= max_chunks:
                    break

                # Memory management
                if chunks_processed % 10 == 0:  # Every 10 chunks
                    gc.collect()

        except InterruptedError:
            warnings.warn("Data loading cancelled by user")
            raise
        except Exception as e:
            raise ValueError(f"Error during optimized chunked loading: {str(e)}")

    def _estimate_total_chunks(self) -> int:
        """Estimate total number of chunks."""
        bytes_per_sample = 8 * self.config.data_loader.bytes_per_sample_overhead
        chunk_bytes = self.chunk_size * bytes_per_sample
        return max(1, int(self.file_size / chunk_bytes))

    def _estimate_remaining_time(
        self, bytes_processed: int, total_bytes: int, elapsed: float
    ) -> float:
        """Estimate remaining time in seconds."""
        if bytes_processed == 0:
            return 0.0

        rate = bytes_processed / elapsed  # bytes per second
        remaining_bytes = total_bytes - bytes_processed
        return remaining_bytes / rate

    def _parquet_chunked_reader(self) -> Generator[pd.DataFrame, None, None]:
        """Create optimized chunked reader for Parquet files."""
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self.file_path)

        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            yield batch.to_pandas()

    def _hdf5_chunked_reader(self) -> Generator[pd.DataFrame, None, None]:
        """Create optimized chunked reader for HDF5 files."""
        import h5py

        key = self.kwargs.get("key", "data")

        with h5py.File(self.file_path, "r") as f:
            dataset = f[key]
            total_rows = dataset.shape[0]

            for start_idx in range(0, total_rows, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_rows)
                chunk_data = dataset[start_idx:end_idx]
                yield pd.DataFrame(chunk_data)

    def _update_performance_stats(self, chunk_size: int, chunk_bytes: int):
        """Update performance statistics."""
        self._performance_stats["total_chunks_processed"] += 1
        self._performance_stats["total_bytes_processed"] += chunk_bytes

        # Update peak memory usage
        current_memory = psutil.virtual_memory().used / (1024**2)
        if current_memory > self._performance_stats["peak_memory_usage_mb"]:
            self._performance_stats["peak_memory_usage_mb"] = current_memory

    def load_all(
        self, progress_callback: Optional[Callable[[ProgressInfo], None]] = None
    ) -> pd.DataFrame:
        """
        Load entire file using optimized chunked loading.
        """
        chunks = []
        for chunk in self.load_chunks(progress_callback=progress_callback):
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True)

    def get_info(self) -> Dict[str, Any]:
        """Get loader information with performance stats."""
        return {
            **self.metadata,
            "file_size_mb": self.file_size / (1024**2),
            "estimated_chunks": self._estimate_total_chunks(),
            "chunk_size_mb": (
                self.chunk_size * 8 * self.config.data_loader.bytes_per_sample_overhead
            )
            / (1024**2),
            "loading_strategy": "optimized_chunked",
            "performance_stats": self._performance_stats.copy(),
        }


class OptimizedMemoryMappedLoader:
    """
    Optimized memory-mapped data loader with dynamic configuration.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        dtype: Union[str, np.dtype] = "float64",
        mode: str = "r",
        shape: Optional[tuple] = None,
        offset: int = 0,
        cancellation_token: Optional[CancellationToken] = None,
        config: Optional[DynamicConfig] = None,
    ):
        """
        Initialize OptimizedMemoryMappedLoader.
        """
        self.file_path = Path(file_path)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self.offset = offset
        self.cancellation_token = cancellation_token or CancellationToken()
        self.config = config or get_config()

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.file_size = self.file_path.stat().st_size

        # Detect if it's a .npy file or raw binary
        self.is_npy_format = self.file_path.suffix == ".npy"

        if self.is_npy_format:
            # Load .npy file (shape and dtype auto-detected)
            self.mmap = np.load(self.file_path, mmap_mode=mode)
            self.shape = self.mmap.shape
            self.size = self.mmap.size
        else:
            # Raw binary file - shape must be provided
            if shape is None:
                # Try to infer shape from file size
                element_size = self.dtype.itemsize
                total_elements = (self.file_size - offset) // element_size
                shape = (total_elements,)

            self.shape = shape
            self.size = np.prod(shape)

            # Create memory map
            self.mmap = np.memmap(
                self.file_path, dtype=self.dtype, mode=mode, shape=shape, offset=offset
            )

        # Metadata
        self.metadata = {
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "size": self.size,
            "is_npy_format": self.is_npy_format,
            "config_used": self.config.environment.value,
        }

        # Performance tracking
        self._performance_stats = {
            "total_segments_accessed": 0,
            "total_bytes_accessed": 0,
            "peak_memory_usage_mb": 0.0,
        }

    def get_segment(self, start: int, end: int, copy: bool = False) -> np.ndarray:
        """
        Get a segment of the data with performance tracking.
        """
        self.cancellation_token.throw_if_cancelled()

        if start < 0 or end > self.size:
            raise ValueError(f"Invalid segment range: [{start}, {end})")

        segment = self.mmap[start:end]

        # Update performance stats
        segment_bytes = len(segment) * self.dtype.itemsize
        self._performance_stats["total_segments_accessed"] += 1
        self._performance_stats["total_bytes_accessed"] += segment_bytes

        if copy:
            return np.array(segment)  # Create a copy in memory
        return segment  # Return view (zero-copy)

    def iterate_chunks(
        self,
        chunk_size: int = None,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        copy: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data in optimized chunks.
        """
        if chunk_size is None:
            chunk_size = self.config.get_optimal_chunk_size(
                file_size_mb=self.file_size / (1024**2)
            )

        start_time = time.time()
        total_chunks = (self.size + chunk_size - 1) // chunk_size
        last_progress_update = start_time

        for chunk_idx in range(0, self.size, chunk_size):
            self.cancellation_token.throw_if_cancelled()

            end_idx = min(chunk_idx + chunk_size, self.size)
            chunk = self.mmap[chunk_idx:end_idx]

            if copy:
                chunk = np.array(chunk)

            # Update performance stats
            chunk_bytes = len(chunk) * self.dtype.itemsize
            self._performance_stats["total_segments_accessed"] += 1
            self._performance_stats["total_bytes_accessed"] += chunk_bytes

            # Progress update with throttling
            current_time = time.time()
            if (
                progress_callback
                and current_time - last_progress_update
                >= self.config.data_loader.progress_update_interval
            ):

                elapsed = current_time - start_time
                samples_processed = end_idx
                remaining = self._estimate_remaining(
                    samples_processed, self.size, elapsed
                )

                # Get system metrics
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()

                progress_info = ProgressInfo(
                    bytes_processed=samples_processed * self.dtype.itemsize,
                    total_bytes=self.size * self.dtype.itemsize,
                    chunks_processed=(chunk_idx // chunk_size) + 1,
                    total_chunks=total_chunks,
                    elapsed_time=elapsed,
                    estimated_remaining=remaining,
                    current_chunk_size=len(chunk),
                    loading_strategy="optimized_memory_mapped",
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                )
                progress_callback(progress_info)
                last_progress_update = current_time

            yield chunk

    def get_time_segment(
        self,
        start_time: float,
        end_time: float,
        sampling_rate: float,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Get segment by time range with optimized sampling rate handling.
        """
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)

        return self.get_segment(start_idx, end_idx, copy=copy)

    def _estimate_remaining(self, processed: int, total: int, elapsed: float) -> float:
        """Estimate remaining time."""
        if processed == 0:
            return 0.0
        rate = processed / elapsed
        remaining = total - processed
        return remaining / rate

    def get_info(self) -> Dict[str, Any]:
        """Get loader information with performance stats."""
        return {
            **self.metadata,
            "file_size_mb": self.file_size / (1024**2),
            "file_size_gb": self.file_size / (1024**3),
            "memory_footprint_mb": (self.size * self.dtype.itemsize) / (1024**2),
            "memory_footprint_gb": (self.size * self.dtype.itemsize) / (1024**3),
            "loading_strategy": "optimized_memory_mapped",
            "performance_stats": self._performance_stats.copy(),
        }

    def close(self):
        """Close the memory-mapped file."""
        if hasattr(self.mmap, "_mmap"):
            del self.mmap

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
        return False


def select_optimal_loader(
    file_path: Union[str, Path], config: Optional[DynamicConfig] = None, **kwargs
) -> Union[OptimizedChunkedDataLoader, OptimizedMemoryMappedLoader, None]:
    """
    Automatically select optimal loader based on file size and configuration.
    """
    config = config or get_config()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024**2)

    # Determine file format
    extension = file_path.suffix.lower()

    # Use configuration thresholds
    small_threshold = config.data_loader.small_file_threshold
    medium_threshold = config.data_loader.medium_file_threshold

    # For very large files, prefer memory mapping if binary format
    if file_size_mb > medium_threshold:
        if extension in config.data_loader.memory_map_supported_formats:
            return OptimizedMemoryMappedLoader(file_path, config=config, **kwargs)
        else:
            # Use chunked loading for text formats
            return OptimizedChunkedDataLoader(file_path, config=config, **kwargs)

    # For medium files, use chunked loading
    elif file_size_mb > small_threshold:
        return OptimizedChunkedDataLoader(file_path, config=config, **kwargs)

    # For small files, return None (use standard loading)
    else:
        return None


# Backward compatibility aliases
ChunkedDataLoader = OptimizedChunkedDataLoader
MemoryMappedLoader = OptimizedMemoryMappedLoader

# Example usage and tests
if __name__ == "__main__":
    print("Optimized Advanced Data Loaders Module")
    print("=" * 50)
    print("\nThis module provides optimized data loading")
    print("with dynamic configuration and performance optimization.")
    print("\nFeatures:")
    print("  - Dynamic configuration system")
    print("  - Performance monitoring")
    print("  - Memory optimization")
    print("  - Environment-based tuning")
