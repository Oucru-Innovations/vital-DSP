"""
Advanced Data Loaders for Large-Scale Physiological Signal Processing

This module implements high-performance data loading strategies for handling
large physiological datasets efficiently:

- ChunkedDataLoader: Adaptive chunking for medium-large files (100MB-2GB)
- MemoryMappedLoader: Zero-copy loading for very large files (>2GB)
- ProgressCallback: Real-time progress tracking and cancellation support

Features:
- Adaptive chunk sizing based on available memory
- Progress callbacks for UI integration
- Cancellation support for long-running operations
- Memory-efficient loading with minimal overhead
- Integration with existing DataLoader functionality

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Week 1)
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


class LoadingStrategy(Enum):
    """Loading strategy selection based on file size."""

    STANDARD = "standard"  # < 100MB: Load entirely into memory
    CHUNKED = "chunked"  # 100MB-2GB: Chunked loading
    MEMORY_MAPPED = "memory_mapped"  # > 2GB: Memory-mapped access


@dataclass
class ProgressInfo:
    """
    Progress information for data loading operations.

    Attributes:
        bytes_processed: Number of bytes processed so far
        total_bytes: Total bytes to process
        chunks_processed: Number of chunks processed
        total_chunks: Estimated total chunks
        elapsed_time: Time elapsed since start (seconds)
        estimated_remaining: Estimated time remaining (seconds)
        current_chunk_size: Size of current chunk in bytes
        loading_strategy: Strategy being used
    """

    bytes_processed: int
    total_bytes: int
    chunks_processed: int
    total_chunks: int
    elapsed_time: float
    estimated_remaining: float
    current_chunk_size: int
    loading_strategy: str

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

    Example:
        >>> token = CancellationToken()
        >>> loader = ChunkedDataLoader('large_file.csv', cancellation_token=token)
        >>> # In another thread:
        >>> token.cancel()
    """

    def __init__(self):
        self._is_cancelled = False
        self._cancel_message = None

    def cancel(self, message: Optional[str] = None):
        """Cancel the operation."""
        self._is_cancelled = True
        self._cancel_message = message or "Operation cancelled by user"

    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        return self._is_cancelled

    def throw_if_cancelled(self):
        """Raise exception if operation is cancelled."""
        if self._is_cancelled:
            raise InterruptedError(self._cancel_message)

    def reset(self):
        """Reset cancellation state."""
        self._is_cancelled = False
        self._cancel_message = None


class ChunkedDataLoader:
    """
    Adaptive chunked data loader for medium to large files (100MB-2GB).

    Features:
    - Adaptive chunk sizing based on available memory
    - Progress callbacks for UI integration
    - Cancellation support
    - Memory-efficient processing
    - Automatic chunk size optimization

    The loader automatically determines optimal chunk sizes based on:
    - Available system memory
    - File size
    - Number of CPU cores
    - Memory usage patterns

    Example:
        >>> def progress_callback(info: ProgressInfo):
        ...     print(f"Progress: {info.progress_percent:.1f}%")
        ...
        >>> loader = ChunkedDataLoader('large_ecg.csv', chunk_size='auto')
        >>> for chunk in loader.load_chunks(progress_callback=progress_callback):
        ...     process(chunk)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        chunk_size: Union[int, str] = "auto",
        file_format: str = "csv",
        sampling_rate: Optional[float] = None,
        overlap_samples: int = 0,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ):
        """
        Initialize ChunkedDataLoader.

        Args:
            file_path: Path to data file
            chunk_size: Chunk size in samples ('auto' for adaptive sizing)
            file_format: File format ('csv', 'parquet', 'hdf5', etc.)
            sampling_rate: Signal sampling rate in Hz
            overlap_samples: Number of samples to overlap between chunks
            cancellation_token: Token for cancellation support
            **kwargs: Additional format-specific parameters
        """
        self.file_path = Path(file_path)
        self.file_format = file_format.lower()
        self.sampling_rate = sampling_rate
        self.overlap_samples = overlap_samples
        self.cancellation_token = cancellation_token or CancellationToken()
        self.kwargs = kwargs

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
        }

    def _determine_optimal_chunk_size(self) -> int:
        """
        Adaptively determine optimal chunk size based on system resources.

        Strategy:
        1. Get available system memory
        2. Estimate memory per sample (8 bytes for float64)
        3. Use 10% of available memory per chunk
        4. Scale by number of CPU cores
        5. Ensure minimum and maximum bounds

        Returns:
            Optimal chunk size in samples
        """
        # Get system resources
        available_mem = psutil.virtual_memory().available
        cpu_count = psutil.cpu_count(logical=False) or 1

        # Estimate bytes per sample (assuming float64 + some overhead)
        bytes_per_sample = 12  # 8 bytes + 50% overhead for DataFrames

        # Target: Use 10% of available memory per chunk
        target_chunk_bytes = available_mem * 0.10

        # Calculate chunk size in samples
        chunk_size_samples = int(target_chunk_bytes / bytes_per_sample)

        # Scale by CPU cores (but not linearly)
        scaling_factor = min(cpu_count / 4, 2.0)  # Max 2x scaling
        chunk_size_samples = int(chunk_size_samples * scaling_factor)

        # Apply bounds
        min_chunk = 10000  # Minimum 10k samples
        max_chunk = 10000000  # Maximum 10M samples

        chunk_size_samples = max(min_chunk, min(chunk_size_samples, max_chunk))

        # If we have sampling rate, align to whole seconds
        if self.sampling_rate:
            seconds_per_chunk = chunk_size_samples / self.sampling_rate
            # Round to nearest 10 seconds
            seconds_per_chunk = round(seconds_per_chunk / 10) * 10
            chunk_size_samples = int(seconds_per_chunk * self.sampling_rate)

        return chunk_size_samples

    def load_chunks(
        self,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        start_chunk: int = 0,
        max_chunks: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load data in chunks with progress tracking.

        Args:
            progress_callback: Function to call with progress updates
            start_chunk: Starting chunk index (for resumption)
            max_chunks: Maximum number of chunks to load (None = all)

        Yields:
            DataFrames containing chunk data

        Raises:
            InterruptedError: If operation is cancelled
            ValueError: If file format is unsupported
        """
        start_time = time.time()
        bytes_processed = 0
        chunks_processed = 0

        # Estimate total chunks
        estimated_total_chunks = self._estimate_total_chunks()

        try:
            if self.file_format == "csv":
                reader = pd.read_csv(
                    self.file_path, chunksize=self.chunk_size, **self.kwargs
                )
            elif self.file_format == "parquet":
                # Parquet requires different approach
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

                # Update progress
                chunk_bytes = chunk.memory_usage(deep=True).sum()
                bytes_processed += chunk_bytes
                chunks_processed += 1

                if progress_callback:
                    elapsed = time.time() - start_time
                    remaining = self._estimate_remaining_time(
                        bytes_processed, self.file_size, elapsed
                    )

                    progress_info = ProgressInfo(
                        bytes_processed=bytes_processed,
                        total_bytes=self.file_size,
                        chunks_processed=chunks_processed,
                        total_chunks=estimated_total_chunks,
                        elapsed_time=elapsed,
                        estimated_remaining=remaining,
                        current_chunk_size=len(chunk),
                        loading_strategy="chunked",
                    )
                    progress_callback(progress_info)

                # Yield chunk
                yield chunk
                prev_chunk = chunk

                # Check max_chunks limit
                if max_chunks and chunks_processed >= max_chunks:
                    break

        except InterruptedError:
            warnings.warn("Data loading cancelled by user")
            raise
        except Exception as e:
            raise ValueError(f"Error during chunked loading: {str(e)}")

    def _estimate_total_chunks(self) -> int:
        """Estimate total number of chunks."""
        # Rough estimate: file_size / (chunk_size * bytes_per_sample)
        bytes_per_sample = 12  # Estimate
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
        """Create chunked reader for Parquet files."""
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self.file_path)

        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            yield batch.to_pandas()

    def _hdf5_chunked_reader(self) -> Generator[pd.DataFrame, None, None]:
        """Create chunked reader for HDF5 files."""
        import h5py

        key = self.kwargs.get("key", "data")

        with h5py.File(self.file_path, "r") as f:
            dataset = f[key]
            total_rows = dataset.shape[0]

            for start_idx in range(0, total_rows, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_rows)
                chunk_data = dataset[start_idx:end_idx]
                yield pd.DataFrame(chunk_data)

    def load_all(
        self, progress_callback: Optional[Callable[[ProgressInfo], None]] = None
    ) -> pd.DataFrame:
        """
        Load entire file using chunked loading, then concatenate.

        Args:
            progress_callback: Function to call with progress updates

        Returns:
            Complete DataFrame with all data
        """
        chunks = []
        for chunk in self.load_chunks(progress_callback=progress_callback):
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True)

    def get_info(self) -> Dict[str, Any]:
        """Get loader information."""
        return {
            **self.metadata,
            "file_size_mb": self.file_size / (1024**2),
            "estimated_chunks": self._estimate_total_chunks(),
            "chunk_size_mb": (self.chunk_size * 12) / (1024**2),
            "loading_strategy": "chunked",
        }


class MemoryMappedLoader:
    """
    Memory-mapped data loader for very large files (>2GB).

    Uses numpy memory mapping for zero-copy file access. Ideal for:
    - Files too large to fit in memory
    - Random access patterns
    - Repeated access to same file
    - Read-only operations on large datasets

    Features:
    - Zero-copy access (no loading into RAM)
    - Fast random access to any segment
    - Minimal memory footprint
    - Support for multiple array formats

    Limitations:
    - Requires files in binary format (NumPy .npy, raw binary)
    - Read-only access (use copy() for modifications)
    - Less efficient for sequential full-file scans

    Example:
        >>> loader = MemoryMappedLoader('huge_signal.npy')
        >>> # Access specific segment without loading entire file
        >>> segment = loader.get_segment(start=1000000, end=1100000)
        >>> # Or iterate in chunks
        >>> for chunk in loader.iterate_chunks(chunk_size=100000):
        ...     process(chunk)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        dtype: Union[str, np.dtype] = "float64",
        mode: str = "r",
        shape: Optional[tuple] = None,
        offset: int = 0,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        """
        Initialize MemoryMappedLoader.

        Args:
            file_path: Path to binary data file
            dtype: Data type of array elements
            mode: File access mode ('r' for read-only, 'r+' for read-write)
            shape: Shape of array (required for raw binary files)
            offset: Byte offset in file (default: 0)
            cancellation_token: Token for cancellation support
        """
        self.file_path = Path(file_path)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self.offset = offset
        self.cancellation_token = cancellation_token or CancellationToken()

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
        }

    def get_segment(self, start: int, end: int, copy: bool = False) -> np.ndarray:
        """
        Get a segment of the data without loading entire file.

        Args:
            start: Starting index
            end: Ending index
            copy: If True, return a copy (allows modifications)

        Returns:
            Array segment
        """
        self.cancellation_token.throw_if_cancelled()

        if start < 0 or end > self.size:
            raise ValueError(f"Invalid segment range: [{start}, {end})")

        segment = self.mmap[start:end]

        if copy:
            return np.array(segment)  # Create a copy in memory
        return segment  # Return view (zero-copy)

    def iterate_chunks(
        self,
        chunk_size: int = 100000,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        copy: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data in chunks.

        Args:
            chunk_size: Size of each chunk
            progress_callback: Function to call with progress updates
            copy: If True, yield copies instead of views

        Yields:
            Array chunks
        """
        start_time = time.time()
        total_chunks = (self.size + chunk_size - 1) // chunk_size

        for chunk_idx in range(0, self.size, chunk_size):
            self.cancellation_token.throw_if_cancelled()

            end_idx = min(chunk_idx + chunk_size, self.size)
            chunk = self.mmap[chunk_idx:end_idx]

            if copy:
                chunk = np.array(chunk)

            # Progress update
            if progress_callback:
                elapsed = time.time() - start_time
                samples_processed = end_idx
                remaining = self._estimate_remaining(
                    samples_processed, self.size, elapsed
                )

                progress_info = ProgressInfo(
                    bytes_processed=samples_processed * self.dtype.itemsize,
                    total_bytes=self.size * self.dtype.itemsize,
                    chunks_processed=(chunk_idx // chunk_size) + 1,
                    total_chunks=total_chunks,
                    elapsed_time=elapsed,
                    estimated_remaining=remaining,
                    current_chunk_size=len(chunk),
                    loading_strategy="memory_mapped",
                )
                progress_callback(progress_info)

            yield chunk

    def get_time_segment(
        self,
        start_time: float,
        end_time: float,
        sampling_rate: float,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Get segment by time range (requires sampling rate).

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            sampling_rate: Sampling rate in Hz
            copy: If True, return a copy

        Returns:
            Array segment for specified time range
        """
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)

        return self.get_segment(start_idx, end_idx, copy=copy)

    def _estimate_remaining(self, processed: int, total: int, elapsed: float) -> float:
        """Estimate remaining time."""
        if processed == 0 or elapsed == 0:
            return 0.0
        rate = processed / elapsed
        remaining = total - processed
        return remaining / rate

    def get_info(self) -> Dict[str, Any]:
        """Get loader information."""
        return {
            **self.metadata,
            "file_size_mb": self.file_size / (1024**2),
            "file_size_gb": self.file_size / (1024**3),
            "memory_footprint_mb": (self.size * self.dtype.itemsize) / (1024**2),
            "memory_footprint_gb": (self.size * self.dtype.itemsize) / (1024**3),
            "loading_strategy": "memory_mapped",
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
    file_path: Union[str, Path], **kwargs
) -> Union[ChunkedDataLoader, MemoryMappedLoader]:
    """
    Automatically select optimal loader based on file size and format.

    Strategy:
    - < 100MB: Return None (use standard DataLoader)
    - 100MB - 2GB: ChunkedDataLoader
    - > 2GB: MemoryMappedLoader (if binary format)

    Args:
        file_path: Path to data file
        **kwargs: Additional parameters for loader

    Returns:
        Optimal loader instance

    Example:
        >>> loader = select_optimal_loader('large_data.csv')
        >>> print(f"Using {loader.get_info()['loading_strategy']} strategy")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024**2)

    # Determine file format
    extension = file_path.suffix.lower()

    # For very large files, prefer memory mapping if binary format
    if file_size_mb > 2000:  # > 2GB
        if extension in [".npy", ".dat", ".bin"]:
            return MemoryMappedLoader(file_path, **kwargs)
        else:
            # Use chunked loading for text formats
            return ChunkedDataLoader(file_path, **kwargs)

    # For medium files, use chunked loading
    elif file_size_mb > 100:  # 100MB - 2GB
        return ChunkedDataLoader(file_path, **kwargs)

    # For small files, return None (use standard loading)
    else:
        return None


# Example usage and tests
if __name__ == "__main__":
    print("Advanced Data Loaders Module")
    print("=" * 50)
    print("\nThis module provides high-performance data loading")
    print("for large-scale physiological signal processing.")
    print("\nFeatures:")
    print("  - ChunkedDataLoader: Adaptive chunking (100MB-2GB)")
    print("  - MemoryMappedLoader: Zero-copy loading (>2GB)")
    print("  - Progress callbacks and cancellation support")
    print("  - Memory-efficient processing")
