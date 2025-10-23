"""
Enhanced Data Service for vitalDSP Webapp - Phase 3A Implementation

This module implements the enhanced data service with chunked loading, memory-mapped access,
and progressive loading capabilities specifically designed for the webapp environment.

Features:
- ChunkedDataService: LRU cache with adaptive chunking for medium-large files
- MemoryMappedDataService: Zero-copy access for very large files
- ProgressiveDataLoader: Background processing with real-time updates
- Integration with existing vitalDSP data loaders
- Webapp-specific optimizations and memory management

Author: vitalDSP Development Team
Date: January 11, 2025
Phase: 3A - Core Infrastructure Enhancement (Week 1)
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Generator, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict
import psutil
import gc
from queue import Queue, Empty
import json
import hashlib
from datetime import datetime

# Add vitalDSP to path for imports
current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

try:
    from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
    from vitalDSP.utils.core_infrastructure.data_loaders import (
        ChunkedDataLoader,
        MemoryMappedLoader,
        ProgressInfo,
        CancellationToken,
        select_optimal_loader,
    )
    from vitalDSP.utils.core_infrastructure.optimized_memory_manager import (
        OptimizedMemoryManager,
    )

    VITALDSP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"vitalDSP modules not available: {e}")
    VITALDSP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LoadingStrategy(Enum):
    """Loading strategy selection based on file size and webapp requirements."""

    STANDARD = "standard"  # < 50MB: Standard loading
    CHUNKED = "chunked"  # 50MB-500MB: Chunked loading with cache
    MEMORY_MAPPED = "memory_mapped"  # > 500MB: Memory-mapped access
    PROGRESSIVE = "progressive"  # Any size: Progressive background loading


class FileSizeWarning(Enum):
    """File size warning levels for user feedback."""

    NONE = "none"  # < 50MB: No warning
    INFO = "info"  # 50-200MB: Informational
    WARNING = "warning"  # 200MB-1GB: Warning
    CRITICAL = "critical"  # > 1GB: Critical (may take long time)


@dataclass
class DataSegment:
    """Data segment with metadata for webapp processing."""

    data: Union[np.ndarray, pd.DataFrame]
    start_time: float
    end_time: float
    sampling_rate: float
    segment_id: str
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time

    @property
    def sample_count(self) -> int:
        """Get number of samples in segment."""
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        return len(self.data)


@dataclass
class FileAnalysis:
    """File analysis results for user warnings and time estimation."""

    file_path: str
    file_size_bytes: int
    file_size_mb: float
    warning_level: FileSizeWarning
    recommended_strategy: LoadingStrategy
    estimated_load_time_seconds: float
    warning_message: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": self.file_size_mb,
            "warning_level": self.warning_level.value,
            "recommended_strategy": self.recommended_strategy.value,
            "estimated_load_time_seconds": self.estimated_load_time_seconds,
            "warning_message": self.warning_message,
            "recommendations": self.recommendations,
        }


@dataclass
class LoadingProgress:
    """Enhanced progress information for webapp UI."""

    task_id: str
    progress_percent: float
    bytes_processed: int
    total_bytes: int
    chunks_processed: int
    total_chunks: int
    elapsed_time: float
    estimated_remaining: float
    current_chunk_size: int
    loading_strategy: str
    status: str  # 'loading', 'processing', 'completed', 'error'
    message: str
    data_preview: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class LRUCache:
    """LRU Cache implementation for data chunks."""

    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.access_times = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                return value
            return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.access_times.pop(oldest_key, None)

            self.cache[key] = value
            self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        total_size = 0
        for value in self.cache.values():
            if isinstance(value, pd.DataFrame):
                total_size += value.memory_usage(deep=True).sum()
            elif isinstance(value, np.ndarray):
                total_size += value.nbytes
            else:
                total_size += sys.getsizeof(value)
        return total_size


class ChunkedDataService:
    """
    Enhanced chunked data service with LRU cache and webapp optimizations.

    Features:
    - LRU cache for frequently accessed chunks
    - Adaptive chunk sizing based on available memory
    - Progress callbacks for webapp UI
    - Memory usage monitoring and optimization
    - Integration with vitalDSP ChunkedDataLoader
    """

    def __init__(self, max_cache_size: int = 100, max_memory_mb: int = 200):
        """
        Initialize chunked data service.

        Args:
            max_cache_size: Maximum number of chunks to cache
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_cache_size = max_cache_size
        self.max_memory_mb = max_memory_mb
        self.chunk_cache = LRUCache(max_cache_size)
        self.metadata_cache = {}
        self.loading_tasks = {}
        self._lock = threading.Lock()

        # Performance tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "chunks_loaded": 0,
            "total_loading_time": 0.0,
            "memory_optimizations": 0,
        }

        logger.info(
            f"ChunkedDataService initialized with {max_cache_size} cache size, {max_memory_mb}MB memory limit"
        )

    def load_data_chunked(
        self,
        file_path: str,
        chunk_index: int = 0,
        chunk_size: Union[int, str] = "auto",
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data chunk with caching and progress tracking.

        Args:
            file_path: Path to data file
            chunk_index: Index of chunk to load
            chunk_size: Size of chunks ('auto' for adaptive sizing)
            progress_callback: Callback for progress updates
            task_id: Task ID for progress tracking

        Returns:
            DataFrame containing chunk data
        """
        cache_key = f"{file_path}_{chunk_index}_{chunk_size}"

        # Check cache first
        cached_chunk = self.chunk_cache.get(cache_key)
        if cached_chunk is not None:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for chunk {chunk_index} of {file_path}")
            return cached_chunk

        self.stats["cache_misses"] += 1
        start_time = time.time()

        try:
            # Use vitalDSP ChunkedDataLoader if available
            if VITALDSP_AVAILABLE:
                loader = ChunkedDataLoader(
                    file_path=file_path,
                    chunk_size=chunk_size,
                    file_format=self._detect_format(file_path),
                )

                # Load specific chunk
                chunks = list(loader.load_chunks(max_chunks=chunk_index + 1))
                if chunk_index < len(chunks):
                    chunk = chunks[chunk_index]
                else:
                    raise IndexError(f"Chunk index {chunk_index} out of range")
            else:
                # Fallback to pandas chunking
                chunk = self._load_chunk_pandas(file_path, chunk_index, chunk_size)

            # Cache the chunk
            self.chunk_cache.put(cache_key, chunk)
            self.stats["chunks_loaded"] += 1

            loading_time = time.time() - start_time
            self.stats["total_loading_time"] += loading_time

            # Update progress if callback provided
            if progress_callback and task_id:
                progress = LoadingProgress(
                    task_id=task_id,
                    progress_percent=min(
                        100.0, (chunk_index + 1) * 10
                    ),  # Rough estimate
                    bytes_processed=chunk.memory_usage(deep=True).sum(),
                    total_bytes=Path(file_path).stat().st_size,
                    chunks_processed=chunk_index + 1,
                    total_chunks=10,  # Rough estimate
                    elapsed_time=loading_time,
                    estimated_remaining=0.0,
                    current_chunk_size=len(chunk),
                    loading_strategy="chunked",
                    status="completed",
                    message=f"Loaded chunk {chunk_index + 1}",
                )
                progress_callback(progress)

            logger.debug(f"Loaded chunk {chunk_index} in {loading_time:.3f}s")
            return chunk

        except Exception as e:
            logger.error(f"Error loading chunk {chunk_index}: {e}")
            if progress_callback and task_id:
                progress = LoadingProgress(
                    task_id=task_id,
                    progress_percent=0.0,
                    bytes_processed=0,
                    total_bytes=0,
                    chunks_processed=0,
                    total_chunks=0,
                    elapsed_time=time.time() - start_time,
                    estimated_remaining=0.0,
                    current_chunk_size=0,
                    loading_strategy="chunked",
                    status="error",
                    message=f"Error loading chunk: {str(e)}",
                )
                progress_callback(progress)
            raise

    def get_data_preview(
        self,
        file_path: str,
        preview_size: int = 1000,
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get data preview without loading entire file.

        Args:
            file_path: Path to data file
            preview_size: Number of rows to preview
            progress_callback: Callback for progress updates
            task_id: Task ID for progress tracking

        Returns:
            DataFrame with preview data
        """
        cache_key = f"preview_{file_path}_{preview_size}"

        # Check cache first
        cached_preview = self.chunk_cache.get(cache_key)
        if cached_preview is not None:
            self.stats["cache_hits"] += 1
            return cached_preview

        self.stats["cache_misses"] += 1
        start_time = time.time()

        try:
            # Load preview using pandas
            if file_path.endswith(".csv"):
                preview = pd.read_csv(file_path, nrows=preview_size)
            elif file_path.endswith(".parquet"):
                preview = pd.read_parquet(file_path).head(preview_size)
            else:
                # Fallback to first chunk
                preview = self.load_data_chunked(file_path, 0, preview_size)

            # Cache the preview
            self.chunk_cache.put(cache_key, preview)

            # Update progress if callback provided
            if progress_callback and task_id:
                progress = LoadingProgress(
                    task_id=task_id,
                    progress_percent=100.0,
                    bytes_processed=preview.memory_usage(deep=True).sum(),
                    total_bytes=Path(file_path).stat().st_size,
                    chunks_processed=1,
                    total_chunks=1,
                    elapsed_time=time.time() - start_time,
                    estimated_remaining=0.0,
                    current_chunk_size=len(preview),
                    loading_strategy="preview",
                    status="completed",
                    message=f"Preview loaded: {len(preview)} rows",
                )
                progress_callback(progress)

            logger.debug(f"Loaded preview in {time.time() - start_time:.3f}s")
            return preview

        except Exception as e:
            logger.error(f"Error loading preview: {e}")
            if progress_callback and task_id:
                progress = LoadingProgress(
                    task_id=task_id,
                    progress_percent=0.0,
                    bytes_processed=0,
                    total_bytes=0,
                    chunks_processed=0,
                    total_chunks=0,
                    elapsed_time=time.time() - start_time,
                    estimated_remaining=0.0,
                    current_chunk_size=0,
                    loading_strategy="preview",
                    status="error",
                    message=f"Error loading preview: {str(e)}",
                )
                progress_callback(progress)
            raise

    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        ext = Path(file_path).suffix.lower()
        format_map = {
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".xlsx": "excel",
            ".xls": "excel",
        }
        return format_map.get(ext, "csv")

    def _load_chunk_pandas(
        self, file_path: str, chunk_index: int, chunk_size: int
    ) -> pd.DataFrame:
        """Fallback chunk loading using pandas."""
        skip_rows = chunk_index * chunk_size
        return pd.read_csv(file_path, skiprows=skip_rows, nrows=chunk_size)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            "cache_size": self.chunk_cache.size(),
            "cache_memory_mb": self.chunk_cache.memory_usage() / (1024**2),
            "cache_hit_rate": self.stats["cache_hits"]
            / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]),
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.chunk_cache.clear()
        self.metadata_cache.clear()
        logger.info("Cache cleared")


class MemoryMappedDataService:
    """
    Enhanced memory-mapped data service for very large files.

    Features:
    - Zero-copy access to large files
    - Segment-based access with time indexing
    - Memory usage monitoring
    - Integration with vitalDSP MemoryMappedLoader
    """

    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize memory-mapped data service.

        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.active_maps = {}
        self.segment_cache = LRUCache(50)  # Smaller cache for segments
        self._lock = threading.Lock()

        # Performance tracking
        self.stats = {
            "files_mapped": 0,
            "segments_accessed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_usage_mb": 0.0,
        }

        logger.info(
            f"MemoryMappedDataService initialized with {max_memory_mb}MB memory limit"
        )

    def map_file(
        self,
        file_path: str,
        dtype: str = "float64",
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Map a file for memory-mapped access.

        Args:
            file_path: Path to data file
            dtype: Data type of array elements
            progress_callback: Callback for progress updates
            task_id: Task ID for progress tracking

        Returns:
            Map ID for accessing the mapped file
        """
        map_id = hashlib.md5(f"{file_path}_{dtype}".encode()).hexdigest()[:16]

        with self._lock:
            if map_id in self.active_maps:
                logger.debug(f"File already mapped: {map_id}")
                return map_id

            try:
                if VITALDSP_AVAILABLE:
                    # Use vitalDSP MemoryMappedLoader
                    loader = MemoryMappedLoader(
                        file_path=file_path, dtype=dtype, mode="r"
                    )
                    self.active_maps[map_id] = {
                        "loader": loader,
                        "file_path": file_path,
                        "dtype": dtype,
                        "shape": loader.shape,
                        "size": loader.size,
                        "created_at": time.time(),
                    }
                else:
                    # Fallback to numpy memory mapping
                    mmap = np.memmap(file_path, dtype=dtype, mode="r")
                    self.active_maps[map_id] = {
                        "loader": None,
                        "mmap": mmap,
                        "file_path": file_path,
                        "dtype": dtype,
                        "shape": mmap.shape,
                        "size": mmap.size,
                        "created_at": time.time(),
                    }

                self.stats["files_mapped"] += 1

                # Update progress if callback provided
                if progress_callback and task_id:
                    progress = LoadingProgress(
                        task_id=task_id,
                        progress_percent=100.0,
                        bytes_processed=0,
                        total_bytes=0,
                        chunks_processed=1,
                        total_chunks=1,
                        elapsed_time=0.0,
                        estimated_remaining=0.0,
                        current_chunk_size=0,
                        loading_strategy="memory_mapped",
                        status="completed",
                        message=f"File mapped: {self.active_maps[map_id]['size']} samples",
                    )
                    progress_callback(progress)

                logger.info(f"File mapped successfully: {map_id}")
                return map_id

            except Exception as e:
                logger.error(f"Error mapping file {file_path}: {e}")
                if progress_callback and task_id:
                    progress = LoadingProgress(
                        task_id=task_id,
                        progress_percent=0.0,
                        bytes_processed=0,
                        total_bytes=0,
                        chunks_processed=0,
                        total_chunks=0,
                        elapsed_time=0.0,
                        estimated_remaining=0.0,
                        current_chunk_size=0,
                        loading_strategy="memory_mapped",
                        status="error",
                        message=f"Error mapping file: {str(e)}",
                    )
                    progress_callback(progress)
                raise

    def get_segment(
        self,
        map_id: str,
        start_idx: int,
        end_idx: int,
        sampling_rate: Optional[float] = None,
    ) -> DataSegment:
        """
        Get a segment from mapped file.

        Args:
            map_id: Map ID from map_file()
            start_idx: Starting index
            end_idx: Ending index
            sampling_rate: Sampling rate for time calculation

        Returns:
            DataSegment with segment data and metadata
        """
        cache_key = f"{map_id}_{start_idx}_{end_idx}"

        # Check cache first
        cached_segment = self.segment_cache.get(cache_key)
        if cached_segment is not None:
            self.stats["cache_hits"] += 1
            return cached_segment

        self.stats["cache_misses"] += 1

        with self._lock:
            if map_id not in self.active_maps:
                raise ValueError(f"Map ID {map_id} not found")

            map_info = self.active_maps[map_id]

            try:
                if map_info["loader"] is not None:
                    # Use vitalDSP loader
                    data = map_info["loader"].get_segment(start_idx, end_idx, copy=True)
                else:
                    # Use numpy memory map
                    data = np.array(map_info["mmap"][start_idx:end_idx])

                # Calculate time range
                if sampling_rate:
                    start_time = start_idx / sampling_rate
                    end_time = end_idx / sampling_rate
                else:
                    start_time = start_idx
                    end_time = end_idx

                segment = DataSegment(
                    data=data,
                    start_time=start_time,
                    end_time=end_time,
                    sampling_rate=sampling_rate or 1.0,
                    segment_id=cache_key,
                    metadata={
                        "map_id": map_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "file_path": map_info["file_path"],
                    },
                )

                # Cache the segment
                self.segment_cache.put(cache_key, segment)
                self.stats["segments_accessed"] += 1

                logger.debug(
                    f"Accessed segment {start_idx}-{end_idx} from map {map_id}"
                )
                return segment

            except Exception as e:
                logger.error(f"Error accessing segment {start_idx}-{end_idx}: {e}")
                raise

    def get_time_segment(
        self, map_id: str, start_time: float, end_time: float, sampling_rate: float
    ) -> DataSegment:
        """
        Get segment by time range.

        Args:
            map_id: Map ID from map_file()
            start_time: Start time in seconds
            end_time: End time in seconds
            sampling_rate: Sampling rate in Hz

        Returns:
            DataSegment with segment data and metadata
        """
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)

        return self.get_segment(map_id, start_idx, end_idx, sampling_rate)

    def get_statistics(self, map_id: str) -> Dict[str, Any]:
        """Get file statistics without loading data."""
        with self._lock:
            if map_id not in self.active_maps:
                raise ValueError(f"Map ID {map_id} not found")

            map_info = self.active_maps[map_id]
            return {
                "file_path": map_info["file_path"],
                "shape": map_info["shape"],
                "size": map_info["size"],
                "dtype": map_info["dtype"],
                "file_size_mb": Path(map_info["file_path"]).stat().st_size / (1024**2),
                "created_at": map_info["created_at"],
            }

    def unmap_file(self, map_id: str) -> None:
        """Unmap a file and free resources."""
        with self._lock:
            if map_id in self.active_maps:
                map_info = self.active_maps[map_id]

                # Close vitalDSP loader if available
                if map_info["loader"] is not None:
                    map_info["loader"].close()

                del self.active_maps[map_id]
                logger.info(f"File unmapped: {map_id}")

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "active_maps": len(self.active_maps),
            "cache_size": self.segment_cache.size(),
            "cache_memory_mb": self.segment_cache.memory_usage() / (1024**2),
        }

    def cleanup(self) -> None:
        """Cleanup all mapped files and cache."""
        with self._lock:
            for map_id in list(self.active_maps.keys()):
                self.unmap_file(map_id)
            self.segment_cache.clear()
        logger.info("MemoryMappedDataService cleaned up")


class ProgressiveDataLoader:
    """
    Progressive data loader with background processing and real-time updates.

    Features:
    - Background loading with threading
    - Real-time progress updates
    - Queue-based request handling
    - Integration with ChunkedDataService and MemoryMappedDataService
    - Automatic strategy selection based on file size
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize progressive data loader.

        Args:
            max_workers: Maximum number of background workers
        """
        self.max_workers = max_workers
        self.loading_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.running = False
        self.task_callbacks = {}

        # Services
        self.chunked_service = ChunkedDataService()
        self.memory_mapped_service = MemoryMappedDataService()

        # Performance tracking
        self.stats = {
            "tasks_processed": 0,
            "background_loads": 0,
            "queue_size": 0,
            "average_load_time": 0.0,
        }

        logger.info(f"ProgressiveDataLoader initialized with {max_workers} workers")

    def start(self) -> None:
        """Start background workers."""
        if self.running:
            return

        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.max_workers} background workers")

    def stop(self) -> None:
        """Stop background workers."""
        self.running = False

        # Send shutdown signals
        for _ in range(self.max_workers):
            self.loading_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()
        logger.info("Background workers stopped")

    def request_data_segment(
        self,
        file_path: str,
        start_time: float,
        end_time: float,
        sampling_rate: float,
        callback: Callable[[DataSegment], None],
        task_id: Optional[str] = None,
    ) -> str:
        """
        Request data segment with callback when ready.

        Args:
            file_path: Path to data file
            start_time: Start time in seconds
            end_time: End time in seconds
            sampling_rate: Sampling rate in Hz
            callback: Function to call when data is ready
            task_id: Task ID for tracking

        Returns:
            Request ID for tracking
        """
        if not self.running:
            self.start()

        request_id = task_id or f"req_{int(time.time() * 1000)}"

        request = {
            "request_id": request_id,
            "file_path": file_path,
            "start_time": start_time,
            "end_time": end_time,
            "sampling_rate": sampling_rate,
            "callback": callback,
            "created_at": time.time(),
        }

        self.loading_queue.put(request)
        self.task_callbacks[request_id] = callback

        logger.debug(f"Queued request {request_id} for {file_path}")
        return request_id

    def request_data_preview(
        self,
        file_path: str,
        callback: Callable[[pd.DataFrame], None],
        preview_size: int = 1000,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Request data preview with callback when ready.

        Args:
            file_path: Path to data file
            preview_size: Number of rows to preview
            callback: Function to call when preview is ready
            task_id: Task ID for tracking

        Returns:
            Request ID for tracking
        """
        if not self.running:
            self.start()

        request_id = task_id or f"preview_{int(time.time() * 1000)}"

        request = {
            "request_id": request_id,
            "file_path": file_path,
            "preview_size": preview_size,
            "callback": callback,
            "created_at": time.time(),
        }

        self.loading_queue.put(request)
        self.task_callbacks[request_id] = callback

        logger.debug(f"Queued preview request {request_id} for {file_path}")
        return request_id

    def _worker(self) -> None:
        """Background worker for processing requests."""
        while self.running:
            try:
                request = self.loading_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    break

                self._process_request(request)
                self.stats["tasks_processed"] += 1

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _process_request(self, request: Dict[str, Any]) -> None:
        """Process a single request."""
        request_id = request["request_id"]
        start_time = time.time()

        try:
            if "preview_size" in request:
                # Preview request
                preview = self.chunked_service.get_data_preview(
                    request["file_path"], request["preview_size"]
                )
                request["callback"](preview)

            else:
                # Segment request
                file_path = request["file_path"]
                file_size = Path(file_path).stat().st_size

                # Select optimal strategy
                if file_size > 500 * 1024 * 1024:  # > 500MB
                    # Use memory-mapped access
                    map_id = self.memory_mapped_service.map_file(file_path)
                    segment = self.memory_mapped_service.get_time_segment(
                        map_id,
                        request["start_time"],
                        request["end_time"],
                        request["sampling_rate"],
                    )
                else:
                    # Use chunked access
                    # Calculate chunk indices
                    start_idx = int(request["start_time"] * request["sampling_rate"])
                    end_idx = int(request["end_time"] * request["sampling_rate"])
                    chunk_size = end_idx - start_idx

                    # Load chunk
                    chunk = self.chunked_service.load_data_chunked(
                        file_path, 0, chunk_size
                    )

                    # Extract segment
                    segment_data = (
                        chunk.iloc[start_idx:end_idx]
                        if isinstance(chunk, pd.DataFrame)
                        else chunk[start_idx:end_idx]
                    )

                    segment = DataSegment(
                        data=segment_data,
                        start_time=request["start_time"],
                        end_time=request["end_time"],
                        sampling_rate=request["sampling_rate"],
                        segment_id=request_id,
                        metadata={"file_path": file_path},
                    )

                request["callback"](segment)

            # Update statistics
            load_time = time.time() - start_time
            self.stats["background_loads"] += 1
            self.stats["average_load_time"] = (
                self.stats["average_load_time"] * (self.stats["background_loads"] - 1)
                + load_time
            ) / self.stats["background_loads"]

            logger.debug(f"Processed request {request_id} in {load_time:.3f}s")

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            # Call callback with error
            try:
                request["callback"](None)
            except Exception as e:
                logger.error(f"Error calling callback: {e}")
                pass

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            **self.stats,
            "queue_size": self.loading_queue.qsize(),
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "running": self.running,
        }

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""
        if request_id in self.task_callbacks:
            del self.task_callbacks[request_id]
            logger.debug(f"Cancelled request {request_id}")
            return True
        return False


class EnhancedDataService:
    """
    Enhanced data service integrating all loading strategies for webapp.

    This is the main service that webapp components should use.
    It automatically selects the optimal loading strategy and provides
    a unified interface for all data loading operations.
    """

    def load_data(
        self,
        file_path: str,
        strategy: Optional[LoadingStrategy] = None,
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> Union[pd.DataFrame, DataSegment]:
        """
        Load data using optimal strategy.

        Args:
            file_path: Path to data file
            strategy: Loading strategy (auto-selected if None)
            progress_callback: Callback for progress updates
            task_id: Task ID for tracking

        Returns:
            Loaded data (DataFrame or DataSegment)
        """
        self.stats["total_requests"] += 1

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = Path(file_path).stat().st_size
        file_size_mb = file_size / (1024**2)

        # Auto-select strategy if not provided
        if strategy is None:
            if file_size_mb < 50:
                strategy = LoadingStrategy.STANDARD
            elif file_size_mb < 500:
                strategy = LoadingStrategy.CHUNKED
            else:
                strategy = LoadingStrategy.MEMORY_MAPPED

        self.stats["strategy_selections"][strategy.value] += 1

        try:
            if strategy == LoadingStrategy.STANDARD:
                # Use standard DataLoader
                if VITALDSP_AVAILABLE:
                    loader = DataLoader(file_path)
                    return loader.load()
                else:
                    # Fallback to pandas
                    if file_path.endswith(".csv"):
                        return pd.read_csv(file_path)
                    else:
                        return pd.read_parquet(file_path)

            elif strategy == LoadingStrategy.CHUNKED:
                # Use chunked loading
                return self.chunked_service.load_data_chunked(
                    file_path, progress_callback=progress_callback, task_id=task_id
                )

            elif strategy == LoadingStrategy.MEMORY_MAPPED:
                # Use memory-mapped access
                map_id = self.memory_mapped_service.map_file(
                    file_path, progress_callback=progress_callback, task_id=task_id
                )
                return self.memory_mapped_service.get_segment(map_id, 0, 1000)

            elif strategy == LoadingStrategy.PROGRESSIVE:
                # Use progressive loading
                # This would typically be used for background loading
                # For now, fall back to chunked
                return self.chunked_service.load_data_chunked(
                    file_path, progress_callback=progress_callback, task_id=task_id
                )

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def get_data_preview(
        self,
        file_path: str,
        preview_size: int = 1000,
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
        task_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get data preview.

        Args:
            file_path: Path to data file
            preview_size: Number of rows to preview
            progress_callback: Callback for progress updates
            task_id: Task ID for tracking

        Returns:
            DataFrame with preview data
        """
        return self.chunked_service.get_data_preview(
            file_path, preview_size, progress_callback, task_id
        )

    def request_data_segment(
        self,
        file_path: str,
        start_time: float,
        end_time: float,
        sampling_rate: float,
        callback: Callable[[DataSegment], None],
        task_id: Optional[str] = None,
    ) -> str:
        """
        Request data segment with background loading.

        Args:
            file_path: Path to data file
            start_time: Start time in seconds
            end_time: End time in seconds
            sampling_rate: Sampling rate in Hz
            callback: Function to call when data is ready
            task_id: Task ID for tracking

        Returns:
            Request ID for tracking
        """
        return self.progressive_loader.request_data_segment(
            file_path, start_time, end_time, sampling_rate, callback, task_id
        )

    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            **self.stats,
            "chunked_service": self.chunked_service.get_cache_stats(),
            "memory_mapped_service": self.memory_mapped_service.get_service_stats(),
            "progressive_loader": self.progressive_loader.get_queue_status(),
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024**2),
        }

    def cleanup(self) -> None:
        """Cleanup all services and resources."""
        self.progressive_loader.stop()
        self.chunked_service.clear_cache()
        self.memory_mapped_service.cleanup()
        gc.collect()
        logger.info("EnhancedDataService cleaned up")

    # ============================================================================
    # COMPATIBILITY LAYER - Methods to match old DataService interface
    # ============================================================================
    
    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize enhanced data service with compatibility layer.
        """
        # Initialize the original EnhancedDataService
        self.max_memory_mb = max_memory_mb

        # Initialize services
        self.chunked_service = ChunkedDataService(max_memory_mb=max_memory_mb // 2)
        self.memory_mapped_service = MemoryMappedDataService(
            max_memory_mb=max_memory_mb // 2
        )
        self.progressive_loader = ProgressiveDataLoader()

        # Start progressive loader
        self.progressive_loader.start()

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "strategy_selections": {
                "standard": 0,
                "chunked": 0,
                "memory_mapped": 0,
                "progressive": 0,
            },
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        # Compatibility attributes for old interface
        self.current_headers = None
        self.current_metadata = None
        self.current_data = None
        self.data_config = {}
        
        # Data store for compatibility (maps data_id to data/info)
        self._data_store = {}
        self._column_mappings = {}
        self._next_id = 1
        
        logger.info(f"EnhancedDataService initialized with {max_memory_mb}MB memory limit")
        logger.debug("EnhancedDataService compatibility layer initialized")

    def store_data(self, df: pd.DataFrame, info: Dict[str, Any]) -> str:
        """
        Store data with a unique ID and return the ID.
        Compatibility method for old DataService interface.
        """
        try:
            data_id = f"data_{self._next_id}"
            self._next_id += 1

            logger.debug("=== STORING DATA (Enhanced Service) ===")
            logger.debug(f"Data ID: {data_id}")
            logger.debug(f"Data shape: {df.shape}")
            logger.debug(f"Data columns: {list(df.columns)}")
            logger.debug(f"Data info: {info}")

            # Store in compatibility store
            self._data_store[data_id] = {
                "data": df,
                "info": info,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Auto-generate column mapping if not provided
            if "column_mapping" in info and info["column_mapping"]:
                logger.debug("Using custom column mapping from info")
                column_mapping = info["column_mapping"]
            else:
                logger.debug("Auto-detecting columns...")
                column_mapping = self._auto_detect_columns(df)

            self._column_mappings[data_id] = column_mapping

            logger.info(f"Data stored with ID: {data_id}")
            logger.debug(f"Column mapping: {column_mapping}")
            return data_id

        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return None

    def get_data(self, data_id: str) -> Optional[pd.DataFrame]:
        """
        Get data by ID.
        Compatibility method for old DataService interface.
        """
        if data_id in self._data_store:
            return self._data_store[data_id]["data"]
        return None

    def get_all_data(self) -> Dict[str, Any]:
        """
        Get all stored data.
        Compatibility method for old DataService interface.
        """
        return self._data_store

    def get_data_info(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data info by ID.
        Compatibility method for old DataService interface.
        """
        if data_id in self._data_store:
            return self._data_store[data_id]["info"]
        return None

    def get_column_mapping(self, data_id: str) -> Dict[str, str]:
        """
        Get column mapping for a specific data ID.
        Compatibility method for old DataService interface.
        """
        mapping = self._column_mappings.get(data_id, {})
        logger.debug(f"Getting column mapping for {data_id}: {mapping}")
        return mapping

    def store_filtered_data(
        self, data_id: str, filtered_signal: np.ndarray, filter_info: Dict[str, Any]
    ) -> bool:
        """
        Store filtered signal data from filtering screen.
        Compatibility method for old DataService interface.
        """
        try:
            if data_id not in self._data_store:
                logger.error(f"Data ID {data_id} not found")
                return False

            logger.debug(f"Storing filtered data for ID: {data_id}")
            logger.debug(f"Filtered signal shape: {filtered_signal.shape}")
            logger.debug(f"Filter info: {filter_info}")

            self._data_store[data_id]["filtered_signal"] = filtered_signal
            self._data_store[data_id]["filter_info"] = filter_info
            self._data_store[data_id]["has_filtered_data"] = True
            self._data_store[data_id][
                "filtered_timestamp"
            ] = pd.Timestamp.now().isoformat()

            logger.debug(f"Filtered data stored successfully for ID: {data_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing filtered data: {e}")
            return False

    def update_config(self, config: Dict[str, Any]):
        """
        Update data configuration.
        Compatibility method for old DataService interface.
        """
        self.data_config.update(config)
        logger.debug(f"Config updated: {config}")

    def _auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect column types based on data characteristics.
        Compatibility method for old DataService interface.
        """
        mapping = {}

        if len(df.columns) >= 1:
            # Look for specific column types based on names first
            for col in df.columns:
                col_lower = col.lower()

                # Time-related columns
                if "time" not in mapping and (
                    any(keyword in col_lower for keyword in ["time", "timestamp"])
                    or col_lower == "t"
                ):
                    mapping["time"] = col

                # Signal columns - prioritize waveform/pleth columns
                elif any(
                    keyword in col_lower for keyword in ["waveform", "pleth", "pl"]
                ):
                    mapping["signal"] = col
                    logger.debug(f"Found waveform/pleth column: {col}")

                # Other signal columns
                elif any(keyword in col_lower for keyword in ["signal", "ppg", "ecg"]):
                    if "signal" not in mapping:
                        mapping["signal"] = col
                        logger.debug(f"Found signal column: {col}")

                # RED channel (for pulse oximetry)
                elif "red" not in mapping and any(
                    keyword in col_lower for keyword in ["red"]
                ):
                    mapping["red"] = col

                # IR channel (for pulse oximetry)
                elif "ir" not in mapping and any(
                    keyword in col_lower for keyword in ["ir", "infrared"]
                ):
                    mapping["ir"] = col

            # If no specific columns found, use defaults based on position
            if "time" not in mapping and len(df.columns) > 0:
                mapping["time"] = df.columns[0]
                logger.debug(f"Using default time column: {df.columns[0]}")

            if "signal" not in mapping:
                if len(df.columns) > 1:
                    mapping["signal"] = df.columns[1]
                    logger.debug(f"Using default signal column: {df.columns[1]}")
                elif len(df.columns) == 1:
                    mapping["signal"] = df.columns[0]
                    logger.debug(f"Using single column for signal: {df.columns[0]}")

        # Special case: single column dataframe
        if len(df.columns) == 1:
            single_col = df.columns[0]
            mapping["time"] = single_col
            mapping["signal"] = single_col
            logger.debug(f"Single column detected, mapping both time and signal to: {single_col}")

        logger.debug(f"Auto-detected column mapping: {mapping}")
        return mapping

    def get_filtered_data(self, data_id: str) -> Optional[np.ndarray]:
        """
        Retrieve filtered signal data.
        Compatibility method for old DataService interface.
        """
        try:
            if data_id not in self._data_store:
                logger.warning(f"Data ID {data_id} not found")
                return None

            data = self._data_store[data_id]
            if data.get("has_filtered_data", False):
                logger.debug(f"Retrieved filtered data for ID: {data_id}")
                return data.get("filtered_signal")
            else:
                logger.debug(f"No filtered data available for ID: {data_id}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving filtered data: {e}")
            return None

    def has_filtered_data(self, data_id: str) -> bool:
        """
        Check if filtered data is available.
        Compatibility method for old DataService interface.
        """
        try:
            if data_id not in self._data_store:
                return False

            has_filtered = self._data_store[data_id].get("has_filtered_data", False)
            logger.debug(f"Filtered data available for ID {data_id}: {has_filtered}")
            return has_filtered

        except Exception as e:
            logger.error(f"Error checking filtered data availability: {e}")
            return False

    def get_filter_info(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get filter information for a specific data ID.
        Compatibility method for old DataService interface.
        """
        try:
            if data_id not in self._data_store:
                return None

            data = self._data_store[data_id]
            if data.get("has_filtered_data", False):
                return data.get("filter_info")
            return None

        except Exception as e:
            logger.error(f"Error retrieving filter info: {e}")
            return None

    def clear_filtered_data(self, data_id: str) -> bool:
        """
        Clear filtered data for a specific data ID.
        Compatibility method for old DataService interface.
        """
        try:
            if data_id not in self._data_store:
                logger.warning(f"Data ID {data_id} not found")
                return False

            data = self._data_store[data_id]
            if data.get("has_filtered_data", False):
                data["has_filtered_data"] = False
                data.pop("filtered_signal", None)
                data.pop("filter_info", None)
                logger.debug(f"Cleared filtered data for ID: {data_id}")
                return True
            else:
                logger.debug(f"No filtered data to clear for ID: {data_id}")
                return False

        except Exception as e:
            logger.error(f"Error clearing filtered data: {e}")
            return False

    def get_current_data(self) -> Optional[pd.DataFrame]:
        """
        Get current data.
        Compatibility method for old DataService interface.
        """
        return self.current_data

    def get_config(self) -> Dict[str, Any]:
        """
        Get current data configuration.
        Compatibility method for old DataService interface.
        """
        return self.data_config

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current data configuration (copy).
        Compatibility method for old DataService interface.
        """
        return self.data_config.copy()

    def clear_data(self, data_id: str):
        """
        Clear data by ID.
        Compatibility method for old DataService interface.
        """
        if data_id in self._data_store:
            del self._data_store[data_id]
        if data_id in self._column_mappings:
            del self._column_mappings[data_id]
        logger.debug(f"Data cleared for ID: {data_id}")

    def clear_all_data(self):
        """
        Clear all stored data.
        Compatibility method for old DataService interface.
        """
        self.current_data = None
        self.data_config.clear()
        self._data_store.clear()
        self._column_mappings.clear()
        self._next_id = 1
        logger.info("All data cleared")

    def set_column_mapping(self, data_id: str, mapping: Dict[str, str]):
        """
        Set column mapping for a specific data ID.
        Compatibility method for old DataService interface.
        """
        self._column_mappings[data_id] = mapping
        logger.debug(f"Column mapping set for {data_id}: {mapping}")

    def update_column_mapping(self, data_id: str, mapping: Dict[str, str]):
        """
        Update column mapping for a specific data ID.
        Compatibility method for old DataService interface.
        """
        self._column_mappings[data_id] = mapping
        logger.debug(f"Column mapping updated for {data_id}: {mapping}")

    def get_data_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary of current data.
        Compatibility method for old DataService interface.
        """
        if self.current_data is None or self.current_data.empty:
            return None

        return {
            "shape": self.current_data.shape,
            "columns": self.current_data.columns.tolist(),
            "data_config": self.data_config,
        }

    def has_data(self) -> bool:
        """
        Check if any data is available.
        Compatibility method for old DataService interface.
        """
        return len(self._data_store) > 0

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from file path.
        Compatibility method for old DataService interface.
        """
        try:
            file_path = Path(file_path)
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == ".txt":
                df = pd.read_csv(file_path, sep="\t")
            elif file_path.suffix.lower() == ".mat":
                # For .mat files, we'd need scipy.io
                logger.warning(".mat files not yet supported")
                return None
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None

            self.current_data = df
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def process_data(
        self, df: pd.DataFrame, sampling_freq: float, time_unit: str = "seconds"
    ) -> Dict[str, Any]:
        """
        Process uploaded data and return metadata.
        Compatibility method for old DataService interface.
        """
        try:
            # Basic data validation
            if df.empty:
                return {"error": "Data is empty"}

            # Calculate basic statistics
            signal_data = (
                df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
            )

            # Convert time unit if needed
            if time_unit == "milliseconds":
                sampling_freq = sampling_freq / 1000
            elif time_unit == "minutes":
                sampling_freq = sampling_freq * 60

            duration = len(signal_data) / sampling_freq

            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "sampling_freq": sampling_freq,
                "time_unit": time_unit,
                "duration": duration,
                "signal_length": len(signal_data),
                "mean": float(np.mean(signal_data)) if len(signal_data) > 0 else 0.0,
                "std": float(np.std(signal_data)) if len(signal_data) > 0 else 0.0,
                "min": float(np.min(signal_data)) if len(signal_data) > 0 else 0.0,
                "max": float(np.max(signal_data)) if len(signal_data) > 0 else 0.0,
            }

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {"error": str(e)}

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in destructor: {e}")
            pass


# Global instance for webapp use
_enhanced_data_service = None


def get_enhanced_data_service(max_memory_mb: int = 500) -> EnhancedDataService:
    """Get global enhanced data service instance."""
    global _enhanced_data_service
    if _enhanced_data_service is None:
        _enhanced_data_service = EnhancedDataService(max_memory_mb=max_memory_mb)
    return _enhanced_data_service


# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Data Service for vitalDSP Webapp")
    print("=" * 50)
    print("\nThis module provides enhanced data loading capabilities")
    print("specifically designed for the webapp environment.")
    print("\nFeatures:")
    print("  - ChunkedDataService: LRU cache with adaptive chunking")
    print("  - MemoryMappedDataService: Zero-copy access for large files")
    print("  - ProgressiveDataLoader: Background processing with real-time updates")
    print(
        "  - EnhancedDataService: Unified interface with automatic strategy selection"
    )
    print("\nIntegration with vitalDSP:")
    print(f"  - vitalDSP modules available: {VITALDSP_AVAILABLE}")
    if VITALDSP_AVAILABLE:
        print("  - ChunkedDataLoader: Available")
        print("  - MemoryMappedLoader: Available")
        print("  - DataLoader: Available")
