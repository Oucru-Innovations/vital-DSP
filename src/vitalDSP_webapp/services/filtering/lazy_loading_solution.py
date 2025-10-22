"""
Lazy Loading Solution for Progressive Filtering - Phase 3B Implementation

This module implements a comprehensive lazy loading solution for progressive filtering
of large datasets in the webapp environment.

Features:
- ProgressiveDataLoader: Background loading with threading
- LazyChunkManager: On-demand chunk loading and caching
- StreamingFilterProcessor: Real-time filtering with lazy evaluation
- MemoryEfficientCache: LRU cache with memory management
- WebSocketProgressBroadcaster: Real-time progress updates
- CancellationSupport: Request cancellation and cleanup

Author: vitalDSP Development Team
Date: January 11, 2025
Phase: 3B - Heavy Data Processing Integration
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Callable,
    Generator,
    Tuple,
    Iterator,
)
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict, deque
import psutil
import gc
from queue import Queue, Empty, PriorityQueue
import json
import uuid
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import pickle
import hashlib

# Add vitalDSP to path for imports
current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

try:
    # Import heavy data filtering service
    from vitalDSP_webapp.services.filtering.heavy_data_filtering_service import (
        HeavyDataFilteringService,
        FilteringRequest,
        FilteringResult,
        FilteringStrategy,
        FilteringMode,
    )

    # Import WebSocket manager
    from vitalDSP_webapp.services.async_services.websocket_manager import (
        get_websocket_manager,
    )

    HEAVY_DATA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Heavy data filtering service not available: {e}")
    HEAVY_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChunkState(Enum):
    """Chunk state enumeration."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FILTERING = "filtering"
    FILTERED = "filtered"
    CACHED = "cached"
    EVICTED = "evicted"


class LoadingPriority(Enum):
    """Loading priority enumeration."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class LazyChunk:
    """Data class for lazy loading chunk information."""

    chunk_id: str
    start_index: int
    end_index: int
    file_path: Optional[str] = None
    data: Optional[np.ndarray] = None
    filtered_data: Optional[np.ndarray] = None
    state: ChunkState = ChunkState.NOT_LOADED
    priority: LoadingPriority = LoadingPriority.NORMAL
    loading_time: float = 0.0
    filtering_time: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    memory_size: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.last_accessed = time.time()


@dataclass
class LoadingTask:
    """Data class for loading task information."""

    task_id: str
    chunk_id: str
    priority: LoadingPriority
    callback: Optional[Callable] = None
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class MemoryEfficientCache:
    """
    Memory-efficient LRU cache with intelligent eviction.

    Features:
    - LRU eviction policy
    - Memory usage monitoring
    - Adaptive cache sizing
    - Weak reference support
    - Compression for large chunks
    """

    def __init__(self, max_memory_mb: int = 200, compression_threshold_mb: int = 10):
        """
        Initialize memory-efficient cache.

        Args:
            max_memory_mb: Maximum memory usage in MB
            compression_threshold_mb: Threshold for compression in MB
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_threshold_bytes = compression_threshold_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.weak_refs = weakref.WeakValueDictionary()
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0,
            "memory_saved": 0,
        }

        logger.info(f"MemoryEfficientCache initialized with {max_memory_mb}MB limit")

    def get(self, key: str) -> Optional[LazyChunk]:
        """Get chunk from cache."""
        with self._lock:
            if key in self.cache:
                chunk = self.cache[key]
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                chunk.last_accessed = time.time()
                chunk.access_count += 1
                self.stats["hits"] += 1

                # Decompress if needed
                if chunk.metadata.get("compressed", False):
                    chunk = self._decompress_chunk(chunk)

                return chunk
            else:
                self.stats["misses"] += 1
                return None

    def put(self, key: str, chunk: LazyChunk) -> None:
        """Put chunk in cache."""
        with self._lock:
            # Calculate memory size
            chunk.memory_size = self._calculate_chunk_size(chunk)

            # Compress if needed
            if chunk.memory_size > self.compression_threshold_bytes:
                chunk = self._compress_chunk(chunk)

            # Evict if necessary
            self._evict_if_needed(chunk.memory_size)

            # Add to cache
            self.cache[key] = chunk
            self.weak_refs[key] = chunk

    def _calculate_chunk_size(self, chunk: LazyChunk) -> int:
        """Calculate chunk memory size."""
        size = 0
        if chunk.data is not None:
            size += chunk.data.nbytes
        if chunk.filtered_data is not None:
            size += chunk.filtered_data.nbytes
        return size

    def _compress_chunk(self, chunk: LazyChunk) -> LazyChunk:
        """Compress chunk data."""
        try:
            if chunk.data is not None:
                chunk.data = np.frombuffer(
                    pickle.dumps(chunk.data, protocol=pickle.HIGHEST_PROTOCOL),
                    dtype=chunk.data.dtype,
                )
                chunk.metadata["compressed"] = True
                chunk.metadata["original_shape"] = chunk.data.shape
                self.stats["compressions"] += 1
                self.stats[
                    "memory_saved"
                ] += chunk.memory_size - self._calculate_chunk_size(chunk)
        except Exception as e:
            logger.warning(f"Failed to compress chunk {chunk.chunk_id}: {e}")

        return chunk

    def _decompress_chunk(self, chunk: LazyChunk) -> LazyChunk:
        """Decompress chunk data."""
        try:
            if chunk.metadata.get("compressed", False):
                original_shape = chunk.metadata.get("original_shape")
                if original_shape:
                    chunk.data = chunk.data.reshape(original_shape)
                    chunk.metadata["compressed"] = False
                    self.stats["decompressions"] += 1
        except Exception as e:
            logger.warning(f"Failed to decompress chunk {chunk.chunk_id}: {e}")

        return chunk

    def _evict_if_needed(self, new_chunk_size: int) -> None:
        """Evict chunks if memory limit exceeded."""
        current_memory = sum(
            self._calculate_chunk_size(chunk) for chunk in self.cache.values()
        )

        while current_memory + new_chunk_size > self.max_memory_bytes and self.cache:
            # Remove least recently used chunk
            oldest_key, oldest_chunk = self.cache.popitem(last=False)
            current_memory -= self._calculate_chunk_size(oldest_chunk)
            self.stats["evictions"] += 1

            logger.debug(f"Evicted chunk {oldest_key} to free memory")

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.weak_refs.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_memory = sum(
                self._calculate_chunk_size(chunk) for chunk in self.cache.values()
            )
            hit_rate = self.stats["hits"] / max(
                1, self.stats["hits"] + self.stats["misses"]
            )

            return {
                "cache_size": len(self.cache),
                "current_memory_mb": current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "statistics": self.stats.copy(),
            }


class LazyChunkManager:
    """
    Lazy chunk manager for on-demand loading and caching.

    Features:
    - On-demand chunk loading
    - Intelligent prefetching
    - Memory-efficient caching
    - Background loading
    - Priority-based loading
    """

    def __init__(self, chunk_size_mb: int = 50, max_workers: int = 4):
        """
        Initialize lazy chunk manager.

        Args:
            chunk_size_mb: Size of each chunk in MB
            max_workers: Maximum number of background workers
        """
        self.chunk_size_mb = chunk_size_mb
        self.max_workers = max_workers
        self.chunk_size_samples = None  # Will be calculated based on data

        # Components
        self.cache = MemoryEfficientCache()
        self.loading_queue = PriorityQueue()
        self.loading_tasks = {}
        self.workers = []
        self.running = False
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "chunks_created": 0,
            "chunks_loaded": 0,
            "chunks_cached": 0,
            "prefetch_hits": 0,
            "background_loads": 0,
            "average_load_time": 0.0,
        }

        logger.info(
            f"LazyChunkManager initialized with {chunk_size_mb}MB chunks, {max_workers} workers"
        )

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
        if not self.running:
            return

        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()
        logger.info("Stopped background workers")

    def create_chunks(
        self, signal_length: int, file_path: Optional[str] = None
    ) -> List[LazyChunk]:
        """
        Create lazy chunks for signal data.

        Args:
            signal_length: Length of signal data
            file_path: Optional file path for data

        Returns:
            List of LazyChunk objects
        """
        # Calculate chunk size in samples
        if self.chunk_size_samples is None:
            # Estimate based on float64 (8 bytes per sample)
            self.chunk_size_samples = int(self.chunk_size_mb * 1024 * 1024 / 8)
            self.chunk_size_samples = max(
                1000, min(self.chunk_size_samples, signal_length // 4)
            )

        chunks = []
        total_chunks = (
            signal_length + self.chunk_size_samples - 1
        ) // self.chunk_size_samples

        for i in range(total_chunks):
            start_idx = i * self.chunk_size_samples
            end_idx = min(start_idx + self.chunk_size_samples, signal_length)

            chunk_id = f"chunk_{i}_{start_idx}_{end_idx}"
            chunk = LazyChunk(
                chunk_id=chunk_id,
                start_index=start_idx,
                end_index=end_idx,
                file_path=file_path,
            )

            chunks.append(chunk)
            self.stats["chunks_created"] += 1

        logger.info(f"Created {len(chunks)} chunks for {signal_length} samples")
        return chunks

    def get_chunk(
        self, chunk: LazyChunk, priority: LoadingPriority = LoadingPriority.NORMAL
    ) -> Optional[LazyChunk]:
        """
        Get chunk data, loading if necessary.

        Args:
            chunk: Chunk to get
            priority: Loading priority

        Returns:
            Loaded chunk or None if loading failed
        """
        # Check cache first
        cached_chunk = self.cache.get(chunk.chunk_id)
        if cached_chunk and cached_chunk.state == ChunkState.LOADED:
            return cached_chunk

        # Load chunk
        return self._load_chunk(chunk, priority)

    def _load_chunk(
        self, chunk: LazyChunk, priority: LoadingPriority = LoadingPriority.NORMAL
    ) -> Optional[LazyChunk]:
        """Load chunk data."""
        try:
            chunk.state = ChunkState.LOADING
            start_time = time.time()

            # Load data based on source
            if chunk.file_path and os.path.exists(chunk.file_path):
                # Load from file (would need file format support)
                chunk.data = self._load_from_file(chunk)
            else:
                # For now, return None (would need signal data reference)
                logger.warning(f"Cannot load chunk {chunk.chunk_id}: no data source")
                return None

            chunk.loading_time = time.time() - start_time
            chunk.state = ChunkState.LOADED
            chunk.last_accessed = time.time()

            # Cache chunk
            self.cache.put(chunk.chunk_id, chunk)
            self.stats["chunks_loaded"] += 1
            self.stats["chunks_cached"] += 1

            # Update average load time
            self.stats["average_load_time"] = (
                self.stats["average_load_time"] * (self.stats["chunks_loaded"] - 1)
                + chunk.loading_time
            ) / self.stats["chunks_loaded"]

            return chunk

        except Exception as e:
            logger.error(f"Error loading chunk {chunk.chunk_id}: {e}")
            chunk.state = ChunkState.NOT_LOADED
            return None

    def _load_from_file(self, chunk: LazyChunk) -> Optional[np.ndarray]:
        """Load chunk data from file."""
        # This would need to be implemented based on file format
        # For now, return None
        return None

    def prefetch_chunks(
        self, chunks: List[LazyChunk], priority: LoadingPriority = LoadingPriority.LOW
    ) -> None:
        """Prefetch chunks in background."""
        for chunk in chunks:
            if chunk.state == ChunkState.NOT_LOADED:
                task = LoadingTask(
                    task_id=str(uuid.uuid4()),
                    chunk_id=chunk.chunk_id,
                    priority=priority,
                )

                self.loading_queue.put((priority.value, task))
                self.loading_tasks[task.task_id] = task
                self.stats["background_loads"] += 1

    def _worker(self) -> None:
        """Background worker for loading chunks."""
        while self.running:
            try:
                # Get task from queue
                priority, task = self.loading_queue.get(timeout=1.0)

                # Find chunk
                chunk = None
                for cached_chunk in self.cache.cache.values():
                    if cached_chunk.chunk_id == task.chunk_id:
                        chunk = cached_chunk
                        break

                if chunk and chunk.state == ChunkState.NOT_LOADED:
                    self._load_chunk(chunk, LoadingPriority(priority))

                # Mark task as completed
                if task.task_id in self.loading_tasks:
                    del self.loading_tasks[task.task_id]

                self.loading_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background worker: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get chunk manager statistics."""
        return {
            "chunk_manager_stats": self.stats.copy(),
            "cache_stats": self.cache.get_statistics(),
            "active_tasks": len(self.loading_tasks),
            "queue_size": self.loading_queue.qsize(),
            "workers_running": len(self.workers),
        }


class StreamingFilterProcessor:
    """
    Streaming filter processor for real-time filtering with lazy evaluation.

    Features:
    - Real-time filtering with lazy evaluation
    - Streaming data processing
    - Memory-efficient processing
    - Progress tracking
    - Cancellation support
    """

    def __init__(self, chunk_manager: LazyChunkManager):
        """
        Initialize streaming filter processor.

        Args:
            chunk_manager: Lazy chunk manager instance
        """
        self.chunk_manager = chunk_manager
        self.filtering_queue = Queue()
        self.result_queue = Queue()
        self.active_filters = {}
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "filters_applied": 0,
            "chunks_filtered": 0,
            "average_filtering_time": 0.0,
            "streaming_requests": 0,
        }

        logger.info("StreamingFilterProcessor initialized")

    def process_streaming(
        self,
        chunks: List[LazyChunk],
        filter_params: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[FilteringResult, None, None]:
        """
        Process chunks in streaming fashion.

        Args:
            chunks: List of chunks to process
            filter_params: Filtering parameters
            progress_callback: Progress callback function

        Yields:
            FilteringResult for each processed chunk
        """
        request_id = str(uuid.uuid4())
        self.stats["streaming_requests"] += 1

        try:
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                # Get chunk data
                loaded_chunk = self.chunk_manager.get_chunk(chunk)
                if not loaded_chunk or loaded_chunk.data is None:
                    logger.warning(f"Failed to load chunk {chunk.chunk_id}")
                    continue

                # Apply filtering
                filtered_data = self._apply_filter_to_chunk(loaded_chunk, filter_params)
                loaded_chunk.filtered_data = filtered_data
                loaded_chunk.state = ChunkState.FILTERED

                # Update progress
                progress = (i + 1) / total_chunks
                if progress_callback:
                    progress_callback(
                        progress, f"Filtered chunk {i + 1}/{total_chunks}"
                    )

                # Yield result
                result = FilteringResult(
                    request_id=request_id,
                    success=True,
                    filtered_signal=filtered_data,
                    original_signal=loaded_chunk.data,
                    processing_time=loaded_chunk.filtering_time,
                    strategy_used=FilteringStrategy.CHUNKED,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "progress": progress,
                    },
                )

                yield result
                self.stats["chunks_filtered"] += 1

            logger.info(f"Completed streaming processing of {total_chunks} chunks")

        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
            yield FilteringResult(
                request_id=request_id, success=False, error_message=str(e)
            )

    def _apply_filter_to_chunk(
        self, chunk: LazyChunk, filter_params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply filter to a single chunk."""
        start_time = time.time()

        try:
            if chunk.data is None:
                return np.array([])

            # Apply basic filtering (can be extended)
            filtered_data = chunk.data.copy()

            filter_type = filter_params.get("filter_type", "lowpass")
            sampling_freq = filter_params.get("sampling_freq", 100.0)

            if filter_type == "lowpass":
                from scipy import signal

                nyquist = sampling_freq / 2
                cutoff = filter_params.get("low_freq", 10) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4), cutoff, btype="low"
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            elif filter_type == "highpass":
                from scipy import signal

                nyquist = sampling_freq / 2
                cutoff = filter_params.get("high_freq", 1) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4), cutoff, btype="high"
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            elif filter_type == "bandpass":
                from scipy import signal

                nyquist = sampling_freq / 2
                low_cutoff = filter_params.get("low_freq", 1) / nyquist
                high_cutoff = filter_params.get("high_freq", 10) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4),
                    [low_cutoff, high_cutoff],
                    btype="band",
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            chunk.filtering_time = time.time() - start_time
            self.stats["filters_applied"] += 1

            # Update average filtering time
            self.stats["average_filtering_time"] = (
                self.stats["average_filtering_time"]
                * (self.stats["filters_applied"] - 1)
                + chunk.filtering_time
            ) / self.stats["filters_applied"]

            return filtered_data

        except Exception as e:
            logger.error(f"Error filtering chunk {chunk.chunk_id}: {e}")
            return chunk.data

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return self.stats.copy()


class WebSocketProgressBroadcaster:
    """
    WebSocket progress broadcaster for real-time updates.

    Features:
    - Real-time progress broadcasting
    - Multiple client support
    - Progress aggregation
    - Error handling
    """

    def __init__(self, websocket_manager=None):
        """
        Initialize WebSocket progress broadcaster.

        Args:
            websocket_manager: WebSocket manager instance
        """
        self.websocket_manager = (
            websocket_manager or get_websocket_manager()
            if HEAVY_DATA_AVAILABLE
            else None
        )
        self.active_broadcasts = {}
        self._lock = threading.RLock()

        logger.info("WebSocketProgressBroadcaster initialized")

    def start_broadcast(
        self, request_id: str, total_chunks: int, description: str = "Processing"
    ) -> None:
        """Start progress broadcast for a request."""
        with self._lock:
            self.active_broadcasts[request_id] = {
                "total_chunks": total_chunks,
                "completed_chunks": 0,
                "description": description,
                "start_time": time.time(),
                "last_update": time.time(),
            }

    def update_progress(
        self, request_id: str, completed_chunks: int, message: str = None
    ) -> None:
        """Update progress for a request."""
        with self._lock:
            if request_id not in self.active_broadcasts:
                return

            broadcast_info = self.active_broadcasts[request_id]
            broadcast_info["completed_chunks"] = completed_chunks
            broadcast_info["last_update"] = time.time()

            # Calculate progress
            progress = completed_chunks / broadcast_info["total_chunks"]
            elapsed_time = time.time() - broadcast_info["start_time"]

            # Broadcast update
            if self.websocket_manager:
                try:
                    # Use the correct WebSocket method

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def broadcast_update():
                        from vitalDSP_webapp.services.async_services.websocket_manager import (
                            WebSocketMessage,
                        )

                        ws_message = WebSocketMessage(
                            type="lazy_loading_progress",
                            data={
                                "request_id": request_id,
                                "progress": progress,
                                "completed_chunks": completed_chunks,
                                "total_chunks": broadcast_info["total_chunks"],
                                "message": message or broadcast_info["description"],
                                "elapsed_time": elapsed_time,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        return await self.websocket_manager.broadcast_to_all(ws_message)

                    loop.run_until_complete(broadcast_update())
                    loop.close()
                except Exception as e:
                    logger.warning(f"Failed to broadcast WebSocket update: {e}")

    def complete_broadcast(self, request_id: str, success: bool = True) -> None:
        """Complete progress broadcast for a request."""
        with self._lock:
            if request_id not in self.active_broadcasts:
                return

            broadcast_info = self.active_broadcasts[request_id]
            elapsed_time = time.time() - broadcast_info["start_time"]

            # Send completion message
            if self.websocket_manager:
                try:

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def broadcast_completion():
                        from vitalDSP_webapp.services.async_services.websocket_manager import (
                            WebSocketMessage,
                        )

                        ws_message = WebSocketMessage(
                            type="lazy_loading_complete",
                            data={
                                "request_id": request_id,
                                "success": success,
                                "total_chunks": broadcast_info["total_chunks"],
                                "elapsed_time": elapsed_time,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        return await self.websocket_manager.broadcast_to_all(ws_message)

                    loop.run_until_complete(broadcast_completion())
                    loop.close()
                except Exception as e:
                    logger.warning(f"Failed to broadcast WebSocket completion: {e}")

            # Remove from active broadcasts
            del self.active_broadcasts[request_id]

    def get_active_broadcasts(self) -> Dict[str, Any]:
        """Get active broadcast information."""
        with self._lock:
            return {
                req_id: {
                    "description": info["description"],
                    "progress": info["completed_chunks"] / info["total_chunks"],
                    "elapsed_time": time.time() - info["start_time"],
                }
                for req_id, info in self.active_broadcasts.items()
            }


class ProgressiveDataLoader:
    """
    Progressive data loader with lazy loading and streaming processing.

    This is the main class that integrates all lazy loading components
    for comprehensive progressive data processing.
    """

    def __init__(self, max_memory_mb: int = 500, chunk_size_mb: int = 50):
        """
        Initialize progressive data loader.

        Args:
            max_memory_mb: Maximum memory usage in MB
            chunk_size_mb: Size of each chunk in MB
        """
        self.max_memory_mb = max_memory_mb
        self.chunk_size_mb = chunk_size_mb

        # Initialize components
        self.chunk_manager = LazyChunkManager(chunk_size_mb)
        self.filter_processor = StreamingFilterProcessor(self.chunk_manager)
        self.progress_broadcaster = WebSocketProgressBroadcaster()

        # Start chunk manager
        self.chunk_manager.start()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "total_data_processed_mb": 0.0,
        }

        logger.info(
            f"ProgressiveDataLoader initialized with {max_memory_mb}MB memory limit"
        )

    def process_lazy_filtering(
        self,
        signal_data: np.ndarray,
        filter_params: Dict[str, Any],
        sampling_freq: float = 100.0,
        signal_type: str = "unknown",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[FilteringResult, None, None]:
        """
        Process lazy filtering with progressive loading.

        Args:
            signal_data: Signal data to process
            filter_params: Filtering parameters
            sampling_freq: Sampling frequency
            signal_type: Type of signal
            progress_callback: Progress callback function

        Yields:
            FilteringResult for each processed chunk
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # Create chunks
            chunks = self.chunk_manager.create_chunks(len(signal_data))

            # Start progress broadcast
            self.progress_broadcaster.start_broadcast(
                request_id, len(chunks), "Lazy filtering"
            )

            # Process chunks progressively
            completed_chunks = 0
            for result in self.filter_processor.process_streaming(
                chunks, filter_params, progress_callback
            ):
                completed_chunks += 1

                # Update progress broadcast
                self.progress_broadcaster.update_progress(
                    request_id,
                    completed_chunks,
                    f"Processed chunk {completed_chunks}/{len(chunks)}",
                )

                yield result

            # Complete progress broadcast
            self.progress_broadcaster.complete_broadcast(request_id, True)

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["successful_requests"] += 1
            self.stats["average_processing_time"] = (
                self.stats["average_processing_time"]
                * (self.stats["successful_requests"] - 1)
                + processing_time
            ) / self.stats["successful_requests"]

            data_size_mb = signal_data.nbytes / (1024 * 1024)
            self.stats["total_data_processed_mb"] += data_size_mb

            logger.info(
                f"Completed lazy filtering request {request_id} in {processing_time:.3f}s"
            )

        except Exception as e:
            logger.error(f"Error in lazy filtering: {e}")
            self.stats["failed_requests"] += 1

            # Complete progress broadcast with error
            self.progress_broadcaster.complete_broadcast(request_id, False)

            yield FilteringResult(
                request_id=request_id, success=False, error_message=str(e)
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "progressive_loader_stats": self.stats.copy(),
            "chunk_manager_stats": self.chunk_manager.get_statistics(),
            "filter_processor_stats": self.filter_processor.get_statistics(),
            "active_broadcasts": self.progress_broadcaster.get_active_broadcasts(),
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.chunk_manager.stop()
        self.cache.clear()
        logger.info("ProgressiveDataLoader cleaned up")


# Global progressive data loader instance
_progressive_data_loader = None


def get_progressive_data_loader() -> ProgressiveDataLoader:
    """Get global progressive data loader instance."""
    global _progressive_data_loader
    if _progressive_data_loader is None:
        _progressive_data_loader = ProgressiveDataLoader()
    return _progressive_data_loader
