"""
Heavy Data Filtering Service for vitalDSP Webapp - Phase 3B Implementation

This module implements the heavy data filtering service that leverages the core vitalDSP
heavy data processing strategies for efficient processing of large datasets in the webapp.

Features:
- HeavyDataFilteringService: Integrates OptimizedStandardProcessingPipeline
- LazyLoadingFilteringService: Progressive filtering with lazy loading
- IntelligentStrategySelector: Auto-selects optimal processing strategy
- MemoryOptimizedFiltering: Uses OptimizedMemoryManager for large datasets
- ProgressiveFilteringPipeline: Background processing with real-time updates
- Integration with existing webapp filtering callbacks

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
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add vitalDSP to path for imports
current_dir = Path(__file__).parent
vitaldsp_path = current_dir.parent.parent.parent / "vitalDSP"
if str(vitaldsp_path) not in sys.path:
    sys.path.insert(0, str(vitaldsp_path))

try:
    # Core vitalDSP heavy data processing imports
    from vitalDSP.utils.core_infrastructure.optimized_processing_pipeline import (
        OptimizedStandardProcessingPipeline,
        ProcessingStage,
        ProcessingCheckpoint,
    )
    from vitalDSP.utils.core_infrastructure.processing_pipeline import (
        StandardProcessingPipeline,
    )
    from vitalDSP.utils.core_infrastructure.data_loaders import (
        ChunkedDataLoader,
        MemoryMappedLoader,
        ProgressInfo,
        CancellationToken,
        select_optimal_loader,
    )
    from vitalDSP.utils.core_infrastructure.optimized_memory_manager import (
        OptimizedMemoryManager,
        MemoryStrategy,
        MemoryInfo,
    )
    from vitalDSP.utils.core_infrastructure.parallel_pipeline import ParallelPipeline
    from vitalDSP.utils.core_infrastructure.quality_screener import QualityScreener
    from vitalDSP.utils.config_utilities.dynamic_config import DynamicConfigManager

    # Filtering imports
    from vitalDSP.filtering.signal_filtering import SignalFiltering
    from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
    from vitalDSP.filtering.neural_network_filtering import NeuralNetworkFiltering

    VITALDSP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"vitalDSP modules not available: {e}")
    VITALDSP_AVAILABLE = False

logger = logging.getLogger(__name__)


class FilteringStrategy(Enum):
    """Filtering strategy selection based on data size and requirements."""

    STANDARD = "standard"  # < 100MB: Direct processing
    CHUNKED = "chunked"  # 100MB-2GB: Chunked processing
    MEMORY_MAPPED = "memory_mapped"  # >2GB: Memory-mapped processing
    PROGRESSIVE = "progressive"  # Background processing with lazy loading


class FilteringMode(Enum):
    """Filtering mode for different processing approaches."""

    SYNCHRONOUS = "synchronous"  # Immediate processing
    ASYNCHRONOUS = "asynchronous"  # Background processing
    LAZY = "lazy"  # On-demand processing


@dataclass
class FilteringRequest:
    """Data class for filtering request information."""

    request_id: str
    signal_data: Optional[np.ndarray] = None
    sampling_freq: float = 100.0
    signal_type: str = "unknown"
    filter_params: Dict[str, Any] = None
    strategy: FilteringStrategy = FilteringStrategy.STANDARD
    mode: FilteringMode = FilteringMode.SYNCHRONOUS
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    file_path: Optional[str] = None

    def __post_init__(self):
        if self.filter_params is None:
            self.filter_params = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FilteringResult:
    """Data class for filtering result information."""

    request_id: str
    success: bool
    filtered_signal: Optional[np.ndarray] = None
    original_signal: Optional[np.ndarray] = None
    processing_time: float = 0.0
    memory_used: int = 0
    strategy_used: FilteringStrategy = FilteringStrategy.STANDARD
    quality_metrics: Dict[str, Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LazyLoadingChunk:
    """Data class for lazy loading chunk information."""

    chunk_id: str
    start_index: int
    end_index: int
    data: Optional[np.ndarray] = None
    filtered_data: Optional[np.ndarray] = None
    is_loaded: bool = False
    is_filtered: bool = False
    loading_time: float = 0.0
    filtering_time: float = 0.0


class IntelligentStrategySelector:
    """
    Intelligent strategy selector for optimal filtering approach.

    Selects the best filtering strategy based on:
    - Data size and memory requirements
    - Available system resources
    - User preferences and requirements
    - Processing complexity
    """

    def __init__(self):
        """Initialize strategy selector."""
        self.memory_manager = OptimizedMemoryManager() if VITALDSP_AVAILABLE else None
        self.config_manager = DynamicConfigManager() if VITALDSP_AVAILABLE else None

        # Strategy thresholds (in MB)
        self.thresholds = {
            "standard_max": 100,  # < 100MB: Standard processing
            "chunked_max": 2048,  # 100MB-2GB: Chunked processing
            "memory_mapped_min": 2048,  # >2GB: Memory-mapped processing
        }

        logger.info("IntelligentStrategySelector initialized")

    def select_strategy(
        self,
        data_size_mb: float,
        available_memory_mb: float,
        processing_complexity: str = "medium",
        user_preference: Optional[FilteringStrategy] = None,
    ) -> Tuple[FilteringStrategy, FilteringMode]:
        """
        Select optimal filtering strategy and mode.

        Args:
            data_size_mb: Size of data to process in MB
            available_memory_mb: Available system memory in MB
            processing_complexity: Complexity level (low, medium, high)
            user_preference: User's preferred strategy

        Returns:
            Tuple of (strategy, mode)
        """
        # Use user preference if valid for data size
        if user_preference and self._is_strategy_valid(
            user_preference, data_size_mb, available_memory_mb
        ):
            strategy = user_preference
        else:
            # Auto-select based on data size and available memory
            if data_size_mb < self.thresholds["standard_max"]:
                strategy = FilteringStrategy.STANDARD
            elif data_size_mb < self.thresholds["chunked_max"]:
                strategy = FilteringStrategy.CHUNKED
            else:
                strategy = FilteringStrategy.MEMORY_MAPPED

        # Select mode based on strategy and complexity
        if strategy == FilteringStrategy.STANDARD:
            mode = FilteringMode.SYNCHRONOUS
        elif processing_complexity == "high" or data_size_mb > 500:
            mode = FilteringMode.ASYNCHRONOUS
        else:
            mode = FilteringMode.LAZY

        logger.info(
            f"Selected strategy: {strategy.value}, mode: {mode.value} for {data_size_mb:.1f}MB data"
        )
        return strategy, mode

    def _is_strategy_valid(
        self,
        strategy: FilteringStrategy,
        data_size_mb: float,
        available_memory_mb: float,
    ) -> bool:
        """Check if strategy is valid for given constraints."""
        if strategy == FilteringStrategy.STANDARD:
            return (
                data_size_mb < self.thresholds["standard_max"]
                and data_size_mb < available_memory_mb * 0.5
            )
        elif strategy == FilteringStrategy.CHUNKED:
            return data_size_mb < self.thresholds["chunked_max"]
        elif strategy == FilteringStrategy.MEMORY_MAPPED:
            return data_size_mb >= self.thresholds["memory_mapped_min"]
        return True


class LazyLoadingFilteringService:
    """
    Lazy loading filtering service for progressive processing.

    Features:
    - On-demand loading and filtering of data chunks
    - Memory-efficient processing of large datasets
    - Real-time progress updates
    - Cancellation support
    - Integration with vitalDSP processing pipelines
    """

    def __init__(self, chunk_size_mb: int = 50, max_concurrent_chunks: int = 4):
        """
        Initialize lazy loading filtering service.

        Args:
            chunk_size_mb: Size of each chunk in MB
            max_concurrent_chunks: Maximum concurrent chunk processing
        """
        self.chunk_size_mb = chunk_size_mb
        self.max_concurrent_chunks = max_concurrent_chunks
        self.chunk_cache = OrderedDict()
        self.processing_queue = Queue()
        self.result_queue = Queue()
        self.cancellation_tokens = {}
        self._lock = threading.Lock()

        # Initialize vitalDSP components
        if VITALDSP_AVAILABLE:
            self.memory_manager = OptimizedMemoryManager()
            self.config_manager = DynamicConfigManager()
            self.quality_screener = QualityScreener(self.config_manager)
        else:
            self.memory_manager = None
            self.config_manager = None
            self.quality_screener = None

        # Performance tracking
        self.stats = {
            "chunks_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "memory_optimizations": 0,
        }

        logger.info(
            f"LazyLoadingFilteringService initialized with {chunk_size_mb}MB chunks"
        )

    def process_lazy(
        self,
        request: FilteringRequest,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[FilteringResult, None, None]:
        """
        Process data lazily with progressive filtering.

        Args:
            request: Filtering request
            progress_callback: Progress callback function

        Yields:
            FilteringResult for each processed chunk
        """
        try:
            # Calculate chunk parameters
            signal_length = (
                len(request.signal_data) if request.signal_data is not None else 0
            )
            chunk_size_samples = int(
                self.chunk_size_mb * 1024 * 1024 / (signal_length * 8)
            )  # Assuming float64
            chunk_size_samples = max(
                1000, min(chunk_size_samples, signal_length // 4)
            )  # Reasonable bounds

            total_chunks = (
                signal_length + chunk_size_samples - 1
            ) // chunk_size_samples

            logger.info(
                f"Processing {signal_length} samples in {total_chunks} chunks of {chunk_size_samples} samples"
            )

            # Process chunks progressively
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size_samples
                end_idx = min(start_idx + chunk_size_samples, signal_length)

                # Create chunk
                chunk_id = f"{request.request_id}_chunk_{chunk_idx}"
                chunk = LazyLoadingChunk(
                    chunk_id=chunk_id, start_index=start_idx, end_index=end_idx
                )

                # Load chunk data
                if request.signal_data is not None:
                    chunk.data = request.signal_data[start_idx:end_idx].copy()
                    chunk.is_loaded = True

                # Apply filtering to chunk
                filtered_chunk = self._filter_chunk(chunk, request.filter_params)
                chunk.filtered_data = filtered_chunk
                chunk.is_filtered = True

                # Update progress
                progress = (chunk_idx + 1) / total_chunks
                if progress_callback:
                    progress_callback(
                        progress, f"Processed chunk {chunk_idx + 1}/{total_chunks}"
                    )

                # Yield result
                result = FilteringResult(
                    request_id=request.request_id,
                    success=True,
                    filtered_signal=filtered_chunk,
                    original_signal=chunk.data,
                    processing_time=chunk.filtering_time,
                    strategy_used=FilteringStrategy.CHUNKED,
                    metadata={
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks,
                        "progress": progress,
                    },
                )

                yield result

                # Cache chunk for potential reuse
                self._cache_chunk(chunk)

                # Memory optimization
                if self.memory_manager:
                    self.memory_manager.optimize_memory_usage()
                    self.stats["memory_optimizations"] += 1

            logger.info(f"Completed lazy processing of {total_chunks} chunks")

        except Exception as e:
            logger.error(f"Error in lazy processing: {e}")
            yield FilteringResult(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def _filter_chunk(
        self, chunk: LazyLoadingChunk, filter_params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply filtering to a single chunk."""
        start_time = time.time()

        try:
            if not VITALDSP_AVAILABLE or chunk.data is None:
                return chunk.data

            # Apply basic filtering (can be extended with advanced filters)
            filtered_data = chunk.data.copy()

            # Apply filter based on parameters
            if filter_params.get("filter_type") == "lowpass":
                from scipy import signal

                nyquist = filter_params.get("sampling_freq", 100) / 2
                cutoff = filter_params.get("low_freq", 10) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4), cutoff, btype="low"
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            elif filter_params.get("filter_type") == "highpass":
                from scipy import signal

                nyquist = filter_params.get("sampling_freq", 100) / 2
                cutoff = filter_params.get("high_freq", 1) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4), cutoff, btype="high"
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            elif filter_params.get("filter_type") == "bandpass":
                from scipy import signal

                nyquist = filter_params.get("sampling_freq", 100) / 2
                low_cutoff = filter_params.get("low_freq", 1) / nyquist
                high_cutoff = filter_params.get("high_freq", 10) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4),
                    [low_cutoff, high_cutoff],
                    btype="band",
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            chunk.filtering_time = time.time() - start_time
            self.stats["chunks_processed"] += 1

            return filtered_data

        except Exception as e:
            logger.error(f"Error filtering chunk {chunk.chunk_id}: {e}")
            return chunk.data

    def _cache_chunk(self, chunk: LazyLoadingChunk) -> None:
        """Cache chunk for potential reuse."""
        with self._lock:
            # Simple LRU cache implementation
            if len(self.chunk_cache) >= 100:  # Max 100 chunks in cache
                self.chunk_cache.popitem(last=False)  # Remove oldest

            self.chunk_cache[chunk.chunk_id] = chunk
            self.stats["cache_hits"] += 1


class HeavyDataFilteringService:
    """
    Heavy data filtering service integrating core vitalDSP heavy data processing strategies.

    This is the main service that webapp components should use for heavy data filtering.
    It automatically selects the optimal processing strategy and provides a unified
    interface for all heavy data filtering operations.
    """

    def __init__(self, max_memory_mb: int = 1000):
        """
        Initialize heavy data filtering service.

        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb

        # Initialize components
        self.strategy_selector = IntelligentStrategySelector()
        self.lazy_service = LazyLoadingFilteringService()

        # Initialize vitalDSP components
        if VITALDSP_AVAILABLE:
            self.config_manager = DynamicConfigManager()
            self.memory_manager = OptimizedMemoryManager()
            self.processing_pipeline = OptimizedStandardProcessingPipeline(
                self.config_manager
            )
            self.parallel_pipeline = ParallelPipeline(self.config_manager)
            self.quality_screener = QualityScreener(self.config_manager)
        else:
            self.config_manager = None
            self.memory_manager = None
            self.processing_pipeline = None
            self.parallel_pipeline = None
            self.quality_screener = None

        # Processing statistics
        self.stats = {
            "total_requests": 0,
            "strategy_usage": {
                "standard": 0,
                "chunked": 0,
                "memory_mapped": 0,
                "progressive": 0,
            },
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "total_memory_used": 0,
        }

        logger.info(
            f"HeavyDataFilteringService initialized with {max_memory_mb}MB memory limit"
        )

    def process_filtering_request(
        self,
        request: FilteringRequest,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Union[FilteringResult, Generator[FilteringResult, None, None]]:
        """
        Process filtering request using optimal strategy.

        Args:
            request: Filtering request
            progress_callback: Progress callback function

        Returns:
            FilteringResult or Generator of FilteringResults for progressive processing
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # Calculate data size
            data_size_mb = self._calculate_data_size(request)
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Select optimal strategy
            strategy, mode = self.strategy_selector.select_strategy(
                data_size_mb,
                available_memory_mb,
                request.filter_params.get("complexity", "medium"),
            )

            request.strategy = strategy
            request.mode = mode

            logger.info(
                f"Processing request {request.request_id} with strategy: {strategy.value}, mode: {mode.value}"
            )

            # Process based on selected strategy
            if strategy == FilteringStrategy.STANDARD:
                result = self._process_standard(request, progress_callback)
            elif strategy == FilteringStrategy.CHUNKED:
                result = self._process_chunked(request, progress_callback)
            elif strategy == FilteringStrategy.MEMORY_MAPPED:
                result = self._process_memory_mapped(request, progress_callback)
            elif strategy == FilteringStrategy.PROGRESSIVE:
                result = self._process_progressive(request, progress_callback)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["strategy_usage"][strategy.value] += 1
            self.stats["successful_requests"] += 1
            self.stats["average_processing_time"] = (
                self.stats["average_processing_time"]
                * (self.stats["successful_requests"] - 1)
                + processing_time
            ) / self.stats["successful_requests"]

            logger.info(
                f"Completed request {request.request_id} in {processing_time:.3f}s using {strategy.value}"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            self.stats["failed_requests"] += 1

            return FilteringResult(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

    def _calculate_data_size(self, request: FilteringRequest) -> float:
        """Calculate data size in MB."""
        if request.signal_data is not None:
            return request.signal_data.nbytes / (1024 * 1024)
        elif request.file_path:
            try:
                return os.path.getsize(request.file_path) / (1024 * 1024)
            except OSError:
                return 0.0
        return 0.0

    def _process_standard(
        self, request: FilteringRequest, progress_callback: Optional[Callable] = None
    ) -> FilteringResult:
        """Process using standard strategy."""
        if progress_callback:
            progress_callback(0.1, "Initializing standard processing...")

        try:
            if not VITALDSP_AVAILABLE or request.signal_data is None:
                # Fallback to basic filtering
                filtered_signal = self._apply_basic_filter(
                    request.signal_data, request.filter_params
                )
            else:
                # Use vitalDSP processing pipeline
                result = self.processing_pipeline.process_signal(
                    request.signal_data,
                    request.sampling_freq,
                    request.signal_type,
                    request.metadata,
                )
                filtered_signal = result.get("filtered_signal", request.signal_data)

            if progress_callback:
                progress_callback(1.0, "Standard processing completed")

            return FilteringResult(
                request_id=request.request_id,
                success=True,
                filtered_signal=filtered_signal,
                original_signal=request.signal_data,
                strategy_used=FilteringStrategy.STANDARD,
                metadata={"processing_method": "standard"},
            )

        except Exception as e:
            logger.error(f"Error in standard processing: {e}")
            raise

    def _process_chunked(
        self, request: FilteringRequest, progress_callback: Optional[Callable] = None
    ) -> Generator[FilteringResult, None, None]:
        """Process using chunked strategy."""
        if progress_callback:
            progress_callback(0.1, "Initializing chunked processing...")

        try:
            # Use lazy loading service for chunked processing
            for result in self.lazy_service.process_lazy(request, progress_callback):
                yield result

        except Exception as e:
            logger.error(f"Error in chunked processing: {e}")
            yield FilteringResult(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def _process_memory_mapped(
        self, request: FilteringRequest, progress_callback: Optional[Callable] = None
    ) -> Generator[FilteringResult, None, None]:
        """Process using memory-mapped strategy."""
        if progress_callback:
            progress_callback(0.1, "Initializing memory-mapped processing...")

        try:
            if not VITALDSP_AVAILABLE or not request.file_path:
                raise ValueError(
                    "Memory-mapped processing requires file path and vitalDSP"
                )

            # Use memory-mapped loader
            loader = MemoryMappedLoader()

            # Process in segments
            segment_size = 1000000  # 1M samples per segment
            total_samples = (
                len(request.signal_data) if request.signal_data is not None else 0
            )

            for start_idx in range(0, total_samples, segment_size):
                end_idx = min(start_idx + segment_size, total_samples)

                if progress_callback:
                    progress = (start_idx + segment_size) / total_samples
                    progress_callback(
                        progress, f"Processing segment {start_idx}-{end_idx}"
                    )

                # Load segment
                segment_data = request.signal_data[start_idx:end_idx]

                # Apply filtering
                filtered_segment = self._apply_basic_filter(
                    segment_data, request.filter_params
                )

                # Yield result
                yield FilteringResult(
                    request_id=request.request_id,
                    success=True,
                    filtered_signal=filtered_segment,
                    original_signal=segment_data,
                    strategy_used=FilteringStrategy.MEMORY_MAPPED,
                    metadata={
                        "segment_start": start_idx,
                        "segment_end": end_idx,
                        "processing_method": "memory_mapped",
                    },
                )

        except Exception as e:
            logger.error(f"Error in memory-mapped processing: {e}")
            yield FilteringResult(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def _process_progressive(
        self, request: FilteringRequest, progress_callback: Optional[Callable] = None
    ) -> Generator[FilteringResult, None, None]:
        """Process using progressive strategy with background processing."""
        if progress_callback:
            progress_callback(0.1, "Initializing progressive processing...")

        try:
            # Use chunked processing with background threading
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit processing task
                future = executor.submit(self._process_chunked_background, request)

                # Monitor progress
                while not future.done():
                    if progress_callback:
                        progress_callback(0.5, "Processing in background...")
                    time.sleep(0.1)

                # Get results
                results = future.result()
                for result in results:
                    yield result

        except Exception as e:
            logger.error(f"Error in progressive processing: {e}")
            yield FilteringResult(
                request_id=request.request_id, success=False, error_message=str(e)
            )

    def _process_chunked_background(
        self, request: FilteringRequest
    ) -> List[FilteringResult]:
        """Process chunks in background."""
        results = []
        for result in self.lazy_service.process_lazy(request):
            results.append(result)
        return results

    def _apply_basic_filter(
        self, signal_data: np.ndarray, filter_params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply basic filtering using scipy."""
        try:
            from scipy import signal

            filtered_data = signal_data.copy()

            # Apply filter based on parameters
            if filter_params.get("filter_type") == "lowpass":
                nyquist = filter_params.get("sampling_freq", 100) / 2
                cutoff = filter_params.get("low_freq", 10) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4), cutoff, btype="low"
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            elif filter_params.get("filter_type") == "highpass":
                nyquist = filter_params.get("sampling_freq", 100) / 2
                cutoff = filter_params.get("high_freq", 1) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4), cutoff, btype="high"
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            elif filter_params.get("filter_type") == "bandpass":
                nyquist = filter_params.get("sampling_freq", 100) / 2
                low_cutoff = filter_params.get("low_freq", 1) / nyquist
                high_cutoff = filter_params.get("high_freq", 10) / nyquist
                b, a = signal.butter(
                    filter_params.get("filter_order", 4),
                    [low_cutoff, high_cutoff],
                    btype="band",
                )
                filtered_data = signal.filtfilt(b, a, filtered_data)

            return filtered_data

        except Exception as e:
            logger.error(f"Error applying basic filter: {e}")
            return signal_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "service_stats": self.stats,
            "lazy_service_stats": self.lazy_service.stats,
            "memory_info": (
                self.memory_manager.get_memory_info() if self.memory_manager else None
            ),
            "vitaldsp_available": VITALDSP_AVAILABLE,
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.memory_manager:
            self.memory_manager.cleanup()

        logger.info("HeavyDataFilteringService cleaned up")


# Global service instance
_heavy_data_filtering_service = None


def get_heavy_data_filtering_service() -> HeavyDataFilteringService:
    """Get global heavy data filtering service instance."""
    global _heavy_data_filtering_service
    if _heavy_data_filtering_service is None:
        _heavy_data_filtering_service = HeavyDataFilteringService()
    return _heavy_data_filtering_service


def create_filtering_request(
    request_id: str,
    signal_data: np.ndarray,
    sampling_freq: float,
    filter_params: Dict[str, Any],
    signal_type: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> FilteringRequest:
    """Create a filtering request."""
    return FilteringRequest(
        request_id=request_id,
        signal_data=signal_data,
        sampling_freq=sampling_freq,
        signal_type=signal_type,
        filter_params=filter_params,
        metadata=metadata or {},
    )
