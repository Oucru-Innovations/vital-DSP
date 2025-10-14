"""
Data Compression Manager for Memory Optimization

This module implements intelligent data compression with multiple algorithms,
adaptive compression selection, and performance optimization for the vitalDSP webapp.

Author: vitalDSP Development Team
Date: January 14, 2025
Version: 1.0.0 (Phase 3B)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time
import pickle
import zlib
import gzip
import lz4.frame
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import optional compression libraries
try:
    import blosc
    BLOSC_AVAILABLE = True
except ImportError:
    BLOSC_AVAILABLE = False
    logger.warning("blosc not available - using fallback compression")

try:
    import lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("lz4 not available - using fallback compression")


class CompressionMethod(Enum):
    """Enumeration of available compression methods."""
    
    LZ4 = "lz4"           # Fast compression
    ZLIB = "zlib"         # Balanced compression
    GZIP = "gzip"         # Good compression ratio
    BLOSC = "blosc"       # High-performance compression
    PICKLE = "pickle"     # Python-specific compression
    NONE = "none"         # No compression


class CompressionLevel(Enum):
    """Enumeration of compression levels."""
    
    FAST = 1      # Fast compression, lower ratio
    BALANCED = 2  # Balanced speed and ratio
    BEST = 3      # Best compression ratio, slower


@dataclass
class CompressionResult:
    """Data class for compression results."""
    
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    method_used: str
    compression_time: float
    decompression_time: float
    quality_score: float


@dataclass
class CompressionStats:
    """Data class for compression statistics."""
    
    total_compressions: int
    total_decompressions: int
    total_bytes_saved: int
    average_compression_ratio: float
    average_compression_time: float
    average_decompression_time: float
    method_performance: Dict[str, Dict[str, float]]


class DataCompressionManager:
    """
    Intelligent data compression manager with multiple algorithms.
    
    Provides efficient data compression with automatic method selection,
    performance monitoring, and adaptive compression strategies.
    """
    
    def __init__(
        self, 
        default_method: CompressionMethod = CompressionMethod.LZ4,
        default_level: CompressionLevel = CompressionLevel.BALANCED,
        enable_adaptive_selection: bool = True
    ):
        """
        Initialize the data compression manager.
        
        Args:
            default_method: Default compression method
            default_level: Default compression level
            enable_adaptive_selection: Whether to use adaptive method selection
        """
        self.default_method = default_method
        self.default_level = default_level
        self.enable_adaptive_selection = enable_adaptive_selection
        
        # Available methods based on installed libraries
        self.available_methods = self._get_available_methods()
        
        # Performance tracking
        self.stats = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "total_bytes_saved": 0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
            "method_performance": {}
        }
        
        # Compression cache
        self.compression_cache = {}
        self.max_cache_size = 100
        
        logger.info(f"DataCompressionManager initialized with {len(self.available_methods)} methods")
    
    def compress_data(
        self, 
        data: Any, 
        method: Optional[CompressionMethod] = None,
        level: Optional[CompressionLevel] = None,
        cache_result: bool = True
    ) -> CompressionResult:
        """
        Compress data using specified or optimal method.
        
        Args:
            data: Data to compress
            method: Compression method (None for auto-selection)
            level: Compression level (None for default)
            cache_result: Whether to cache the result
            
        Returns:
            CompressionResult with compressed data and metadata
        """
        start_time = time.time()
        
        # Use defaults if not specified
        if method is None:
            method = self._select_optimal_method(data) if self.enable_adaptive_selection else self.default_method
        
        if level is None:
            level = self.default_level
        
        # Check cache first
        cache_key = self._get_cache_key(data, method, level)
        if cache_result and cache_key in self.compression_cache:
            logger.debug(f"Using cached compression for {method.value}")
            return self.compression_cache[cache_key]
        
        # Serialize data if needed
        serialized_data = self._serialize_data(data)
        original_size = len(serialized_data)
        
        # Compress data
        compressed_data = self._compress_bytes(serialized_data, method, level)
        compressed_size = len(compressed_data)
        
        # Calculate metrics
        compression_time = time.time() - start_time
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Test decompression speed
        decomp_start = time.time()
        self._decompress_bytes(compressed_data, method)
        decompression_time = time.time() - decomp_start
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            compression_ratio, compression_time, decompression_time
        )
        
        result = CompressionResult(
            compressed_data=compressed_data,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            method_used=method.value,
            compression_time=compression_time,
            decompression_time=decompression_time,
            quality_score=quality_score
        )
        
        # Update statistics
        self._update_stats(result)
        
        # Cache result if requested
        if cache_result:
            self._cache_result(cache_key, result)
        
        logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes "
                   f"using {method.value} (ratio: {compression_ratio:.2f}, "
                   f"time: {compression_time:.3f}s)")
        
        return result
    
    def decompress_data(
        self, 
        compressed_data: bytes, 
        method: CompressionMethod,
        original_type: Optional[type] = None
    ) -> Any:
        """
        Decompress data using specified method.
        
        Args:
            compressed_data: Compressed data bytes
            method: Compression method used
            original_type: Optional original data type for deserialization
            
        Returns:
            Decompressed data
        """
        start_time = time.time()
        
        # Decompress bytes
        decompressed_bytes = self._decompress_bytes(compressed_data, method)
        
        # Deserialize data
        data = self._deserialize_data(decompressed_bytes, original_type)
        
        # Update statistics
        decompression_time = time.time() - start_time
        self.stats["total_decompressions"] += 1
        self.stats["total_decompression_time"] += decompression_time
        
        logger.debug(f"Decompressed data using {method.value} in {decompression_time:.3f}s")
        
        return data
    
    def _compress_bytes(
        self, 
        data: bytes, 
        method: CompressionMethod, 
        level: CompressionLevel
    ) -> bytes:
        """Compress bytes using specified method and level."""
        level_value = self._get_level_value(level)
        
        if method == CompressionMethod.LZ4:
            if LZ4_AVAILABLE:
                return lz4.frame.compress(data, compression_level=level_value)
            else:
                # Fallback to zlib
                return zlib.compress(data, level_value)
        
        elif method == CompressionMethod.ZLIB:
            return zlib.compress(data, level_value)
        
        elif method == CompressionMethod.GZIP:
            return gzip.compress(data, compresslevel=level_value)
        
        elif method == CompressionMethod.BLOSC:
            if BLOSC_AVAILABLE:
                return blosc.compress(data, clevel=level_value)
            else:
                # Fallback to zlib
                return zlib.compress(data, level_value)
        
        elif method == CompressionMethod.PICKLE:
            # Pickle compression (uses zlib internally)
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif method == CompressionMethod.NONE:
            return data
        
        else:
            raise ValueError(f"Unsupported compression method: {method}")
    
    def _decompress_bytes(self, compressed_data: bytes, method: CompressionMethod) -> bytes:
        """Decompress bytes using specified method."""
        if method == CompressionMethod.LZ4:
            if LZ4_AVAILABLE:
                return lz4.frame.decompress(compressed_data)
            else:
                # Fallback to zlib
                return zlib.decompress(compressed_data)
        
        elif method == CompressionMethod.ZLIB:
            return zlib.decompress(compressed_data)
        
        elif method == CompressionMethod.GZIP:
            return gzip.decompress(compressed_data)
        
        elif method == CompressionMethod.BLOSC:
            if BLOSC_AVAILABLE:
                return blosc.decompress(compressed_data)
            else:
                # Fallback to zlib
                return zlib.decompress(compressed_data)
        
        elif method == CompressionMethod.PICKLE:
            return pickle.loads(compressed_data)
        
        elif method == CompressionMethod.NONE:
            return compressed_data
        
        else:
            raise ValueError(f"Unsupported compression method: {method}")
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, np.ndarray):
            return data.tobytes()
        elif isinstance(data, pd.DataFrame):
            return data.to_pickle()
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str).encode('utf-8')
        else:
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes, original_type: Optional[type] = None) -> Any:
        """Deserialize bytes back to original data type."""
        if original_type is None:
            # Try to detect type
            try:
                return pickle.loads(data)
            except:
                try:
                    return json.loads(data.decode('utf-8'))
                except:
                    return data
        
        elif original_type == np.ndarray:
            return np.frombuffer(data, dtype=np.float64)
        elif original_type == pd.DataFrame:
            return pd.read_pickle(data)
        elif original_type == bytes:
            return data
        else:
            return pickle.loads(data)
    
    def _get_level_value(self, level: CompressionLevel) -> int:
        """Get numeric level value for compression methods."""
        if level == CompressionLevel.FAST:
            return 1
        elif level == CompressionLevel.BALANCED:
            return 6
        elif level == CompressionLevel.BEST:
            return 9
        else:
            return 6
    
    def _select_optimal_method(self, data: Any) -> CompressionMethod:
        """Select optimal compression method based on data characteristics."""
        # Analyze data characteristics
        data_size = len(self._serialize_data(data))
        data_type = type(data).__name__
        
        # Use performance history to select method
        best_method = self.default_method
        best_score = 0.0
        
        for method_name, stats in self.stats["method_performance"].items():
            if stats["count"] > 0:
                # Score based on compression ratio and speed
                avg_ratio = stats["avg_compression_ratio"]
                avg_time = stats["avg_compression_time"]
                
                # Prefer methods with good compression ratio and reasonable speed
                speed_score = 1.0 / (1.0 + avg_time)  # Faster is better
                ratio_score = min(1.0, avg_ratio / 10.0)  # Cap ratio score
                
                combined_score = 0.6 * ratio_score + 0.4 * speed_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_method = CompressionMethod(method_name)
        
        # Override based on data characteristics
        if data_size < 1024:  # Small data
            return CompressionMethod.ZLIB
        elif data_size > 10 * 1024 * 1024:  # Large data
            return CompressionMethod.LZ4
        elif data_type == "ndarray":  # NumPy arrays
            return CompressionMethod.BLOSC if BLOSC_AVAILABLE else CompressionMethod.LZ4
        else:
            return best_method
    
    def _calculate_quality_score(
        self, 
        compression_ratio: float, 
        compression_time: float, 
        decompression_time: float
    ) -> float:
        """Calculate quality score for compression result."""
        # Normalize metrics
        ratio_score = min(1.0, compression_ratio / 10.0)  # Cap at 10x compression
        speed_score = 1.0 / (1.0 + compression_time + decompression_time)
        
        # Weighted combination
        quality_score = 0.7 * ratio_score + 0.3 * speed_score
        
        return min(1.0, max(0.0, quality_score))
    
    def _get_cache_key(self, data: Any, method: CompressionMethod, level: CompressionLevel) -> str:
        """Generate cache key for compression result."""
        data_hash = hash(str(data))
        return f"{data_hash}_{method.value}_{level.value}"
    
    def _cache_result(self, cache_key: str, result: CompressionResult):
        """Cache compression result."""
        if len(self.compression_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.compression_cache))
            del self.compression_cache[oldest_key]
        
        self.compression_cache[cache_key] = result
    
    def _update_stats(self, result: CompressionResult):
        """Update compression statistics."""
        self.stats["total_compressions"] += 1
        self.stats["total_bytes_saved"] += (result.original_size - result.compressed_size)
        self.stats["total_compression_time"] += result.compression_time
        
        # Update method-specific stats
        method_name = result.method_used
        if method_name not in self.stats["method_performance"]:
            self.stats["method_performance"][method_name] = {
                "count": 0,
                "avg_compression_ratio": 0.0,
                "avg_compression_time": 0.0,
                "avg_quality_score": 0.0
            }
        
        method_stats = self.stats["method_performance"][method_name]
        method_stats["count"] += 1
        
        # Update running averages
        method_stats["avg_compression_ratio"] = (
            (method_stats["avg_compression_ratio"] * (method_stats["count"] - 1) + result.compression_ratio) 
            / method_stats["count"]
        )
        method_stats["avg_compression_time"] = (
            (method_stats["avg_compression_time"] * (method_stats["count"] - 1) + result.compression_time) 
            / method_stats["count"]
        )
        method_stats["avg_quality_score"] = (
            (method_stats["avg_quality_score"] * (method_stats["count"] - 1) + result.quality_score) 
            / method_stats["count"]
        )
    
    def _get_available_methods(self) -> List[CompressionMethod]:
        """Get list of available compression methods."""
        methods = [CompressionMethod.NONE, CompressionMethod.ZLIB, CompressionMethod.GZIP]
        
        if LZ4_AVAILABLE:
            methods.append(CompressionMethod.LZ4)
        
        if BLOSC_AVAILABLE:
            methods.append(CompressionMethod.BLOSC)
        
        methods.append(CompressionMethod.PICKLE)
        
        return methods
    
    def get_compression_stats(self) -> CompressionStats:
        """Get comprehensive compression statistics."""
        total_compressions = self.stats["total_compressions"]
        total_decompressions = self.stats["total_decompressions"]
        
        avg_compression_ratio = 0.0
        avg_compression_time = 0.0
        avg_decompression_time = 0.0
        
        if total_compressions > 0:
            avg_compression_time = self.stats["total_compression_time"] / total_compressions
        
        if total_decompressions > 0:
            avg_decompression_time = self.stats["total_decompression_time"] / total_decompressions
        
        # Calculate average compression ratio across all methods
        if self.stats["method_performance"]:
            total_ratio = sum(
                stats["avg_compression_ratio"] * stats["count"] 
                for stats in self.stats["method_performance"].values()
            )
            total_count = sum(
                stats["count"] 
                for stats in self.stats["method_performance"].values()
            )
            if total_count > 0:
                avg_compression_ratio = total_ratio / total_count
        
        return CompressionStats(
            total_compressions=total_compressions,
            total_decompressions=total_decompressions,
            total_bytes_saved=self.stats["total_bytes_saved"],
            average_compression_ratio=avg_compression_ratio,
            average_compression_time=avg_compression_time,
            average_decompression_time=avg_decompression_time,
            method_performance=self.stats["method_performance"]
        )
    
    def get_method_recommendations(self, data_size: int) -> List[Tuple[CompressionMethod, float]]:
        """Get compression method recommendations based on data size."""
        recommendations = []
        
        for method in self.available_methods:
            if method.value in self.stats["method_performance"]:
                stats = self.stats["method_performance"][method.value]
                if stats["count"] > 0:
                    score = stats["avg_quality_score"]
                    recommendations.append((method, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def clear_cache(self):
        """Clear compression cache."""
        self.compression_cache.clear()
        logger.info("Compression cache cleared")
    
    def reset_stats(self):
        """Reset compression statistics."""
        self.stats = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "total_bytes_saved": 0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
            "method_performance": {}
        }
        logger.info("Compression statistics reset")


# Global instance for webapp use
_data_compression_manager = None


def get_data_compression_manager() -> DataCompressionManager:
    """Get global data compression manager instance."""
    global _data_compression_manager
    if _data_compression_manager is None:
        _data_compression_manager = DataCompressionManager()
    return _data_compression_manager
