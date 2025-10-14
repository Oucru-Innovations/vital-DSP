"""
Adaptive Downsampling for Large Dataset Visualization

This module implements intelligent data downsampling algorithms optimized for
visualization performance while preserving important signal features.

Author: vitalDSP Development Team
Date: January 14, 2025
Version: 1.0.0 (Phase 3B)
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)


class DownsamplingMethod(Enum):
    """Enumeration of available downsampling methods."""
    
    LTTB = "lttb"  # Largest Triangle Three Buckets
    UNIFORM = "uniform"  # Uniform sampling
    PEAK_PRESERVING = "peak_preserving"  # Peak-preserving sampling
    ADAPTIVE = "adaptive"  # Adaptive based on signal characteristics


@dataclass
class DownsamplingResult:
    """Data class for downsampling results."""
    
    downsampled_data: np.ndarray
    downsampled_time: Optional[np.ndarray]
    original_indices: np.ndarray
    compression_ratio: float
    method_used: str
    processing_time: float
    quality_score: float


class AdaptiveDownsampler:
    """
    Intelligent data downsampling for visualization optimization.
    
    Implements multiple downsampling algorithms with automatic method selection
    based on signal characteristics and visualization requirements.
    """
    
    def __init__(self, max_points: int = 10000, quality_threshold: float = 0.95):
        """
        Initialize the adaptive downsampler.
        
        Args:
            max_points: Maximum number of points for visualization
            quality_threshold: Minimum quality score to accept downsampling
        """
        self.max_points = max_points
        self.quality_threshold = quality_threshold
        self.downsampling_cache = {}
        self.method_performance = {
            DownsamplingMethod.LTTB: {"avg_time": 0.0, "avg_quality": 0.0, "count": 0},
            DownsamplingMethod.UNIFORM: {"avg_time": 0.0, "avg_quality": 0.0, "count": 0},
            DownsamplingMethod.PEAK_PRESERVING: {"avg_time": 0.0, "avg_quality": 0.0, "count": 0},
            DownsamplingMethod.ADAPTIVE: {"avg_time": 0.0, "avg_quality": 0.0, "count": 0},
        }
        
        logger.info(f"AdaptiveDownsampler initialized with max_points={max_points}")
    
    def downsample_for_display(
        self, 
        data: np.ndarray, 
        time_axis: Optional[np.ndarray] = None,
        method: Optional[DownsamplingMethod] = None,
        preserve_features: bool = True
    ) -> DownsamplingResult:
        """
        Downsample data while preserving important features.
        
        Args:
            data: Input signal data
            time_axis: Optional time axis data
            method: Specific downsampling method to use
            preserve_features: Whether to preserve signal features
            
        Returns:
            DownsamplingResult with downsampled data and metadata
        """
        start_time = time.time()
        
        if len(data) <= self.max_points:
            # No downsampling needed
            return DownsamplingResult(
                downsampled_data=data,
                downsampled_time=time_axis,
                original_indices=np.arange(len(data)),
                compression_ratio=1.0,
                method_used="none",
                processing_time=time.time() - start_time,
                quality_score=1.0
            )
        
        # Select optimal method if not specified
        if method is None:
            method = self._select_optimal_method(data, preserve_features)
        
        # Check cache first
        cache_key = self._get_cache_key(data, method, preserve_features)
        if cache_key in self.downsampling_cache:
            logger.debug(f"Using cached downsampling result for {method.value}")
            return self.downsampling_cache[cache_key]
        
        # Perform downsampling
        if method == DownsamplingMethod.LTTB:
            result = self._lttb_downsample(data, time_axis)
        elif method == DownsamplingMethod.UNIFORM:
            result = self._uniform_downsample(data, time_axis)
        elif method == DownsamplingMethod.PEAK_PRESERVING:
            result = self._peak_preserving_downsample(data, time_axis)
        elif method == DownsamplingMethod.ADAPTIVE:
            result = self._adaptive_downsample(data, time_axis, preserve_features)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(data, result.downsampled_data)
        result.quality_score = quality_score
        result.processing_time = time.time() - start_time
        
        # Update performance statistics
        self._update_method_performance(method, result.processing_time, quality_score)
        
        # Cache result if quality is acceptable
        if quality_score >= self.quality_threshold:
            self.downsampling_cache[cache_key] = result
        
        logger.info(f"Downsampled {len(data)} points to {len(result.downsampled_data)} "
                   f"using {method.value} (quality: {quality_score:.3f}, "
                   f"time: {result.processing_time:.3f}s)")
        
        return result
    
    def _lttb_downsample(
        self, 
        data: np.ndarray, 
        time_axis: Optional[np.ndarray] = None
    ) -> DownsamplingResult:
        """
        Largest Triangle Three Buckets (LTTB) downsampling algorithm.
        
        Preserves peaks, valleys, and trends while reducing data points.
        """
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        if len(data) <= self.max_points:
            return DownsamplingResult(
                downsampled_data=data,
                downsampled_time=time_axis,
                original_indices=np.arange(len(data)),
                compression_ratio=1.0,
                method_used="lttb",
                processing_time=0.0,
                quality_score=1.0
            )
        
        # LTTB algorithm implementation
        sampled_indices = [0]  # Always include first point
        
        bucket_size = (len(data) - 2) / (self.max_points - 2)
        
        for i in range(self.max_points - 2):
            bucket_start = int(i * bucket_size) + 1
            bucket_end = int((i + 1) * bucket_size)
            
            # Find point with largest triangle area
            max_area = -1
            max_area_point = bucket_start
            
            for j in range(bucket_start, bucket_end):
                area = self._calculate_triangle_area(
                    data[sampled_indices[-1]], data[j], 
                    np.mean(data[bucket_start:bucket_end])
                )
                if area > max_area:
                    max_area = area
                    max_area_point = j
            
            sampled_indices.append(max_area_point)
        
        sampled_indices.append(len(data) - 1)  # Always include last point
        
        return DownsamplingResult(
            downsampled_data=data[sampled_indices],
            downsampled_time=time_axis[sampled_indices],
            original_indices=np.array(sampled_indices),
            compression_ratio=len(data) / len(sampled_indices),
            method_used="lttb",
            processing_time=0.0,
            quality_score=0.0  # Will be calculated later
        )
    
    def _uniform_downsample(
        self, 
        data: np.ndarray, 
        time_axis: Optional[np.ndarray] = None
    ) -> DownsamplingResult:
        """Uniform downsampling with equal intervals."""
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        step = len(data) // self.max_points
        indices = np.arange(0, len(data), step)[:self.max_points]
        
        return DownsamplingResult(
            downsampled_data=data[indices],
            downsampled_time=time_axis[indices],
            original_indices=indices,
            compression_ratio=len(data) / len(indices),
            method_used="uniform",
            processing_time=0.0,
            quality_score=0.0
        )
    
    def _peak_preserving_downsample(
        self, 
        data: np.ndarray, 
        time_axis: Optional[np.ndarray] = None
    ) -> DownsamplingResult:
        """Peak-preserving downsampling that maintains signal peaks."""
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        # Find peaks using simple peak detection
        peaks = self._find_simple_peaks(data)
        
        # Calculate base sampling rate
        base_step = len(data) // self.max_points
        
        # Create sampling indices
        sampled_indices = set()
        
        # Always include first and last points
        sampled_indices.add(0)
        sampled_indices.add(len(data) - 1)
        
        # Add peaks
        for peak in peaks:
            sampled_indices.add(peak)
        
        # Add uniform samples between peaks
        for i in range(0, len(data), base_step):
            sampled_indices.add(i)
        
        # Convert to sorted array and limit to max_points
        indices = sorted(list(sampled_indices))
        if len(indices) > self.max_points:
            # Keep most important points (peaks + uniform distribution)
            peak_indices = [i for i in indices if i in peaks]
            uniform_indices = [i for i in indices if i not in peaks]
            
            # Keep all peaks and fill remaining with uniform samples
            remaining_slots = self.max_points - len(peak_indices) - 2  # -2 for first/last
            if remaining_slots > 0:
                uniform_step = len(uniform_indices) // remaining_slots
                selected_uniform = uniform_indices[::uniform_step][:remaining_slots]
                indices = sorted([0] + peak_indices + selected_uniform + [len(data) - 1])
            else:
                indices = sorted([0] + peak_indices[:self.max_points-2] + [len(data) - 1])
        
        return DownsamplingResult(
            downsampled_data=data[indices],
            downsampled_time=time_axis[indices],
            original_indices=np.array(indices),
            compression_ratio=len(data) / len(indices),
            method_used="peak_preserving",
            processing_time=0.0,
            quality_score=0.0
        )
    
    def _adaptive_downsample(
        self, 
        data: np.ndarray, 
        time_axis: Optional[np.ndarray] = None,
        preserve_features: bool = True
    ) -> DownsamplingResult:
        """Adaptive downsampling based on signal characteristics."""
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        # Analyze signal characteristics
        signal_stats = self._analyze_signal_characteristics(data)
        
        # Choose method based on characteristics
        if signal_stats["has_peaks"] and preserve_features:
            return self._peak_preserving_downsample(data, time_axis)
        elif signal_stats["is_smooth"]:
            return self._uniform_downsample(data, time_axis)
        else:
            return self._lttb_downsample(data, time_axis)
    
    def _select_optimal_method(
        self, 
        data: np.ndarray, 
        preserve_features: bool
    ) -> DownsamplingMethod:
        """Select optimal downsampling method based on signal characteristics."""
        signal_stats = self._analyze_signal_characteristics(data)
        
        # Use performance history to select method
        best_method = DownsamplingMethod.LTTB
        best_score = 0.0
        
        for method, stats in self.method_performance.items():
            if stats["count"] > 0:
                # Score based on quality and speed
                quality_score = stats["avg_quality"]
                speed_score = 1.0 / (1.0 + stats["avg_time"])  # Faster is better
                combined_score = 0.7 * quality_score + 0.3 * speed_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_method = method
        
        # Override based on signal characteristics
        if signal_stats["has_peaks"] and preserve_features:
            return DownsamplingMethod.PEAK_PRESERVING
        elif signal_stats["is_smooth"]:
            return DownsamplingMethod.UNIFORM
        else:
            return best_method
    
    def _analyze_signal_characteristics(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze signal characteristics to guide downsampling method selection."""
        # Calculate basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Detect peaks
        peaks = self._find_simple_peaks(data)
        has_peaks = len(peaks) > len(data) * 0.01  # More than 1% peaks
        
        # Check smoothness
        diff = np.diff(data)
        smoothness = 1.0 / (1.0 + np.std(diff) / std_val) if std_val > 0 else 1.0
        is_smooth = smoothness > 0.8
        
        # Check variability
        variability = std_val / abs(mean_val) if mean_val != 0 else float('inf')
        is_variable = variability > 0.1
        
        return {
            "has_peaks": has_peaks,
            "is_smooth": is_smooth,
            "is_variable": is_variable,
            "peak_count": len(peaks),
            "smoothness": smoothness,
            "variability": variability
        }
    
    def _find_simple_peaks(self, data: np.ndarray, min_height: float = None) -> List[int]:
        """Find peaks using simple threshold-based detection."""
        if min_height is None:
            min_height = np.std(data) * 0.5
        
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and 
                data[i] > data[i+1] and 
                data[i] > np.mean(data) + min_height):
                peaks.append(i)
        
        return peaks
    
    def _calculate_triangle_area(self, y1: float, y2: float, y3: float) -> float:
        """Calculate triangle area for LTTB algorithm."""
        # Simplified triangle area calculation
        return abs((y1 - y2) * (y2 - y3))
    
    def _calculate_quality_score(
        self, 
        original: np.ndarray, 
        downsampled: np.ndarray
    ) -> float:
        """Calculate quality score for downsampling result."""
        if len(original) == len(downsampled):
            return 1.0
        
        # Calculate correlation coefficient
        if len(downsampled) < 2:
            return 0.0
        
        # Interpolate downsampled data to original length for comparison
        indices = np.linspace(0, len(downsampled) - 1, len(original))
        interpolated = np.interp(np.arange(len(original)), indices, downsampled)
        
        # Calculate correlation
        correlation = np.corrcoef(original, interpolated)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Calculate relative error
        mse = np.mean((original - interpolated) ** 2)
        relative_error = mse / np.var(original) if np.var(original) > 0 else 1.0
        
        # Combine correlation and error into quality score
        quality_score = 0.7 * max(0, correlation) + 0.3 * max(0, 1.0 - relative_error)
        
        return min(1.0, max(0.0, quality_score))
    
    def _get_cache_key(
        self, 
        data: np.ndarray, 
        method: DownsamplingMethod, 
        preserve_features: bool
    ) -> str:
        """Generate cache key for downsampling result."""
        # Use data hash and parameters for cache key
        data_hash = hash(data.tobytes())
        return f"{data_hash}_{method.value}_{preserve_features}_{self.max_points}"
    
    def _update_method_performance(
        self, 
        method: DownsamplingMethod, 
        processing_time: float, 
        quality_score: float
    ):
        """Update performance statistics for method selection."""
        stats = self.method_performance[method]
        stats["count"] += 1
        stats["avg_time"] = (stats["avg_time"] * (stats["count"] - 1) + processing_time) / stats["count"]
        stats["avg_quality"] = (stats["avg_quality"] * (stats["count"] - 1) + quality_score) / stats["count"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all methods."""
        return {
            "method_performance": self.method_performance,
            "cache_size": len(self.downsampling_cache),
            "max_points": self.max_points,
            "quality_threshold": self.quality_threshold
        }
    
    def clear_cache(self):
        """Clear the downsampling cache."""
        self.downsampling_cache.clear()
        logger.info("Downsampling cache cleared")


# Global instance for webapp use
_adaptive_downsampler = None


def get_adaptive_downsampler(max_points: int = 10000) -> AdaptiveDownsampler:
    """Get global adaptive downsampler instance."""
    global _adaptive_downsampler
    if _adaptive_downsampler is None:
        _adaptive_downsampler = AdaptiveDownsampler(max_points=max_points)
    return _adaptive_downsampler
