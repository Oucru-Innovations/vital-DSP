"""
Garbage Collection Optimization for Webapp Performance

This module implements intelligent garbage collection optimization with
automatic cleanup, memory leak detection, and performance monitoring.

Author: vitalDSP Development Team
Date: January 14, 2025
Version: 1.0.0 (Phase 3B)
"""

import gc
import psutil
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
import weakref
from collections import defaultdict, deque
import tracemalloc
import sys

logger = logging.getLogger(__name__)


class GCStrategy(Enum):
    """Enumeration of garbage collection strategies."""
    
    AUTOMATIC = "automatic"      # Automatic GC based on thresholds
    MANUAL = "manual"           # Manual GC triggers only
    AGGRESSIVE = "aggressive"   # Frequent GC for memory-constrained environments
    CONSERVATIVE = "conservative"  # Minimal GC for performance-critical environments


class MemoryLeakType(Enum):
    """Enumeration of memory leak types."""
    
    CIRCULAR_REFERENCE = "circular_reference"
    UNRELEASED_RESOURCE = "unreleased_resource"
    CACHE_GROWTH = "cache_growth"
    EVENT_LISTENER = "event_listener"
    THREAD_LEAK = "thread_leak"


@dataclass
class GCStats:
    """Data class for garbage collection statistics."""
    
    total_collections: int
    total_objects_collected: int
    total_time_spent: float
    average_collection_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_freed_mb: float
    collection_frequency: float


@dataclass
class MemoryLeakInfo:
    """Data class for memory leak information."""
    
    leak_type: MemoryLeakType
    description: str
    severity: str  # low, medium, high, critical
    objects_count: int
    memory_size_mb: float
    detection_time: float
    suggested_action: str


class GarbageCollectionOptimizer:
    """
    Intelligent garbage collection optimizer with leak detection.
    
    Provides automatic garbage collection optimization, memory leak detection,
    and performance monitoring for the vitalDSP webapp.
    """
    
    def __init__(
        self, 
        strategy: GCStrategy = GCStrategy.AUTOMATIC,
        gc_threshold: float = 0.8,
        leak_detection_enabled: bool = True
    ):
        """
        Initialize the garbage collection optimizer.
        
        Args:
            strategy: Garbage collection strategy
            gc_threshold: Memory threshold for triggering GC (0.0 to 1.0)
            leak_detection_enabled: Whether to enable memory leak detection
        """
        self.strategy = strategy
        self.gc_threshold = gc_threshold
        self.leak_detection_enabled = leak_detection_enabled
        
        # GC statistics
        self.stats = {
            "total_collections": 0,
            "total_objects_collected": 0,
            "total_time_spent": 0.0,
            "memory_freed_total_mb": 0.0,
            "leak_detections": 0,
            "false_positives": 0
        }
        
        # Memory monitoring
        self.memory_history = deque(maxlen=100)
        self.gc_history = deque(maxlen=50)
        
        # Leak detection
        self.leak_detector = None
        self.detected_leaks = []
        
        # Monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # Weak reference tracking
        self._weak_refs = {}
        self._ref_callbacks = {}
        
        # Initialize leak detection if enabled
        if self.leak_detection_enabled:
            self._initialize_leak_detection()
        
        logger.info(f"GarbageCollectionOptimizer initialized with strategy: {strategy.value}")
    
    def optimize_gc_settings(self):
        """Optimize garbage collection settings based on system and strategy."""
        if self.strategy == GCStrategy.AUTOMATIC:
            # Enable automatic garbage collection
            gc.enable()
            
            # Set thresholds based on system memory
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if total_memory_gb < 4:
                # Low memory system - more aggressive GC
                gc.set_threshold(100, 5, 5)
            elif total_memory_gb < 8:
                # Medium memory system - balanced GC
                gc.set_threshold(200, 10, 10)
            else:
                # High memory system - less frequent GC
                gc.set_threshold(500, 20, 20)
        
        elif self.strategy == GCStrategy.AGGRESSIVE:
            # Aggressive garbage collection
            gc.enable()
            gc.set_threshold(50, 3, 3)
        
        elif self.strategy == GCStrategy.CONSERVATIVE:
            # Conservative garbage collection
            gc.enable()
            gc.set_threshold(1000, 50, 50)
        
        elif self.strategy == GCStrategy.MANUAL:
            # Disable automatic garbage collection
            gc.disable()
        
        logger.info(f"GC settings optimized for {self.strategy.value} strategy")
    
    def force_garbage_collection(self, generation: Optional[int] = None) -> GCStats:
        """
        Force garbage collection and return statistics.
        
        Args:
            generation: Specific generation to collect (0, 1, 2, or None for all)
            
        Returns:
            GCStats with collection information
        """
        with self._lock:
            start_time = time.time()
            memory_before = self._get_memory_usage_mb()
            
            # Count objects before collection
            objects_before = len(gc.get_objects())
            
            # Perform garbage collection
            if generation is not None:
                collected = gc.collect(generation)
            else:
                collected = gc.collect()
            
            # Calculate metrics
            collection_time = time.time() - start_time
            memory_after = self._get_memory_usage_mb()
            memory_freed = memory_before - memory_after
            
            # Update statistics
            self.stats["total_collections"] += 1
            self.stats["total_objects_collected"] += collected
            self.stats["total_time_spent"] += collection_time
            self.stats["memory_freed_total_mb"] += memory_freed
            
            # Record in history
            gc_stats = GCStats(
                total_collections=self.stats["total_collections"],
                total_objects_collected=collected,
                total_time_spent=collection_time,
                average_collection_time=self.stats["total_time_spent"] / self.stats["total_collections"],
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_freed_mb=memory_freed,
                collection_frequency=1.0 / max(1, collection_time)
            )
            
            self.gc_history.append(gc_stats)
            
            logger.info(f"GC collected {collected} objects, freed {memory_freed:.1f}MB "
                       f"in {collection_time:.3f}s")
            
            return gc_stats
    
    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start garbage collection monitoring in background thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory_and_gc,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Started GC monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop garbage collection monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Stopped GC monitoring")
    
    def _monitor_memory_and_gc(self, interval_seconds: float):
        """Background monitoring for memory and garbage collection."""
        while self._monitoring_active:
            try:
                # Record current memory usage
                current_memory = self._get_memory_usage_mb()
                self.memory_history.append({
                    "timestamp": time.time(),
                    "memory_mb": current_memory,
                    "objects_count": len(gc.get_objects())
                })
                
                # Check if GC should be triggered
                if self.strategy == GCStrategy.AUTOMATIC:
                    memory_percentage = current_memory / (psutil.virtual_memory().total / (1024**2))
                    
                    if memory_percentage > self.gc_threshold:
                        logger.debug(f"Memory usage {memory_percentage:.1%} exceeds threshold, triggering GC")
                        self.force_garbage_collection()
                
                # Detect memory leaks
                if self.leak_detection_enabled:
                    self._detect_memory_leaks()
                
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in GC monitoring: {e}")
                time.sleep(interval_seconds)
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        if len(self.memory_history) < 10:
            return
        
        # Analyze memory trend
        recent_memory = [entry["memory_mb"] for entry in list(self.memory_history)[-10:]]
        memory_trend = self._calculate_trend(recent_memory)
        
        # Detect growing memory usage
        if memory_trend > 0.1:  # Growing by more than 10% over recent samples
            leak_info = MemoryLeakInfo(
                leak_type=MemoryLeakType.CACHE_GROWTH,
                description=f"Memory usage growing by {memory_trend:.1%} over recent samples",
                severity="medium",
                objects_count=len(gc.get_objects()),
                memory_size_mb=recent_memory[-1],
                detection_time=time.time(),
                suggested_action="Review cache sizes and implement cleanup policies"
            )
            
            self._record_leak_detection(leak_info)
        
        # Detect circular references
        self._detect_circular_references()
    
    def _detect_circular_references(self):
        """Detect circular references in the object graph."""
        try:
            # Get objects that are not collectable
            uncollectable = gc.garbage
            
            if len(uncollectable) > 0:
                leak_info = MemoryLeakInfo(
                    leak_type=MemoryLeakType.CIRCULAR_REFERENCE,
                    description=f"Found {len(uncollectable)} uncollectable objects",
                    severity="high",
                    objects_count=len(uncollectable),
                    memory_size_mb=self._estimate_objects_size_mb(uncollectable),
                    detection_time=time.time(),
                    suggested_action="Review object references and break circular dependencies"
                )
                
                self._record_leak_detection(leak_info)
        
        except Exception as e:
            logger.warning(f"Error detecting circular references: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values (positive = growing, negative = shrinking)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope / values[0] if values[0] != 0 else 0.0
    
    def _record_leak_detection(self, leak_info: MemoryLeakInfo):
        """Record memory leak detection."""
        self.detected_leaks.append(leak_info)
        self.stats["leak_detections"] += 1
        
        logger.warning(f"Memory leak detected: {leak_info.description} "
                      f"(severity: {leak_info.severity})")
    
    def _estimate_objects_size_mb(self, objects: List[Any]) -> float:
        """Estimate memory size of objects in MB."""
        try:
            total_size = 0
            for obj in objects:
                total_size += sys.getsizeof(obj)
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _initialize_leak_detection(self):
        """Initialize memory leak detection."""
        try:
            # Start memory tracing if available
            if hasattr(tracemalloc, 'start'):
                tracemalloc.start()
                logger.info("Memory tracing enabled for leak detection")
        except Exception as e:
            logger.warning(f"Could not enable memory tracing: {e}")
    
    def register_weak_reference(
        self, 
        obj: Any, 
        callback: Optional[Callable] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Register a weak reference for automatic cleanup tracking.
        
        Args:
            obj: Object to track
            callback: Optional callback when object is garbage collected
            name: Optional name for the reference
            
        Returns:
            Reference ID
        """
        ref_id = name or f"ref_{len(self._weak_refs)}"
        
        def cleanup_callback(weak_ref):
            if callback:
                callback(ref_id)
            if ref_id in self._ref_callbacks:
                del self._ref_callbacks[ref_id]
        
        self._weak_refs[ref_id] = weakref.ref(obj, cleanup_callback)
        if callback:
            self._ref_callbacks[ref_id] = callback
        
        logger.debug(f"Registered weak reference: {ref_id}")
        return ref_id
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get comprehensive garbage collection statistics."""
        with self._lock:
            avg_collection_time = (
                self.stats["total_time_spent"] / self.stats["total_collections"]
                if self.stats["total_collections"] > 0 else 0.0
            )
            
            return {
                **self.stats,
                "average_collection_time": avg_collection_time,
                "monitoring_active": self._monitoring_active,
                "strategy": self.strategy.value,
                "gc_threshold": self.gc_threshold,
                "weak_refs_count": len(self._weak_refs),
                "detected_leaks_count": len(self.detected_leaks)
            }
    
    def get_memory_leaks(self) -> List[MemoryLeakInfo]:
        """Get list of detected memory leaks."""
        return self.detected_leaks.copy()
    
    def clear_leak_history(self):
        """Clear memory leak detection history."""
        self.detected_leaks.clear()
        logger.info("Memory leak history cleared")
    
    def get_memory_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Get memory usage trend analysis."""
        if len(self.memory_history) < window_size:
            return {"trend": 0.0, "samples": len(self.memory_history)}
        
        recent_entries = list(self.memory_history)[-window_size:]
        memory_values = [entry["memory_mb"] for entry in recent_entries]
        
        trend = self._calculate_trend(memory_values)
        
        return {
            "trend": trend,
            "samples": len(recent_entries),
            "current_memory_mb": memory_values[-1],
            "average_memory_mb": sum(memory_values) / len(memory_values),
            "memory_range_mb": max(memory_values) - min(memory_values)
        }
    
    def optimize_for_webapp(self):
        """Optimize garbage collection specifically for webapp usage."""
        # Set webapp-specific GC thresholds
        gc.set_threshold(300, 15, 15)
        
        # Enable automatic collection
        gc.enable()
        
        # Set up periodic cleanup
        if not self._monitoring_active:
            self.start_monitoring(interval_seconds=60.0)  # Check every minute
        
        logger.info("GC optimized for webapp usage")
    
    def cleanup_webapp_resources(self):
        """Clean up webapp-specific resources."""
        # Clear any webapp-specific caches
        # This would be called during webapp shutdown
        
        # Force final garbage collection
        self.force_garbage_collection()
        
        logger.info("Webapp resources cleaned up")


# Global instance for webapp use
_gc_optimizer = None


def get_gc_optimizer(strategy: GCStrategy = GCStrategy.AUTOMATIC) -> GarbageCollectionOptimizer:
    """Get global garbage collection optimizer instance."""
    global _gc_optimizer
    if _gc_optimizer is None:
        _gc_optimizer = GarbageCollectionOptimizer(strategy=strategy)
    return _gc_optimizer
