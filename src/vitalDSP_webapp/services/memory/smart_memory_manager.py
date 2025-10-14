"""
Smart Memory Manager for Webapp Performance Optimization

This module implements intelligent memory management with priority-based eviction,
adaptive resource allocation, and performance optimization for the vitalDSP webapp.

Author: vitalDSP Development Team
Date: January 14, 2025
Version: 1.0.0 (Phase 3B)
"""

import psutil
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import gc
from collections import OrderedDict
import weakref

logger = logging.getLogger(__name__)


class MemoryPriority(Enum):
    """Enumeration of memory priority levels."""
    
    CRITICAL = 1    # Critical system data
    HIGH = 2        # User data, active processing
    MEDIUM = 3      # Cached results, background processing
    LOW = 4         # Temporary data, logs
    CLEANUP = 5     # Marked for cleanup


class MemoryStrategy(Enum):
    """Enumeration of memory management strategies."""
    
    CONSERVATIVE = "conservative"  # Use minimal memory
    BALANCED = "balanced"         # Balance memory and performance
    AGGRESSIVE = "aggressive"     # Use maximum available memory


@dataclass
class MemoryItem:
    """Data class for memory item tracking."""
    
    data_id: str
    data: Any
    size_mb: float
    priority: MemoryPriority
    access_time: float
    access_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Data class for memory statistics."""
    
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percentage: float
    registered_items: int
    total_registered_mb: float
    cache_hit_rate: float
    eviction_count: int
    gc_count: int


class SmartMemoryManager:
    """
    Intelligent memory manager with priority-based eviction and adaptive strategies.
    
    Provides efficient memory management for the webapp with automatic cleanup,
    priority-based eviction, and performance monitoring.
    """
    
    def __init__(
        self, 
        max_memory_mb: int = 500,
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
        gc_threshold: float = 0.8
    ):
        """
        Initialize the smart memory manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            strategy: Memory management strategy
            gc_threshold: Threshold for garbage collection (0.0 to 1.0)
        """
        self.max_memory_mb = max_memory_mb
        self.strategy = strategy
        self.gc_threshold = gc_threshold
        
        # Memory registry
        self.memory_registry: OrderedDict[str, MemoryItem] = OrderedDict()
        self.priority_queues: Dict[MemoryPriority, List[str]] = {
            priority: [] for priority in MemoryPriority
        }
        
        # Performance tracking
        self.stats = {
            "total_registrations": 0,
            "total_evictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gc_triggers": 0,
            "memory_warnings": 0,
            "total_memory_saved_mb": 0.0,
            "average_access_time": 0.0
        }
        
        # Monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # Weak references for automatic cleanup
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        logger.info(f"SmartMemoryManager initialized with {max_memory_mb}MB limit, "
                   f"strategy: {strategy.value}")
    
    def register_data(
        self, 
        data_id: str, 
        data: Any, 
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register data with memory manager.
        
        Args:
            data_id: Unique identifier for data
            data: Data to register
            priority: Memory priority level
            metadata: Optional metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            # Calculate data size
            size_mb = self._estimate_size_mb(data)
            
            # Check if we need to make space
            if not self._can_allocate_memory(size_mb):
                if not self._make_space(size_mb):
                    logger.warning(f"Cannot allocate {size_mb:.1f}MB for {data_id}")
                    return False
            
            # Create memory item
            memory_item = MemoryItem(
                data_id=data_id,
                data=data,
                size_mb=size_mb,
                priority=priority,
                access_time=time.time(),
                metadata=metadata or {}
            )
            
            # Register data
            self.memory_registry[data_id] = memory_item
            self.priority_queues[priority].append(data_id)
            
            # Create weak reference for automatic cleanup
            self._weak_refs[data_id] = weakref.ref(data, self._cleanup_callback)
            
            # Update statistics
            self.stats["total_registrations"] += 1
            
            logger.debug(f"Registered {data_id} ({size_mb:.1f}MB, priority: {priority.name})")
            
            return True
    
    def get_data(self, data_id: str) -> Optional[Any]:
        """
        Get registered data by ID.
        
        Args:
            data_id: Data identifier
            
        Returns:
            Data if found, None otherwise
        """
        with self._lock:
            if data_id in self.memory_registry:
                memory_item = self.memory_registry[data_id]
                
                # Update access statistics
                memory_item.access_count += 1
                memory_item.last_accessed = time.time()
                
                # Move to end (LRU)
                self.memory_registry.move_to_end(data_id)
                
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {data_id}")
                
                return memory_item.data
            else:
                self.stats["cache_misses"] += 1
                logger.debug(f"Cache miss for {data_id}")
                return None
    
    def unregister_data(self, data_id: str) -> bool:
        """
        Unregister data from memory manager.
        
        Args:
            data_id: Data identifier
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        with self._lock:
            if data_id in self.memory_registry:
                memory_item = self.memory_registry[data_id]
                
                # Remove from priority queue
                if data_id in self.priority_queues[memory_item.priority]:
                    self.priority_queues[memory_item.priority].remove(data_id)
                
                # Remove from registry
                del self.memory_registry[data_id]
                
                # Remove weak reference
                if data_id in self._weak_refs:
                    del self._weak_refs[data_id]
                
                logger.debug(f"Unregistered {data_id}")
                return True
            
            return False
    
    def update_priority(self, data_id: str, new_priority: MemoryPriority) -> bool:
        """
        Update priority of registered data.
        
        Args:
            data_id: Data identifier
            new_priority: New priority level
            
        Returns:
            True if updated successfully, False otherwise
        """
        with self._lock:
            if data_id in self.memory_registry:
                memory_item = self.memory_registry[data_id]
                old_priority = memory_item.priority
                
                # Remove from old priority queue
                if data_id in self.priority_queues[old_priority]:
                    self.priority_queues[old_priority].remove(data_id)
                
                # Add to new priority queue
                memory_item.priority = new_priority
                self.priority_queues[new_priority].append(data_id)
                
                logger.debug(f"Updated priority for {data_id}: {old_priority.name} -> {new_priority.name}")
                return True
            
            return False
    
    def _can_allocate_memory(self, required_mb: float) -> bool:
        """Check if memory can be allocated."""
        current_usage = self._get_current_usage_mb()
        return current_usage + required_mb <= self.max_memory_mb
    
    def _make_space(self, required_mb: float) -> bool:
        """
        Make space by evicting low-priority data.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if space was made, False otherwise
        """
        freed_mb = 0.0
        
        # Evict by priority order (lowest first)
        for priority in sorted(MemoryPriority, key=lambda x: x.value, reverse=True):
            if freed_mb >= required_mb:
                break
            
            # Get items in this priority queue
            items_to_evict = []
            for data_id in self.priority_queues[priority]:
                if data_id in self.memory_registry:
                    memory_item = self.memory_registry[data_id]
                    items_to_evict.append((data_id, memory_item))
            
            # Sort by access time (oldest first)
            items_to_evict.sort(key=lambda x: x[1].last_accessed)
            
            # Evict items until we have enough space
            for data_id, memory_item in items_to_evict:
                if freed_mb >= required_mb:
                    break
                
                freed_mb += memory_item.size_mb
                self._evict_data(data_id)
        
        # If still not enough space, trigger garbage collection
        if freed_mb < required_mb:
            self._trigger_garbage_collection()
            freed_mb += self._get_current_usage_mb() * 0.1  # Estimate GC benefit
        
        success = freed_mb >= required_mb
        if success:
            self.stats["total_evictions"] += 1
            logger.info(f"Made {freed_mb:.1f}MB space (required: {required_mb:.1f}MB)")
        else:
            logger.warning(f"Could only free {freed_mb:.1f}MB (required: {required_mb:.1f}MB)")
        
        return success
    
    def _evict_data(self, data_id: str):
        """Evict specific data from memory."""
        if data_id in self.memory_registry:
            memory_item = self.memory_registry[data_id]
            
            # Remove from priority queue
            if data_id in self.priority_queues[memory_item.priority]:
                self.priority_queues[memory_item.priority].remove(data_id)
            
            # Remove from registry
            del self.memory_registry[data_id]
            
            # Remove weak reference
            if data_id in self._weak_refs:
                del self._weak_refs[data_id]
            
            logger.debug(f"Evicted {data_id} ({memory_item.size_mb:.1f}MB)")
    
    def _trigger_garbage_collection(self):
        """Trigger garbage collection."""
        logger.debug("Triggering garbage collection")
        
        # Force garbage collection
        collected = gc.collect()
        
        self.stats["gc_triggers"] += 1
        logger.info(f"Garbage collection freed {collected} objects")
    
    def _cleanup_callback(self, weak_ref):
        """Callback for automatic cleanup when weak reference is deleted."""
        logger.debug("Weak reference cleanup triggered")
    
    def _estimate_size_mb(self, data: Any) -> float:
        """Estimate memory size of data in MB."""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes / (1024 * 1024)
            elif isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size_mb(item) for item in data) / len(data)
            elif isinstance(data, dict):
                return sum(self._estimate_size_mb(v) for v in data.values())
            else:
                # Rough estimate for other types
                return len(str(data)) / (1024 * 1024)
        except Exception:
            return 1.0  # Default estimate
    
    def _get_current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return sum(item.size_mb for item in self.memory_registry.values())
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        with self._lock:
            # System memory info
            system_memory = psutil.virtual_memory()
            
            # Registry stats
            registered_items = len(self.memory_registry)
            total_registered_mb = self._get_current_usage_mb()
            
            # Cache hit rate
            total_accesses = self.stats["cache_hits"] + self.stats["cache_misses"]
            cache_hit_rate = (
                self.stats["cache_hits"] / total_accesses 
                if total_accesses > 0 else 0.0
            )
            
            return MemoryStats(
                total_memory_mb=system_memory.total / (1024 * 1024),
                used_memory_mb=system_memory.used / (1024 * 1024),
                available_memory_mb=system_memory.available / (1024 * 1024),
                memory_percentage=system_memory.percent,
                registered_items=registered_items,
                total_registered_mb=total_registered_mb,
                cache_hit_rate=cache_hit_rate,
                eviction_count=self.stats["total_evictions"],
                gc_count=self.stats["gc_triggers"]
            )
    
    def get_registered_items(self) -> List[Dict[str, Any]]:
        """Get list of all registered items with metadata."""
        with self._lock:
            items = []
            for data_id, memory_item in self.memory_registry.items():
                items.append({
                    "data_id": data_id,
                    "size_mb": memory_item.size_mb,
                    "priority": memory_item.priority.name,
                    "access_count": memory_item.access_count,
                    "last_accessed": memory_item.last_accessed,
                    "creation_time": memory_item.creation_time,
                    "metadata": memory_item.metadata
                })
            
            return sorted(items, key=lambda x: x["last_accessed"], reverse=True)
    
    def cleanup_by_priority(self, priority: MemoryPriority) -> int:
        """
        Clean up all data with specific priority.
        
        Args:
            priority: Priority level to clean up
            
        Returns:
            Number of items cleaned up
        """
        with self._lock:
            items_to_cleanup = self.priority_queues[priority].copy()
            cleaned_count = 0
            
            for data_id in items_to_cleanup:
                if data_id in self.memory_registry:
                    self._evict_data(data_id)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} items with priority {priority.name}")
            return cleaned_count
    
    def cleanup_old_data(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up data older than specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of items cleaned up
        """
        with self._lock:
            current_time = time.time()
            items_to_cleanup = []
            
            for data_id, memory_item in self.memory_registry.items():
                if current_time - memory_item.creation_time > max_age_seconds:
                    items_to_cleanup.append(data_id)
            
            for data_id in items_to_cleanup:
                self._evict_data(data_id)
            
            logger.info(f"Cleaned up {len(items_to_cleanup)} old items")
            return len(items_to_cleanup)
    
    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start memory monitoring in background thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Started memory monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Stopped memory monitoring")
    
    def _monitor_memory(self, interval_seconds: float):
        """Background memory monitoring."""
        while self._monitoring_active:
            try:
                # Check memory usage
                current_usage = self._get_current_usage_mb()
                usage_percentage = current_usage / self.max_memory_mb
                
                if usage_percentage > self.gc_threshold:
                    logger.warning(f"Memory usage high: {usage_percentage:.1%}")
                    self._trigger_garbage_collection()
                    
                    # Clean up old data if still high
                    if self._get_current_usage_mb() / self.max_memory_mb > self.gc_threshold:
                        self.cleanup_old_data(max_age_seconds=1800)  # 30 minutes
                
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval_seconds)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            "current_usage_mb": self._get_current_usage_mb(),
            "max_memory_mb": self.max_memory_mb,
            "usage_percentage": self._get_current_usage_mb() / self.max_memory_mb,
            "registered_items": len(self.memory_registry),
            "monitoring_active": self._monitoring_active
        }
    
    def clear_all(self):
        """Clear all registered data."""
        with self._lock:
            self.memory_registry.clear()
            for priority_queue in self.priority_queues.values():
                priority_queue.clear()
            self._weak_refs.clear()
            
            logger.info("Cleared all registered data")


# Global instance for webapp use
_smart_memory_manager = None


def get_smart_memory_manager(max_memory_mb: int = 500) -> SmartMemoryManager:
    """Get global smart memory manager instance."""
    global _smart_memory_manager
    if _smart_memory_manager is None:
        _smart_memory_manager = SmartMemoryManager(max_memory_mb=max_memory_mb)
    return _smart_memory_manager
