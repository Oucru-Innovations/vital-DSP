"""
Virtual Scrolling Manager for Large Dataset Visualization

This module implements virtual scrolling capabilities for handling large datasets
in web visualizations, providing smooth interaction with millions of data points.

Author: vitalDSP Development Team
Date: January 14, 2025
Version: 1.0.0 (Phase 3B)
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)


class ScrollDirection(Enum):
    """Enumeration of scroll directions."""
    
    FORWARD = "forward"
    BACKWARD = "backward"
    UP = "up"
    DOWN = "down"


@dataclass
class ViewportInfo:
    """Data class for viewport information."""
    
    start_index: int
    end_index: int
    visible_points: int
    total_points: int
    scroll_position: float  # 0.0 to 1.0
    zoom_level: float
    data_range: Tuple[float, float]
    time_range: Optional[Tuple[float, float]] = None


@dataclass
class ScrollEvent:
    """Data class for scroll events."""
    
    direction: ScrollDirection
    amount: float
    timestamp: float
    viewport_before: ViewportInfo
    viewport_after: ViewportInfo


class VirtualScrollingManager:
    """
    Virtual scrolling manager for large dataset visualization.
    
    Provides efficient handling of large datasets by only rendering visible portions,
    with smooth scrolling, zooming, and data loading capabilities.
    """
    
    def __init__(
        self, 
        viewport_size: int = 1000,
        buffer_size: int = 2000,
        zoom_levels: List[float] = None
    ):
        """
        Initialize the virtual scrolling manager.
        
        Args:
            viewport_size: Number of points visible in viewport
            buffer_size: Number of points to keep in buffer around viewport
            zoom_levels: Available zoom levels (multipliers)
        """
        self.viewport_size = viewport_size
        self.buffer_size = buffer_size
        self.zoom_levels = zoom_levels or [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        
        # Current state
        self.current_position = 0
        self.current_zoom = 1.0
        self.total_data_size = 0
        self.data_cache = {}
        self.scroll_history = deque(maxlen=100)
        
        # Performance tracking
        self.stats = {
            "scroll_events": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "data_loads": 0,
            "average_scroll_time": 0.0,
            "total_scroll_time": 0.0
        }
        
        # Callbacks
        self.data_load_callback: Optional[Callable] = None
        self.viewport_change_callback: Optional[Callable] = None
        
        logger.info(f"VirtualScrollingManager initialized with viewport_size={viewport_size}")
    
    def set_data_size(self, total_size: int):
        """Set the total size of the dataset."""
        self.total_data_size = total_size
        self.current_position = min(self.current_position, max(0, total_size - self.viewport_size))
        logger.info(f"Data size set to {total_size} points")
    
    def get_viewport_data(
        self, 
        data: np.ndarray, 
        position: Optional[int] = None,
        time_axis: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ViewportInfo]:
        """
        Get data for current viewport.
        
        Args:
            data: Full dataset
            position: Optional position override
            time_axis: Optional time axis data
            
        Returns:
            Tuple of (viewport_data, viewport_info)
        """
        if position is None:
            position = self.current_position
        
        # Calculate viewport bounds
        start_idx = max(0, position)
        end_idx = min(len(data), start_idx + self.viewport_size)
        
        # Get viewport data
        viewport_data = data[start_idx:end_idx]
        
        # Calculate scroll position (0.0 to 1.0)
        scroll_position = position / max(1, self.total_data_size - self.viewport_size)
        
        # Calculate data range
        data_range = (float(np.min(viewport_data)), float(np.max(viewport_data)))
        
        # Calculate time range if time axis provided
        time_range = None
        if time_axis is not None:
            time_range = (float(time_axis[start_idx]), float(time_axis[end_idx-1]))
        
        viewport_info = ViewportInfo(
            start_index=start_idx,
            end_index=end_idx,
            visible_points=len(viewport_data),
            total_points=self.total_data_size,
            scroll_position=scroll_position,
            zoom_level=self.current_zoom,
            data_range=data_range,
            time_range=time_range
        )
        
        return viewport_data, viewport_info
    
    def scroll(
        self, 
        direction: ScrollDirection, 
        amount: float = 0.1,
        data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ViewportInfo]:
        """
        Scroll the viewport in specified direction.
        
        Args:
            direction: Scroll direction
            amount: Scroll amount (0.0 to 1.0)
            data: Optional data for immediate return
            
        Returns:
            Tuple of (viewport_data, viewport_info)
        """
        start_time = time.time()
        
        # Record viewport before scroll
        viewport_before = self._get_current_viewport_info()
        
        # Calculate new position
        scroll_distance = int(amount * self.viewport_size)
        
        if direction == ScrollDirection.FORWARD:
            new_position = min(
                self.total_data_size - self.viewport_size,
                self.current_position + scroll_distance
            )
        elif direction == ScrollDirection.BACKWARD:
            new_position = max(0, self.current_position - scroll_distance)
        else:
            # For UP/DOWN, treat as FORWARD/BACKWARD
            if direction == ScrollDirection.DOWN:
                new_position = min(
                    self.total_data_size - self.viewport_size,
                    self.current_position + scroll_distance
                )
            else:  # UP
                new_position = max(0, self.current_position - scroll_distance)
        
        # Update position
        self.current_position = new_position
        
        # Record scroll event
        scroll_event = ScrollEvent(
            direction=direction,
            amount=amount,
            timestamp=time.time(),
            viewport_before=viewport_before,
            viewport_after=self._get_current_viewport_info()
        )
        self.scroll_history.append(scroll_event)
        
        # Update statistics
        self.stats["scroll_events"] += 1
        scroll_time = time.time() - start_time
        self.stats["total_scroll_time"] += scroll_time
        self.stats["average_scroll_time"] = (
            self.stats["total_scroll_time"] / self.stats["scroll_events"]
        )
        
        # Get viewport data
        if data is not None:
            viewport_data, viewport_info = self.get_viewport_data(data)
        else:
            viewport_data = np.array([])
            viewport_info = self._get_current_viewport_info()
        
        # Trigger callbacks
        if self.viewport_change_callback:
            self.viewport_change_callback(viewport_info)
        
        logger.debug(f"Scrolled {direction.value} by {amount:.2f}, "
                   f"position: {self.current_position}")
        
        return viewport_data, viewport_info
    
    def zoom(self, zoom_factor: float, center_position: Optional[float] = None) -> ViewportInfo:
        """
        Zoom the viewport.
        
        Args:
            zoom_factor: Zoom factor (1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out)
            center_position: Optional center position for zoom (0.0 to 1.0)
            
        Returns:
            Updated viewport info
        """
        # Clamp zoom factor to available levels
        available_zooms = sorted(self.zoom_levels)
        zoom_factor = max(available_zooms[0], min(available_zooms[-1], zoom_factor))
        
        # Find closest zoom level
        closest_zoom = min(available_zooms, key=lambda x: abs(x - zoom_factor))
        
        if closest_zoom == self.current_zoom:
            return self._get_current_viewport_info()
        
        # Calculate new viewport size based on zoom
        old_viewport_size = self.viewport_size
        self.viewport_size = int(old_viewport_size / closest_zoom)
        self.current_zoom = closest_zoom
        
        # Adjust position if center position specified
        if center_position is not None:
            center_idx = int(center_position * self.total_data_size)
            self.current_position = max(0, min(
                self.total_data_size - self.viewport_size,
                center_idx - self.viewport_size // 2
            ))
        
        # Restore viewport size
        self.viewport_size = old_viewport_size
        
        logger.info(f"Zoomed to {closest_zoom:.2f}x, viewport size: {self.viewport_size}")
        
        return self._get_current_viewport_info()
    
    def jump_to_position(self, position: float) -> ViewportInfo:
        """
        Jump to specific position in dataset.
        
        Args:
            position: Position as fraction (0.0 to 1.0)
            
        Returns:
            Updated viewport info
        """
        position = max(0.0, min(1.0, position))
        self.current_position = int(position * (self.total_data_size - self.viewport_size))
        
        logger.info(f"Jumped to position {position:.2f} (index {self.current_position})")
        
        return self._get_current_viewport_info()
    
    def get_buffer_data(
        self, 
        data: np.ndarray,
        buffer_multiplier: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data including buffer around current viewport.
        
        Args:
            data: Full dataset
            buffer_multiplier: Buffer size multiplier
            
        Returns:
            Tuple of (buffer_data, buffer_indices)
        """
        buffer_size = int(self.viewport_size * buffer_multiplier)
        
        start_idx = max(0, self.current_position - buffer_size // 2)
        end_idx = min(len(data), start_idx + buffer_size)
        
        buffer_data = data[start_idx:end_idx]
        buffer_indices = np.arange(start_idx, end_idx)
        
        return buffer_data, buffer_indices
    
    def preload_data(
        self, 
        data_loader: Callable[[int, int], np.ndarray],
        direction: ScrollDirection = ScrollDirection.FORWARD
    ):
        """
        Preload data in scroll direction for smooth scrolling.
        
        Args:
            data_loader: Function that loads data for given range (start, end)
            direction: Direction to preload
        """
        if direction == ScrollDirection.FORWARD:
            start_idx = self.current_position + self.viewport_size
            end_idx = min(self.total_data_size, start_idx + self.buffer_size)
        else:  # BACKWARD
            end_idx = self.current_position
            start_idx = max(0, end_idx - self.buffer_size)
        
        if start_idx < end_idx:
            try:
                preloaded_data = data_loader(start_idx, end_idx)
                cache_key = f"{start_idx}_{end_idx}"
                self.data_cache[cache_key] = preloaded_data
                self.stats["data_loads"] += 1
                logger.debug(f"Preloaded data for range [{start_idx}, {end_idx})")
            except Exception as e:
                logger.warning(f"Failed to preload data: {e}")
    
    def get_scroll_suggestions(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get suggestions for interesting scroll positions.
        
        Args:
            data: Dataset to analyze
            
        Returns:
            List of scroll suggestions with metadata
        """
        suggestions = []
        
        # Find peaks for suggestions
        peaks = self._find_interesting_points(data)
        
        for peak_idx in peaks[:10]:  # Limit to 10 suggestions
            position = peak_idx / len(data)
            suggestions.append({
                "position": position,
                "index": peak_idx,
                "value": float(data[peak_idx]),
                "description": f"Peak at index {peak_idx}",
                "type": "peak"
            })
        
        # Add start, middle, end suggestions
        suggestions.extend([
            {
                "position": 0.0,
                "index": 0,
                "value": float(data[0]),
                "description": "Start of dataset",
                "type": "landmark"
            },
            {
                "position": 0.5,
                "index": len(data) // 2,
                "value": float(data[len(data) // 2]),
                "description": "Middle of dataset",
                "type": "landmark"
            },
            {
                "position": 1.0,
                "index": len(data) - 1,
                "value": float(data[-1]),
                "description": "End of dataset",
                "type": "landmark"
            }
        ])
        
        return sorted(suggestions, key=lambda x: x["position"])
    
    def _find_interesting_points(self, data: np.ndarray) -> List[int]:
        """Find interesting points in the data (peaks, valleys, etc.)."""
        interesting_points = []
        
        # Simple peak detection
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and 
                data[i] > np.mean(data) + np.std(data)):
                interesting_points.append(i)
        
        return interesting_points
    
    def _get_current_viewport_info(self) -> ViewportInfo:
        """Get current viewport information."""
        return ViewportInfo(
            start_index=self.current_position,
            end_index=self.current_position + self.viewport_size,
            visible_points=self.viewport_size,
            total_points=self.total_data_size,
            scroll_position=self.current_position / max(1, self.total_data_size - self.viewport_size),
            zoom_level=self.current_zoom,
            data_range=(0.0, 0.0),  # Will be updated when data is available
            time_range=None
        )
    
    def set_data_load_callback(self, callback: Callable[[int, int], np.ndarray]):
        """Set callback for data loading."""
        self.data_load_callback = callback
    
    def set_viewport_change_callback(self, callback: Callable[[ViewportInfo], None]):
        """Set callback for viewport changes."""
        self.viewport_change_callback = callback
    
    def get_scroll_history(self, limit: int = 10) -> List[ScrollEvent]:
        """Get recent scroll history."""
        return list(self.scroll_history)[-limit:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            "cache_size": len(self.data_cache),
            "viewport_size": self.viewport_size,
            "buffer_size": self.buffer_size,
            "current_position": self.current_position,
            "current_zoom": self.current_zoom,
            "total_data_size": self.total_data_size
        }
    
    def clear_cache(self):
        """Clear data cache."""
        self.data_cache.clear()
        logger.info("Virtual scrolling cache cleared")
    
    def reset(self):
        """Reset virtual scrolling state."""
        self.current_position = 0
        self.current_zoom = 1.0
        self.total_data_size = 0
        self.data_cache.clear()
        self.scroll_history.clear()
        logger.info("Virtual scrolling state reset")


# Global instance for webapp use
_virtual_scrolling_manager = None


def get_virtual_scrolling_manager(viewport_size: int = 1000) -> VirtualScrollingManager:
    """Get global virtual scrolling manager instance."""
    global _virtual_scrolling_manager
    if _virtual_scrolling_manager is None:
        _virtual_scrolling_manager = VirtualScrollingManager(viewport_size=viewport_size)
    return _virtual_scrolling_manager
