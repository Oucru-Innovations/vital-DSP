"""
Optimized Plot Manager for Large Dataset Visualization

This module implements optimized plot management with memory control,
template reuse, and performance optimization for Plotly visualizations.

Author: vitalDSP Development Team
Date: January 14, 2025
Version: 1.0.0 (Phase 3B)
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
from collections import OrderedDict
import json
import hashlib

logger = logging.getLogger(__name__)


class PlotType(Enum):
    """Enumeration of supported plot types."""
    
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    SUBPLOT = "subplot"
    CANDLESTICK = "candlestick"
    BOX = "box"
    VIOLIN = "violin"


@dataclass
class PlotConfig:
    """Data class for plot configuration."""
    
    plot_type: PlotType
    title: str
    x_label: str = ""
    y_label: str = ""
    height: int = 400
    width: int = 800
    template: str = "plotly_white"
    show_legend: bool = True
    max_points: int = 10000
    enable_downsampling: bool = True
    enable_virtual_scrolling: bool = False
    colors: List[str] = None
    annotations: List[Dict] = None
    layout_updates: Dict[str, Any] = None


@dataclass
class PlotPerformance:
    """Data class for plot performance metrics."""
    
    creation_time: float
    data_points: int
    memory_usage_mb: float
    template_reused: bool
    downsampling_applied: bool
    cache_hit: bool


class OptimizedPlotManager:
    """
    Optimized plot manager with memory control and performance optimization.
    
    Provides efficient plot creation with template reuse, memory management,
    and intelligent caching for large dataset visualizations.
    """
    
    def __init__(
        self, 
        max_cache_size: int = 50,
        max_memory_mb: int = 100,
        enable_compression: bool = True
    ):
        """
        Initialize the optimized plot manager.
        
        Args:
            max_cache_size: Maximum number of plots to cache
            max_memory_mb: Maximum memory usage for plots
            enable_compression: Whether to compress cached plots
        """
        self.max_cache_size = max_cache_size
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        
        # Caches
        self.plot_cache = OrderedDict()
        self.figure_templates = {}
        self.data_cache = {}
        
        # Performance tracking
        self.stats = {
            "plots_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "templates_reused": 0,
            "downsampling_applied": 0,
            "memory_saved_mb": 0.0,
            "average_creation_time": 0.0,
            "total_creation_time": 0.0
        }
        
        # Initialize default templates
        self._initialize_default_templates()
        
        logger.info(f"OptimizedPlotManager initialized with max_cache_size={max_cache_size}")
    
    def create_optimized_plot(
        self, 
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        config: PlotConfig,
        time_axis: Optional[np.ndarray] = None
    ) -> Tuple[go.Figure, PlotPerformance]:
        """
        Create optimized plot with performance tracking.
        
        Args:
            data: Input data for plotting
            config: Plot configuration
            time_axis: Optional time axis data
            
        Returns:
            Tuple of (plotly_figure, performance_metrics)
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(data, config)
        if cache_key in self.plot_cache:
            self.stats["cache_hits"] += 1
            cached_fig, cached_perf = self.plot_cache[cache_key]
            
            # Move to end (LRU)
            self.plot_cache.move_to_end(cache_key)
            
            logger.debug(f"Using cached plot for {config.plot_type.value}")
            return cached_fig, cached_perf
        
        self.stats["cache_misses"] += 1
        
        # Create plot
        fig = self._create_plot(data, config, time_axis)
        
        # Calculate performance metrics
        creation_time = time.time() - start_time
        data_points = self._count_data_points(data)
        memory_usage = self._estimate_memory_usage(fig)
        
        # Check if template was reused
        template_reused = config.plot_type in self.figure_templates
        
        # Check if downsampling was applied
        downsampling_applied = (
            config.enable_downsampling and 
            data_points > config.max_points
        )
        
        performance = PlotPerformance(
            creation_time=creation_time,
            data_points=data_points,
            memory_usage_mb=memory_usage,
            template_reused=template_reused,
            downsampling_applied=downsampling_applied,
            cache_hit=False
        )
        
        # Update statistics
        self._update_stats(performance)
        
        # Cache plot if memory allows
        if self._can_cache_plot(memory_usage):
            self._cache_plot(cache_key, fig, performance)
        
        logger.info(f"Created {config.plot_type.value} plot with {data_points} points "
                   f"in {creation_time:.3f}s (memory: {memory_usage:.1f}MB)")
        
        return fig, performance
    
    def create_line_plot(
        self,
        data: np.ndarray,
        title: str = "Line Plot",
        time_axis: Optional[np.ndarray] = None,
        **kwargs
    ) -> go.Figure:
        """Create optimized line plot."""
        config = PlotConfig(
            plot_type=PlotType.LINE,
            title=title,
            **kwargs
        )
        
        fig, _ = self.create_optimized_plot(data, config, time_axis)
        return fig
    
    def create_scatter_plot(
        self,
        data: np.ndarray,
        title: str = "Scatter Plot",
        time_axis: Optional[np.ndarray] = None,
        **kwargs
    ) -> go.Figure:
        """Create optimized scatter plot."""
        config = PlotConfig(
            plot_type=PlotType.SCATTER,
            title=title,
            **kwargs
        )
        
        fig, _ = self.create_optimized_plot(data, config, time_axis)
        return fig
    
    def create_subplot(
        self,
        data_dict: Dict[str, np.ndarray],
        subplot_titles: List[str],
        title: str = "Multi-Signal Plot",
        rows: int = 2,
        cols: int = 2,
        **kwargs
    ) -> go.Figure:
        """Create optimized subplot with multiple signals."""
        config = PlotConfig(
            plot_type=PlotType.SUBPLOT,
            title=title,
            **kwargs
        )
        
        fig = self._create_subplot(data_dict, subplot_titles, rows, cols, config)
        return fig
    
    def create_heatmap(
        self,
        data: np.ndarray,
        title: str = "Heatmap",
        **kwargs
    ) -> go.Figure:
        """Create optimized heatmap."""
        config = PlotConfig(
            plot_type=PlotType.HEATMAP,
            title=title,
            **kwargs
        )
        
        fig, _ = self.create_optimized_plot(data, config)
        return fig
    
    def _create_plot(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        config: PlotConfig,
        time_axis: Optional[np.ndarray] = None
    ) -> go.Figure:
        """Create plot based on configuration."""
        
        if config.plot_type == PlotType.LINE:
            return self._create_line_plot(data, config, time_axis)
        elif config.plot_type == PlotType.SCATTER:
            return self._create_scatter_plot(data, config, time_axis)
        elif config.plot_type == PlotType.BAR:
            return self._create_bar_plot(data, config)
        elif config.plot_type == PlotType.HISTOGRAM:
            return self._create_histogram_plot(data, config)
        elif config.plot_type == PlotType.HEATMAP:
            return self._create_heatmap_plot(data, config)
        elif config.plot_type == PlotType.SUBPLOT:
            return self._create_subplot(data, [], 2, 2, config)
        else:
            raise ValueError(f"Unsupported plot type: {config.plot_type}")
    
    def _create_line_plot(
        self,
        data: np.ndarray,
        config: PlotConfig,
        time_axis: Optional[np.ndarray] = None
    ) -> go.Figure:
        """Create optimized line plot."""
        # Use template if available
        if config.plot_type in self.figure_templates:
            fig = self.figure_templates[config.plot_type].copy()
            self.stats["templates_reused"] += 1
        else:
            fig = go.Figure()
            self.figure_templates[config.plot_type] = fig
        
        # Prepare data
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        # Apply downsampling if needed
        if config.enable_downsampling and len(data) > config.max_points:
            from .adaptive_downsampler import get_adaptive_downsampler
            
            downsampler = get_adaptive_downsampler(config.max_points)
            result = downsampler.downsample_for_display(data, time_axis)
            
            data = result.downsampled_data
            time_axis = result.downsampled_time
            self.stats["downsampling_applied"] += 1
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=data,
            mode='lines',
            name='Signal',
            line=dict(width=1),
            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
        ))
        
        # Update layout
        self._update_layout(fig, config)
        
        return fig
    
    def _create_scatter_plot(
        self,
        data: np.ndarray,
        config: PlotConfig,
        time_axis: Optional[np.ndarray] = None
    ) -> go.Figure:
        """Create optimized scatter plot."""
        if config.plot_type in self.figure_templates:
            fig = self.figure_templates[config.plot_type].copy()
            self.stats["templates_reused"] += 1
        else:
            fig = go.Figure()
            self.figure_templates[config.plot_type] = fig
        
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        # Apply downsampling if needed
        if config.enable_downsampling and len(data) > config.max_points:
            from .adaptive_downsampler import get_adaptive_downsampler
            
            downsampler = get_adaptive_downsampler(config.max_points)
            result = downsampler.downsample_for_display(data, time_axis)
            
            data = result.downsampled_data
            time_axis = result.downsampled_time
            self.stats["downsampling_applied"] += 1
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=data,
            mode='markers',
            name='Signal',
            marker=dict(size=3),
            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
        ))
        
        self._update_layout(fig, config)
        return fig
    
    def _create_bar_plot(
        self,
        data: np.ndarray,
        config: PlotConfig
    ) -> go.Figure:
        """Create optimized bar plot."""
        fig = go.Figure()
        
        # For bar plots, limit data points more aggressively
        if len(data) > config.max_points // 2:
            step = len(data) // (config.max_points // 2)
            data = data[::step]
        
        fig.add_trace(go.Bar(
            y=data,
            name='Values',
            hovertemplate='<b>Index:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
        ))
        
        self._update_layout(fig, config)
        return fig
    
    def _create_histogram_plot(
        self,
        data: np.ndarray,
        config: PlotConfig
    ) -> go.Figure:
        """Create optimized histogram plot."""
        fig = go.Figure()
        
        # Calculate histogram
        hist, bins = np.histogram(data, bins=min(50, len(data) // 10))
        
        fig.add_trace(go.Bar(
            x=bins[:-1],
            y=hist,
            name='Histogram',
            hovertemplate='<b>Bin:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))
        
        self._update_layout(fig, config)
        return fig
    
    def _create_heatmap_plot(
        self,
        data: np.ndarray,
        config: PlotConfig
    ) -> go.Figure:
        """Create optimized heatmap plot."""
        fig = go.Figure()
        
        # Reshape data for heatmap if needed
        if data.ndim == 1:
            # Convert 1D to 2D for heatmap
            size = int(np.sqrt(len(data)))
            if size * size != len(data):
                # Pad data to square size
                padded_size = size + 1
                padded_data = np.zeros(padded_size * padded_size)
                padded_data[:len(data)] = data
                data = padded_data.reshape(padded_size, padded_size)
            else:
                data = data.reshape(size, size)
        
        fig.add_trace(go.Heatmap(
            z=data,
            hovertemplate='<b>Row:</b> %{y}<br><b>Col:</b> %{x}<br><b>Value:</b> %{z}<extra></extra>'
        ))
        
        self._update_layout(fig, config)
        return fig
    
    def _create_subplot(
        self,
        data_dict: Dict[str, np.ndarray],
        subplot_titles: List[str],
        rows: int,
        cols: int,
        config: PlotConfig
    ) -> go.Figure:
        """Create optimized subplot."""
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        plot_idx = 1
        for signal_name, data in data_dict.items():
            row = (plot_idx - 1) // cols + 1
            col = (plot_idx - 1) % cols + 1
            
            # Apply downsampling if needed
            if config.enable_downsampling and len(data) > config.max_points:
                from .adaptive_downsampler import get_adaptive_downsampler
                
                downsampler = get_adaptive_downsampler(config.max_points)
                result = downsampler.downsample_for_display(data)
                data = result.downsampled_data
                self.stats["downsampling_applied"] += 1
            
            fig.add_trace(
                go.Scatter(
                    y=data,
                    mode='lines',
                    name=signal_name,
                    line=dict(width=1)
                ),
                row=row,
                col=col
            )
            
            plot_idx += 1
        
        # Update layout
        fig.update_layout(
            title=config.title,
            height=config.height,
            template=config.template,
            showlegend=config.show_legend
        )
        
        return fig
    
    def _update_layout(self, fig: go.Figure, config: PlotConfig):
        """Update plot layout with configuration."""
        layout_updates = {
            'title': config.title,
            'height': config.height,
            'width': config.width,
            'template': config.template,
            'showlegend': config.show_legend,
            'xaxis_title': config.x_label,
            'yaxis_title': config.y_label
        }
        
        # Add custom layout updates
        if config.layout_updates:
            layout_updates.update(config.layout_updates)
        
        # Add annotations
        if config.annotations:
            layout_updates['annotations'] = config.annotations
        
        fig.update_layout(**layout_updates)
    
    def _initialize_default_templates(self):
        """Initialize default figure templates."""
        # Line plot template
        self.figure_templates[PlotType.LINE] = go.Figure(
            data=go.Scatter(
                mode='lines',
                line=dict(width=1),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
            )
        )
        
        # Scatter plot template
        self.figure_templates[PlotType.SCATTER] = go.Figure(
            data=go.Scatter(
                mode='markers',
                marker=dict(size=3),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
            )
        )
    
    def _get_cache_key(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]],
        config: PlotConfig
    ) -> str:
        """Generate cache key for plot."""
        # Create hash of data and config
        data_hash = hashlib.md5(str(data).encode()).hexdigest()
        config_hash = hashlib.md5(json.dumps(config.__dict__, sort_keys=True).encode()).hexdigest()
        
        return f"{data_hash}_{config_hash}"
    
    def _count_data_points(self, data: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]) -> int:
        """Count total data points."""
        if isinstance(data, np.ndarray):
            return len(data)
        elif isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, dict):
            return sum(len(v) for v in data.values())
        else:
            return 0
    
    def _estimate_memory_usage(self, fig: go.Figure) -> float:
        """Estimate memory usage of plotly figure."""
        try:
            # Convert figure to JSON and estimate size
            fig_json = fig.to_json()
            size_bytes = len(fig_json.encode('utf-8'))
            return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 1.0  # Default estimate
    
    def _can_cache_plot(self, memory_usage: float) -> bool:
        """Check if plot can be cached based on memory constraints."""
        current_memory = sum(perf.memory_usage_mb for perf in self.plot_cache.values())
        return current_memory + memory_usage <= self.max_memory_mb
    
    def _cache_plot(self, cache_key: str, fig: go.Figure, performance: PlotPerformance):
        """Cache plot with LRU eviction."""
        # Remove oldest if cache is full
        if len(self.plot_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.plot_cache))
            del self.plot_cache[oldest_key]
        
        self.plot_cache[cache_key] = (fig, performance)
        self.stats["memory_saved_mb"] += performance.memory_usage_mb
    
    def _update_stats(self, performance: PlotPerformance):
        """Update performance statistics."""
        self.stats["plots_created"] += 1
        self.stats["total_creation_time"] += performance.creation_time
        self.stats["average_creation_time"] = (
            self.stats["total_creation_time"] / self.stats["plots_created"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            "cache_size": len(self.plot_cache),
            "template_count": len(self.figure_templates),
            "cache_hit_rate": (
                self.stats["cache_hits"] / 
                max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
        }
    
    def clear_cache(self):
        """Clear plot cache."""
        self.plot_cache.clear()
        logger.info("Plot cache cleared")
    
    def clear_templates(self):
        """Clear figure templates."""
        self.figure_templates.clear()
        self._initialize_default_templates()
        logger.info("Figure templates cleared")


# Global instance for webapp use
_optimized_plot_manager = None


def get_optimized_plot_manager(max_cache_size: int = 50) -> OptimizedPlotManager:
    """Get global optimized plot manager instance."""
    global _optimized_plot_manager
    if _optimized_plot_manager is None:
        _optimized_plot_manager = OptimizedPlotManager(max_cache_size=max_cache_size)
    return _optimized_plot_manager
