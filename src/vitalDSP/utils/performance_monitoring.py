# src/vitalDSP/utils/performance_monitoring.py
"""
Performance monitoring utilities for vitalDSP functions.

This module provides comprehensive performance monitoring capabilities
for critical signal processing operations, including execution time,
memory usage, and performance metrics collection.
"""

import time
import psutil
import os
import threading
import warnings
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    signal_length: int
    parameters: Dict[str, Any]
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class PerformanceMonitor:
    """
    Performance monitoring system for vitalDSP functions.
    
    This class provides comprehensive performance monitoring capabilities
    including execution time tracking, memory usage monitoring, and
    performance metrics collection.
    """
    
    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize the performance monitor.
        
        Parameters
        ----------
        enable_monitoring : bool, optional
            Whether to enable performance monitoring (default: True)
        """
        self.enable_monitoring = enable_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        
        # Performance thresholds
        self.execution_time_threshold = 30.0  # seconds
        self.memory_usage_threshold = 1000.0  # MB
        self.cpu_percent_threshold = 80.0  # percent
    
    def monitor_function(self, function_name: str = None, 
                        signal_length: int = None,
                        parameters: Dict[str, Any] = None):
        """
        Decorator for monitoring function performance.
        
        Parameters
        ----------
        function_name : str, optional
            Name of the function being monitored
        signal_length : int, optional
            Length of the input signal
        parameters : dict, optional
            Function parameters
        
        Returns
        -------
        callable
            Decorated function with performance monitoring
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_monitoring:
                    return func(*args, **kwargs)
                
                # Extract function name
                func_name = function_name or func.__name__
                
                # Extract signal length from args
                sig_length = signal_length
                if sig_length is None and len(args) > 0:
                    if hasattr(args[0], '__len__'):
                        sig_length = len(args[0])
                    elif isinstance(args[0], np.ndarray):
                        sig_length = len(args[0])
                
                # Extract parameters
                func_params = parameters or kwargs.copy()
                
                # Monitor performance
                with self._monitor_execution(func_name, sig_length, func_params):
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        # Record error in metrics
                        self._record_error(func_name, str(e), sig_length, func_params)
                        raise
            
            return wrapper
        return decorator
    
    @contextmanager
    def _monitor_execution(self, function_name: str, signal_length: int, parameters: Dict[str, Any]):
        """Context manager for monitoring function execution."""
        # Record initial state
        start_time = time.time()
        start_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self._process.cpu_percent()
        
        try:
            yield
        finally:
            # Record final state
            end_time = time.time()
            end_memory = self._process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self._process.cpu_percent()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Create metrics object
            metrics = PerformanceMetrics(
                function_name=function_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_percent=end_cpu,
                signal_length=signal_length or 0,
                parameters=parameters,
                timestamp=start_time,
                success=True
            )
            
            # Record metrics
            self._record_metrics(metrics)
            
            # Check for performance issues
            self._check_performance_thresholds(metrics)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 metrics to prevent memory issues
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
    
    def _record_error(self, function_name: str, error_message: str, 
                     signal_length: int, parameters: Dict[str, Any]):
        """Record error metrics."""
        metrics = PerformanceMetrics(
            function_name=function_name,
            execution_time=0.0,
            memory_usage_mb=0.0,
            cpu_percent=0.0,
            signal_length=signal_length or 0,
            parameters=parameters,
            timestamp=time.time(),
            success=False,
            error_message=error_message
        )
        
        self._record_metrics(metrics)
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and issue warnings."""
        if metrics.execution_time > self.execution_time_threshold:
            warnings.warn(
                f"Performance Warning: {metrics.function_name} took "
                f"{metrics.execution_time:.2f}s (threshold: {self.execution_time_threshold}s)"
            )
        
        if metrics.memory_usage_mb > self.memory_usage_threshold:
            warnings.warn(
                f"Memory Warning: {metrics.function_name} used "
                f"{metrics.memory_usage_mb:.2f}MB (threshold: {self.memory_usage_threshold}MB)"
            )
        
        if metrics.cpu_percent > self.cpu_percent_threshold:
            warnings.warn(
                f"CPU Warning: {metrics.function_name} used "
                f"{metrics.cpu_percent:.2f}% CPU (threshold: {self.cpu_percent_threshold}%)"
            )
    
    def get_performance_summary(self, function_name: str = None) -> Dict[str, Any]:
        """
        Get performance summary for a function or all functions.
        
        Parameters
        ----------
        function_name : str, optional
            Name of function to summarize (if None, summarize all)
        
        Returns
        -------
        dict
            Performance summary statistics
        """
        with self._lock:
            if function_name:
                metrics = [m for m in self.metrics_history if m.function_name == function_name]
            else:
                metrics = self.metrics_history
            
            if not metrics:
                return {"message": "No metrics available"}
            
            # Filter successful executions
            successful_metrics = [m for m in metrics if m.success]
            
            if not successful_metrics:
                return {"message": "No successful executions found"}
            
            # Calculate statistics
            execution_times = [m.execution_time for m in successful_metrics]
            memory_usages = [m.memory_usage_mb for m in successful_metrics]
            signal_lengths = [m.signal_length for m in successful_metrics]
            
            summary = {
                "function_name": function_name or "all_functions",
                "total_executions": len(metrics),
                "successful_executions": len(successful_metrics),
                "success_rate": len(successful_metrics) / len(metrics) * 100,
                "execution_time": {
                    "mean": np.mean(execution_times),
                    "median": np.median(execution_times),
                    "std": np.std(execution_times),
                    "min": np.min(execution_times),
                    "max": np.max(execution_times)
                },
                "memory_usage": {
                    "mean": np.mean(memory_usages),
                    "median": np.median(memory_usages),
                    "std": np.std(memory_usages),
                    "min": np.min(memory_usages),
                    "max": np.max(memory_usages)
                },
                "signal_length": {
                    "mean": np.mean(signal_lengths),
                    "median": np.median(signal_lengths),
                    "std": np.std(signal_lengths),
                    "min": np.min(signal_lengths),
                    "max": np.max(signal_lengths)
                }
            }
            
            return summary
    
    def get_performance_trends(self, function_name: str = None) -> Dict[str, List]:
        """
        Get performance trends over time.
        
        Parameters
        ----------
        function_name : str, optional
            Name of function to analyze (if None, analyze all)
        
        Returns
        -------
        dict
            Performance trends data
        """
        with self._lock:
            if function_name:
                metrics = [m for m in self.metrics_history if m.function_name == function_name]
            else:
                metrics = self.metrics_history
            
            if not metrics:
                return {"message": "No metrics available"}
            
            # Sort by timestamp
            metrics.sort(key=lambda x: x.timestamp)
            
            # Extract trend data
            timestamps = [m.timestamp for m in metrics]
            execution_times = [m.execution_time for m in metrics]
            memory_usages = [m.memory_usage_mb for m in metrics]
            signal_lengths = [m.signal_length for m in metrics]
            
            trends = {
                "timestamps": timestamps,
                "execution_times": execution_times,
                "memory_usages": memory_usages,
                "signal_lengths": signal_lengths,
                "function_name": function_name or "all_functions"
            }
            
            return trends
    
    def clear_metrics(self):
        """Clear all performance metrics."""
        with self._lock:
            self.metrics_history.clear()
    
    def set_thresholds(self, execution_time: float = None, 
                      memory_usage: float = None, 
                      cpu_percent: float = None):
        """
        Set performance thresholds.
        
        Parameters
        ----------
        execution_time : float, optional
            Execution time threshold in seconds
        memory_usage : float, optional
            Memory usage threshold in MB
        cpu_percent : float, optional
            CPU usage threshold in percent
        """
        if execution_time is not None:
            self.execution_time_threshold = execution_time
        
        if memory_usage is not None:
            self.memory_usage_threshold = memory_usage
        
        if cpu_percent is not None:
            self.cpu_percent_threshold = cpu_percent


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def monitor_performance(function_name: str = None, 
                       signal_length: int = None,
                       parameters: Dict[str, Any] = None):
    """
    Global performance monitoring decorator.
    
    Parameters
    ----------
    function_name : str, optional
        Name of the function being monitored
    signal_length : int, optional
        Length of the input signal
    parameters : dict, optional
        Function parameters
    
    Returns
    -------
    callable
        Decorated function with performance monitoring
    """
    return _global_monitor.monitor_function(function_name, signal_length, parameters)


def get_performance_summary(function_name: str = None) -> Dict[str, Any]:
    """
    Get performance summary from global monitor.
    
    Parameters
    ----------
    function_name : str, optional
        Name of function to summarize (if None, summarize all)
    
    Returns
    -------
    dict
        Performance summary statistics
    """
    return _global_monitor.get_performance_summary(function_name)


def get_performance_trends(function_name: str = None) -> Dict[str, List]:
    """
    Get performance trends from global monitor.
    
    Parameters
    ----------
    function_name : str, optional
        Name of function to analyze (if None, analyze all)
    
    Returns
    -------
    dict
        Performance trends data
    """
    return _global_monitor.get_performance_trends(function_name)


def clear_performance_metrics():
    """Clear all performance metrics from global monitor."""
    _global_monitor.clear_metrics()


def set_performance_thresholds(execution_time: float = None, 
                              memory_usage: float = None, 
                              cpu_percent: float = None):
    """
    Set performance thresholds for global monitor.
    
    Parameters
    ----------
    execution_time : float, optional
        Execution time threshold in seconds
    memory_usage : float, optional
        Memory usage threshold in MB
    cpu_percent : float, optional
        CPU usage threshold in percent
    """
    _global_monitor.set_thresholds(execution_time, memory_usage, cpu_percent)


def enable_performance_monitoring(enable: bool = True):
    """
    Enable or disable performance monitoring.
    
    Parameters
    ----------
    enable : bool, optional
        Whether to enable performance monitoring (default: True)
    """
    _global_monitor.enable_monitoring = enable


# Convenience functions for specific operations
def monitor_filtering_operation(func):
    """Monitor filtering operations."""
    return monitor_performance(
        function_name=f"filtering_{func.__name__}",
        parameters={"operation": "filtering"}
    )(func)


def monitor_transform_operation(func):
    """Monitor transform operations."""
    return monitor_performance(
        function_name=f"transform_{func.__name__}",
        parameters={"operation": "transform"}
    )(func)


def monitor_feature_extraction_operation(func):
    """Monitor feature extraction operations."""
    return monitor_performance(
        function_name=f"feature_extraction_{func.__name__}",
        parameters={"operation": "feature_extraction"}
    )(func)


def monitor_analysis_operation(func):
    """Monitor analysis operations."""
    return monitor_performance(
        function_name=f"analysis_{func.__name__}",
        parameters={"operation": "analysis"}
    )(func)


# Performance monitoring context manager
@contextmanager
def performance_monitoring_context(operation_name: str, 
                                  signal_length: int = None,
                                  parameters: Dict[str, Any] = None):
    """
    Context manager for performance monitoring.
    
    Parameters
    ----------
    operation_name : str
        Name of the operation being monitored
    signal_length : int, optional
        Length of the input signal
    parameters : dict, optional
        Operation parameters
    """
    with _global_monitor._monitor_execution(operation_name, signal_length, parameters or {}):
        yield


# Performance reporting utilities
def generate_performance_report(function_name: str = None) -> str:
    """
    Generate a formatted performance report.
    
    Parameters
    ----------
    function_name : str, optional
        Name of function to report on (if None, report on all)
    
    Returns
    -------
    str
        Formatted performance report
    """
    summary = get_performance_summary(function_name)
    
    if "message" in summary:
        return f"Performance Report: {summary['message']}"
    
    report = f"""
Performance Report for {summary['function_name']}
===============================================

Execution Statistics:
  Total Executions: {summary['total_executions']}
  Successful Executions: {summary['successful_executions']}
  Success Rate: {summary['success_rate']:.1f}%

Execution Time (seconds):
  Mean: {summary['execution_time']['mean']:.3f}
  Median: {summary['execution_time']['median']:.3f}
  Std Dev: {summary['execution_time']['std']:.3f}
  Min: {summary['execution_time']['min']:.3f}
  Max: {summary['execution_time']['max']:.3f}

Memory Usage (MB):
  Mean: {summary['memory_usage']['mean']:.2f}
  Median: {summary['memory_usage']['median']:.2f}
  Std Dev: {summary['memory_usage']['std']:.2f}
  Min: {summary['memory_usage']['min']:.2f}
  Max: {summary['memory_usage']['max']:.2f}

Signal Length:
  Mean: {summary['signal_length']['mean']:.0f}
  Median: {summary['signal_length']['median']:.0f}
  Std Dev: {summary['signal_length']['std']:.0f}
  Min: {summary['signal_length']['min']:.0f}
  Max: {summary['signal_length']['max']:.0f}
"""
    
    return report
