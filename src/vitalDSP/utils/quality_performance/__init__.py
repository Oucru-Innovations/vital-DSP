"""
Quality and Performance Module

This module provides quality assessment and performance monitoring utilities for physiological signal processing.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Performance monitoring
- Quality assessment
- Performance optimization
- Monitoring utilities

Examples:
--------
Performance monitoring:
    >>> from vitalDSP.utils.quality_performance import PerformanceMonitor
    >>> monitor = PerformanceMonitor()
"""

from .performance_monitoring import (
    PerformanceMonitor,
    PerformanceMetrics,
    monitor_performance,
    get_performance_summary,
)

__all__ = [
    "PerformanceMonitor",
    "PerformanceMetrics",
    "monitor_performance",
    "get_performance_summary",
]