"""
Plot utilities for vitalDSP webapp.

This module provides utility functions for optimizing plot data.
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def limit_plot_data(
    time_axis: np.ndarray,
    signal_data: np.ndarray,
    max_duration: float = 300.0,
    max_points: int = 10000,
    start_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limit plot data to maximum duration and maximum number of points.

    This function optimizes plot data for better performance by:
    1. Limiting the time duration to a maximum (default 5 minutes)
    2. Downsampling if there are too many points (default 10,000 max)

    Args:
        time_axis: Time values array
        signal_data: Signal values array
        max_duration: Maximum duration in seconds (default 300 = 5 minutes)
        max_points: Maximum number of points to plot (default 10,000)
        start_time: Optional start time to begin the window from

    Returns:
        tuple: (limited_time_axis, limited_signal_data)

    Examples:
        >>> time = np.linspace(0, 600, 60000)  # 10 minutes at 100Hz
        >>> signal = np.random.randn(60000)
        >>> time_limited, signal_limited = limit_plot_data(time, signal)
        >>> len(time_limited)  # Will be ~10,000 points (5 minutes downsampled)
        10000
    """
    if len(time_axis) == 0 or len(signal_data) == 0:
        logger.warning("Empty data provided to limit_plot_data")
        return time_axis, signal_data

    if len(time_axis) != len(signal_data):
        logger.error(
            f"Time and signal data length mismatch: {len(time_axis)} vs {len(signal_data)}"
        )
        # Truncate to shorter length
        min_len = min(len(time_axis), len(signal_data))
        time_axis = time_axis[:min_len]
        signal_data = signal_data[:min_len]

    original_length = len(signal_data)
    original_duration = time_axis[-1] - time_axis[0] if len(time_axis) > 0 else 0

    # Step 1: Limit duration to max_duration
    if original_duration > max_duration:
        if start_time is not None:
            # Use specified start time
            end_time = start_time + max_duration
            mask = (time_axis >= start_time) & (time_axis <= end_time)
        else:
            # Use first max_duration seconds from the START of the provided data
            start_time = time_axis[0]
            end_time = start_time + max_duration
            mask = (time_axis >= start_time) & (time_axis <= end_time)

        time_axis = time_axis[mask]
        signal_data = signal_data[mask]

        logger.warning(
            f"Plot duration limited to {max_duration}s "
            f"(original: {original_duration:.1f}s, "
            f"reduced from {original_length} to {len(signal_data)} points)"
        )

    # Step 2: Downsample if too many points
    if len(signal_data) > max_points:
        # Calculate downsample factor
        factor = int(np.ceil(len(signal_data) / max_points))

        # Use simple decimation (every Nth point)
        time_axis = time_axis[::factor]
        signal_data = signal_data[::factor]

        logger.info(
            f"Plot downsampled by factor {factor}x "
            f"(from {original_length} to {len(signal_data)} points)"
        )

    final_duration = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0
    logger.info(
        f"Plot data limited: {original_length} points ({original_duration:.1f}s) → "
        f"{len(signal_data)} points ({final_duration:.1f}s)"
    )

    # Final check: ensure we have data to plot
    if len(time_axis) == 0 or len(signal_data) == 0:
        logger.warning("limit_plot_data resulted in empty arrays!")

    return time_axis, signal_data


def smart_downsample(
    time_axis: np.ndarray, signal_data: np.ndarray, target_points: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intelligently downsample signal data while preserving important features.

    Uses min-max downsampling to preserve peaks and valleys.

    Args:
        time_axis: Time values array
        signal_data: Signal values array
        target_points: Target number of points after downsampling

    Returns:
        tuple: (downsampled_time, downsampled_signal)
    """
    if len(signal_data) <= target_points:
        return time_axis, signal_data

    # Calculate window size for downsampling
    window_size = len(signal_data) // (target_points // 2)  # Divide by 2 for min-max pairs

    if window_size < 2:
        # Use simple decimation if window too small
        factor = len(signal_data) // target_points
        return time_axis[::factor], signal_data[::factor]

    # Min-max downsampling: preserve peaks and valleys
    downsampled_time = []
    downsampled_signal = []

    for i in range(0, len(signal_data), window_size):
        window = signal_data[i : i + window_size]
        time_window = time_axis[i : i + window_size]

        if len(window) == 0:
            continue

        # Find min and max in window
        min_idx = np.argmin(window)
        max_idx = np.argmax(window)

        # Add min and max points (in time order)
        if time_window[min_idx] < time_window[max_idx]:
            downsampled_time.extend([time_window[min_idx], time_window[max_idx]])
            downsampled_signal.extend([window[min_idx], window[max_idx]])
        else:
            downsampled_time.extend([time_window[max_idx], time_window[min_idx]])
            downsampled_signal.extend([window[max_idx], window[min_idx]])

    logger.info(
        f"Smart downsample: {len(signal_data)} → {len(downsampled_signal)} points "
        f"(target: {target_points})"
    )

    return np.array(downsampled_time), np.array(downsampled_signal)


def check_plot_data_size(
    time_axis: np.ndarray, signal_data: np.ndarray, max_points: int = 10000
) -> bool:
    """
    Check if plot data size is within acceptable limits.

    Args:
        time_axis: Time values array
        signal_data: Signal values array
        max_points: Maximum acceptable number of points

    Returns:
        bool: True if data size is acceptable, False otherwise
    """
    if len(signal_data) > max_points:
        duration = time_axis[-1] - time_axis[0] if len(time_axis) > 0 else 0
        logger.warning(
            f"Plot data size ({len(signal_data)} points, {duration:.1f}s) "
            f"exceeds recommended maximum ({max_points} points). "
            f"Consider using limit_plot_data() or smart_downsample()."
        )
        return False
    return True


def estimate_plot_size(num_points: int) -> str:
    """
    Estimate the size of plot data in JSON format.

    Args:
        num_points: Number of data points

    Returns:
        str: Estimated size (e.g., "5.2 MB")
    """
    # Rough estimate: ~50 bytes per point in Plotly JSON
    size_bytes = num_points * 50
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024

    if size_mb >= 1:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_kb:.1f} KB"


def get_recommended_downsampling(
    num_points: int, sampling_freq: float, duration: float
) -> dict:
    """
    Get recommended downsampling parameters based on data characteristics.

    Args:
        num_points: Number of data points
        sampling_freq: Sampling frequency in Hz
        duration: Duration in seconds

    Returns:
        dict: Recommendations including target_points, method, and warnings
    """
    recommendations = {
        "needs_downsampling": False,
        "needs_duration_limit": False,
        "target_points": num_points,
        "method": "none",
        "warnings": [],
    }

    # Check duration
    if duration > 300:  # 5 minutes
        recommendations["needs_duration_limit"] = True
        recommendations["warnings"].append(
            f"Duration ({duration:.1f}s) exceeds 5-minute limit. "
            "Consider limiting to 300s."
        )

    # Check number of points
    if num_points > 50000:
        recommendations["needs_downsampling"] = True
        recommendations["target_points"] = 10000
        recommendations["method"] = "smart"
        recommendations["warnings"].append(
            f"Too many points ({num_points}). Recommend smart downsampling to 10,000 points."
        )
    elif num_points > 10000:
        recommendations["needs_downsampling"] = True
        recommendations["target_points"] = 10000
        recommendations["method"] = "simple"
        recommendations["warnings"].append(
            f"Many points ({num_points}). Recommend simple downsampling to 10,000 points."
        )

    # Estimate plot size
    estimated_size = estimate_plot_size(num_points)
    if num_points > 10000:
        recommended_size = estimate_plot_size(recommendations["target_points"])
        recommendations["size_info"] = {
            "current": estimated_size,
            "recommended": recommended_size,
        }

    return recommendations
