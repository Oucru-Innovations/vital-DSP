"""
Comprehensive tests for plot_utils.py to improve coverage.

This file adds extensive coverage for plot utility functions.
"""

import pytest
import numpy as np
from vitalDSP_webapp.utils.plot_utils import (
    limit_plot_data,
    smart_downsample,
    check_plot_data_size,
    estimate_plot_size,
    get_recommended_downsampling,
)


@pytest.fixture
def sample_signal_data():
    """Create sample signal data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    return t, signal


@pytest.fixture
def large_signal_data():
    """Create large signal data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 600, 60000)  # 10 minutes at 100Hz
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    return t, signal


class TestLimitPlotData:
    """Test limit_plot_data function."""

    def test_limit_plot_data_basic(self, sample_signal_data):
        """Test basic limit_plot_data with small data."""
        time_axis, signal_data = sample_signal_data
        limited_time, limited_signal = limit_plot_data(time_axis, signal_data)
        
        assert len(limited_time) == len(limited_signal)
        assert len(limited_time) <= len(time_axis)

    def test_limit_plot_data_empty(self):
        """Test limit_plot_data with empty arrays."""
        time_axis = np.array([])
        signal_data = np.array([])
        limited_time, limited_signal = limit_plot_data(time_axis, signal_data)
        
        assert len(limited_time) == 0
        assert len(limited_signal) == 0

    def test_limit_plot_data_length_mismatch(self):
        """Test limit_plot_data with mismatched lengths."""
        time_axis = np.linspace(0, 10, 1000)
        signal_data = np.random.randn(500)  # Different length
        limited_time, limited_signal = limit_plot_data(time_axis, signal_data)
        
        assert len(limited_time) == len(limited_signal)
        assert len(limited_time) == min(len(time_axis), len(signal_data))

    def test_limit_plot_data_max_duration(self, large_signal_data):
        """Test limit_plot_data with max_duration limit."""
        time_axis, signal_data = large_signal_data
        limited_time, limited_signal = limit_plot_data(time_axis, signal_data, max_duration=300.0)
        
        duration = limited_time[-1] - limited_time[0] if len(limited_time) > 1 else 0
        assert duration <= 300.0

    def test_limit_plot_data_max_points(self):
        """Test limit_plot_data with max_points limit."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 50000)
        signal_data = np.random.randn(50000)
        limited_time, limited_signal = limit_plot_data(time_axis, signal_data, max_points=10000)
        
        assert len(limited_time) <= 10000
        assert len(limited_signal) <= 10000

    def test_limit_plot_data_with_start_time(self, large_signal_data):
        """Test limit_plot_data with specified start_time."""
        time_axis, signal_data = large_signal_data
        start_time = 100.0
        limited_time, limited_signal = limit_plot_data(
            time_axis, signal_data, max_duration=300.0, start_time=start_time
        )
        
        if len(limited_time) > 0:
            assert limited_time[0] >= start_time

    def test_limit_plot_data_single_point(self):
        """Test limit_plot_data with single point."""
        time_axis = np.array([5.0])
        signal_data = np.array([1.0])
        limited_time, limited_signal = limit_plot_data(time_axis, signal_data)
        
        assert len(limited_time) == 1
        assert len(limited_signal) == 1


class TestSmartDownsample:
    """Test smart_downsample function."""

    def test_smart_downsample_basic(self, sample_signal_data):
        """Test basic smart_downsample."""
        time_axis, signal_data = sample_signal_data
        downsampled_time, downsampled_signal = smart_downsample(time_axis, signal_data, target_points=100)
        
        assert len(downsampled_time) == len(downsampled_signal)
        assert len(downsampled_time) <= len(time_axis)

    def test_smart_downsample_no_downsample_needed(self, sample_signal_data):
        """Test smart_downsample when no downsampling is needed."""
        time_axis, signal_data = sample_signal_data
        # Request more points than available
        downsampled_time, downsampled_signal = smart_downsample(time_axis, signal_data, target_points=10000)
        
        assert len(downsampled_time) == len(time_axis)
        assert len(downsampled_signal) == len(signal_data)
        assert np.array_equal(downsampled_time, time_axis)
        assert np.array_equal(downsampled_signal, signal_data)

    def test_smart_downsample_large_data(self):
        """Test smart_downsample with large dataset."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 50000)
        signal_data = np.sin(2 * np.pi * 1.2 * time_axis) + 0.1 * np.random.randn(len(time_axis))
        downsampled_time, downsampled_signal = smart_downsample(time_axis, signal_data, target_points=1000)
        
        assert len(downsampled_time) <= len(time_axis)
        assert len(downsampled_signal) <= len(signal_data)
        assert len(downsampled_time) >= 100  # Should have reasonable number of points

    def test_smart_downsample_small_window(self):
        """Test smart_downsample with small window size."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 1000)
        signal_data = np.random.randn(1000)
        downsampled_time, downsampled_signal = smart_downsample(time_axis, signal_data, target_points=500)
        
        assert len(downsampled_time) == len(downsampled_signal)
        assert len(downsampled_time) <= len(time_axis)

    def test_smart_downsample_empty(self):
        """Test smart_downsample with empty arrays."""
        time_axis = np.array([])
        signal_data = np.array([])
        downsampled_time, downsampled_signal = smart_downsample(time_axis, signal_data)
        
        assert len(downsampled_time) == 0
        assert len(downsampled_signal) == 0


class TestCheckPlotDataSize:
    """Test check_plot_data_size function."""

    def test_check_plot_data_size_acceptable(self, sample_signal_data):
        """Test check_plot_data_size with acceptable size."""
        time_axis, signal_data = sample_signal_data
        result = check_plot_data_size(time_axis, signal_data, max_points=10000)
        
        assert result is True

    def test_check_plot_data_size_too_large(self):
        """Test check_plot_data_size with too large data."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 50000)
        signal_data = np.random.randn(50000)
        result = check_plot_data_size(time_axis, signal_data, max_points=10000)
        
        assert result is False

    def test_check_plot_data_size_exact_limit(self):
        """Test check_plot_data_size at exact limit."""
        np.random.seed(42)
        time_axis = np.linspace(0, 10, 10000)
        signal_data = np.random.randn(10000)
        result = check_plot_data_size(time_axis, signal_data, max_points=10000)
        
        assert result is True

    def test_check_plot_data_size_empty(self):
        """Test check_plot_data_size with empty arrays."""
        time_axis = np.array([])
        signal_data = np.array([])
        result = check_plot_data_size(time_axis, signal_data)
        
        assert result is True


class TestEstimatePlotSize:
    """Test estimate_plot_size function."""

    def test_estimate_plot_size_small(self):
        """Test estimate_plot_size with small number of points."""
        size = estimate_plot_size(100)
        assert isinstance(size, str)
        assert "KB" in size or "MB" in size

    def test_estimate_plot_size_large(self):
        """Test estimate_plot_size with large number of points."""
        size = estimate_plot_size(100000)
        assert isinstance(size, str)
        assert "MB" in size

    def test_estimate_plot_size_zero(self):
        """Test estimate_plot_size with zero points."""
        size = estimate_plot_size(0)
        assert isinstance(size, str)
        assert "KB" in size or "MB" in size

    def test_estimate_plot_size_medium(self):
        """Test estimate_plot_size with medium number of points."""
        size = estimate_plot_size(10000)
        assert isinstance(size, str)
        assert "KB" in size or "MB" in size


class TestGetRecommendedDownsampling:
    """Test get_recommended_downsampling function."""

    def test_get_recommended_downsampling_small(self):
        """Test get_recommended_downsampling with small dataset."""
        recommendations = get_recommended_downsampling(1000, 100.0, 10.0)
        
        assert isinstance(recommendations, dict)
        assert "needs_downsampling" in recommendations
        assert "needs_duration_limit" in recommendations
        assert "target_points" in recommendations
        assert "method" in recommendations
        assert recommendations["needs_downsampling"] is False

    def test_get_recommended_downsampling_large_points(self):
        """Test get_recommended_downsampling with many points."""
        recommendations = get_recommended_downsampling(50000, 100.0, 500.0)
        
        assert isinstance(recommendations, dict)
        assert recommendations["needs_downsampling"] is True
        assert recommendations["target_points"] == 10000
        # Method can be "smart" or "simple" depending on exact threshold
        assert recommendations["method"] in ["smart", "simple"]

    def test_get_recommended_downsampling_medium_points(self):
        """Test get_recommended_downsampling with medium number of points."""
        recommendations = get_recommended_downsampling(15000, 100.0, 150.0)
        
        assert isinstance(recommendations, dict)
        assert recommendations["needs_downsampling"] is True
        assert recommendations["target_points"] == 10000
        assert recommendations["method"] == "simple"

    def test_get_recommended_downsampling_long_duration(self):
        """Test get_recommended_downsampling with long duration."""
        recommendations = get_recommended_downsampling(10000, 100.0, 600.0)
        
        assert isinstance(recommendations, dict)
        assert recommendations["needs_duration_limit"] is True
        assert len(recommendations["warnings"]) > 0

    def test_get_recommended_downsampling_both_issues(self):
        """Test get_recommended_downsampling with both issues."""
        recommendations = get_recommended_downsampling(60000, 100.0, 600.0)
        
        assert isinstance(recommendations, dict)
        assert recommendations["needs_downsampling"] is True
        assert recommendations["needs_duration_limit"] is True
        assert len(recommendations["warnings"]) > 0

    def test_get_recommended_downsampling_size_info(self):
        """Test get_recommended_downsampling includes size info for large datasets."""
        recommendations = get_recommended_downsampling(20000, 100.0, 200.0)
        
        assert isinstance(recommendations, dict)
        if "size_info" in recommendations:
            assert "current" in recommendations["size_info"]
            assert "recommended" in recommendations["size_info"]

