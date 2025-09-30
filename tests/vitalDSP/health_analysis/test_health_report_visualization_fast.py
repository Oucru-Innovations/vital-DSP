"""
Fast unit tests for HealthReportVisualizer using mocks.
This replaces the slow comprehensive tests.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, Mock
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer

# Use Agg backend to prevent display issues
import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def visualizer():
    """Create a HealthReportVisualizer instance for testing."""
    config = {
        "sdnn": {
            "normal_range": {"1_min": [20, 50], "5_min": [50, 130]},
            "description": "Test SDNN feature"
        }
    }
    return HealthReportVisualizer(config, "1_min")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return {
        "sdnn": np.random.normal(35.0, 5.0, 100).tolist(),
        "rmssd": np.random.normal(25.0, 3.0, 100).tolist(),
        "nn50": np.random.poisson(20, 100).tolist()
    }


def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert visualizer.config is not None
    assert visualizer.segment_duration == "1_min"


def test_normalize_web_path(visualizer):
    """Test web path normalization."""
    windows_path = "C:\\Users\\test\\images\\plot.png"
    normalized = visualizer._normalize_web_path(windows_path)
    assert normalized == "C:/Users/test/images/plot.png"

    unix_path = "/home/user/images/plot.png"
    normalized = visualizer._normalize_web_path(unix_path)
    assert normalized == unix_path

    normalized = visualizer._normalize_web_path(None)
    assert normalized is None


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_bell_shape_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test bell shape plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_bell_shape_plot("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


def test_create_bell_shape_plot_with_empty_data(visualizer):
    """Test bell shape plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_bell_shape_plot("sdnn", [], temp_dir)
        assert plot_path is not None
        assert "Error" in plot_path


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_violin_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test violin plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_violin_plot("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path


def test_create_violin_plot_with_empty_data(visualizer):
    """Test violin plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_violin_plot("sdnn", [], temp_dir)
        assert plot_path is not None
        assert "Error" in plot_path


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.clf')
def test_thread_safe_matplotlib_operation(mock_clf, mock_close, mock_savefig, visualizer):
    """Test thread-safe matplotlib operation (lines 65-81)."""
    # Mock function to execute
    def mock_func(*args, **kwargs):
        return "test_result"

    result = visualizer._thread_safe_matplotlib_operation(mock_func, "arg1", key="value")

    # Should execute the function
    assert result == "test_result"
    # Should clean up
    assert mock_clf.call_count >= 2  # Called before and after
    mock_close.assert_called_once()


@patch('vitalDSP.health_analysis.health_report_visualization.plt.clf')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_thread_safe_matplotlib_operation_with_exception(mock_close, mock_clf, visualizer):
    """Test thread-safe matplotlib operation with exception (lines 77-81)."""
    # Mock function that raises exception
    def mock_func_error(*args, **kwargs):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        visualizer._thread_safe_matplotlib_operation(mock_func_error)

    # Should still clean up on error
    mock_clf.assert_called()
    # close() is called twice: once at line 69 and once at line 80 in exception handler
    assert mock_close.call_count == 2


def test_create_visualizations_with_none_output_dir(visualizer, sample_data):
    """Test create_visualizations with None output_dir (lines 157-158)."""
    # Should create default directory
    result = visualizer.create_visualizations(sample_data, output_dir=None)

    assert isinstance(result, dict)
    # Should have created visualizations for all features
    assert "sdnn" in result
    assert "rmssd" in result
    assert "nn50" in result


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_box_swarm_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test box swarm plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_box_swarm_plot("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path


def test_create_box_swarm_plot_with_empty_data(visualizer):
    """Test box swarm plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_box_swarm_plot("sdnn", [], temp_dir)
        assert plot_path is not None
        assert "Error" in plot_path


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_heatmap_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test heatmap plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_heatmap_plot("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path


def test_create_heatmap_plot_with_empty_data(visualizer):
    """Test heatmap plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_heatmap_plot("sdnn", [], temp_dir)
        assert plot_path is not None
        assert "Error" in plot_path


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_radar_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test radar plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_radar_plot("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path


@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_bell_shape_plot')
@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_violin_plot')
@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_heatmap_plot')
def test_create_visualizations_basic(mock_heatmap, mock_violin, mock_bell, visualizer, sample_data):
    """Test basic visualization creation with mocked plot functions."""
    # Mock plot functions to return paths
    mock_bell.return_value = "/tmp/sdnn_bell.png"
    mock_violin.return_value = "/tmp/sdnn_violin.png"
    mock_heatmap.return_value = "/tmp/sdnn_heatmap.png"

    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations(sample_data, temp_dir)

        assert isinstance(visualizations, dict)
        assert "sdnn" in visualizations


def test_create_visualizations_with_empty_data(visualizer):
    """Test visualization creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations({}, temp_dir)

        assert isinstance(visualizations, dict)
        assert len(visualizations) == 0


def test_fetch_and_validate_normal_range(visualizer):
    """Test fetching and validating normal range."""
    range_info = visualizer._fetch_and_validate_normal_range("sdnn", 35.0)
    assert range_info is not None

    try:
        range_info = visualizer._fetch_and_validate_normal_range("unknown", 1.0)
        assert range_info is None
    except ValueError:
        pass


def test_parse_inf_values(visualizer):
    """Test parsing infinite values."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    parsed = visualizer._parse_inf_values(values)
    assert parsed == values

    values_with_inf = [1.0, float('inf'), 3.0, float('-inf'), 5.0]
    parsed = visualizer._parse_inf_values(values_with_inf)
    assert len(parsed) == len(values_with_inf)


def test_get_normal_range_for_feature(visualizer):
    """Test getting normal range for feature."""
    normal_range = visualizer._get_normal_range_for_feature("sdnn")
    assert normal_range is not None
    assert len(normal_range) == 2

    normal_range = visualizer._get_normal_range_for_feature("unknown")
    assert normal_range is None


def test_auto_detect_roi(visualizer):
    """Test automatic ROI detection."""
    Sxx = np.random.rand(10, 20)
    times = np.linspace(0, 10, 20)
    frequencies = np.linspace(0, 5, 10)

    roi = visualizer.auto_detect_roi(Sxx, times, frequencies)

    assert isinstance(roi, tuple)
    assert len(roi) == 2


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_spectrogram_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test spectrogram plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_plot_spectrogram("sdnn", sample_data["sdnn"], temp_dir)

        # May return error if data is insufficient for spectrogram
        assert plot_path is not None


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_lag_plot(mock_close, mock_savefig, visualizer, sample_data):
    """Test lag plot creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_plot_lag("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_line_plot_with_rolling_stats(mock_close, mock_savefig, visualizer, sample_data):
    """Test line plot with rolling stats creation with mocked matplotlib."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_plot_line_with_rolling_stats("sdnn", sample_data["sdnn"], temp_dir)

        assert plot_path is not None
        assert '.png' in plot_path


def test_create_line_plot_with_rolling_stats_empty_data(visualizer):
    """Test line plot with rolling stats with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_plot_line_with_rolling_stats("sdnn", [], temp_dir)

        assert plot_path is not None
        assert "Error" in plot_path