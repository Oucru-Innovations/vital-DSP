"""
Comprehensive tests for HealthReportVisualizer.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer


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
    return {
        "sdnn": [25.0, 30.0, 35.0, 40.0, 45.0],
        "rmssd": [15.0, 20.0, 25.0, 30.0, 35.0],
        "nn50": [10, 15, 20, 25, 30]
    }


def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert visualizer.config is not None
    assert visualizer.segment_duration == "1_min"


def test_normalize_web_path(visualizer):
    """Test web path normalization."""
    # Test Windows path
    windows_path = "C:\\Users\\test\\images\\plot.png"
    normalized = visualizer._normalize_web_path(windows_path)
    assert normalized == "C:/Users/test/images/plot.png"
    
    # Test Unix path (should remain unchanged)
    unix_path = "/home/user/images/plot.png"
    normalized = visualizer._normalize_web_path(unix_path)
    assert normalized == unix_path
    
    # Test None path
    normalized = visualizer._normalize_web_path(None)
    assert normalized is None


def test_create_bell_shape_plot(visualizer, sample_data):
    """Test bell shape plot creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_bell_shape_plot("sdnn", sample_data["sdnn"], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


def test_create_bell_shape_plot_with_empty_data(visualizer):
    """Test bell shape plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_bell_shape_plot("sdnn", [], temp_dir)
        
        # Should return error message for empty data
        assert plot_path is not None
        assert "Error" in plot_path


def test_create_bell_shape_plot_with_single_value(visualizer):
    """Test bell shape plot creation with single value."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_bell_shape_plot("sdnn", [35.0], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)


def test_create_violin_plot(visualizer, sample_data):
    """Test violin plot creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_violin_plot("sdnn", sample_data["sdnn"], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


def test_create_violin_plot_with_empty_data(visualizer):
    """Test violin plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_violin_plot("sdnn", [], temp_dir)
        
        # Should return error message for empty data
        assert plot_path is not None
        assert "Error" in plot_path


def test_create_box_swarm_plot(visualizer, sample_data):
    """Test box swarm plot creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_box_swarm_plot("sdnn", sample_data["sdnn"], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


def test_create_box_swarm_plot_with_empty_data(visualizer):
    """Test box swarm plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_box_swarm_plot("sdnn", [], temp_dir)
        
        # Should return error message for empty data
        assert plot_path is not None
        assert "Error" in plot_path


def test_create_heatmap_plot(visualizer, sample_data):
    """Test heatmap plot creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_heatmap_plot("sdnn", sample_data["sdnn"], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


def test_create_heatmap_plot_with_empty_data(visualizer):
    """Test heatmap plot creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_heatmap_plot("sdnn", [], temp_dir)
        
        # Should return error message for empty data
        assert plot_path is not None
        assert "Error" in plot_path


def test_create_spectral_density_plot(visualizer, sample_data):
    """Test spectral density plot creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_spectral_density_plot("sdnn", sample_data["sdnn"], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


def test_create_radar_plot(visualizer, sample_data):
    """Test radar plot creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = visualizer._create_radar_plot("sdnn", sample_data["sdnn"], temp_dir)
        
        assert plot_path is not None
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


def test_create_visualizations_basic(visualizer, sample_data):
    """Test basic visualization creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations(sample_data, temp_dir)
        
        assert isinstance(visualizations, dict)
        assert "sdnn" in visualizations
        assert isinstance(visualizations["sdnn"], dict)


def test_create_visualizations_with_empty_data(visualizer):
    """Test visualization creation with empty data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations({}, temp_dir)
        
        assert isinstance(visualizations, dict)
        assert len(visualizations) == 0


def test_create_visualizations_with_none_values(visualizer):
    """Test visualization creation with None values."""
    data_with_none = {
        "sdnn": [25.0, None, 35.0, None, 45.0],
        "rmssd": [15.0, 20.0, None, 30.0, 35.0]
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations(data_with_none, temp_dir)
        
        assert isinstance(visualizations, dict)
        assert "sdnn" in visualizations


def test_create_visualizations_with_large_dataset(visualizer):
    """Test visualization creation with large dataset."""
    large_data = {
        "sdnn": np.random.normal(35, 5, 1000).tolist(),
        "rmssd": np.random.normal(25, 3, 1000).tolist()
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations(large_data, temp_dir)
        
        assert isinstance(visualizations, dict)
        assert "sdnn" in visualizations


def test_create_visualizations_with_unknown_feature(visualizer, sample_data):
    """Test visualization creation with unknown feature."""
    data_with_unknown = {
        "unknown_feature": [1, 2, 3, 4, 5],
        "sdnn": sample_data["sdnn"]
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizations = visualizer.create_visualizations(data_with_unknown, temp_dir)
        
        assert isinstance(visualizations, dict)
        assert "sdnn" in visualizations
        # Unknown feature should be handled gracefully


def test_create_visualizations_with_plot_errors(visualizer):
    """Test visualization creation with plot generation errors."""
    # Mock matplotlib to raise an error
    with patch('matplotlib.pyplot.savefig', side_effect=Exception("Plot error")):
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizations = visualizer.create_visualizations({"sdnn": [1, 2, 3]}, temp_dir)
            
            # Should handle errors gracefully
            assert isinstance(visualizations, dict)


def test_fetch_and_validate_normal_range(visualizer):
    """Test fetching and validating normal range."""
    # Test with valid feature and single value
    range_info = visualizer._fetch_and_validate_normal_range("sdnn", 35.0)
    assert range_info is not None
    
    # Test with invalid feature
    try:
        range_info = visualizer._fetch_and_validate_normal_range("unknown", 1.0)
        assert range_info is None
    except ValueError:
        # Expected for invalid features
        pass


def test_parse_inf_values(visualizer):
    """Test parsing infinite values."""
    # Test with normal values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    parsed = visualizer._parse_inf_values(values)
    assert parsed == values
    
    # Test with inf values
    values_with_inf = [1.0, float('inf'), 3.0, float('-inf'), 5.0]
    parsed = visualizer._parse_inf_values(values_with_inf)
    assert len(parsed) == len(values_with_inf)  # Should keep all values, just convert inf to NaN


def test_get_normal_range_for_feature(visualizer):
    """Test getting normal range for feature."""
    # Test with valid feature
    normal_range = visualizer._get_normal_range_for_feature("sdnn")
    assert normal_range is not None
    assert len(normal_range) == 2
    
    # Test with invalid feature
    normal_range = visualizer._get_normal_range_for_feature("unknown")
    assert normal_range is None


def test_auto_detect_roi(visualizer):
    """Test automatic ROI detection."""
    # Create mock spectrogram data
    Sxx = np.random.rand(10, 20)
    times = np.linspace(0, 10, 20)
    frequencies = np.linspace(0, 5, 10)
    
    roi = visualizer.auto_detect_roi(Sxx, times, frequencies)
    
    assert isinstance(roi, tuple)
    assert len(roi) == 2  # Should return (time_range, freq_range)