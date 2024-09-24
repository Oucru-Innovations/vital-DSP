import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer

# Force matplotlib to use the 'Agg' backend for tests
matplotlib.use("Agg")


@pytest.fixture
def mock_config():
    """Provides a mock config for the visualizer."""
    return {
        "rmssd": {"normal_range": {"1_min": [20, 60]}},
        "sdnn": {"normal_range": {"1_min": [30, 70]}},
    }


@pytest.fixture
def visualizer(mock_config):
    """Initializes the HealthReportVisualizer with mock config."""
    return HealthReportVisualizer(mock_config)


def test_initialize_visualizer(mock_config):
    """Test visualizer initialization."""
    visualizer = HealthReportVisualizer(mock_config)
    assert visualizer.config == mock_config


def test_invalid_config_type():
    """Test visualizer raises TypeError for invalid config."""
    with pytest.raises(TypeError):
        HealthReportVisualizer("invalid config")


def test_fetch_and_validate_normal_range_valid_feature(visualizer):
    """Test fetching and validating normal range for a valid feature."""
    normal_min, normal_max = visualizer._fetch_and_validate_normal_range("rmssd", 50)
    assert normal_min == 20
    assert normal_max == 60


def test_fetch_and_validate_normal_range_invalid_feature(visualizer):
    """Test fetching and validating normal range for an invalid feature."""
    with pytest.raises(ValueError):
        visualizer._fetch_and_validate_normal_range("invalid_feature", 50)


def test_create_visualizations_creates_files(visualizer, tmpdir):
    """Test creating visualizations and ensuring files are generated."""
    feature_data = {"rmssd": 45, "sdnn": [30, 40, 50, 60]}

    output_dir = tmpdir.mkdir("visualizations")
    visualizations = visualizer.create_visualizations(
        feature_data, output_dir=str(output_dir)
    )

    # Check that the correct files were created
    assert os.path.exists(visualizations["rmssd"]["heatmap"])
    assert os.path.exists(visualizations["rmssd"]["bell_plot"])
    assert os.path.exists(visualizations["rmssd"]["radar_plot"])
    assert os.path.exists(visualizations["rmssd"]["violin_plot"])
    assert os.path.exists(visualizations["sdnn"])


def test_heatmap_plot(visualizer, tmpdir):
    """Test heatmap plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_heatmap_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)


def test_bell_shape_plot(visualizer, tmpdir):
    """Test bell shape plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_bell_shape_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)


def test_violin_plot(visualizer, tmpdir):
    """Test violin plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_violin_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)


def test_radar_plot(visualizer, tmpdir):
    """Test radar plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_radar_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)


def test_line_plot(visualizer, tmpdir):
    """Test line plot creation for time-series data."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_line_plot("sdnn", [30, 40, 50, 60], str(output_dir))

    assert os.path.exists(filepath)


def test_donut_plot(visualizer, tmpdir):
    """Test donut plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_donut_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)


def test_waterfall_plot(visualizer, tmpdir):
    """Test waterfall plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_waterfall_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)


def test_difference_plot(visualizer, tmpdir):
    """Test difference plot creation."""
    output_dir = tmpdir.mkdir("visualizations")
    filepath = visualizer._create_difference_plot("rmssd", 45, str(output_dir))

    assert os.path.exists(filepath)
