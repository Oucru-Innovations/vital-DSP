import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer

# Force matplotlib to use the 'Agg' backend for tests
matplotlib.use("Agg")


@pytest.fixture
def visualizer():
    # Dummy config with a normal range for testing purposes
    config = {
        "sdnn": {"normal_range": {"1_min": [30, 70]}},
        "rmssd": {"normal_range": {"1_min": [20, 60]}},
    }
    return HealthReportVisualizer(config)


@pytest.fixture
def feature_data():
    return {
        "sdnn": [35.0, 40.0, 45.0, 1000.0, 35.0, 40.0, 50.0, 60.0, 10000.0],
        "rmssd": [25.0, 30.0, 35.0, 40.0, 100.0, 25.0, 40.0, 60.0, 200.0],
    }


@pytest.fixture
def output_dir(tmp_path):
    # Temporary directory for saving visualizations
    return tmp_path


def test_visualizer_initialization(visualizer):
    assert isinstance(
        visualizer.config, dict
    ), "Visualizer should initialize with a dictionary config."


def test_invalid_config_initialization():
    with pytest.raises(TypeError):
        HealthReportVisualizer("invalid_config")


def test_create_visualizations(visualizer, feature_data, output_dir):
    visualizations = visualizer.create_visualizations(feature_data, output_dir)

    assert "sdnn" in visualizations, "SDNN visualization should be created."
    assert "rmssd" in visualizations, "RMSSD visualization should be created."

    # Check that all types of visualizations are generated
    for plot_type in ["heatmap", "bell_plot", "radar_plot", "violin_plot"]:
        assert plot_type in visualizations["sdnn"], f"SDNN should have {plot_type}."
        assert os.path.exists(
            visualizations["sdnn"][plot_type]
        ), f"{plot_type} should be saved."


def test_bell_shape_plot(visualizer, feature_data, output_dir):
    file_path = visualizer._create_bell_shape_plot(
        "sdnn", feature_data["sdnn"], output_dir
    )
    assert os.path.exists(file_path), "Bell shape plot should be created and saved."


def test_violin_plot(visualizer, feature_data, output_dir):
    file_path = visualizer._create_violin_plot(
        "rmssd", feature_data["rmssd"], output_dir
    )
    assert os.path.exists(file_path), "Violin plot should be created and saved."


def test_heatmap_plot(visualizer, feature_data, output_dir):
    file_path = visualizer._create_heatmap_plot(
        "sdnn", feature_data["sdnn"], output_dir
    )
    assert os.path.exists(file_path), "Heatmap plot should be created and saved."


def test_radar_plot(visualizer, feature_data, output_dir):
    file_path = visualizer._create_radar_plot(
        "rmssd", feature_data["rmssd"], output_dir
    )
    assert os.path.exists(file_path), "Radar plot should be created and saved."


def test_fetch_normal_range(visualizer):
    normal_range = visualizer._fetch_and_validate_normal_range("sdnn", 40)
    assert normal_range == (30, 70), "Should return the correct normal range for SDNN."


def test_handle_inf_values(visualizer):
    assert (
        visualizer._parse_inf_values("inf") == np.inf
    ), "Should return np.inf for 'inf' string."
    assert (
        visualizer._parse_inf_values("-inf") == -np.inf
    ), "Should return -np.inf for '-inf' string."
    assert (
        visualizer._parse_inf_values(50) == 50
    ), "Should return the original value if not 'inf' or '-inf'."
