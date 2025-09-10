import pytest
import os
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock
import matplotlib
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer

# Force matplotlib to use the 'Agg' backend for tests
matplotlib.use("Agg")

# Filter out expected warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Data has no positive values.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More than 20 figures have been opened.*")


@pytest.fixture
def visualizer():
    # Dummy config with a normal range for testing purposes
    config = {
        "sdnn": {"normal_range": {"1_min": [30, 70]}},
        "rmssd": {"normal_range": {"1_min": [20, 60]}},
    }
    return HealthReportVisualizer(config)

@pytest.fixture
def visualizer_inf():
    # Dummy config with a normal range for testing purposes
    config = {
        "sdnn": {"normal_range": {"1_min": [-np.inf, np.inf]}},  # Use Inf values here
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
def feature_data_single():
    return {
        "sdnn": 35.0,
        "rmssd": 25.0,
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


def test_create_visualizations_single(visualizer, feature_data_single, output_dir):
    visualizations = visualizer.create_visualizations(feature_data_single, output_dir)

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

def test_create_plot_exception(visualizer, feature_data):
    # Mock the logger on the visualizer instance
    with patch.object(visualizer, 'logger') as mock_logger:
        # Mock os.path.join to raise an exception when called
        with patch('os.path.join', side_effect=Exception("Normal range for feature 'sdnn_dump' not found.")):
            # Call the _create_bell_shape_plot function, which should raise an exception
            result = visualizer._create_bell_shape_plot("sdnn_dump", feature_data["sdnn"], "output_dir")
            # Assert that the logger was called with the correct error message
            # mock_logger.error.assert_called_once_with("Error generating heatmap for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            # Assert that the function returns the expected error message
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_heatmap_plot("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating heatmap plot for: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_radar_plot("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error generating radar for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_violin_plot("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error generating violin for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_plot_lag("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_spectral_density_plot("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_plot_periodogram("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_plot_line_with_rolling_stats("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"

            result = visualizer._create_plot_autocorrelation("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_box_swarm_plot("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
            result = visualizer._create_plot_seasonal_decomposition("sdnn_dump", feature_data["sdnn"], "output_dir")
            # mock_logger.error.assert_called_once_with("Error creating enhanced lag plot for sdnn_dump: Normal range for feature 'sdnn_dump' not found.")
            assert result == "Error generating plot for sdnn_dump"
            
def test_normal_range_not_defined(visualizer):
    with pytest.raises(ValueError):
        visualizer._fetch_and_validate_normal_range("non_existing_features", 40)

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


def test_nan_inf_handling(visualizer_inf):
    # Test Inf in normal_min and normal_max
    normal_range = visualizer_inf._fetch_and_validate_normal_range('sdnn', 100)
    assert normal_range == (-100, 300), "Should replace Inf values with appropriate values."

def test_empty_valid_values(visualizer_inf):
    # Test Inf in normal_min and normal_max
    with pytest.raises(ValueError):
        normal_range = visualizer_inf._fetch_and_validate_normal_range('sdnn', np.inf)

def test_statistical_values_finite(visualizer, output_dir):
    values = [np.inf, -np.inf]
    with patch.object(visualizer, '_fetch_and_validate_normal_range', return_value=(30, 70)):
        file_path = visualizer._create_radar_plot('sdnn', values, output_dir)
    assert os.path.exists(file_path), "Radar plot should be created even with infinite values."


def test_normal_range_not_none(visualizer):
    normal_range = visualizer._get_normal_range_for_feature('sdnn')
    assert normal_range == [30, 70], "Normal range should not be None."


def test_parse_inf_values(visualizer):
    assert visualizer._parse_inf_values('-inf') == -np.inf, "Should handle '-inf' string."
    assert visualizer._parse_inf_values('inf') == np.inf, "Should handle 'inf' string."


def test_nan_inf_handling_in_normal_range(visualizer_inf):
    result = visualizer_inf._fetch_and_validate_normal_range('sdnn', 10)
    # Ensure Inf is replaced with the correct values
    assert result == (-10, 30), "Should handle Inf values correctly."


@pytest.fixture
def output_dir(tmp_path):
    # Create a valid directory path for saving visualizations
    dir_path = tmp_path / "visualizations"
    dir_path.mkdir()
    return dir_path


def test_highlight_freqs_and_peaks(visualizer, output_dir):
    values = np.random.randn(100)
    freqs = [(1, 5), (10, 15)]
    file_path = visualizer._create_spectral_density_plot('sdnn', values, output_dir, highlight_freqs=freqs)
    assert os.path.exists(file_path), "Spectral density plot should be created with highlighted frequencies."
    
    file_path = visualizer._create_spectral_density_plot('sdnn', values, output_dir, highlight_freqs=freqs,peak_threshold=0.)
    assert os.path.exists(file_path), "Spectral density plot should be created with highlighted frequencies."
    
def test_threshold_and_time_none(visualizer, output_dir):
    values = np.random.randn(100)
    file_path = visualizer._create_plot_periodogram('sdnn', values, output_dir, threshold=0.5)
    assert os.path.exists(file_path), "Periodogram plot should be created."


def test_create_plot_autocorrelation(visualizer, output_dir):
    values = np.random.randn(100)
    file_path = visualizer._create_plot_autocorrelation('sdnn', values, output_dir, lags=5)
    assert os.path.exists(file_path), "Autocorrelation plot should be created."


def test_create_plot_seasonal_decomposition(visualizer, output_dir):
    values = np.random.randn(100)
    file_path = visualizer._create_plot_seasonal_decomposition('sdnn', values, output_dir, freq=12)
    assert os.path.exists(file_path), "Seasonal decomposition plot should be created."


def test_value_capping(visualizer, output_dir):
    values = [-100, 0, 50, 100, 150]  # Outlier values
    file_path = visualizer._create_box_swarm_plot('sdnn', values, output_dir)
    assert os.path.exists(file_path), "Box swarm plot should be created with capped values."
