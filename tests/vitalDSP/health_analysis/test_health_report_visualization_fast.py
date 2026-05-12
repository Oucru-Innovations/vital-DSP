"""
Fast unit tests for HealthReportVisualizer using mocks.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer

import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def visualizer():
    config = {
        "sdnn": {
            "normal_range": {"1_min": [20, 50], "5_min": [50, 130]},
            "description": "Test SDNN feature"
        }
    }
    return HealthReportVisualizer(config, "1_min")


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return {
        "sdnn": np.random.normal(35.0, 5.0, 20).tolist(),
    }


def test_visualizer_initialization(visualizer):
    assert visualizer.config is not None
    assert visualizer.segment_duration == "1_min"


def test_normalize_web_path(visualizer):
    assert visualizer._normalize_web_path("C:\\Users\\test\\plot.png") == "C:/Users/test/plot.png"
    assert visualizer._normalize_web_path("/home/user/plot.png") == "/home/user/plot.png"
    assert visualizer._normalize_web_path(None) is None


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_gauge_chart(mock_close, mock_savefig, visualizer, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = visualizer._create_gauge_chart("sdnn", sample_data["sdnn"], temp_dir)
        assert path is not None
        assert ".png" in path
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


def test_create_gauge_chart_error_on_unknown_feature(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer._create_gauge_chart("unknown_feature", [35.0], temp_dir)
        assert "Error" in result


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_violin_plot(mock_close, mock_savefig, visualizer, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = visualizer._create_violin_plot("sdnn", sample_data["sdnn"], temp_dir)
        assert path is not None
        assert ".png" in path


def test_create_violin_plot_empty_data(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer._create_violin_plot("sdnn", [], temp_dir)
        assert result is not None
        assert "Error" in result


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_box_swarm_plot(mock_close, mock_savefig, visualizer, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = visualizer._create_box_swarm_plot("sdnn", sample_data["sdnn"], temp_dir)
        assert path is not None
        assert ".png" in path


def test_create_box_swarm_plot_empty_data(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer._create_box_swarm_plot("sdnn", [], temp_dir)
        assert result is not None
        assert "Error" in result


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_create_line_plot_with_rolling_stats(mock_close, mock_savefig, visualizer, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = visualizer._create_plot_line_with_rolling_stats("sdnn", sample_data["sdnn"], temp_dir)
        assert path is not None
        assert ".png" in path


def test_create_line_plot_empty_data(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer._create_plot_line_with_rolling_stats("sdnn", [], temp_dir)
        assert result is not None
        assert "Error" in result


def test_trend_sparkline_none_for_short_data(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        assert visualizer._create_trend_sparkline("sdnn", [35.0], temp_dir) is None
        assert visualizer._create_trend_sparkline("sdnn", [35.0, 40.0], temp_dir) is None


@patch('vitalDSP.health_analysis.health_report_visualization.plt.savefig')
@patch('vitalDSP.health_analysis.health_report_visualization.plt.close')
def test_trend_sparkline_with_enough_data(mock_close, mock_savefig, visualizer, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = visualizer._create_trend_sparkline("sdnn", sample_data["sdnn"], temp_dir)
        assert path is not None
        assert ".png" in path


@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_gauge_chart')
@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_violin_plot')
@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_box_swarm_plot')
@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_plot_line_with_rolling_stats')
@patch('vitalDSP.health_analysis.health_report_visualization.HealthReportVisualizer._create_trend_sparkline')
def test_create_visualizations_basic(mock_sparkline, mock_line, mock_box, mock_violin,
                                     mock_gauge, visualizer, sample_data):
    mock_gauge.return_value = "/tmp/sdnn_gauge.png"
    mock_violin.return_value = "/tmp/sdnn_violin.png"
    mock_box.return_value = "/tmp/sdnn_box.png"
    mock_line.return_value = "/tmp/sdnn_line.png"
    mock_sparkline.return_value = "/tmp/sdnn_sparkline.png"

    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer.create_visualizations(sample_data, temp_dir)
        assert isinstance(result, dict)
        assert "sdnn" in result


def test_create_visualizations_empty_data(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer.create_visualizations({}, temp_dir)
        assert isinstance(result, dict)
        assert len(result) == 0


def test_create_visualizations_none_output_dir(visualizer, sample_data):
    result = visualizer.create_visualizations(sample_data, output_dir=None)
    assert isinstance(result, dict)
    assert "sdnn" in result


def test_fetch_and_validate_normal_range(visualizer):
    lo, hi = visualizer._fetch_and_validate_normal_range("sdnn", 35.0)
    assert lo == 20
    assert hi == 50


def test_fetch_and_validate_unknown_feature(visualizer):
    with pytest.raises(ValueError):
        visualizer._fetch_and_validate_normal_range("unknown", 1.0)


def test_parse_inf_values(visualizer):
    assert visualizer._parse_inf_values("inf") == np.inf
    assert visualizer._parse_inf_values("-inf") == -np.inf
    assert visualizer._parse_inf_values(3.14) == 3.14


def test_get_normal_range_for_feature(visualizer):
    r = visualizer._get_normal_range_for_feature("sdnn")
    assert r is not None
    assert len(r) == 2
    assert visualizer._get_normal_range_for_feature("unknown") is None


def test_thread_safe_matplotlib_operation(visualizer):
    def mock_func(*args, **kwargs):
        return "test_result"

    result = visualizer._thread_safe_matplotlib_operation(mock_func, "arg1", key="value")
    assert result == "test_result"


def test_thread_safe_matplotlib_operation_with_exception(visualizer):
    def mock_func_error(*args, **kwargs):
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        visualizer._thread_safe_matplotlib_operation(mock_func_error)


def test_get_plot_bounds(visualizer):
    lo, hi = visualizer._get_plot_bounds(20, 50, [10, 60])
    assert lo < 20
    assert hi > 50


def test_summary_chart_created(visualizer, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = visualizer.create_summary_chart(sample_data, temp_dir)
        assert path is not None
        assert os.path.exists(path)


def test_summary_chart_empty_data(visualizer):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = visualizer.create_summary_chart({}, temp_dir)
        assert result is None
