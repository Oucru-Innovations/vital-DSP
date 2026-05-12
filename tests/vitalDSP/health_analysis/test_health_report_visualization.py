import pytest
import os
import numpy as np
import warnings
from unittest.mock import patch
import matplotlib
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer

matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def visualizer():
    config = {
        "sdnn": {"normal_range": {"1_min": [30, 70]}},
        "rmssd": {"normal_range": {"1_min": [20, 60]}},
    }
    return HealthReportVisualizer(config)


@pytest.fixture
def visualizer_inf():
    config = {
        "sdnn": {"normal_range": {"1_min": [-np.inf, np.inf]}},
        "rmssd": {"normal_range": {"1_min": [20, 60]}},
    }
    return HealthReportVisualizer(config)


@pytest.fixture
def feature_data():
    return {
        "sdnn": [35.0, 40.0, 45.0, 50.0, 38.0, 42.0, 47.0, 55.0, 60.0],
        "rmssd": [25.0, 30.0, 35.0, 40.0, 28.0, 33.0, 38.0, 45.0, 50.0],
    }


@pytest.fixture
def feature_data_single():
    return {"sdnn": 35.0, "rmssd": 25.0}


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "visualizations"
    d.mkdir()
    return d


def test_visualizer_initialization(visualizer):
    assert isinstance(visualizer.config, dict)


def test_invalid_config_initialization():
    with pytest.raises(TypeError):
        HealthReportVisualizer("invalid_config")


def test_create_visualizations_returns_correct_keys(visualizer, feature_data, output_dir):
    result = visualizer.create_visualizations(feature_data, output_dir)

    assert "sdnn" in result
    assert "rmssd" in result

    for key in ["gauge_chart", "violin_plot", "line_with_rolling_stats", "plot_box_swarm"]:
        assert key in result["sdnn"], f"Expected key '{key}' in sdnn visualizations"

    # trend_sparkline should appear when >=3 segments
    assert "trend_sparkline" in result["sdnn"]


def test_create_visualizations_single_value(visualizer, feature_data_single, output_dir):
    result = visualizer.create_visualizations(feature_data_single, output_dir)

    assert "sdnn" in result
    # No trend_sparkline for single-segment
    assert "trend_sparkline" not in result["sdnn"]
    # Core charts still present
    for key in ["gauge_chart", "violin_plot", "line_with_rolling_stats", "plot_box_swarm"]:
        assert key in result["sdnn"]


def test_create_visualizations_no_removed_keys(visualizer, feature_data, output_dir):
    result = visualizer.create_visualizations(feature_data, output_dir)

    for removed_key in ["heatmap", "bell_plot", "radar_plot", "lag_plot",
                        "plot_periodogram", "plot_spectrogram", "plot_spectral_density"]:
        assert removed_key not in result["sdnn"], f"Removed key '{removed_key}' should not be present"


def test_gauge_chart_created(visualizer, feature_data, output_dir):
    path = visualizer._create_gauge_chart("sdnn", feature_data["sdnn"], output_dir)
    assert os.path.exists(path), "Gauge chart file should exist"


def test_gauge_chart_single_value(visualizer, output_dir):
    path = visualizer._create_gauge_chart("sdnn", [40.0], output_dir)
    assert os.path.exists(path)


def test_trend_sparkline_created(visualizer, feature_data, output_dir):
    path = visualizer._create_trend_sparkline("sdnn", feature_data["sdnn"], output_dir)
    assert path is not None
    assert os.path.exists(path)


def test_trend_sparkline_returns_none_for_short_data(visualizer, output_dir):
    assert visualizer._create_trend_sparkline("sdnn", [40.0], output_dir) is None
    assert visualizer._create_trend_sparkline("sdnn", [40.0, 45.0], output_dir) is None


def test_violin_plot_created(visualizer, feature_data, output_dir):
    path = visualizer._create_violin_plot("rmssd", feature_data["rmssd"], output_dir)
    assert os.path.exists(path)


def test_violin_plot_falls_back_for_short_data(visualizer, output_dir):
    # < 4 values falls back to box+swarm
    path = visualizer._create_violin_plot("sdnn", [35.0, 42.0, 38.0], output_dir)
    assert path is not None
    assert "box_swarm" in path or os.path.exists(path)


def test_box_swarm_plot_created(visualizer, feature_data, output_dir):
    path = visualizer._create_box_swarm_plot("sdnn", feature_data["sdnn"], output_dir)
    assert os.path.exists(path)


def test_box_swarm_plot_single_value(visualizer, output_dir):
    path = visualizer._create_box_swarm_plot("sdnn", [40.0], output_dir)
    assert os.path.exists(path)


def test_line_with_rolling_stats_created(visualizer, feature_data, output_dir):
    path = visualizer._create_plot_line_with_rolling_stats(
        "sdnn", feature_data["sdnn"], output_dir
    )
    assert os.path.exists(path)


def test_summary_chart_created(visualizer, feature_data, output_dir):
    path = visualizer.create_summary_chart(feature_data, output_dir)
    assert os.path.exists(path)


def test_fetch_normal_range_returns_config_values(visualizer):
    normal_min, normal_max = visualizer._fetch_and_validate_normal_range("sdnn", 40)
    # Config values must be returned unchanged — value=40 must NOT corrupt them
    assert normal_min == 30
    assert normal_max == 70


def test_fetch_normal_range_outlier_does_not_corrupt(visualizer):
    # Value far outside range; bounds must still come from config
    normal_min, normal_max = visualizer._fetch_and_validate_normal_range("sdnn", 8)
    assert normal_min == 30
    assert normal_max == 70


def test_fetch_normal_range_missing_feature(visualizer):
    with pytest.raises(ValueError):
        visualizer._fetch_and_validate_normal_range("nonexistent_feature", 40)


def test_fetch_normal_range_inf_bounds(visualizer_inf):
    # Inf bounds should be substituted using the value, not crash
    normal_min, normal_max = visualizer_inf._fetch_and_validate_normal_range("sdnn", 100)
    assert np.isfinite(normal_min)
    assert np.isfinite(normal_max)


def test_fetch_normal_range_inf_bounds_with_inf_value(visualizer_inf):
    # Both config and value are inf — should either raise or return finite fallback
    try:
        result = visualizer_inf._fetch_and_validate_normal_range("sdnn", np.inf)
        # If it doesn't raise, result should still be finite or a defined fallback
        assert result is not None
    except ValueError:
        pass  # Acceptable


def test_get_plot_bounds(visualizer):
    lo, hi = visualizer._get_plot_bounds(30, 70, [10, 80])
    assert lo < 30
    assert hi > 70


def test_parse_inf_values(visualizer):
    assert visualizer._parse_inf_values("inf") == np.inf
    assert visualizer._parse_inf_values("-inf") == -np.inf
    assert visualizer._parse_inf_values(50) == 50


def test_normal_range_not_none(visualizer):
    normal_range = visualizer._get_normal_range_for_feature("sdnn")
    assert normal_range == [30, 70]


def test_get_normal_range_missing_feature(visualizer):
    assert visualizer._get_normal_range_for_feature("nonexistent") is None


def test_normalize_web_path(visualizer):
    assert visualizer._normalize_web_path("C:\\Users\\test\\plot.png") == "C:/Users/test/plot.png"
    assert visualizer._normalize_web_path(None) is None


def test_value_capping_in_box_swarm(visualizer, output_dir):
    values = [-100, 0, 50, 100, 150]
    path = visualizer._create_box_swarm_plot("sdnn", values, output_dir)
    assert os.path.exists(path)


def test_gauge_chart_below_range(visualizer, output_dir):
    # Value below normal_min
    path = visualizer._create_gauge_chart("sdnn", [10.0], output_dir)
    assert os.path.exists(path)


def test_gauge_chart_above_range(visualizer, output_dir):
    # Value above normal_max
    path = visualizer._create_gauge_chart("sdnn", [200.0], output_dir)
    assert os.path.exists(path)


def test_create_visualizations_none_output_dir(visualizer, feature_data):
    result = visualizer.create_visualizations(feature_data, output_dir=None)
    assert isinstance(result, dict)
    assert "sdnn" in result


def test_create_visualizations_empty_feature_data(visualizer, output_dir):
    result = visualizer.create_visualizations({}, output_dir)
    assert result == {}


def test_thread_safe_operation(visualizer):
    def mock_func(*args, **kwargs):
        return "result"

    result = visualizer._thread_safe_matplotlib_operation(mock_func)
    assert result == "result"


def test_thread_safe_operation_exception(visualizer):
    def mock_func_error(*args, **kwargs):
        raise ValueError("test error")

    with pytest.raises(ValueError):
        visualizer._thread_safe_matplotlib_operation(mock_func_error)
