"""
Simplified tests for health_report_visualization.py module.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock
import os
import tempfile

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
    HEALTH_VIZ_AVAILABLE = True
except ImportError as e:
    HEALTH_VIZ_AVAILABLE = False


@pytest.mark.skipif(not HEALTH_VIZ_AVAILABLE, reason="Health visualization module not available")
class TestHealthReportVisualizerInitialization:

    def test_init_basic(self):
        config = {'feature_ranges': {}}
        hrv = HealthReportVisualizer(config=config)
        assert isinstance(hrv, HealthReportVisualizer)
        assert hrv.config == config

    def test_init_with_config(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
            'rmssd': {'normal_range': {'1_min': [20, 60]}},
        }
        hrv = HealthReportVisualizer(config=config)
        assert isinstance(hrv, HealthReportVisualizer)
        assert hrv.config == config

    def test_init_with_segment_duration(self):
        config = {'feature_ranges': {}}
        hrv = HealthReportVisualizer(config=config, segment_duration="5_min")
        assert hrv.segment_duration == "5_min"

    def test_init_invalid_config(self):
        with pytest.raises(TypeError):
            HealthReportVisualizer(config="invalid_config")


@pytest.mark.skipif(not HEALTH_VIZ_AVAILABLE, reason="Health visualization module not available")
class TestVisualizationMethods:

    def test_create_visualizations_basic(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
        }
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations({'sdnn': [35, 40, 45]}, output_dir=temp_dir)
            assert isinstance(result, dict)
            assert "sdnn" in result

    def test_create_visualizations_empty_data(self):
        config = {'feature_ranges': {}}
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations({}, output_dir=temp_dir)
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_create_visualizations_single_value(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
        }
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations({'sdnn': 45}, output_dir=temp_dir)
            assert isinstance(result, dict)
            assert "sdnn" in result
            # No sparkline for single segment
            assert "trend_sparkline" not in result["sdnn"]

    def test_create_visualizations_chart_keys(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
        }
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations({'sdnn': [35, 40, 45, 50]}, output_dir=temp_dir)
            assert "gauge_chart" in result["sdnn"]
            assert "violin_plot" in result["sdnn"]
            assert "line_with_rolling_stats" in result["sdnn"]
            assert "plot_box_swarm" in result["sdnn"]
            assert "trend_sparkline" in result["sdnn"]


@pytest.mark.skipif(not HEALTH_VIZ_AVAILABLE, reason="Health visualization module not available")
class TestUtilityMethods:

    def test_fetch_and_validate_normal_range_valid(self):
        config = {'sdnn': {'normal_range': {'1_min': [30, 70]}}}
        hrv = HealthReportVisualizer(config=config)
        lo, hi = hrv._fetch_and_validate_normal_range('sdnn', 50)
        assert lo == 30
        assert hi == 70

    def test_fetch_and_validate_normal_range_missing_feature(self):
        config = {'feature_ranges': {}}
        hrv = HealthReportVisualizer(config=config)
        hrv._get_normal_range_for_feature = Mock(return_value=None)
        with pytest.raises(ValueError):
            hrv._fetch_and_validate_normal_range('invalid_feature', 50)

    def test_fetch_and_validate_normal_range_outlier_doesnt_corrupt(self):
        config = {'sdnn': {'normal_range': {'1_min': [30, 70]}}}
        hrv = HealthReportVisualizer(config=config)
        lo, hi = hrv._fetch_and_validate_normal_range('sdnn', 8)
        assert lo == 30  # must be config value, not min(30, 70, 8)=8
        assert hi == 70

    def test_fetch_and_validate_inf_bounds(self):
        config = {'sdnn': {'normal_range': {'1_min': ['-inf', 'inf']}}}
        hrv = HealthReportVisualizer(config=config)
        lo, hi = hrv._fetch_and_validate_normal_range('sdnn', 50)
        assert np.isfinite(lo)
        assert np.isfinite(hi)

    def test_get_plot_bounds(self):
        config = {'sdnn': {'normal_range': {'1_min': [30, 70]}}}
        hrv = HealthReportVisualizer(config=config)
        lo, hi = hrv._get_plot_bounds(30, 70, [10, 80])
        assert lo < 30
        assert hi > 70

    def test_parse_inf_values(self):
        config = {}
        hrv = HealthReportVisualizer(config=config)
        assert hrv._parse_inf_values("inf") == np.inf
        assert hrv._parse_inf_values("-inf") == -np.inf
        assert hrv._parse_inf_values(3.0) == 3.0


@pytest.mark.skipif(not HEALTH_VIZ_AVAILABLE, reason="Health visualization module not available")
class TestErrorHandling:

    def test_create_visualizations_with_nan_data(self):
        config = {'sdnn': {'normal_range': {'1_min': [30, 70]}}}
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations({'sdnn': np.nan}, output_dir=temp_dir)
            assert isinstance(result, dict)

    def test_create_visualizations_with_infinite_data(self):
        config = {'sdnn': {'normal_range': {'1_min': [30, 70]}}}
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations({'sdnn': np.inf}, output_dir=temp_dir)
            assert isinstance(result, dict)

    def test_gauge_chart_error_returns_string(self):
        config = {}
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv._create_gauge_chart("nonexistent", [40.0], temp_dir)
            assert "Error" in result

    def test_box_swarm_error_returns_string(self):
        config = {}
        hrv = HealthReportVisualizer(config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv._create_box_swarm_plot("nonexistent", [40.0], temp_dir)
            assert "Error" in result


@pytest.mark.skipif(not HEALTH_VIZ_AVAILABLE, reason="Health visualization module not available")
class TestIntegrationScenarios:

    def test_full_workflow_with_multiple_features(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
            'rmssd': {'normal_range': {'1_min': [20, 60]}},
        }
        hrv = HealthReportVisualizer(config=config)
        feature_data = {
            'sdnn': [35, 40, 45, 50, 38],
            'rmssd': [25, 30, 35, 28, 32],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations(feature_data, output_dir=temp_dir)
            assert isinstance(result, dict)
            assert "sdnn" in result
            assert "rmssd" in result

    def test_summary_chart_full_workflow(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
            'rmssd': {'normal_range': {'1_min': [20, 60]}},
        }
        hrv = HealthReportVisualizer(config=config)
        segment_values = {
            'sdnn': [35, 40, 45],
            'rmssd': [25, 30, 35],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = hrv.create_summary_chart(segment_values, output_dir=temp_dir)
            assert path is not None
            assert os.path.exists(path)

    def test_workflow_with_mixed_valid_invalid_data(self):
        config = {
            'sdnn': {'normal_range': {'1_min': [30, 70]}},
        }
        hrv = HealthReportVisualizer(config=config)
        feature_data = {
            'sdnn': [35, 40, 45],
            'unknown_feature': [42, 43, 44],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            result = hrv.create_visualizations(feature_data, output_dir=temp_dir)
            assert isinstance(result, dict)
            # sdnn should work; unknown_feature may error gracefully
            assert "sdnn" in result
