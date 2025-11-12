"""
Comprehensive unit tests for pipeline_callbacks.py module.

This test file adds extensive coverage for pipeline execution callbacks.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from dash.exceptions import PreventUpdate
import json

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.pipeline_callbacks import (
    register_pipeline_callbacks,
    _get_stage_number,
    _execute_pipeline_stage,
    PIPELINE_STAGES,
)


@pytest.fixture
def mock_app():
    """Create a mock Dash app for testing."""
    app = Mock(spec=Dash)
    app.callback = Mock()
    return app


@pytest.fixture
def sample_signal_data():
    """Create sample signal data for testing."""
    np.random.seed(42)
    # Create ECG-like signal
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000  # signal data and sampling frequency


@pytest.fixture
def sample_pipeline_data():
    """Create sample pipeline data dictionary."""
    return {
        "current_stage": 1,
        "stage2_sqi_type": "snr_sqi",
        "stage2_window_size": 1000,
        "stage2_step_size": 500,
        "stage2_threshold_type": "below",
        "stage2_threshold": 0.7,
        "stage2_scale": "zscore",
        "stage3_filter_type": "traditional",
        "stage3_filter_family": "butter",
        "stage3_filter_response": "bandpass",
        "stage3_filter_lowcut": 0.5,
        "stage3_filter_highcut": 40,
        "stage3_filter_order": 4,
    }


class TestHelperFunctions:
    """Test helper functions in pipeline_callbacks."""

    def test_get_stage_number_with_int(self):
        """Test _get_stage_number with integer input."""
        result = _get_stage_number(5)
        assert result == 5

    def test_get_stage_number_with_string_id(self):
        """Test _get_stage_number with string pipeline_run_id."""
        # Mock the register_pipeline_callbacks module attribute
        with patch('vitalDSP_webapp.callbacks.analysis.pipeline_callbacks.register_pipeline_callbacks') as mock_reg:
            mock_reg.pipeline_data = {
                "12345678": {"current_stage": 3}
            }
            result = _get_stage_number("12345678")
            assert result == 3

    def test_get_stage_number_with_invalid_string(self):
        """Test _get_stage_number with invalid string."""
        result = _get_stage_number("invalid")
        assert result == 0

    def test_get_stage_number_with_nonexistent_id(self):
        """Test _get_stage_number with non-existent pipeline_run_id."""
        with patch('vitalDSP_webapp.callbacks.analysis.pipeline_callbacks.register_pipeline_callbacks') as mock_reg:
            mock_reg.pipeline_data = {}
            result = _get_stage_number("12345678")
            assert result == 0

    def test_execute_pipeline_stage_1(self, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage for stage 1 (Data Ingestion)."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 1, signal_data, fs, "ecg"
        )
        assert success is True
        assert "samples" in result
        assert "duration" in result
        assert "fs" in result
        assert result["fs"] == fs
        assert result["signal_type"] == "ecg"
        assert result["samples"] == len(signal_data)

    @patch('vitalDSP_webapp.callbacks.analysis.quality_sqi_functions.compute_sqi')
    def test_execute_pipeline_stage_2_success(self, mock_compute_sqi, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage for stage 2 (Quality Screening) - success case."""
        signal_data, fs = sample_signal_data
        mock_compute_sqi.return_value = {
            "sqi_values": [0.8, 0.9, 0.85],
            "normal_segments": [(0, 1000), (2000, 3000)],
            "abnormal_segments": [(1000, 2000)],
            "overall_sqi": 0.85
        }
        
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 2, signal_data, fs, "ecg"
        )
        assert success is True
        assert "sqi_type" in result
        assert "quality_scores" in result
        assert "overall_quality" in result
        assert result["overall_quality"] == 0.85
        assert result["passed"] is True  # 0.85 >= 0.7

    @patch('vitalDSP_webapp.callbacks.analysis.quality_sqi_functions.compute_sqi')
    def test_execute_pipeline_stage_2_failure(self, mock_compute_sqi, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage for stage 2 (Quality Screening) - failure case."""
        signal_data, fs = sample_signal_data
        mock_compute_sqi.side_effect = Exception("SQI computation failed")
        
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 2, signal_data, fs, "ecg"
        )
        assert success is False
        assert "error" in result
        assert result["passed"] is False

    @patch('vitalDSP.filtering.signal_filtering.SignalFiltering')
    def test_execute_pipeline_stage_3_traditional_bandpass(self, mock_sf_class, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage for stage 3 (Parallel Processing) - traditional bandpass."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.bandpass.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True
        assert "filtered" in result or "paths" in result

    @patch('vitalDSP.filtering.signal_filtering.SignalFiltering')
    def test_execute_pipeline_stage_3_traditional_lowpass(self, mock_sf_class, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage for stage 3 - traditional lowpass."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_response"] = "lowpass"
        
        mock_sf = Mock()
        mock_sf.lowpass.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.signal_filtering.SignalFiltering')
    def test_execute_pipeline_stage_3_traditional_highpass(self, mock_sf_class, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage for stage 3 - traditional highpass."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_response"] = "highpass"
        
        mock_sf = Mock()
        mock_sf.highpass.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    def test_execute_pipeline_stage_invalid_stage(self, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage with invalid stage number."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 99, signal_data, fs, "ecg"
        )
        assert success is False
        assert isinstance(result, str)


class TestPipelineCallbacksRegistration:
    """Test the callback registration functionality."""

    def test_register_pipeline_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_pipeline_callbacks(mock_app)
        
        # Verify that callback decorator was called multiple times
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1


class TestPipelineCallbacks:
    """Test individual pipeline callbacks."""

    def test_update_stage3_params_visibility_registered(self, mock_app):
        """Test that update_stage3_params_visibility callback is registered."""
        register_pipeline_callbacks(mock_app)
        
        # Check that callback was registered
        callback_found = False
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 5:
                if any("stage3-traditional-params" in str(out) for out in outputs):
                    callback_found = True
                    break
        
        assert callback_found, "update_stage3_params_visibility callback should be registered"

    def test_update_advanced_method_params_registered(self, mock_app):
        """Test that update_advanced_method_params callback is registered."""
        register_pipeline_callbacks(mock_app)
        
        # Check that callback was registered
        callback_found = False
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 2:
                if any("kalman-params" in str(out) for out in outputs):
                    callback_found = True
                    break
        
        assert callback_found, "update_advanced_method_params callback should be registered"


class TestPipelineExecutionCallback:
    """Test the main pipeline execution callback."""

    def test_handle_pipeline_execution_registered(self, mock_app):
        """Test that handle_pipeline_execution callback is registered."""
        register_pipeline_callbacks(mock_app)
        
        # Check that callback was registered
        callback_found = False
        for call in mock_app.callback.call_args_list:
            inputs = call[0][0]
            if isinstance(inputs, list):
                if any("pipeline-run-btn" in str(inp) for inp in inputs):
                    callback_found = True
                    break
        
        assert callback_found, "handle_pipeline_execution callback should be registered"


class TestPipelineConstants:
    """Test pipeline constants."""

    def test_pipeline_stages_defined(self):
        """Test that PIPELINE_STAGES is properly defined."""
        assert isinstance(PIPELINE_STAGES, list)
        assert len(PIPELINE_STAGES) == 8
        assert "Data Ingestion" in PIPELINE_STAGES
        assert "Quality Screening" in PIPELINE_STAGES
        assert "Output Package" in PIPELINE_STAGES

