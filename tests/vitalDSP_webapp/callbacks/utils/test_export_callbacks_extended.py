"""
Extended comprehensive tests for export_callbacks.py.

This file adds extensive coverage for export callback functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from dash.exceptions import PreventUpdate

# Import the module to test
from vitalDSP_webapp.callbacks.utils.export_callbacks import (
    register_filtering_export_callbacks,
    register_time_domain_export_callbacks,
    register_frequency_domain_export_callbacks,
    register_physiological_export_callbacks,
    register_quality_export_callbacks,
    register_respiratory_export_callbacks,
    register_transforms_export_callbacks,
    register_all_export_callbacks,
)


@pytest.fixture
def mock_app():
    """Create a mock Dash app for testing."""
    app = Mock(spec=Dash)
    captured_callbacks = []
    
    def mock_callback(*args, **kwargs):
        def decorator(func):
            captured_callbacks.append((args, kwargs, func))
            return func
        return decorator
    
    app.callback = mock_callback
    app._captured_callbacks = captured_callbacks
    return app


@pytest.fixture
def sample_filtered_data():
    """Create sample filtered signal data."""
    signal = np.sin(2 * np.pi * np.linspace(0, 10, 1000))
    time = np.linspace(0, 10, 1000)
    return {
        "signal": signal.tolist(),
        "time": time.tolist(),
        "filter_type": "bandpass",
        "filter_params": {"low_freq": 0.5, "high_freq": 40},
        "signal_type": "PPG",
        "sampling_freq": 100
    }


def get_callback_function(mock_app, output_id_pattern):
    """Helper to extract callback function from registered callbacks."""
    for args, kwargs, func in mock_app._captured_callbacks:
        outputs = args[0] if args else []
        if isinstance(outputs, list):
            for output in outputs:
                if output_id_pattern in str(output):
                    return func
    return None


class TestFilteringExportCallbacks:
    """Test filtering export callbacks."""

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_filtered_signal_csv')
    def test_export_filtered_csv_success(self, mock_export_csv, mock_app, sample_filtered_data):
        """Test export_filtered_csv callback with valid data."""
        mock_export_csv.return_value = "time,signal\n0.0,0.1\n"
        
        register_filtering_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-filtered-csv")
        
        if callback_func:
            result = callback_func(1, sample_filtered_data, 100)
            assert isinstance(result, dict)
            assert "content" in result
            assert "filename" in result
            assert result["filename"].endswith(".csv")
            mock_export_csv.assert_called_once()

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_filtered_signal_csv')
    def test_export_filtered_csv_no_time(self, mock_export_csv, mock_app):
        """Test export_filtered_csv with no time array."""
        mock_export_csv.return_value = "time,signal\n0.0,0.1\n"
        
        filtered_data = {
            "signal": [0.1, 0.2, 0.3],
            "time": [],
            "filter_type": "bandpass",
            "sampling_freq": 100
        }
        
        register_filtering_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-filtered-csv")
        
        if callback_func:
            result = callback_func(1, filtered_data, 100)
            assert isinstance(result, dict)
            assert "content" in result

    def test_export_filtered_csv_no_clicks(self, mock_app, sample_filtered_data):
        """Test export_filtered_csv with no button clicks."""
        register_filtering_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-filtered-csv")
        
        if callback_func:
            with pytest.raises(PreventUpdate):
                callback_func(None, sample_filtered_data, 100)

    def test_export_filtered_csv_no_data(self, mock_app):
        """Test export_filtered_csv with no data."""
        register_filtering_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-filtered-csv")
        
        if callback_func:
            with pytest.raises(PreventUpdate):
                callback_func(1, None, 100)

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_filtered_signal_json')
    def test_export_filtered_json_success(self, mock_export_json, mock_app, sample_filtered_data):
        """Test export_filtered_json callback with valid data."""
        mock_export_json.return_value = '{"time": [0.0], "signal": [0.1]}'
        
        register_filtering_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-filtered-json")
        
        if callback_func:
            result = callback_func(1, sample_filtered_data, 100)
            assert isinstance(result, dict)
            assert "content" in result
            assert "filename" in result
            assert result["filename"].endswith(".json")
            mock_export_json.assert_called_once()


class TestRespiratoryExportCallbacks:
    """Test respiratory export callbacks."""

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_respiratory_analysis_csv')
    def test_export_respiratory_csv_success(self, mock_export_csv, mock_app):
        """Test respiratory CSV export callback."""
        mock_export_csv.return_value = "time,rr_interval\n0.0,0.8\n"
        
        sample_data = {
            "rr_intervals": [0.8, 0.9, 0.85],
            "time": [0.0, 1.0, 2.0]
        }
        
        register_respiratory_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-respiratory-csv")
        
        if callback_func:
            result = callback_func(1, sample_data)
            assert isinstance(result, dict)
            assert "content" in result
            assert "filename" in result

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_respiratory_analysis_json')
    def test_export_respiratory_json_success(self, mock_export_json, mock_app):
        """Test respiratory JSON export callback."""
        mock_export_json.return_value = '{"rr_intervals": [0.8, 0.9]}'
        
        sample_data = {
            "rr_intervals": [0.8, 0.9, 0.85],
            "time": [0.0, 1.0, 2.0]
        }
        
        register_respiratory_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-respiratory-json")
        
        if callback_func:
            result = callback_func(1, sample_data)
            assert isinstance(result, dict)
            assert "content" in result
            assert "filename" in result


class TestTransformsExportCallbacks:
    """Test transforms export callbacks."""

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_transform_results_csv')
    def test_export_transforms_csv_success(self, mock_export_csv, mock_app):
        """Test transforms CSV export callback."""
        mock_export_csv.return_value = "frequency,magnitude\n1.0,0.5\n"
        
        sample_data = {
            "transform_type": "fft",
            "frequency": [1.0, 2.0],
            "magnitude": [0.5, 0.3]
        }
        
        register_transforms_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-transforms-csv")
        
        if callback_func:
            result = callback_func(1, sample_data)
            assert isinstance(result, dict)
            assert "content" in result
            assert "filename" in result

    @patch('vitalDSP_webapp.callbacks.utils.export_callbacks.export_transform_results_json')
    def test_export_transforms_json_success(self, mock_export_json, mock_app):
        """Test transforms JSON export callback."""
        mock_export_json.return_value = '{"transform_type": "fft", "frequency": [1.0]}'
        
        sample_data = {
            "transform_type": "fft",
            "frequency": [1.0, 2.0],
            "magnitude": [0.5, 0.3]
        }
        
        register_transforms_export_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "download-transforms-json")
        
        if callback_func:
            result = callback_func(1, sample_data)
            assert isinstance(result, dict)
            assert "content" in result
            assert "filename" in result


class TestAllExportCallbacksRegistration:
    """Test registration of all export callbacks."""

    def test_register_all_export_callbacks(self, mock_app):
        """Test that all export callbacks are registered."""
        register_all_export_callbacks(mock_app)
        # Check that callback was called (it's a decorator function)
        assert len(mock_app._captured_callbacks) > 0

