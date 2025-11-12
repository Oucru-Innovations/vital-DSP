"""
Direct tests for transform_callbacks.py functions.

This file tests callback functions directly after registration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.transform_callbacks import (
    register_transform_callbacks,
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
def sample_signal_data():
    """Create sample signal data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000


def get_callback_function(mock_app, output_id_pattern):
    """Helper to extract callback function from registered callbacks."""
    for args, kwargs, func in mock_app._captured_callbacks:
        outputs = args[0] if args else []
        if isinstance(outputs, list):
            for output in outputs:
                if output_id_pattern in str(output):
                    return func
    return None


class TestTransformParametersDirect:
    """Test transform parameters callback directly."""

    def test_update_transform_parameters_direct_none(self, mock_app):
        """Test update_transform_parameters with None."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func(None)
            assert result == []

    def test_update_transform_parameters_direct_fft(self, mock_app):
        """Test update_transform_parameters for FFT."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func("fft")
            assert isinstance(result, list)
            assert len(result) > 0
            # Check for FFT-specific elements
            result_str = str(result)
            assert "FFT" in result_str or "fft" in result_str.lower()

    def test_update_transform_parameters_direct_stft(self, mock_app):
        """Test update_transform_parameters for STFT."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func("stft")
            assert isinstance(result, list)
            assert len(result) > 0

    def test_update_transform_parameters_direct_wavelet(self, mock_app):
        """Test update_transform_parameters for Wavelet."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func("wavelet")
            assert isinstance(result, list)
            assert len(result) > 0

    def test_update_transform_parameters_direct_hilbert(self, mock_app):
        """Test update_transform_parameters for Hilbert."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func("hilbert")
            assert isinstance(result, list)
            assert len(result) > 0

    def test_update_transform_parameters_direct_mfcc(self, mock_app):
        """Test update_transform_parameters for MFCC."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func("mfcc")
            assert isinstance(result, list)
            assert len(result) > 0

    def test_update_transform_parameters_direct_invalid(self, mock_app):
        """Test update_transform_parameters with invalid type."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-parameters-container")
        
        if callback_func:
            result = callback_func("invalid_transform_type")
            assert result == []


class TestStoreParametersDirect:
    """Test store parameters callback directly."""

    def test_store_parameters_direct_empty(self, mock_app):
        """Test store_parameters with empty inputs."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "store-transforms-data")
        
        if callback_func:
            result = callback_func([], [], "fft")
            assert isinstance(result, dict)
            assert result["transform_type"] == "fft"

    def test_store_parameters_direct_with_values(self, mock_app):
        """Test store_parameters with parameter values."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "store-transforms-data")
        
        if callback_func:
            param_values = [1024, "hann"]
            param_ids = [
                {"type": "transform-param", "param": "fft-npoints"},
                {"type": "transform-param", "param": "fft-window"}
            ]
            result = callback_func(param_values, param_ids, "fft")
            assert isinstance(result, dict)
            assert result["transform_type"] == "fft"
            assert result.get("fft-npoints") == 1024
            assert result.get("fft-window") == "hann"

    def test_store_parameters_direct_mismatched_lengths(self, mock_app):
        """Test store_parameters with mismatched param_values and param_ids."""
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "store-transforms-data")
        
        if callback_func:
            param_values = [1024]
            param_ids = [
                {"type": "transform-param", "param": "fft-npoints"},
                {"type": "transform-param", "param": "fft-window"}
            ]
            result = callback_func(param_values, param_ids, "fft")
            assert isinstance(result, dict)
            assert result["transform_type"] == "fft"


class TestTimePositionNudgeDirect:
    """Test time position nudge callback directly."""

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_m10(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with -10 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-m10.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(1, 0, 0, 0, 0, 50)
            assert result == 40  # 50 - 10

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_m5(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with -5 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-m5.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(0, 1, 0, 0, 0, 50)
            assert result == 45  # 50 - 5

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_center(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with center button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-center.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(0, 0, 1, 0, 0, 50)
            assert result == 50  # Center at 50%

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_p5(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with +5 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p5.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(0, 0, 0, 1, 0, 50)
            assert result == 55  # 50 + 5

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_p10(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with +10 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p10.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(0, 0, 0, 0, 1, 50)
            assert result == 60  # 50 + 10

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_no_trigger(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with no trigger."""
        mock_ctx.triggered = []
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            with pytest.raises(PreventUpdate):
                callback_func(0, 0, 0, 0, 0, 50)

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_boundary_min(self, mock_ctx, mock_app):
        """Test update_time_position_nudge respects minimum boundary."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-m10.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(1, 0, 0, 0, 0, 5)
            assert result == 0  # Should not go below 0

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_boundary_max(self, mock_ctx, mock_app):
        """Test update_time_position_nudge respects maximum boundary."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p10.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(0, 0, 0, 0, 1, 95)
            assert result == 100  # Should not go above 100

    @patch('dash.callback_context')
    def test_update_time_position_nudge_direct_none_value(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with None start_position."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p5.n_clicks"}]
        
        register_transform_callbacks(mock_app)
        callback_func = get_callback_function(mock_app, "transforms-start-position")
        
        if callback_func:
            result = callback_func(0, 0, 0, 1, 0, None)
            assert result == 5  # None defaults to 0, then +5

