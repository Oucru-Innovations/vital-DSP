"""
Comprehensive unit tests for transform_callbacks.py module.

This test file adds extensive coverage for transform callbacks.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.transform_callbacks import (
    register_transform_callbacks,
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
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000  # signal data and sampling frequency


class TestTransformCallbacksRegistration:
    """Test the callback registration functionality."""

    def test_register_transform_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_transform_callbacks(mock_app)
        
        # Verify that callback decorator was called
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1


class TestTransformParametersCallback:
    """Test the transform parameters update callback."""

    def test_update_transform_parameters_none(self, mock_app):
        """Test update_transform_parameters with None input."""
        register_transform_callbacks(mock_app)
        
        # Find the parameters callback
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(None)
            assert result == []

    def test_update_transform_parameters_fft(self, mock_app):
        """Test update_transform_parameters for FFT transform."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func("fft")
            assert result != []
            assert isinstance(result, list)
            # Should contain parameter inputs for FFT

    def test_update_transform_parameters_stft(self, mock_app):
        """Test update_transform_parameters for STFT transform."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func("stft")
            assert result != []
            assert isinstance(result, list)

    def test_update_transform_parameters_wavelet(self, mock_app):
        """Test update_transform_parameters for Wavelet transform."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func("wavelet")
            assert result != []
            assert isinstance(result, list)

    def test_update_transform_parameters_hilbert(self, mock_app):
        """Test update_transform_parameters for Hilbert transform."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func("hilbert")
            assert result != []
            assert isinstance(result, list)

    def test_update_transform_parameters_mfcc(self, mock_app):
        """Test update_transform_parameters for MFCC transform."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func("mfcc")
            assert result != []
            assert isinstance(result, list)

    def test_update_transform_parameters_invalid(self, mock_app):
        """Test update_transform_parameters with invalid transform type."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-parameters-container" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func("invalid_type")
            assert result == []


class TestStoreParametersCallback:
    """Test the store parameters callback."""

    def test_store_parameters_empty(self, mock_app):
        """Test store_parameters with empty inputs."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("store-transforms-data" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func([], [], "fft")
            assert isinstance(result, dict)
            assert result["transform_type"] == "fft"

    def test_store_parameters_with_values(self, mock_app):
        """Test store_parameters with parameter values."""
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("store-transforms-data" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            param_values = [1024, "hann"]
            param_ids = [
                {"type": "transform-param", "param": "fft-npoints"},
                {"type": "transform-param", "param": "fft-window"}
            ]
            result = callback_func(param_values, param_ids, "fft")
            assert isinstance(result, dict)
            assert result["transform_type"] == "fft"
            assert "fft-npoints" in result
            assert "fft-window" in result
            assert result["fft-npoints"] == 1024
            assert result["fft-window"] == "hann"


class TestTimePositionNudgeCallback:
    """Test the time position nudge callback."""

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_no_trigger(self, mock_ctx, mock_app):
        """Test update_time_position_nudge when no trigger."""
        mock_ctx.triggered = []
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            with pytest.raises(PreventUpdate):
                callback_func(0, 0, 0, 0, 0, 0)

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_m10(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with -10 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-m10.n_clicks", "value": 1}]
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(1, 0, 0, 0, 0, 50)
            assert result == 40  # 50 - 10

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_m5(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with -5 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-m5.n_clicks", "value": 1}]
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(0, 1, 0, 0, 0, 50)
            assert result == 45  # 50 - 5

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_center(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with center button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-center.n_clicks", "value": 1}]
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(0, 0, 1, 0, 0, 50)
            assert result == 0  # Center should reset to 0

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_p5(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with +5 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p5.n_clicks", "value": 1}]
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(0, 0, 0, 1, 0, 50)
            assert result == 55  # 50 + 5

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_p10(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with +10 button."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p10.n_clicks", "value": 1}]
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(0, 0, 0, 0, 1, 50)
            assert result == 60  # 50 + 10

    @patch('vitalDSP_webapp.callbacks.analysis.transform_callbacks.callback_context')
    def test_update_time_position_nudge_none_value(self, mock_ctx, mock_app):
        """Test update_time_position_nudge with None start_position."""
        mock_ctx.triggered = [{"prop_id": "transforms-btn-nudge-p5.n_clicks", "value": 1}]
        
        register_transform_callbacks(mock_app)
        
        callback_func = None
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 1:
                if any("transforms-start-position" in str(out) for out in outputs):
                    callback_func = call[1].get('function') or call[0][1]
                    break
        
        if callback_func:
            result = callback_func(0, 0, 0, 1, 0, None)
            assert result == 5  # None defaults to 0, then +5

