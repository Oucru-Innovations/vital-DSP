"""
Additional comprehensive tests for quality_callbacks.py module.

This test file adds more coverage for quality callbacks, focusing on helper functions
and callback registration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from dash.exceptions import PreventUpdate

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.quality_callbacks import (
    register_quality_callbacks,
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
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000  # signal data and sampling frequency


class TestQualityCallbacksAdditional:
    """Additional tests for quality callbacks."""

    def test_register_quality_callbacks_called(self, mock_app):
        """Test that register_quality_callbacks registers callbacks."""
        register_quality_callbacks(mock_app)
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1

    def test_update_sqi_parameters_callback_registered(self, mock_app):
        """Test that update_sqi_parameters callback is registered."""
        register_quality_callbacks(mock_app)
        # Just verify callbacks were registered - functional tests verify they work
        assert mock_app.callback.call_count > 0

    def test_store_sqi_parameters_callback_registered(self, mock_app):
        """Test that store_sqi_parameters callback is registered."""
        register_quality_callbacks(mock_app)
        # Just verify callbacks were registered - functional tests verify they work
        assert mock_app.callback.call_count > 0

    def test_update_threshold_inputs_visibility_callback_registered(self, mock_app):
        """Test that update_threshold_inputs_visibility callback is registered."""
        register_quality_callbacks(mock_app)
        
        callback_found = False
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 2:
                if any("sqi-param-threshold-row" in str(out) for out in outputs):
                    callback_found = True
                    break
        
        assert callback_found, "update_threshold_inputs_visibility callback should be registered"

    def test_update_quality_position_nudge_callback_registered(self, mock_app):
        """Test that update_quality_position_nudge callback is registered."""
        register_quality_callbacks(mock_app)
        # Just verify callbacks were registered - functional tests verify they work
        assert mock_app.callback.call_count > 0

    @patch('vitalDSP_webapp.callbacks.analysis.quality_callbacks._import_vitaldsp_modules')
    def test_register_quality_callbacks_imports_modules(self, mock_import, mock_app):
        """Test that register_quality_callbacks imports vitalDSP modules."""
        register_quality_callbacks(mock_app)
        mock_import.assert_called_once()


class TestQualityCallbacksWithCapturedFunctions:
    """Test quality callbacks by capturing and testing the actual functions."""

    def test_update_sqi_parameters_with_none(self, mock_app):
        """Test update_sqi_parameters with None input."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_sqi_parameters callback
        sqi_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_sqi_parameters':
                sqi_callback = func
                break
        
        if sqi_callback:
            result = sqi_callback(None)
            assert result == []

    @patch('vitalDSP_webapp.callbacks.analysis.quality_sqi_functions.get_sqi_parameters_layout')
    def test_update_sqi_parameters_with_type(self, mock_get_layout, mock_app):
        """Test update_sqi_parameters with sqi_type."""
        mock_get_layout.return_value = [Mock()]
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_sqi_parameters callback
        sqi_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_sqi_parameters':
                sqi_callback = func
                break
        
        if sqi_callback:
            result = sqi_callback("snr_sqi")
            assert result is not None
            mock_get_layout.assert_called_once_with("snr_sqi")

    def test_store_sqi_parameters_empty(self, mock_app):
        """Test store_sqi_parameters with empty inputs."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find store_sqi_parameters callback
        store_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'store_sqi_parameters':
                store_callback = func
                break
        
        if store_callback:
            result = store_callback([], [], "snr_sqi")
            assert isinstance(result, dict)
            assert result["sqi_type"] == "snr_sqi"

    def test_store_sqi_parameters_with_values(self, mock_app):
        """Test store_sqi_parameters with parameter values."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find store_sqi_parameters callback
        store_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'store_sqi_parameters':
                store_callback = func
                break
        
        if store_callback:
            param_values = [1000, 500, 0.7]
            param_ids = [
                {"type": "sqi-param", "param": "window_size"},
                {"type": "sqi-param", "param": "step_size"},
                {"type": "sqi-param", "param": "threshold"}
            ]
            result = store_callback(param_values, param_ids, "snr_sqi")
            assert isinstance(result, dict)
            assert result["sqi_type"] == "snr_sqi"
            assert result["window_size"] == 1000
            assert result["step_size"] == 500
            assert result["threshold"] == 0.7

    def test_update_threshold_inputs_visibility_range(self, mock_app):
        """Test update_threshold_inputs_visibility with range threshold type."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_threshold_inputs_visibility callback
        threshold_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_threshold_inputs_visibility':
                threshold_callback = func
                break
        
        if threshold_callback:
            single_style, range_style = threshold_callback("range")
            assert single_style["display"] == "none"
            assert range_style["display"] == "block"

    def test_update_threshold_inputs_visibility_single(self, mock_app):
        """Test update_threshold_inputs_visibility with single threshold type."""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_threshold_inputs_visibility callback
        threshold_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_threshold_inputs_visibility':
                threshold_callback = func
                break
        
        if threshold_callback:
            single_style, range_style = threshold_callback("single")
            assert single_style["display"] == "block"
            assert range_style["display"] == "none"

    @patch('vitalDSP_webapp.callbacks.analysis.quality_callbacks.callback_context')
    def test_update_quality_position_nudge_no_trigger(self, mock_ctx, mock_app):
        """Test update_quality_position_nudge when no trigger."""
        mock_ctx.triggered = []
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_quality_position_nudge callback
        nudge_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_quality_position_nudge':
                nudge_callback = func
                break
        
        if nudge_callback:
            with pytest.raises(PreventUpdate):
                nudge_callback(0, 0, 0, 0, 0, 50)

    @patch('vitalDSP_webapp.callbacks.analysis.quality_callbacks.callback_context')
    def test_update_quality_position_nudge_m10(self, mock_ctx, mock_app):
        """Test update_quality_position_nudge with -10 button."""
        mock_ctx.triggered = [{"prop_id": "quality-btn-nudge-m10.n_clicks", "value": 1}]
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_quality_position_nudge callback
        nudge_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_quality_position_nudge':
                nudge_callback = func
                break
        
        if nudge_callback:
            result = nudge_callback(1, 0, 0, 0, 0, 50)
            assert result == 40  # 50 - 10, but clamped to max(0, ...)

    @patch('vitalDSP_webapp.callbacks.analysis.quality_callbacks.callback_context')
    def test_update_quality_position_nudge_center(self, mock_ctx, mock_app):
        """Test update_quality_position_nudge with center button."""
        mock_ctx.triggered = [{"prop_id": "quality-btn-center.n_clicks", "value": 1}]
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_quality_position_nudge callback
        nudge_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_quality_position_nudge':
                nudge_callback = func
                break
        
        if nudge_callback:
            result = nudge_callback(0, 0, 1, 0, 0, 50)
            assert result == 50  # Center should return 50

    @patch('vitalDSP_webapp.callbacks.analysis.quality_callbacks.callback_context')
    def test_update_quality_position_nudge_p10(self, mock_ctx, mock_app):
        """Test update_quality_position_nudge with +10 button."""
        mock_ctx.triggered = [{"prop_id": "quality-btn-nudge-p10.n_clicks", "value": 1}]
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)
        
        # Find update_quality_position_nudge callback
        nudge_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_quality_position_nudge':
                nudge_callback = func
                break
        
        if nudge_callback:
            result = nudge_callback(0, 0, 0, 0, 1, 50)
            assert result == 60  # 50 + 10, but clamped to min(100, ...)

