"""
Comprehensive unit tests for quality_callbacks.py module.

This test file adds extensive coverage for signal quality assessment callbacks.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dash import Dash, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

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
    # Create ECG-like signal
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000  # signal data and sampling frequency


@pytest.fixture
def sample_dataframe(sample_signal_data):
    """Create sample dataframe for testing."""
    signal_data, sampling_freq = sample_signal_data
    df = pd.DataFrame({
        'time': np.linspace(0, 10, len(signal_data)),
        'signal': signal_data
    })
    return df


@pytest.fixture
def sample_column_mapping():
    """Create sample column mapping for testing."""
    return {
        'time': 'time',
        'signal': 'signal'
    }


@pytest.fixture
def sample_data_info():
    """Create sample data info for testing."""
    return {
        'sampling_frequency': 1000,
        'duration': 10.0,
        'signal_type': 'ECG',
        'num_samples': 10000
    }


class TestQualityCallbacksRegistration:
    """Test the callback registration functionality."""

    def test_register_quality_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_quality_callbacks(mock_app)

        # Verify that callback decorator was called
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1


class TestQualityAssessmentCallback:
    """Test the main quality assessment callback."""

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    @patch('vitalDSP_webapp.callbacks.analysis.quality_callbacks.callback_context')
    def test_callback_not_on_quality_page(self, mock_ctx, mock_get_data_service, mock_app):
        """Test callback behavior when not on quality page."""
        # Setup mocks
        mock_ctx.triggered = [{"prop_id": "quality-analyze-btn.n_clicks", "value": 1}]

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call with wrong pathname
        result = quality_callback(
            n_clicks=1,
            pathname="/other-page",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr", "artifacts"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=None
        )

        # Should return empty figures with message
        assert isinstance(result, tuple)
        assert len(result) == 8
        assert isinstance(result[0], go.Figure)  # Empty figure
        assert result[2] == "Navigate to Quality page"

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_data_service_none(self, mock_get_data_service, mock_app):
        """Test callback behavior when data service is None."""
        # Setup mocks
        mock_get_data_service.return_value = None

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call with None data service
        result = quality_callback(
            n_clicks=1,
            pathname="/quality",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr", "artifacts"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=None
        )

        # Should return error message
        assert isinstance(result, tuple)
        assert len(result) == 8
        assert "Data service not available" in result[2]

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_no_data_available(self, mock_get_data_service, mock_app):
        """Test callback behavior when no data is available."""
        # Setup mocks
        mock_data_service = Mock()
        mock_data_service.get_all_data.return_value = {}  # No data
        mock_get_data_service.return_value = mock_data_service

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call with no data
        result = quality_callback(
            n_clicks=1,
            pathname="/quality",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr", "artifacts"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=None
        )

        # Should return message about no data
        assert isinstance(result, tuple)
        assert len(result) == 8
        assert "No data available" in result[2]

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_no_button_click(self, mock_get_data_service, mock_app, sample_dataframe):
        """Test callback behavior when button not clicked."""
        # Setup mocks
        mock_data_service = Mock()
        mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
        mock_get_data_service.return_value = mock_data_service

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call with no button click (n_clicks=None)
        result = quality_callback(
            n_clicks=None,
            pathname="/quality",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr", "artifacts"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=None
        )

        # Should return instructions message
        assert isinstance(result, tuple)
        assert len(result) == 8
        assert "Click" in result[2] and "Assess Signal Quality" in result[2]

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_no_column_mapping(self, mock_get_data_service, mock_app, sample_dataframe):
        """Test callback behavior when no column mapping exists."""
        # Setup mocks
        mock_data_service = Mock()
        mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
        mock_data_service.get_column_mapping.return_value = None  # No mapping
        mock_get_data_service.return_value = mock_data_service

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call with no column mapping
        result = quality_callback(
            n_clicks=1,
            pathname="/quality",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr", "artifacts"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=None
        )

        # Should return message about processing needed
        assert isinstance(result, tuple)
        assert len(result) == 8

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_with_valid_data(self, mock_get_data_service, mock_app, sample_dataframe, sample_column_mapping, sample_data_info):
        """Test callback behavior with valid data."""
        # Setup mocks
        mock_data_service = Mock()
        mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
        mock_data_service.get_column_mapping.return_value = sample_column_mapping
        mock_data_service.get_data_info.return_value = sample_data_info
        mock_get_data_service.return_value = mock_data_service

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call with valid data
        result = quality_callback(
            n_clicks=1,
            pathname="/quality",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr", "artifacts", "baseline"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=["detailed_analysis"]
        )

        # Should return quality assessment results
        assert isinstance(result, tuple)
        assert len(result) == 8
        # First two should be figures
        assert isinstance(result[0], go.Figure)
        assert isinstance(result[1], go.Figure)

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_with_different_quality_metrics(self, mock_get_data_service, mock_app, sample_dataframe, sample_column_mapping, sample_data_info):
        """Test callback with different quality metrics combinations."""
        metrics_combinations = [
            ["snr"],
            ["artifacts"],
            ["baseline"],
            ["snr", "artifacts"],
            ["snr", "baseline"],
            ["artifacts", "baseline"],
            ["snr", "artifacts", "baseline"],
        ]

        for metrics in metrics_combinations:
            # Setup mocks
            mock_data_service = Mock()
            mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
            mock_data_service.get_column_mapping.return_value = sample_column_mapping
            mock_data_service.get_data_info.return_value = sample_data_info
            mock_get_data_service.return_value = mock_data_service

            # Register callbacks and capture them
            captured_callbacks = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_callbacks.append((args, kwargs, func))
                    return func
                return decorator

            mock_app.callback = mock_callback
            register_quality_callbacks(mock_app)

            # Find the quality assessment callback
            quality_callback = None
            for args, kwargs, func in captured_callbacks:
                if func.__name__ == 'quality_assessment_callback':
                    quality_callback = func
                    break

            assert quality_callback is not None

            # Call with different metrics
            result = quality_callback(
                n_clicks=1,
                pathname="/quality",
                slider_value=[0, 10],
                nudge_m10=None,
                nudge_m1=None,
                nudge_p1=None,
                nudge_p10=None,
                start_time=0,
                end_time=10,
                signal_type="ecg",
                quality_metrics=metrics,
                snr_threshold=10,
                artifact_threshold=0.5,
                advanced_options=None
            )

            # Should return valid results
            assert isinstance(result, tuple)
            assert len(result) == 8

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_with_different_signal_types(self, mock_get_data_service, mock_app, sample_dataframe, sample_column_mapping, sample_data_info):
        """Test callback with different signal types."""
        signal_types = ["ecg", "ppg", "general"]

        for sig_type in signal_types:
            # Setup mocks
            mock_data_service = Mock()
            mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
            mock_data_service.get_column_mapping.return_value = sample_column_mapping
            mock_data_service.get_data_info.return_value = sample_data_info
            mock_get_data_service.return_value = mock_data_service

            # Register callbacks and capture them
            captured_callbacks = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_callbacks.append((args, kwargs, func))
                    return func
                return decorator

            mock_app.callback = mock_callback
            register_quality_callbacks(mock_app)

            # Find the quality assessment callback
            quality_callback = None
            for args, kwargs, func in captured_callbacks:
                if func.__name__ == 'quality_assessment_callback':
                    quality_callback = func
                    break

            assert quality_callback is not None

            # Call with different signal type
            result = quality_callback(
                n_clicks=1,
                pathname="/quality",
                slider_value=[0, 10],
                nudge_m10=None,
                nudge_m1=None,
                nudge_p1=None,
                nudge_p10=None,
                start_time=0,
                end_time=10,
                signal_type=sig_type,
                quality_metrics=["snr"],
                snr_threshold=10,
                artifact_threshold=0.5,
                advanced_options=None
            )

            # Should return valid results
            assert isinstance(result, tuple)
            assert len(result) == 8

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_with_different_thresholds(self, mock_get_data_service, mock_app, sample_dataframe, sample_column_mapping, sample_data_info):
        """Test callback with different threshold values."""
        threshold_combinations = [
            (5, 0.3),
            (10, 0.5),
            (15, 0.7),
            (20, 0.9),
        ]

        for snr_thresh, artifact_thresh in threshold_combinations:
            # Setup mocks
            mock_data_service = Mock()
            mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
            mock_data_service.get_column_mapping.return_value = sample_column_mapping
            mock_data_service.get_data_info.return_value = sample_data_info
            mock_get_data_service.return_value = mock_data_service

            # Register callbacks and capture them
            captured_callbacks = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_callbacks.append((args, kwargs, func))
                    return func
                return decorator

            mock_app.callback = mock_callback
            register_quality_callbacks(mock_app)

            # Find the quality assessment callback
            quality_callback = None
            for args, kwargs, func in captured_callbacks:
                if func.__name__ == 'quality_assessment_callback':
                    quality_callback = func
                    break

            assert quality_callback is not None

            # Call with different thresholds
            result = quality_callback(
                n_clicks=1,
                pathname="/quality",
                slider_value=[0, 10],
                nudge_m10=None,
                nudge_m1=None,
                nudge_p1=None,
                nudge_p10=None,
                start_time=0,
                end_time=10,
                signal_type="ecg",
                quality_metrics=["snr", "artifacts"],
                snr_threshold=snr_thresh,
                artifact_threshold=artifact_thresh,
                advanced_options=None
            )

            # Should return valid results
            assert isinstance(result, tuple)
            assert len(result) == 8

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_with_advanced_options(self, mock_get_data_service, mock_app, sample_dataframe, sample_column_mapping, sample_data_info):
        """Test callback with advanced options."""
        advanced_options_combinations = [
            ["detailed_analysis"],
            ["frequency_analysis"],
            ["temporal_analysis"],
            ["detailed_analysis", "frequency_analysis"],
        ]

        for adv_options in advanced_options_combinations:
            # Setup mocks
            mock_data_service = Mock()
            mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
            mock_data_service.get_column_mapping.return_value = sample_column_mapping
            mock_data_service.get_data_info.return_value = sample_data_info
            mock_get_data_service.return_value = mock_data_service

            # Register callbacks and capture them
            captured_callbacks = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_callbacks.append((args, kwargs, func))
                    return func
                return decorator

            mock_app.callback = mock_callback
            register_quality_callbacks(mock_app)

            # Find the quality assessment callback
            quality_callback = None
            for args, kwargs, func in captured_callbacks:
                if func.__name__ == 'quality_assessment_callback':
                    quality_callback = func
                    break

            assert quality_callback is not None

            # Call with advanced options
            result = quality_callback(
                n_clicks=1,
                pathname="/quality",
                slider_value=[0, 10],
                nudge_m10=None,
                nudge_m1=None,
                nudge_p1=None,
                nudge_p10=None,
                start_time=0,
                end_time=10,
                signal_type="ecg",
                quality_metrics=["snr"],
                snr_threshold=10,
                artifact_threshold=0.5,
                advanced_options=adv_options
            )

            # Should return valid results
            assert isinstance(result, tuple)
            assert len(result) == 8

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_with_time_window_adjustment(self, mock_get_data_service, mock_app, sample_dataframe, sample_column_mapping, sample_data_info):
        """Test callback with different time windows."""
        time_windows = [
            (0, 5),
            (2, 8),
            (5, 10),
        ]

        for start, end in time_windows:
            # Setup mocks
            mock_data_service = Mock()
            mock_data_service.get_all_data.return_value = {"data_1": sample_dataframe}
            mock_data_service.get_column_mapping.return_value = sample_column_mapping
            mock_data_service.get_data_info.return_value = sample_data_info
            mock_get_data_service.return_value = mock_data_service

            # Register callbacks and capture them
            captured_callbacks = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_callbacks.append((args, kwargs, func))
                    return func
                return decorator

            mock_app.callback = mock_callback
            register_quality_callbacks(mock_app)

            # Find the quality assessment callback
            quality_callback = None
            for args, kwargs, func in captured_callbacks:
                if func.__name__ == 'quality_assessment_callback':
                    quality_callback = func
                    break

            assert quality_callback is not None

            # Call with different time window
            result = quality_callback(
                n_clicks=1,
                pathname="/quality",
                slider_value=[start, end],
                nudge_m10=None,
                nudge_m1=None,
                nudge_p1=None,
                nudge_p10=None,
                start_time=start,
                end_time=end,
                signal_type="ecg",
                quality_metrics=["snr"],
                snr_threshold=10,
                artifact_threshold=0.5,
                advanced_options=None
            )

            # Should return valid results
            assert isinstance(result, tuple)
            assert len(result) == 8

    @patch('vitalDSP_webapp.services.data.data_service.get_data_service', create=True)
    def test_callback_exception_handling(self, mock_get_data_service, mock_app):
        """Test callback exception handling."""
        # Setup mocks to raise an exception
        mock_data_service = Mock()
        mock_data_service.get_all_data.side_effect = Exception("Test error")
        mock_get_data_service.return_value = mock_data_service

        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator

        mock_app.callback = mock_callback
        register_quality_callbacks(mock_app)

        # Find the quality assessment callback
        quality_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'quality_assessment_callback':
                quality_callback = func
                break

        assert quality_callback is not None

        # Call should handle exception gracefully
        result = quality_callback(
            n_clicks=1,
            pathname="/quality",
            slider_value=[0, 10],
            nudge_m10=None,
            nudge_m1=None,
            nudge_p1=None,
            nudge_p10=None,
            start_time=0,
            end_time=10,
            signal_type="ecg",
            quality_metrics=["snr"],
            snr_threshold=10,
            artifact_threshold=0.5,
            advanced_options=None
        )

        # Should return error results
        assert isinstance(result, tuple)
        assert len(result) == 8


if __name__ == "__main__":
    pytest.main([__file__])
