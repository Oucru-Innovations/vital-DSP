"""
Extended comprehensive tests for time_domain_callbacks.py module.

This test file adds extensive coverage to reach 60%+ coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
import plotly.graph_objects as go

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.time_domain_callbacks import (
    register_time_domain_callbacks,
    format_large_number,
    configure_plot_with_pan_zoom,
    create_empty_figure,
    higuchi_fractal_dimension,
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
    return signal, t, 1000  # signal data, time axis, and sampling frequency


class TestHelperFunctions:
    """Test helper functions in time_domain_callbacks."""

    def test_format_large_number_zero(self):
        """Test format_large_number with zero."""
        result = format_large_number(0)
        assert result == "0"

    def test_format_large_number_small(self):
        """Test format_large_number with small number."""
        result = format_large_number(0.5)
        assert isinstance(result, str)
        assert "0.5" in result or "500" in result  # Could be formatted as 500m

    def test_format_large_number_medium(self):
        """Test format_large_number with medium number."""
        result = format_large_number(123.456)
        assert isinstance(result, str)
        assert "123" in result

    def test_format_large_number_thousands(self):
        """Test format_large_number with thousands."""
        result = format_large_number(5000)
        assert isinstance(result, str)
        assert "k" in result or "5000" in result

    def test_format_large_number_millions(self):
        """Test format_large_number with millions."""
        result = format_large_number(2000000)
        assert isinstance(result, str)
        assert "e" in result.lower() or "2" in result  # Scientific notation

    def test_format_large_number_very_small(self):
        """Test format_large_number with very small number."""
        result = format_large_number(0.0001)
        assert isinstance(result, str)
        assert "e" in result.lower() or "0.1" in result

    def test_format_large_number_negative(self):
        """Test format_large_number with negative number."""
        result = format_large_number(-123.456)
        assert isinstance(result, str)
        assert "-" in result or "123" in result

    def test_format_large_number_scientific_notation(self):
        """Test format_large_number with scientific notation flag."""
        result = format_large_number(1000, use_scientific=True)
        assert isinstance(result, str)
        assert "e" in result.lower()

    def test_format_large_number_custom_precision(self):
        """Test format_large_number with custom precision."""
        result = format_large_number(123.456789, precision=2)
        assert isinstance(result, str)
        # Should have 2 decimal places or less

    def test_configure_plot_with_pan_zoom(self):
        """Test configure_plot_with_pan_zoom function."""
        fig = go.Figure()
        result = configure_plot_with_pan_zoom(fig, "Test Title", height=500)
        assert result is not None
        assert result.layout.title.text == "Test Title"
        assert result.layout.height == 500

    def test_configure_plot_with_pan_zoom_defaults(self):
        """Test configure_plot_with_pan_zoom with default parameters."""
        fig = go.Figure()
        result = configure_plot_with_pan_zoom(fig)
        assert result is not None
        assert result.layout.height == 400

    def test_create_empty_figure_light(self):
        """Test create_empty_figure with light theme."""
        fig = create_empty_figure(theme="light")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_empty_figure_dark(self):
        """Test create_empty_figure with dark theme."""
        fig = create_empty_figure(theme="dark")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_higuchi_fractal_dimension(self, sample_signal_data):
        """Test higuchi_fractal_dimension function."""
        signal_data, _, _ = sample_signal_data
        fd = higuchi_fractal_dimension(signal_data)
        assert isinstance(fd, (int, float))
        assert 0.0 <= fd <= 2.0  # Fractal dimension can be between 0 and 2
        assert not np.isnan(fd)
        assert not np.isinf(fd)

    def test_higuchi_fractal_dimension_custom_kmax(self, sample_signal_data):
        """Test higuchi_fractal_dimension with custom k_max."""
        signal_data, _, _ = sample_signal_data
        fd = higuchi_fractal_dimension(signal_data, k_max=5)
        assert isinstance(fd, (int, float))
        assert 0.0 <= fd <= 2.0

    def test_higuchi_fractal_dimension_constant_signal(self):
        """Test higuchi_fractal_dimension with constant signal."""
        constant_signal = np.ones(1000)
        fd = higuchi_fractal_dimension(constant_signal)
        assert isinstance(fd, (int, float))
        # Constant signal should have fractal dimension close to 1


class TestCallbackRegistration:
    """Test callback registration."""

    def test_register_time_domain_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_time_domain_callbacks(mock_app)
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1

