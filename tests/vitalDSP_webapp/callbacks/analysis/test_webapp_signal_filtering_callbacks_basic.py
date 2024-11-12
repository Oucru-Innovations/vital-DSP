"""Basic tests for signal_filtering_callbacks.py module."""

import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import Mock

# Test data
SAMPLE_DATA = np.sin(2 * np.pi * np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
SAMPLE_TIME = np.linspace(0, 10, 1000)
SAMPLE_FREQ = 100

try:
    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
        create_empty_figure, create_original_signal_plot, create_filtered_signal_plot,
        apply_traditional_filter, calculate_snr_improvement, calculate_mse,
        register_signal_filtering_callbacks
    )
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestBasicFunctions:
    def test_create_empty_figure(self):
        fig = create_empty_figure()
        assert isinstance(fig, go.Figure)
    
    def test_create_original_signal_plot(self):
        fig = create_original_signal_plot(SAMPLE_TIME, SAMPLE_DATA)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_filtered_signal_plot(self):
        fig = create_filtered_signal_plot(SAMPLE_TIME, SAMPLE_DATA)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_apply_traditional_filter(self):
        filtered = apply_traditional_filter(
            SAMPLE_DATA, SAMPLE_FREQ, "butterworth", "lowpass", None, 10.0, 4
        )
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(SAMPLE_DATA)
    
    def test_calculate_snr_improvement(self):
        filtered = SAMPLE_DATA * 0.9
        snr = calculate_snr_improvement(SAMPLE_DATA, filtered)
        assert isinstance(snr, (int, float))
    
    def test_calculate_mse(self):
        filtered = SAMPLE_DATA * 0.9
        mse = calculate_mse(SAMPLE_DATA, filtered)
        assert isinstance(mse, (int, float))
        assert mse >= 0
    
    def test_register_callbacks(self):
        mock_app = Mock()
        mock_app.callback = Mock(return_value=lambda f: f)
        register_signal_filtering_callbacks(mock_app)
        assert mock_app.callback.called
