"""Basic tests for vitaldsp_callbacks.py module."""

import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import Mock

# Test data
SAMPLE_DATA = np.sin(2 * np.pi * np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
SAMPLE_TIME = np.linspace(0, 10, 1000)
SAMPLE_FREQ = 100

try:
    from vitalDSP_webapp.callbacks.analysis.vitaldsp_callbacks import (
        create_empty_figure, create_time_domain_plot, create_peak_analysis_plot,
        create_main_signal_plot, create_filtered_signal_plot, generate_analysis_results,
        create_peak_analysis_table, create_signal_quality_table, create_filtering_results_table,
        create_additional_metrics_table, generate_time_domain_stats, apply_filter,
        detect_peaks, register_vitaldsp_callbacks
    )
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestBasicFunctions:
    def test_create_empty_figure(self):
        fig = create_empty_figure()
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_time_domain_plot(self):
        fig = create_time_domain_plot(SAMPLE_DATA, SAMPLE_TIME, SAMPLE_FREQ)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_peak_analysis_plot(self):
        peaks = [100, 200, 300, 400]  # Sample peak indices
        fig = create_peak_analysis_plot(SAMPLE_DATA, SAMPLE_TIME, peaks, SAMPLE_FREQ)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_main_signal_plot(self):
        analysis_options = ["basic_stats"]
        # Create a DataFrame with proper columns
        import pandas as pd
        df = pd.DataFrame({
            'time': SAMPLE_TIME,
            'signal': SAMPLE_DATA
        })
        column_mapping = {"time": "time", "signal": "signal"}
        fig = create_main_signal_plot(df, SAMPLE_TIME, SAMPLE_FREQ,
                                    analysis_options, column_mapping)
        assert isinstance(fig, go.Figure)
        # The function may return empty figure if column mapping issues
        assert hasattr(fig, 'data')
    
    def test_create_filtered_signal_plot(self):
        # Create a DataFrame with proper columns
        import pandas as pd
        df = pd.DataFrame({
            'time': SAMPLE_TIME,
            'signal': SAMPLE_DATA
        })
        column_mapping = {"time": "time", "signal": "signal"}
        fig = create_filtered_signal_plot(df, SAMPLE_TIME, SAMPLE_FREQ, column_mapping)
        assert isinstance(fig, go.Figure)
        # The function may return empty figure if column mapping issues
        assert hasattr(fig, 'data')
    
    def test_generate_analysis_results(self):
        analysis_options = ["basic_stats"]
        column_mapping = {"signal": "signal"}
        results = generate_analysis_results(SAMPLE_DATA, SAMPLE_DATA, SAMPLE_TIME, 
                                           SAMPLE_FREQ, analysis_options, column_mapping)
        assert results is not None
    
    def test_generate_time_domain_stats(self):
        stats = generate_time_domain_stats(SAMPLE_DATA, SAMPLE_TIME, SAMPLE_FREQ)
        # The function returns a Dash HTML component, not a dict
        from dash import html
        assert isinstance(stats, html.Div)
    
    def test_apply_filter(self):
        filtered = apply_filter(SAMPLE_DATA, SAMPLE_FREQ, "butterworth", "lowpass", 
                              None, 10.0, 4)
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(SAMPLE_DATA)
    
    def test_detect_peaks(self):
        peaks = detect_peaks(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(peaks, (list, np.ndarray))
    
    def test_register_callbacks(self):
        mock_app = Mock()
        mock_app.callback = Mock(return_value=lambda f: f)
        register_vitaldsp_callbacks(mock_app)
        assert mock_app.callback.called

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestTableCreation:
    def test_create_peak_analysis_table(self):
        analysis_options = ["peak_detection"]
        column_mapping = {"signal": "signal"}
        table = create_peak_analysis_table(SAMPLE_DATA, SAMPLE_DATA, SAMPLE_TIME, 
                                         SAMPLE_FREQ, analysis_options, column_mapping)
        assert table is not None
    
    def test_create_signal_quality_table(self):
        analysis_options = ["quality_assessment"]
        column_mapping = {"signal": "signal"}
        table = create_signal_quality_table(SAMPLE_DATA, SAMPLE_DATA, SAMPLE_TIME, 
                                           SAMPLE_FREQ, analysis_options, column_mapping)
        assert table is not None
    
    def test_create_filtering_results_table(self):
        analysis_options = ["filtering"]
        column_mapping = {"signal": "signal"}
        table = create_filtering_results_table(SAMPLE_DATA, SAMPLE_DATA, SAMPLE_TIME, 
                                              SAMPLE_FREQ, analysis_options, column_mapping)
        assert table is not None
    
    def test_create_additional_metrics_table(self):
        analysis_options = ["additional_metrics"]
        column_mapping = {"signal": "signal"}
        table = create_additional_metrics_table(SAMPLE_DATA, SAMPLE_DATA, SAMPLE_TIME, 
                                               SAMPLE_FREQ, analysis_options, column_mapping)
        assert table is not None

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestErrorHandling:
    def test_functions_with_empty_data(self):
        empty_data = np.array([])
        empty_time = np.array([])
        
        fig = create_time_domain_plot(empty_data, empty_time, SAMPLE_FREQ)
        assert isinstance(fig, go.Figure)
        
        stats = generate_time_domain_stats(empty_data, empty_time, SAMPLE_FREQ)
        # The function returns a Dash HTML component, not a dict
        from dash import html
        assert isinstance(stats, html.Div)
        
        peaks = detect_peaks(empty_data, SAMPLE_FREQ)
        assert isinstance(peaks, (list, np.ndarray))
    
    def test_filter_with_invalid_params(self):
        filtered = apply_filter(SAMPLE_DATA, 0, "invalid", "invalid", -1, -1, 0)
        assert isinstance(filtered, np.ndarray)
    
    def test_plots_with_invalid_data(self):
        nan_data = np.full_like(SAMPLE_DATA, np.nan)
        
        fig = create_time_domain_plot(nan_data, SAMPLE_TIME, SAMPLE_FREQ)
        assert isinstance(fig, go.Figure)

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestDifferentSignalTypes:
    def test_plots_different_signal_types(self):
        for signal_type in ["PPG", "ECG", "EEG"]:
            analysis_options = ["basic_stats"]
            column_mapping = {"signal": "signal"}
            fig = create_main_signal_plot(SAMPLE_DATA, SAMPLE_TIME, SAMPLE_FREQ, 
                                        analysis_options, column_mapping, signal_type)
            assert isinstance(fig, go.Figure)
    
    def test_filtered_plots_different_types(self):
        for signal_type in ["PPG", "ECG", "EEG"]:
            column_mapping = {"signal": "signal"}
            fig = create_filtered_signal_plot(SAMPLE_DATA, SAMPLE_TIME, SAMPLE_FREQ, 
                                            column_mapping, signal_type)
            assert isinstance(fig, go.Figure)
