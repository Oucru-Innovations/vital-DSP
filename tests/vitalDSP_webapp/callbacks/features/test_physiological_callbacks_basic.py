"""Comprehensive tests for physiological_callbacks.py module."""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock

# Test data
SAMPLE_DATA = np.sin(2 * np.pi * np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
SAMPLE_TIME = np.linspace(0, 10, 1000)
SAMPLE_FREQ = 100

try:
    from vitalDSP_webapp.callbacks.features.physiological_callbacks import (
        normalize_signal_type, create_empty_figure, detect_physiological_signal_type,
        create_physiological_signal_plot, perform_physiological_analysis,
        analyze_hrv, analyze_morphology, analyze_signal_quality, analyze_trends,
        create_comprehensive_results_display, register_physiological_callbacks,
        analyze_beat_to_beat, analyze_energy, analyze_envelope, analyze_segmentation,
        analyze_waveform, analyze_statistical, analyze_frequency
    )
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestSignalTypeNormalization:
    def test_normalize_valid_types(self):
        assert normalize_signal_type("ecg") == "ECG"
        assert normalize_signal_type("ECG") == "ECG"
        assert normalize_signal_type("ppg") == "PPG"
        assert normalize_signal_type("PPG") == "PPG"
        assert normalize_signal_type("eeg") == "EEG"
        assert normalize_signal_type("EEG") == "EEG"
    
    def test_normalize_invalid_types(self):
        assert normalize_signal_type("invalid") == "PPG"
        assert normalize_signal_type("emg") == "PPG"
        assert normalize_signal_type("xyz") == "PPG"
        assert normalize_signal_type(None) == "PPG"
        assert normalize_signal_type("") == "PPG"
    
    def test_normalize_mixed_case(self):
        assert normalize_signal_type("Ecg") == "ECG"
        assert normalize_signal_type("pPg") == "PPG"
        assert normalize_signal_type("EeG") == "EEG"

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestBasicFunctions:
    def test_create_empty_figure(self):
        fig = create_empty_figure()
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_detect_signal_type(self):
        result = detect_physiological_signal_type(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(result, str)
        # The function returns lowercase strings
        assert result in ["ecg", "ppg", "eeg"]
    
    def test_detect_signal_type_empty(self):
        result = detect_physiological_signal_type(np.array([]), SAMPLE_FREQ)
        assert isinstance(result, str)
    
    def test_create_signal_plot(self):
        fig = create_physiological_signal_plot(SAMPLE_TIME, SAMPLE_DATA, "PPG", SAMPLE_FREQ)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestAnalysisFunctions:
    def test_perform_physiological_analysis_basic(self):
        results = perform_physiological_analysis(
            SAMPLE_TIME, SAMPLE_DATA, "PPG", SAMPLE_FREQ,
            ["hrv"], [], [], [], [], [], [], [], []
        )
        # The function returns a Dash HTML component, not a dict
        from dash import html
        assert isinstance(results, html.Div)
    
    def test_perform_physiological_analysis_multiple(self):
        categories = ["hrv", "morphology", "beat2beat", "energy"]
        results = perform_physiological_analysis(
            SAMPLE_TIME, SAMPLE_DATA, "PPG", SAMPLE_FREQ,
            categories, [], [], [], [], [], [], [], []
        )
        # The function returns a Dash HTML component, not a dict
        from dash import html
        assert isinstance(results, html.Div)
    
    def test_analyze_hrv(self):
        results = analyze_hrv(SAMPLE_DATA, SAMPLE_FREQ, [])
        assert isinstance(results, dict)
    
    def test_analyze_morphology(self):
        results = analyze_morphology(SAMPLE_DATA, SAMPLE_FREQ, [])
        assert isinstance(results, dict)
    
    def test_analyze_signal_quality(self):
        results = analyze_signal_quality(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_trends(self):
        results = analyze_trends(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_beat_to_beat(self):
        results = analyze_beat_to_beat(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_energy(self):
        results = analyze_energy(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_envelope(self):
        results = analyze_envelope(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_segmentation(self):
        results = analyze_segmentation(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_waveform(self):
        results = analyze_waveform(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_statistical(self):
        results = analyze_statistical(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)
    
    def test_analyze_frequency(self):
        results = analyze_frequency(SAMPLE_DATA, SAMPLE_FREQ)
        assert isinstance(results, dict)

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestResultsDisplay:
    def test_create_comprehensive_results_display(self):
        results = {"hrv_metrics": {"rmssd": 50, "sdnn": 45}}
        display = create_comprehensive_results_display(results, "PPG", SAMPLE_FREQ)
        assert display is not None
    
    def test_create_results_display_empty(self):
        results = {}
        display = create_comprehensive_results_display(results, "PPG", SAMPLE_FREQ)
        assert display is not None

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestCallbackRegistration:
    def test_register_physiological_callbacks(self):
        mock_app = Mock()
        mock_app.callback = Mock(return_value=lambda f: f)
        
        register_physiological_callbacks(mock_app)
        assert mock_app.callback.called

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestErrorHandling:
    def test_analysis_with_nan_values(self):
        signal_with_nan = SAMPLE_DATA.copy()
        signal_with_nan[100:110] = np.nan
        
        results = analyze_hrv(signal_with_nan, SAMPLE_FREQ, [])
        assert isinstance(results, dict)
    
    def test_analysis_with_zero_sampling_rate(self):
        results = analyze_hrv(SAMPLE_DATA, 0, [])
        assert isinstance(results, dict)
    
    def test_analysis_with_empty_data(self):
        results = analyze_hrv(np.array([]), SAMPLE_FREQ, [])
        assert isinstance(results, dict)
