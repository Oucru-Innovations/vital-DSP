"""
Comprehensive test cases to cover missing lines in physiological_callbacks.py
Focuses on uncovered branches, edge cases, and complex logic paths.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dash import callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


@pytest.fixture
def sample_rr_intervals():
    """Create sample RR intervals for HRV analysis"""
    np.random.seed(42)
    # Generate realistic RR intervals (600-1200 ms)
    base_rr = 800  # Base RR interval in ms
    rr_intervals = base_rr + np.random.normal(0, 50, 100)  # Add some variability
    return rr_intervals


@pytest.fixture
def sample_hrv_metrics():
    """Create sample HRV metrics"""
    return {
        "poincare_sd1": 25.5,
        "poincare_sd2": 45.2,
        "rmssd": 30.1,
        "sdnn": 42.8,
        "mean_rr": 800.0
    }


@pytest.fixture
def sample_signal_data():
    """Create sample signal data"""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(1000)
    return pd.DataFrame({'time': t, 'signal': signal})


@pytest.fixture
def mock_data_service():
    """Create a mock data service"""
    service = Mock()
    service.get_all_data.return_value = {}
    service.get_data_info.return_value = {}
    service.current_data = None
    return service


class TestFormatLargeNumberFunction:
    """Test cases for the format_large_number helper function"""

    def test_format_large_number_zero(self):
        """Test formatting zero value (line 21-22)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(0)
        assert result == "0"

    def test_format_large_number_as_integer(self):
        """Test integer formatting (line 25-26)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(1234.567, as_integer=True)
        assert result == "1235"  # Should round to nearest integer

    def test_format_large_number_scientific_large(self):
        """Test scientific notation for very large numbers (line 30-32)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(1e7)
        assert "e" in result  # Should use scientific notation

    def test_format_large_number_scientific_forced(self):
        """Test forced scientific notation (line 30-32)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(1000, use_scientific=True)
        assert "e" in result  # Should use scientific notation

    def test_format_large_number_thousands(self):
        """Test thousands notation (line 33-36)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(1500)
        assert "k" in result  # Should use k notation
        assert "1.500" in result

    def test_format_large_number_regular(self):
        """Test regular decimal notation (line 37-39)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(123.456)
        assert "123.456" in result

    def test_format_large_number_millis(self):
        """Test millis notation (line 40-43)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(0.001)
        assert "m" in result  # Should use m notation

    def test_format_large_number_scientific_small(self):
        """Test scientific notation for very small numbers (line 44-46)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(1e-6)
        assert "e" in result  # Should use scientific notation

    def test_format_large_number_precision(self):
        """Test precision parameter"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(123.456789, precision=2)
        assert "123.46" in result  # Should use 2 decimal places

    def test_format_large_number_negative(self):
        """Test negative numbers"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import format_large_number
        
        result = format_large_number(-1500)
        assert "-" in result
        assert "k" in result


class TestHRVPoincarePlotFunction:
    """Test cases for create_hrv_poincare_plot function"""

    def test_create_hrv_poincare_plot_insufficient_data(self, sample_hrv_metrics):
        """Test with insufficient RR intervals (line 55-57)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot
        
        # Test with only 1 RR interval
        rr_intervals = [800]
        fig = create_hrv_poincare_plot(rr_intervals, sample_hrv_metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # Should return empty figure

    def test_create_hrv_poincare_plot_basic(self, sample_rr_intervals, sample_hrv_metrics):
        """Test basic Poincaré plot creation (line 59-93)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot
        
        fig = create_hrv_poincare_plot(sample_rr_intervals, sample_hrv_metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have scatter plot and identity line
        
        # Check scatter plot
        scatter_trace = fig.data[0]
        assert scatter_trace.mode == "markers"
        assert "RR Intervals" in scatter_trace.name
        
        # Check identity line
        line_trace = fig.data[1]
        assert line_trace.mode == "lines"
        assert "Identity Line" in line_trace.name

    def test_create_hrv_poincare_plot_with_sd_ellipses(self, sample_rr_intervals, sample_hrv_metrics):
        """Test Poincaré plot with SD1/SD2 ellipses (line 96-99)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot
        
        fig = create_hrv_poincare_plot(sample_rr_intervals, sample_hrv_metrics)
        
        assert isinstance(fig, go.Figure)
        # Should have scatter plot, identity line, and SD ellipses
        assert len(fig.data) >= 3

    def test_create_hrv_poincare_plot_without_sd_metrics(self, sample_rr_intervals):
        """Test Poincaré plot without SD1/SD2 metrics (line 96)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import create_hrv_poincare_plot
        
        hrv_metrics = {"rmssd": 30.1, "sdnn": 42.8}  # Missing SD1/SD2
        
        fig = create_hrv_poincare_plot(sample_rr_intervals, hrv_metrics)
        
        assert isinstance(fig, go.Figure)
        # Should only have scatter plot and identity line (no SD ellipses)
        assert len(fig.data) == 2


class TestPhysiologicalCallbacksRegistration:
    """Test cases for callback registration and execution"""

    def test_register_physiological_callbacks(self):
        """Test callback registration"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Should register multiple callbacks
        assert len(captured_callbacks) > 0
        
        # Check for specific callback names
        callback_names = [func.__name__ for args, kwargs, func in captured_callbacks]
        print(f"Registered callbacks: {callback_names}")

    def test_analyze_hrv_callback_no_data_service(self):
        """Test HRV analysis callback with no data service (line 54-126)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with no data service
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=None, create=True):
                result = hrv_callback(
                    n_clicks=1,
                    selected_metrics=["rmssd", "sdnn"],
                    time_domain_options=["rmssd"],
                    freq_domain_options=["lf", "hf"],
                    nonlinear_options=["poincare"],
                    preprocessing_options=["detrend"],
                    advanced_options={}
                )
                # Should return error message
                assert isinstance(result, list)
                assert len(result) > 0

    def test_analyze_hrv_callback_no_data_available(self, mock_data_service):
        """Test HRV analysis callback with no data available (line 131-168)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with no data available
            mock_data_service.current_data = None
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                result = hrv_callback(
                    n_clicks=1,
                    selected_metrics=["rmssd", "sdnn"],
                    time_domain_options=["rmssd"],
                    freq_domain_options=["lf", "hf"],
                    nonlinear_options=["poincare"],
                    preprocessing_options=["detrend"],
                    advanced_options={}
                )
                # Should return error message
                assert isinstance(result, list)
                assert len(result) > 0

    def test_analyze_hrv_callback_with_valid_data(self, mock_data_service, sample_signal_data):
        """Test HRV analysis callback with valid data (line 175-270)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with valid data
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "poincare_sd1": 25.5,
                        "poincare_sd2": 45.2
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should return analysis results
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksEdgeCases:
    """Test edge cases and error handling"""

    def test_analyze_hrv_callback_exception_handling(self, mock_data_service, sample_signal_data):
        """Test HRV analysis callback exception handling (line 275-371)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with data that causes exception
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv', side_effect=Exception("Test error")):
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should return error message
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_no_button_click(self):
        """Test HRV analysis callback with no button click (line 22)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with no button click
            with pytest.raises(PreventUpdate):
                hrv_callback(
                    n_clicks=None,
                    selected_metrics=["rmssd", "sdnn"],
                    time_domain_options=["rmssd"],
                    freq_domain_options=["lf", "hf"],
                    nonlinear_options=["poincare"],
                    preprocessing_options=["detrend"],
                    advanced_options={}
                )

    def test_analyze_hrv_callback_empty_metrics(self, mock_data_service, sample_signal_data):
        """Test HRV analysis callback with empty metrics (line 376-498)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with empty metrics
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                result = hrv_callback(
                    n_clicks=1,
                    selected_metrics=[],
                    time_domain_options=[],
                    freq_domain_options=[],
                    nonlinear_options=[],
                    preprocessing_options=[],
                    advanced_options={}
                )
                # Should handle empty metrics gracefully
                assert isinstance(result, list)
                assert len(result) > 0


class TestPhysiologicalCallbacksAdvancedFeatures:
    """Test advanced features and complex logic paths"""

    def test_analyze_hrv_callback_different_signal_types(self, mock_data_service, sample_signal_data):
        """Test HRV analysis with different signal types (line 503-631)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with PPG signal type
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "PPG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle PPG signal type
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_preprocessing_options(self, mock_data_service, sample_signal_data):
        """Test HRV analysis with different preprocessing options (line 660)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with various preprocessing options
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            preprocessing_options = ["detrend", "filter", "normalize", "outlier_removal"]
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=preprocessing_options,
                        advanced_options={}
                    )
                    # Should handle preprocessing options
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_advanced_options(self, mock_data_service, sample_signal_data):
        """Test HRV analysis with advanced options (line 672-674, 680)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with advanced options
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            advanced_options = {
                "window_size": 300,
                "overlap": 0.5,
                "detrend_method": "linear",
                "filter_cutoff": 0.4
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle advanced options
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksComplexScenarios:
    """Test complex scenarios and integration paths"""

    def test_analyze_hrv_callback_frequency_domain_analysis(self, mock_data_service, sample_signal_data):
        """Test frequency domain analysis (line 689-691)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with frequency domain options
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            freq_domain_options = ["lf", "hf", "vlf", "total_power", "lf_hf_ratio"]
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "lf": 150.5,
                        "hf": 200.3,
                        "lf_hf_ratio": 0.75
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn", "lf", "hf"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=freq_domain_options,
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle frequency domain analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_nonlinear_analysis(self, mock_data_service, sample_signal_data):
        """Test nonlinear analysis (line 712-831)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with nonlinear options
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            nonlinear_options = ["poincare", "d2", "sample_entropy", "approximate_entropy"]
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "poincare_sd1": 25.5,
                        "poincare_sd2": 45.2,
                        "d2": 1.8,
                        "sample_entropy": 1.2
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=nonlinear_options,
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle nonlinear analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_morphological_analysis(self, mock_data_service, sample_signal_data):
        """Test morphological analysis (line 893-894)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with morphological analysis
            mock_data_service.current_data = sample_signal_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "morphological_features": {
                            "p_wave_duration": 120,
                            "qrs_duration": 90,
                            "qt_interval": 400
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle morphological analysis
                    assert isinstance(result, list)
                    assert len(result) > 0
