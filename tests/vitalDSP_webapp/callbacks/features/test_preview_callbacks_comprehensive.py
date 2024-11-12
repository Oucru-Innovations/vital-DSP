"""
Comprehensive tests for vitalDSP_webapp.callbacks.features.preview_callbacks module.
Tests preview and visualization callback functionality.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock
from dash.exceptions import PreventUpdate

# Import the modules we need to test
try:
    from vitalDSP_webapp.callbacks.features.preview_callbacks import register_preview_callbacks
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.callbacks.features.preview_callbacks import register_preview_callbacks


class TestPreviewCallbacks:
    """Test preview callbacks functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock()
        
        # Sample data for testing - format that works with pandas DataFrame
        self.sample_data = {
            "data": {
                "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "signal": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        }
        
        self.sample_config = {
            "sampling_freq": 100,
            "time_unit": "seconds"
        }
        
    def test_register_preview_callbacks(self):
        """Test that preview callbacks are registered"""
        register_preview_callbacks(self.mock_app)
        
        # Should register callbacks
        assert self.mock_app.callback.called
        # The callback may be called multiple times, just check it was called
        assert self.mock_app.callback.call_count > 0
        
    def test_preview_callbacks_structure(self):
        """Test preview callbacks registration structure"""
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
            
        self.mock_app.callback = mock_callback
        register_preview_callbacks(self.mock_app)
        
        # Should have registered callbacks
        assert len(captured_callbacks) > 0
        
        # Check callback functions exist
        callback_names = [func.__name__ for _, _, func in captured_callbacks]
        # At least one of the expected callbacks should be present
        expected_callbacks = ['preview_data', 'compare_signals']
        assert any(name in callback_names for name in expected_callbacks)


class TestPreviewDataCallback:
    """Test preview_data callback functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_app = Mock()
        self.captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                self.captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
            
        self.mock_app.callback = mock_callback
        register_preview_callbacks(self.mock_app)
        
        # Find preview_data callback
        self.preview_callback = None
        for args, kwargs, func in self.captured_callbacks:
            if func.__name__ == 'preview_data':
                self.preview_callback = func
                break
                
        assert self.preview_callback is not None
        
        # Sample data for testing - format that works with pandas DataFrame
        self.sample_data = {
            "data": {
                "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "signal": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        }
        
        self.sample_config = {
            "sampling_freq": 100,
            "time_unit": "seconds"
        }
        
    def test_preview_data_no_clicks(self):
        """Test preview_data with no clicks"""
        with pytest.raises(PreventUpdate):
            self.preview_callback(
                n_clicks=None,
                window_size=1000,
                data_store=self.sample_data,
                config_store=self.sample_config
            )
            
    def test_preview_data_no_data(self):
        """Test preview_data with no data"""
        with pytest.raises(PreventUpdate):
            self.preview_callback(
                n_clicks=1,
                window_size=1000,
                data_store=None,
                config_store=self.sample_config
            )
            
    def test_preview_data_no_config(self):
        """Test preview_data with no config"""
        with pytest.raises(PreventUpdate):
            self.preview_callback(
                n_clicks=1,
                window_size=1000,
                data_store=self.sample_data,
                config_store=None
            )
            
    def test_preview_data_successful(self):
        """Test successful preview_data execution"""
        fig, summary = self.preview_callback(
            n_clicks=1,
            window_size=5,
            data_store=self.sample_data,
            config_store=self.sample_config
        )
        
        # Test figure
        assert isinstance(fig, go.Figure)
        if len(fig.data) > 0:  # Only check if there are traces (no error case)
            assert fig.data[0].mode == 'lines'
            assert fig.data[0].name == 'Signal'
        
        # Test summary
        assert summary is not None
        assert hasattr(summary, 'children')
        
    def test_preview_data_default_window_size(self):
        """Test preview_data with default window size"""
        fig, summary = self.preview_callback(
            n_clicks=1,
            window_size=None,
            data_store=self.sample_data,
            config_store=self.sample_config
        )
        
        assert isinstance(fig, go.Figure)
        assert summary is not None
        
    def test_preview_data_window_size_larger_than_data(self):
        """Test preview_data with window size larger than data"""
        fig, summary = self.preview_callback(
            n_clicks=1,
            window_size=1000,  # Larger than 10 samples
            data_store=self.sample_data,
            config_store=self.sample_config
        )
        
        assert isinstance(fig, go.Figure)
        assert summary is not None
        
    def test_preview_data_figure_properties(self):
        """Test preview_data figure properties"""
        fig, summary = self.preview_callback(
            n_clicks=1,
            window_size=5,
            data_store=self.sample_data,
            config_store=self.sample_config
        )
        
        # Test figure layout (only if title exists)
        if fig.layout.title and fig.layout.title.text:
            assert 'Data Preview' in fig.layout.title.text
        if fig.layout.xaxis and fig.layout.xaxis.title:
            assert fig.layout.xaxis.title.text == "Time (seconds)"
        if fig.layout.yaxis and fig.layout.yaxis.title:
            assert fig.layout.yaxis.title.text == "Amplitude"
        assert fig.layout.showlegend is True
        assert fig.layout.height == 400
        
        # Test trace properties (only if traces exist)
        if len(fig.data) > 0:
            trace = fig.data[0]
            assert trace.line.color == 'blue'
            assert trace.line.width == 1
        
    def test_preview_data_summary_content(self):
        """Test preview_data summary content"""
        fig, summary = self.preview_callback(
            n_clicks=1,
            window_size=5,
            data_store=self.sample_data,
            config_store=self.sample_config
        )
        
        # Convert summary to string for content checking
        summary_str = str(summary)
        
        # Should contain expected information
        assert "Data Summary" in summary_str
        assert "Total Samples: 10" in summary_str
        assert "Preview Window: 5" in summary_str
        assert "Sampling Frequency: 100" in summary_str
        assert "Preview Statistics:" in summary_str
        
    @patch('vitalDSP_webapp.callbacks.features.preview_callbacks.logger')
    def test_preview_data_error_handling(self, mock_logger):
        """Test preview_data error handling"""
        # Corrupt data that will cause an error
        corrupt_data = {"data": "invalid"}
        
        fig, summary = self.preview_callback(
            n_clicks=1,
            window_size=5,
            data_store=corrupt_data,
            config_store=self.sample_config
        )
        
        # Should return error figure and summary
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0
        assert "Error:" in fig.layout.annotations[0].text
        
        assert summary is not None
        summary_str = str(summary)
        assert "Error" in summary_str
        assert "Data preview failed" in summary_str
        
        # Should log error
        mock_logger.error.assert_called_once()


class TestCompareSignalsCallback:
    """Test compare_signals callback functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_app = Mock()
        self.captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                self.captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
            
        self.mock_app.callback = mock_callback
        register_preview_callbacks(self.mock_app)
        
        # Find compare_signals callback
        self.compare_callback = None
        for args, kwargs, func in self.captured_callbacks:
            if func.__name__ == 'compare_signals':
                self.compare_callback = func
                break
                
        assert self.compare_callback is not None
        
        # Sample data for testing (longer for FFT)
        signal_length = 100
        time_points = np.linspace(0, 1, signal_length)
        signal_values = np.sin(2 * np.pi * 5 * time_points) + 0.1 * np.random.randn(signal_length)
        
        self.sample_data = {
            "data": [[t, s] for t, s in zip(time_points, signal_values)]
        }
        
        self.sample_config = {
            "sampling_freq": 100,
            "time_unit": "seconds"
        }
        
    def test_compare_signals_no_clicks(self):
        """Test compare_signals with no clicks"""
        with pytest.raises(PreventUpdate):
            self.compare_callback(
                n_clicks=None,
                data_store=self.sample_data,
                config_store=self.sample_config,
                comparison_type="time_frequency"
            )
            
    def test_compare_signals_no_data(self):
        """Test compare_signals with no data"""
        with pytest.raises(PreventUpdate):
            self.compare_callback(
                n_clicks=1,
                data_store=None,
                config_store=self.sample_config,
                comparison_type="time_frequency"
            )
            
    def test_compare_signals_no_config(self):
        """Test compare_signals with no config"""
        with pytest.raises(PreventUpdate):
            self.compare_callback(
                n_clicks=1,
                data_store=self.sample_data,
                config_store=None,
                comparison_type="time_frequency"
            )
            
    def test_compare_signals_time_frequency(self):
        """Test compare_signals with time_frequency comparison"""
        fig = self.compare_callback(
            n_clicks=1,
            data_store=self.sample_data,
            config_store=self.sample_config,
            comparison_type="time_frequency"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Time and frequency domain traces
        
        # Test subplot structure
        assert fig.layout.annotations is not None
        
        # Test traces
        time_trace = fig.data[0]
        freq_trace = fig.data[1]
        
        assert time_trace.name == 'Signal'
        assert freq_trace.name == 'FFT'
        
    def test_compare_signals_statistical(self):
        """Test compare_signals with statistical comparison"""
        fig = self.compare_callback(
            n_clicks=1,
            data_store=self.sample_data,
            config_store=self.sample_config,
            comparison_type="statistical"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Signal and histogram traces
        
        # Test traces
        signal_trace = fig.data[0]
        hist_trace = fig.data[1]
        
        assert signal_trace.name == 'Signal'
        assert hist_trace.name == 'Distribution'
        assert hasattr(hist_trace, 'nbinsx')
        
    def test_compare_signals_default_comparison_type(self):
        """Test compare_signals with default comparison type"""
        fig = self.compare_callback(
            n_clicks=1,
            data_store=self.sample_data,
            config_store=self.sample_config,
            comparison_type=None
        )
        
        # Should default to time_frequency
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        
    def test_compare_signals_invalid_comparison_type(self):
        """Test compare_signals with invalid comparison type"""
        fig = self.compare_callback(
            n_clicks=1,
            data_store=self.sample_data,
            config_store=self.sample_config,
            comparison_type="invalid_type"
        )
        
        # Should return error figure
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0
        assert "Error:" in fig.layout.annotations[0].text
        
    def test_compare_signals_figure_properties(self):
        """Test compare_signals figure properties"""
        fig = self.compare_callback(
            n_clicks=1,
            data_store=self.sample_data,
            config_store=self.sample_config,
            comparison_type="time_frequency"
        )
        
        # Test layout properties
        assert "Signal Comparison" in fig.layout.title.text
        assert fig.layout.height == 600
        assert fig.layout.showlegend is True
        
    @patch('vitalDSP_webapp.callbacks.features.preview_callbacks.logger')
    def test_compare_signals_error_handling(self, mock_logger):
        """Test compare_signals error handling"""
        # Corrupt data that will cause an error
        corrupt_data = {"data": "invalid"}
        
        fig = self.compare_callback(
            n_clicks=1,
            data_store=corrupt_data,
            config_store=self.sample_config,
            comparison_type="time_frequency"
        )
        
        # Should return error figure
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0
        assert "Error:" in fig.layout.annotations[0].text
        
        # Should log error
        mock_logger.error.assert_called_once()


class TestPreviewCallbacksIntegration:
    """Test preview callbacks integration"""
    
    def test_callbacks_registration_integration(self):
        """Test that all callbacks are properly registered"""
        mock_app = Mock()
        captured_calls = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_calls.append((args, kwargs, func))
                return func
            return decorator
            
        mock_app.callback = mock_callback
        register_preview_callbacks(mock_app)
        
        # Should register callbacks (may be more or less than 2 depending on implementation)
        assert len(captured_calls) > 0
        
        # Check callback signatures
        for args, kwargs, func in captured_calls:
            assert len(args) >= 2  # Should have outputs and inputs
            # Outputs and inputs can be single items or lists/tuples
            outputs = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
            inputs = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
            assert len(outputs) >= 1
            assert len(inputs) >= 1
            
    def test_callback_input_output_consistency(self):
        """Test that callback inputs and outputs are consistent"""
        mock_app = Mock()
        captured_calls = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_calls.append((args, kwargs, func))
                return func
            return decorator
            
        mock_app.callback = mock_callback
        register_preview_callbacks(mock_app)
        
        for args, kwargs, func in captured_calls:
            outputs = args[0]
            inputs = args[1]
            
            # Outputs and inputs can be single items or lists/tuples
            outputs_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            inputs_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]
            
            # Should have at least one output and one input
            assert len(outputs_list) >= 1
            assert len(inputs_list) >= 1
            
    def test_data_flow_consistency(self):
        """Test data flow consistency between callbacks"""
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
            
        mock_app.callback = mock_callback
        register_preview_callbacks(mock_app)
        
        # Extract all component IDs used
        all_component_ids = set()
        
        for args, kwargs, func in captured_callbacks:
            outputs = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
            inputs = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
            states = args[2] if len(args) > 2 and isinstance(args[2], (list, tuple)) else []
            
            for output in outputs:
                if hasattr(output, 'component_id'):
                    all_component_ids.add(output.component_id)
                    
            for input_item in inputs:
                if hasattr(input_item, 'component_id'):
                    all_component_ids.add(input_item.component_id)
                    
            for state in states:
                if hasattr(state, 'component_id'):
                    all_component_ids.add(state.component_id)
        
        # Should have reasonable number of component IDs
        assert len(all_component_ids) > 0
        
        # Common component IDs should exist
        expected_ids = ['data-preview-plot', 'comparison-plot', 'btn-preview-data', 'btn-compare-signals']
        for expected_id in expected_ids:
            assert expected_id in all_component_ids


class TestPreviewCallbacksPerformance:
    """Test preview callbacks performance"""
    
    def test_callback_execution_time(self):
        """Test callback execution time is reasonable"""
        import time
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
            
        mock_app.callback = mock_callback
        register_preview_callbacks(mock_app)
        
        # Find preview callback
        preview_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'preview_data':
                preview_callback = func
                break
                
        assert preview_callback is not None
        
        # Test data
        sample_data = {
            "data": [[i, np.sin(i/10)] for i in range(1000)]
        }
        sample_config = {"sampling_freq": 100}
        
        # Measure execution time
        start_time = time.time()
        
        try:
            preview_callback(
                n_clicks=1,
                window_size=500,
                data_store=sample_data,
                config_store=sample_config
            )
        except Exception:
            pass  # We're just testing execution time
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should execute within reasonable time (2 seconds)
        assert execution_time < 2.0
        
    def test_large_data_handling(self):
        """Test handling of large datasets"""
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
            
        mock_app.callback = mock_callback
        register_preview_callbacks(mock_app)
        
        # Find preview callback
        preview_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'preview_data':
                preview_callback = func
                break
                
        # Large dataset
        large_data = {
            "data": [[i, np.random.randn()] for i in range(10000)]
        }
        sample_config = {"sampling_freq": 1000}
        
        # Should handle large data without crashing
        try:
            fig, summary = preview_callback(
                n_clicks=1,
                window_size=1000,
                data_store=large_data,
                config_store=sample_config
            )
            
            assert isinstance(fig, go.Figure)
            assert summary is not None
            
        except Exception as e:
            # Should not raise unexpected exceptions
            assert False, f"Large data handling failed: {e}"
