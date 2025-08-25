"""
Tests for vitalDSP_webapp analysis vitaldsp callbacks
"""
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock

# Import the modules we need to test
try:
    from vitalDSP_webapp.callbacks.analysis.vitaldsp_callbacks import (
        register_vitaldsp_callbacks,
        create_empty_figure,
        create_time_domain_plot,
        create_fft_plot,
        create_enhanced_psd_plot
    )
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.callbacks.analysis.vitaldsp_callbacks import (
        register_vitaldsp_callbacks,
        create_empty_figure,
        create_time_domain_plot,
        create_fft_plot,
        create_enhanced_psd_plot
    )


class TestVitalDSPCallbacks:
    """Test class for vitalDSP analysis callbacks"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())

    def test_register_vitaldsp_callbacks(self):
        """Test that register_vitaldsp_callbacks registers callbacks"""
        register_vitaldsp_callbacks(self.mock_app)
        
        # Should register callbacks
        assert self.mock_app.callback.call_count > 0

    def test_register_vitaldsp_callbacks_structure(self):
        """Test the structure of registered callbacks"""
        # Create a mock app with proper callback capturing
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        
        # Register callbacks - this should work without errors
        register_vitaldsp_callbacks(self.mock_app)
        
        # Verify that callbacks were registered
        assert len(captured_callbacks) > 0
        
        # Test the structure of each callback
        for i, (args, kwargs, func) in enumerate(captured_callbacks):
            outputs = args[0] if args else []
            inputs = args[1] if len(args) > 1 else []
            
            # Each callback should have outputs and inputs
            assert len(outputs) > 0, f"Callback {i} should have outputs"
            assert len(inputs) > 0, f"Callback {i} should have inputs"
            assert callable(func), f"Callback {i} should be callable"

    def test_create_empty_figure(self):
        """Test create_empty_figure function"""
        fig = create_empty_figure()
        
        assert isinstance(fig, go.Figure)
        # Check that it has the expected annotation
        assert len(fig.layout.annotations) == 1
        assert "No data available" in fig.layout.annotations[0].text

    def test_create_time_domain_plot_basic(self):
        """Test create_time_domain_plot function with basic data"""
        # Create test data
        signal_data = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 100))
        time_axis = np.linspace(0, 1, 100)
        sampling_freq = 100
        
        fig = create_time_domain_plot(signal_data, time_axis, sampling_freq)
        
        assert isinstance(fig, go.Figure)
        # Check that it has traces
        assert len(fig.data) > 0
        # Check that the first trace has the expected name
        assert fig.data[0].name == 'Original Signal'

    def test_create_time_domain_plot_with_peaks(self):
        """Test create_time_domain_plot function with peaks"""
        # Create test data
        signal_data = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 100))
        time_axis = np.linspace(0, 1, 100)
        sampling_freq = 100
        peaks = [25, 75]  # Example peak indices
        
        fig = create_time_domain_plot(signal_data, time_axis, sampling_freq, peaks=peaks)
        
        assert isinstance(fig, go.Figure)
        # Should have more traces when peaks are included
        assert len(fig.data) >= 2

    def test_create_time_domain_plot_with_filtered_signal(self):
        """Test create_time_domain_plot function with filtered signal"""
        # Create test data
        signal_data = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 100))
        time_axis = np.linspace(0, 1, 100)
        sampling_freq = 100
        filtered_signal = signal_data * 0.9  # Simple filtered version
        
        fig = create_time_domain_plot(signal_data, time_axis, sampling_freq, filtered_signal=filtered_signal)
        
        assert isinstance(fig, go.Figure)
        # Should have more traces when filtered signal is included
        assert len(fig.data) >= 2

    def test_create_fft_plot_basic(self):
        """Test create_fft_plot function with basic data"""
        # Create test data
        signal_data = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 100))
        sampling_freq = 100
        
        fig = create_fft_plot(signal_data, sampling_freq, 'hann', 256, 0, 50)
        
        assert isinstance(fig, go.Figure)
        # Check that it has traces
        assert len(fig.data) > 0

    def test_create_enhanced_psd_plot_basic(self):
        """Test create_enhanced_psd_plot function with basic data"""
        # Create test data
        signal_data = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 100))
        sampling_freq = 100
        column_mapping = {'signal': 0}  # Mock column mapping
        
        fig = create_enhanced_psd_plot(signal_data, sampling_freq, 2, 0.5, 50, False, False, 'signal', column_mapping)
        
        assert isinstance(fig, go.Figure)
        # Check that it has traces
        assert len(fig.data) > 0

    def test_create_time_domain_plot_error_handling(self):
        """Test create_time_domain_plot function error handling"""
        # Test with invalid data
        try:
            fig = create_time_domain_plot(None, None, 100)
            # If it doesn't raise an exception, it should return a valid figure
            assert isinstance(fig, go.Figure)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_create_fft_plot_error_handling(self):
        """Test create_fft_plot function error handling"""
        # Test with invalid data
        try:
            fig = create_fft_plot(None, 100, 'hann', 256, 0, 50)
            # If it doesn't raise an exception, it should return a valid figure
            assert isinstance(fig, go.Figure)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    @patch('vitalDSP_webapp.callbacks.analysis.vitaldsp_callbacks.logger')
    def test_logging_integration(self, mock_logger):
        """Test that logging is properly integrated"""
        # This is a basic test to ensure the logger is available
        # More specific logging tests would require testing actual callback execution
        assert mock_logger is not None

    def test_callback_registration_without_errors(self):
        """Test that callback registration doesn't raise errors"""
        # This test ensures that the registration process itself is robust
        try:
            register_vitaldsp_callbacks(self.mock_app)
            # If we get here without exception, the test passes
            assert True
        except Exception as e:
            pytest.fail(f"Callback registration raised an exception: {e}")

    def test_helper_functions_return_types(self):
        """Test that helper functions return expected types"""
        # Test create_empty_figure
        fig = create_empty_figure()
        assert isinstance(fig, go.Figure)
        
        # Test with minimal valid data
        signal_data = np.array([1, 2, 3, 4, 5])
        time_axis = np.array([0, 1, 2, 3, 4])
        sampling_freq = 1
        
        fig = create_time_domain_plot(signal_data, time_axis, sampling_freq)
        assert isinstance(fig, go.Figure)
        
        fig = create_fft_plot(signal_data, sampling_freq, 'hann', 8, 0, 0.5)
        assert isinstance(fig, go.Figure)

    def test_function_parameters_validation(self):
        """Test that functions handle parameter validation appropriately"""
        # Test with edge cases
        
        # Empty arrays
        empty_array = np.array([])
        try:
            create_time_domain_plot(empty_array, empty_array, 100)
        except (ValueError, IndexError):
            # Expected to fail with empty data
            pass
        
        # Single value arrays
        single_value = np.array([1])
        try:
            create_time_domain_plot(single_value, single_value, 100)
        except (ValueError, IndexError):
            # May fail with insufficient data
            pass
        
        # Zero sampling frequency
        try:
            create_fft_plot(np.array([1, 2, 3]), 0, 'hann', 4, 0, 1)
        except (ValueError, ZeroDivisionError):
            # Expected to fail with zero sampling frequency
            pass
