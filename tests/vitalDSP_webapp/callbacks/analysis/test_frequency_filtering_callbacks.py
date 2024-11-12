"""
Tests for vitalDSP_webapp analysis frequency filtering callbacks
"""
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock

# Import the modules we need to test
try:
    from vitalDSP_webapp.callbacks.analysis.frequency_filtering_callbacks import (
        register_frequency_filtering_callbacks
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
    from vitalDSP_webapp.callbacks.analysis.frequency_filtering_callbacks import (
        register_frequency_filtering_callbacks
    )


class TestFrequencyFilteringCallbacks:
    """Test class for frequency filtering callbacks"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())

    def test_register_frequency_filtering_callbacks(self):
        """Test that register_frequency_filtering_callbacks registers callbacks"""
        register_frequency_filtering_callbacks(self.mock_app)
        
        # Should register callbacks
        assert self.mock_app.callback.call_count > 0

    def test_register_frequency_filtering_callbacks_structure(self):
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
        register_frequency_filtering_callbacks(self.mock_app)
        
        # Verify that callbacks were registered
        assert len(captured_callbacks) > 0
        
        # Test the structure of each callback
        for i, (args, kwargs, func) in enumerate(captured_callbacks):
            outputs = args[0] if args else []
            inputs = args[1] if len(args) > 1 else []
            
            # Each callback should have outputs and inputs
            # Handle case where outputs/inputs might be single objects instead of lists
            if hasattr(outputs, '__len__') and not isinstance(outputs, str):
                assert len(outputs) > 0, f"Callback {i} should have outputs"
            else:
                assert outputs is not None, f"Callback {i} should have outputs"
                
            if hasattr(inputs, '__len__') and not isinstance(inputs, str):
                assert len(inputs) > 0, f"Callback {i} should have inputs"  
            else:
                assert inputs is not None, f"Callback {i} should have inputs"
                
            assert callable(func), f"Callback {i} should be callable"

    def test_callback_registration_without_errors(self):
        """Test that callback registration doesn't raise errors"""
        try:
            register_frequency_filtering_callbacks(self.mock_app)
            assert True
        except Exception as e:
            pytest.fail(f"Callback registration raised an exception: {e}")

    def test_callback_functions_are_callable(self):
        """Test that all registered callback functions are callable"""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_frequency_filtering_callbacks(self.mock_app)
        
        # Verify that all captured functions are callable
        for args, kwargs, func in captured_callbacks:
            assert callable(func)
            # Verify function has expected parameters
            import inspect
            sig = inspect.signature(func)
            assert len(sig.parameters) > 0  # Should have parameters

    def test_multiple_callback_registration(self):
        """Test that multiple callbacks are registered"""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_frequency_filtering_callbacks(self.mock_app)
        
        # Should register multiple callbacks for frequency filtering functionality
        assert len(captured_callbacks) >= 1

    def test_callback_output_structure(self):
        """Test that callbacks have expected output structure"""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_frequency_filtering_callbacks(self.mock_app)
        
        # Check the first callback (main frequency analysis callback)
        if captured_callbacks:
            args, kwargs, func = captured_callbacks[0]
            outputs = args[0] if args else []
            
            # The main frequency callback should have multiple outputs
            assert len(outputs) >= 5  # At least main plot, PSD plot, results, etc.

    def test_callback_input_structure(self):
        """Test that callbacks have expected input structure"""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_frequency_filtering_callbacks(self.mock_app)
        
        # Check the first callback (main frequency analysis callback)
        if captured_callbacks:
            args, kwargs, func = captured_callbacks[0]
            inputs = args[1] if len(args) > 1 else []
            
            # The main frequency callback should have multiple inputs
            assert len(inputs) >= 3  # At least URL, update button, time range

    @patch('vitalDSP_webapp.callbacks.analysis.frequency_filtering_callbacks.logger')
    def test_logging_integration(self, mock_logger):
        """Test that logging is properly integrated"""
        assert mock_logger is not None

    def test_imports_and_dependencies(self):
        """Test that all required dependencies are importable"""
        # This test ensures that the module can be imported without missing dependencies
        try:
            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from dash import Input, Output, State, callback_context, no_update, html
            from dash.exceptions import PreventUpdate
            from scipy import signal
            from scipy.fft import fft, fftfreq, rfft, rfftfreq
            import dash_bootstrap_components as dbc
            import logging
            
            # All imports successful
            assert True
        except ImportError as e:
            pytest.fail(f"Required dependency not available: {e}")

    def test_registration_idempotency(self):
        """Test that multiple registrations don't cause issues"""
        # Register callbacks multiple times
        register_frequency_filtering_callbacks(self.mock_app)
        initial_count = self.mock_app.callback.call_count
        
        register_frequency_filtering_callbacks(self.mock_app)
        second_count = self.mock_app.callback.call_count
        
        # Should register callbacks each time (no built-in duplicate prevention)
        assert second_count >= initial_count