"""
Tests for vitalDSP_webapp core upload callbacks
"""
import pytest
import pandas as pd
import base64
import io
from unittest.mock import Mock, patch, MagicMock
from dash import html, no_update
from dash.exceptions import PreventUpdate

# Import the modules we need to test
try:
    from vitalDSP_webapp.callbacks.core.upload_callbacks import (
        register_upload_callbacks,
        create_data_preview
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
    from vitalDSP_webapp.callbacks.core.upload_callbacks import (
        register_upload_callbacks,
        create_data_preview
    )


class TestUploadCallbacks:
    """Test class for upload callbacks"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())
        # Don't set the flag so callbacks can be registered

    def test_register_upload_callbacks_first_time(self):
        """Test that register_upload_callbacks registers callbacks on first call"""
        register_upload_callbacks(self.mock_app)
        
        # Should register callbacks and set the flag
        assert self.mock_app.callback.call_count >= 5  # Should register 5 callbacks
        assert hasattr(self.mock_app, '_upload_callbacks_registered')
        assert self.mock_app._upload_callbacks_registered is True

    def test_register_upload_callbacks_duplicate_prevention(self):
        """Test that register_upload_callbacks prevents duplicate registration"""
        # Create a fresh mock app for this test
        mock_app = Mock()
        mock_app.callback = Mock(return_value=Mock())
        
        # Set the flag to indicate callbacks are already registered
        mock_app._upload_callbacks_registered = True
        
        register_upload_callbacks(mock_app)
        
        # Should not register any callbacks
        assert mock_app.callback.call_count == 0

    def test_register_upload_callbacks_structure(self):
        """Test the structure of registered callbacks"""
        register_upload_callbacks(self.mock_app)

        # Should register at least 5 callbacks (slim-flow baseline).
        assert self.mock_app.callback.call_count >= 5

        # Each registered callback must have either positional Output/Input
        # args (the historic pattern) or keyword-only args (the
        # background-callback pattern used by process_data_with_columns).
        for call in self.mock_app.callback.call_args_list:
            assert call[0] or call[1]

    def test_handle_csv_upload_integration(self):
        """Test CSV upload integration (simplified)"""
        # This is a simplified integration test that verifies the callback registration
        # without testing the complex internal logic which requires many dependencies
        
        # Create a mock app with proper callback capturing
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        
        # Register callbacks - this should work without errors
        register_upload_callbacks(self.mock_app)
        
        # Verify that callbacks were registered
        assert len(captured_callbacks) > 0
        
        # Verify that the main upload callback was registered with expected structure
        main_callback = captured_callbacks[0]
        outputs = main_callback[0][0]  # First positional argument (outputs)
        inputs = main_callback[0][1]   # Second positional argument (inputs)
        
        # Check that we have multiple outputs (the upload callback has many outputs)
        # Handle case where outputs might be single objects instead of lists
        if hasattr(outputs, '__len__') and not isinstance(outputs, str):
            assert len(outputs) > 10
        else:
            assert outputs is not None
        
        # Check that we have the expected inputs
        # Handle case where inputs might be single objects instead of lists
        if hasattr(inputs, '__len__') and not isinstance(inputs, str):
            assert len(inputs) == 3  # upload-data, btn-load-sample-ppg, btn-load-sample-ecg
        else:
            assert inputs is not None

    def test_callback_registration_structure(self):
        """Test that callbacks are registered with proper structure"""
        # Create a mock app with proper callback capturing
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        
        # Register callbacks - this should work without errors
        register_upload_callbacks(self.mock_app)
        
        # Verify that multiple callbacks were registered
        assert len(captured_callbacks) >= 5
        
        # Test the structure of each callback.  The historic pattern is
        # ``@app.callback(outputs, inputs, ...)`` (positional), but the
        # background-callback path uses ``@app.callback(output=..., inputs=...,
        # state=..., background=True, manager=...)`` (kw-only).  Accept both.
        for i, (args, kwargs, func) in enumerate(captured_callbacks):
            outputs = args[0] if args else kwargs.get("output")
            inputs = (args[1] if len(args) > 1 else None) or kwargs.get("inputs")

            if hasattr(outputs, "__len__") and not isinstance(outputs, str):
                assert len(outputs) > 0, f"Callback {i} should have outputs"
            else:
                assert outputs is not None, f"Callback {i} should have outputs"

            if hasattr(inputs, "__len__") and not isinstance(inputs, str):
                assert len(inputs) > 0, f"Callback {i} should have inputs"
            else:
                assert inputs is not None, f"Callback {i} should have inputs"

            assert callable(func), f"Callback {i} should be callable"

    def test_callback_functions_exist(self):
        """Test that the callback functions are properly defined"""
        # Create a mock app with proper callback capturing
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        
        # Register callbacks
        register_upload_callbacks(self.mock_app)
        
        # Verify that all captured functions are callable
        for args, kwargs, func in captured_callbacks:
            assert callable(func)
            # Verify function has expected parameters (this is a basic check)
            import inspect
            sig = inspect.signature(func)
            assert len(sig.parameters) > 0  # Should have parameters

    def test_create_data_preview(self):
        """Test data preview creation"""
        df = pd.DataFrame({'time': [1, 2, 3], 'signal': [0.1, 0.2, 0.3]})
        data_info = {
            'sampling_freq': 1000,
            'duration': 3.0
        }

        result = create_data_preview(df, data_info)

        assert isinstance(result, html.Div)
        rendered = str(result)
        # Modern chip layout: shape as "3 × 2", duration as "3.0 s".
        assert "Data Preview" in rendered
        assert "3 × 2" in rendered
        assert "1000 Hz" in rendered
        assert "3.0 s" in rendered
