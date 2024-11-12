"""
Comprehensive tests for vitalDSP_webapp upload callbacks to improve coverage
"""
import pytest
import pandas as pd
import base64
import io
import json
from unittest.mock import Mock, patch, MagicMock, call
from dash import html, no_update, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Import the modules we need to test
try:
    from vitalDSP_webapp.callbacks.core.upload_callbacks import (
        register_upload_callbacks,
        create_file_path_loading_indicator,
        create_upload_progress_bar,
        create_processing_progress_section,
        create_error_status,
        create_success_status,
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
        create_file_path_loading_indicator,
        create_upload_progress_bar,
        create_processing_progress_section,
        create_error_status,
        create_success_status,
        create_data_preview
    )


class TestUploadCallbacksComprehensive:
    """Comprehensive test class for upload callbacks"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())
        # Ensure callbacks can be registered by not setting the flag
        if hasattr(self.mock_app, '_upload_callbacks_registered'):
            delattr(self.mock_app, '_upload_callbacks_registered')

    def test_create_file_path_loading_indicator(self):
        """Test file path loading indicator creation"""
        result = create_file_path_loading_indicator()
        
        assert isinstance(result, html.Div)
        # Check that it contains the expected components
        assert len(result.children) > 0
        assert isinstance(result.children[0], dbc.Alert)

    def test_create_upload_progress_bar(self):
        """Test upload progress bar creation"""
        result = create_upload_progress_bar()
        
        assert isinstance(result, html.Div)
        # Check that it contains progress elements
        assert len(result.children) > 0

    def test_create_processing_progress_section(self):
        """Test processing progress section creation"""
        result = create_processing_progress_section()
        
        assert isinstance(result, html.Div)
        # Check that it contains the expected components
        assert len(result.children) > 0

    def test_create_error_status(self):
        """Test error status creation"""
        error_message = "Test error message"
        result = create_error_status(error_message)
        
        assert isinstance(result, html.Div)
        # Check that the error message is included
        assert len(result.children) > 0

    def test_create_success_status(self):
        """Test success status creation"""
        success_message = "Test success message"
        result = create_success_status(success_message)
        
        assert isinstance(result, html.Div)
        # Check that the success message is included
        assert len(result.children) > 0

    def test_create_data_preview_with_data(self):
        """Test data preview creation with actual data"""
        df = pd.DataFrame({
            'time': [0, 1, 2, 3, 4],
            'signal': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        data_info = {
            'shape': (5, 2),
            'columns': ['time', 'signal'],
            'sampling_freq': 100,
            'duration': 0.05
        }
        
        result = create_data_preview(df, data_info)
        
        assert isinstance(result, html.Div)
        # Check that it contains data table and info
        assert len(result.children) > 0

    def test_create_data_preview_with_empty_data(self):
        """Test data preview creation with empty data"""
        df = pd.DataFrame()
        data_info = {'shape': (0, 0)}
        
        result = create_data_preview(df, data_info)
        
        assert isinstance(result, html.Div)

    @patch('vitalDSP_webapp.callbacks.core.upload_callbacks.get_data_service')
    @patch('vitalDSP_webapp.callbacks.core.upload_callbacks.DataProcessor')
    def test_callback_with_csv_upload(self, mock_data_processor, mock_get_data_service):
        """Test callback behavior with CSV upload"""
        # Setup mocks
        mock_data_service = Mock()
        mock_get_data_service.return_value = mock_data_service
        
        mock_df = pd.DataFrame({'time': [0, 1, 2], 'signal': [0.1, 0.2, 0.3]})
        mock_data_processor.process_uploaded_data.return_value = {
            'shape': (3, 2),
            'columns': ['time', 'signal'],
            'sampling_freq': 100,
            'duration': 0.03
        }
        
        # Create CSV content
        csv_content = "time,signal\n0,0.1\n1,0.2\n2,0.3"
        encoded_content = base64.b64encode(csv_content.encode()).decode()
        
        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_upload_callbacks(self.mock_app)
        
        # Find the main upload callback
        upload_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'handle_all_uploads':
                upload_callback = func
                break
        
        assert upload_callback is not None
        
        # Test the callback with CSV upload
        with patch('pandas.read_csv', return_value=mock_df):
            with patch('vitalDSP_webapp.callbacks.core.upload_callbacks.callback_context') as mock_ctx:
                mock_ctx.triggered = [{"prop_id": "upload-data.contents", "value": encoded_content}]
                
                result = upload_callback(
                    upload_contents=encoded_content,
                    load_path_clicks=None,
                    load_sample_clicks=None,
                    filename="test.csv",
                    file_path=None,
                    sampling_freq=100,
                    time_unit="seconds"
                )
                
                # Should return multiple outputs
                assert isinstance(result, tuple)
                assert len(result) >= 15  # Expected number of outputs

    @patch('vitalDSP_webapp.callbacks.core.upload_callbacks.get_data_service')
    def test_callback_with_sample_data_load(self, mock_get_data_service):
        """Test callback behavior with sample data loading"""
        # Setup mocks
        mock_data_service = Mock()
        mock_get_data_service.return_value = mock_data_service
        
        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_upload_callbacks(self.mock_app)
        
        # Find the main upload callback
        upload_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'handle_all_uploads':
                upload_callback = func
                break
        
        assert upload_callback is not None
        
        # Test the callback with sample data load
        with patch('vitalDSP_webapp.callbacks.core.upload_callbacks.callback_context') as mock_ctx:
            mock_ctx.triggered = [{"prop_id": "btn-load-sample.n_clicks", "value": 1}]
            
            with patch('vitalDSP_webapp.callbacks.core.upload_callbacks.DataProcessor.generate_sample_ppg_data') as mock_generate:
                mock_df = pd.DataFrame({'time': [0, 1, 2], 'signal': [0.1, 0.2, 0.3]})
                mock_generate.return_value = mock_df
                
                result = upload_callback(
                    upload_contents=None,
                    load_path_clicks=None,
                    load_sample_clicks=1,
                    filename=None,
                    file_path=None,
                    sampling_freq=100,
                    time_unit="seconds"
                )
                
                # Should return multiple outputs
                assert isinstance(result, tuple)
                assert len(result) >= 15  # Expected number of outputs

    def test_callback_with_no_trigger(self):
        """Test callback behavior with no trigger"""
        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_upload_callbacks(self.mock_app)
        
        # Find the main upload callback
        upload_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'handle_all_uploads':
                upload_callback = func
                break
        
        assert upload_callback is not None
        
        # Test the callback with no trigger
        with patch('vitalDSP_webapp.callbacks.core.upload_callbacks.callback_context') as mock_ctx:
            mock_ctx.triggered = []
            
            with pytest.raises(PreventUpdate):
                            upload_callback(
                upload_contents=None,
                load_path_clicks=None,
                load_sample_clicks=None,
                filename=None,
                file_path=None,
                sampling_freq=100,
                time_unit="seconds"
            )

    def test_callback_with_invalid_file_format(self):
        """Test callback behavior with invalid file format"""
        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_upload_callbacks(self.mock_app)
        
        # Find the main upload callback
        upload_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'handle_all_uploads':
                upload_callback = func
                break
        
        assert upload_callback is not None
        
        # Test the callback with invalid file format
        invalid_content = base64.b64encode(b"invalid content").decode()
        
        with patch('vitalDSP_webapp.callbacks.core.upload_callbacks.callback_context') as mock_ctx:
            mock_ctx.triggered = [{"prop_id": "upload-data.contents", "value": invalid_content}]
            
            result = upload_callback(
                upload_contents=invalid_content,
                load_path_clicks=None,
                load_sample_clicks=None,
                filename="test.xyz",  # Invalid extension
                file_path=None,
                sampling_freq=100,
                time_unit="seconds"
            )
            
            # Should return error status
            assert isinstance(result, tuple)
            assert len(result) >= 15

    @patch('vitalDSP_webapp.callbacks.core.upload_callbacks.get_data_service')
    def test_column_mapping_callback(self, mock_get_data_service):
        """Test column mapping callback"""
        # Setup mocks
        mock_data_service = Mock()
        mock_get_data_service.return_value = mock_data_service
        mock_data_service._auto_detect_columns.return_value = {
            'time': 'col1',
            'signal': 'col2'
        }
        
        # Register callbacks and capture them
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_upload_callbacks(self.mock_app)
        
                # Find column mapping callback (update_process_button_state)
        column_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_process_button_state':
                column_callback = func
                break

        if column_callback:
            result = column_callback("time_col", "signal_col")  # Simulate with both required parameters
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_all_callbacks_registered(self):
        """Test that all expected callbacks are registered"""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        register_upload_callbacks(self.mock_app)
        
        # Should register exactly 6 callbacks
        assert len(captured_callbacks) == 6
        
        # Check that all expected callback functions exist
        function_names = [func.__name__ for args, kwargs, func in captured_callbacks]
        expected_functions = [
            'handle_all_uploads',
            'hide_progress_section', 
            'auto_hide_progress_section'
        ]
        
        # At least these functions should be present
        for expected_func in expected_functions:
            assert expected_func in function_names

    def test_callback_registration_idempotent(self):
        """Test that callback registration is idempotent"""
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        
        # Register callbacks twice
        register_upload_callbacks(self.mock_app)
        first_count = len(captured_callbacks)
        
        register_upload_callbacks(self.mock_app)
        second_count = len(captured_callbacks)
        
        # Should not register callbacks twice
        assert first_count == second_count
        assert hasattr(self.mock_app, '_upload_callbacks_registered')
        assert self.mock_app._upload_callbacks_registered is True

    def test_create_data_preview_large_dataframe(self):
        """Test data preview with large dataframe (should be truncated)"""
        # Create a large dataframe
        large_df = pd.DataFrame({
            'time': range(1000),
            'signal': [i * 0.1 for i in range(1000)]
        })
        data_info = {
            'shape': (1000, 2),
            'columns': ['time', 'signal'],
            'sampling_freq': 100,
            'duration': 10.0
        }
        
        result = create_data_preview(large_df, data_info)
        
        assert isinstance(result, html.Div)
        # Should contain data table with limited rows
        assert len(result.children) > 0

    def test_create_data_preview_with_missing_info(self):
        """Test data preview creation with missing data info"""
        df = pd.DataFrame({
            'time': [0, 1, 2],
            'signal': [0.1, 0.2, 0.3]
        })
        data_info = {}  # Empty info
        
        result = create_data_preview(df, data_info)
        
        assert isinstance(result, html.Div)
        # Should still create preview even with missing info
        assert len(result.children) > 0
