import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import dash_bootstrap_components as dbc
from dash import html

# Add the src directory to the Python path so tests can import vitalDSP_webapp modules
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the module directly and access functions as attributes
import vitalDSP_webapp.callbacks.upload_callbacks as upload_module
from vitalDSP_webapp.config.settings import app_config


class TestUploadCallbacks:
    """Test class for upload callbacks functions."""

    def test_create_success_status_basic(self):
        """Test creating a basic success status."""
        data_info = {
            'filename': 'test.csv',
            'rows': 1000,
            'size_mb': 2.5,
            'duration_sec': 10.0,
            'quality_status': 'Good'
        }
        
        status = upload_module.create_success_status(data_info)
        
        assert isinstance(status, dbc.Alert)
        assert status.color == "success"
        assert "✅ File Upload Successful!" in str(status)
        assert "1,000" in str(status)  # Check for formatted number
        assert "test.csv" in str(status)
        assert "2.50 MB" in str(status)
        assert "10.0s" in str(status)
        assert "Good" in str(status)

    def test_create_success_status_with_large_numbers(self):
        """Test creating success status with large numbers."""
        data_info = {
            'filename': 'large_file.csv',
            'rows': 1000000,
            'size_mb': 150.75,
            'duration_sec': 3600.5,
            'quality_status': 'Excellent'
        }
        
        status = upload_module.create_success_status(data_info)
        
        assert "1,000,000" in str(status)  # Check for formatted large number
        assert "150.75 MB" in str(status)
        assert "3600.5s" in str(status)
        assert "Excellent" in str(status)

    def test_create_error_status_basic(self):
        """Test creating a basic error status."""
        message = "File format not supported"
        
        status = upload_module.create_error_status(message)
        
        assert isinstance(status, dbc.Alert)
        assert status.color == "danger"
        assert "❌ Upload Error" in str(status)
        assert message in str(status)

    def test_create_error_status_with_long_message(self):
        """Test creating error status with a long error message."""
        long_message = "This is a very long error message that contains multiple sentences and should be displayed properly in the error alert component."
        
        status = upload_module.create_error_status(long_message)
        
        assert isinstance(status, dbc.Alert)
        assert long_message in str(status)

    def test_create_processing_success_status_basic(self):
        """Test creating a basic processing success status."""
        data_store = {
            'info': {
                'rows': 5000,
                'filename': 'processed_data.csv'
            }
        }
        config_store = {
            'sampling_freq': 1000,
            'time_unit': 'ms'
        }
        column_mapping = {
            'time': 'timestamp',
            'signal': 'ppg_signal',
            'red': None,
            'ir': None
        }
        
        status = upload_module.create_processing_success_status(data_store, config_store, column_mapping)
        
        assert isinstance(status, dbc.Alert)
        assert status.color == "success"
        assert "✅ Data Processing Complete!" in str(status)
        assert "5,000" in str(status)
        assert "timestamp" in str(status)
        assert "ppg_signal" in str(status)
        assert "1000 Hz" in str(status)

    def test_create_processing_success_status_with_red_ir(self):
        """Test creating processing success status with RED and IR channels."""
        data_store = {
            'info': {
                'rows': 3000,
                'filename': 'multi_channel.csv'
            }
        }
        config_store = {
            'sampling_freq': 500,
            'time_unit': 'ms'
        }
        column_mapping = {
            'time': 'time',
            'signal': 'signal',
            'red': 'red_channel',
            'ir': 'ir_channel'
        }
        
        status = upload_module.create_processing_success_status(data_store, config_store, column_mapping)
        
        assert "red_channel" in str(status)
        assert "ir_channel" in str(status)
        assert "500 Hz" in str(status)

    def test_create_processing_success_status_with_partial_channels(self):
        """Test creating processing success status with only RED channel."""
        data_store = {
            'info': {
                'rows': 2000,
                'filename': 'partial_channels.csv'
            }
        }
        config_store = {
            'sampling_freq': 250,
            'time_unit': 'ms'
        }
        column_mapping = {
            'time': 'time',
            'signal': 'signal',
            'red': 'red_only',
            'ir': None
        }
        
        status = upload_module.create_processing_success_status(data_store, config_store, column_mapping)
        
        assert "red_only" in str(status)
        assert "IR" not in str(status)  # Should not show IR since it's None

    def test_create_processing_success_status_with_waveform(self):
        """Test creating processing success status with waveform column."""
        data_store = {
            'info': {
                'rows': 1500,
                'filename': 'waveform_data.csv'
            }
        }
        config_store = {
            'sampling_freq': 1000,
            'time_unit': 'ms'
        }
        column_mapping = {
            'time': 'timestamp',
            'signal': 'main_signal',
            'red': 'red_channel',
            'ir': 'ir_channel',
            'waveform': 'pleth_waveform'
        }
        
        status = upload_module.create_processing_success_status(data_store, config_store, column_mapping)
        
        assert "pleth_waveform" in str(status)
        assert "Waveform" in str(status)
        assert "1000 Hz" in str(status)

    @patch('vitalDSP_webapp.callbacks.upload_callbacks.create_data_preview')
    def test_create_data_preview_section_basic(self, mock_create_preview):
        """Test creating data preview section with basic data."""
        # Mock the create_data_preview function
        mock_preview = html.Div("Mock preview")
        mock_create_preview.return_value = mock_preview
        
        data_info = {
            'filename': 'test.csv',
            'rows': 1000,
            'columns': 5
        }
        
        # Create a simple DataFrame
        df = pd.DataFrame({
            'time': np.arange(100),
            'signal': np.random.randn(100),
            'red': np.random.randn(100),
            'ir': np.random.randn(100),
            'quality': np.random.randn(100)
        })
        
        result = upload_module.create_data_preview_section(data_info, df, None)
        
        assert result == mock_preview
        mock_create_preview.assert_called_once()

    @patch('vitalDSP_webapp.callbacks.upload_callbacks.create_data_preview')
    def test_create_data_preview_section_with_numeric_data(self, mock_create_preview):
        """Test creating data preview section with numeric data for plotting."""
        # Mock the create_data_preview function
        mock_preview = html.Div("Mock preview with plot")
        mock_create_preview.return_value = mock_preview
        
        data_info = {
            'filename': 'numeric_data.csv',
            'rows': 500,
            'columns': 3
        }
        
        # Create DataFrame with numeric data
        df = pd.DataFrame({
            'time': np.arange(500),
            'signal': np.random.randn(500),
            'quality': np.random.randn(500)
        })
        
        result = upload_module.create_data_preview_section(data_info, df, None)
        
        assert result == mock_preview
        mock_create_preview.assert_called_once()
        
        # Check that data_info was updated with preview components
        assert 'preview_plot' in data_info
        assert 'preview_table' in data_info

    @patch('vitalDSP_webapp.callbacks.upload_callbacks.create_data_preview')
    def test_create_data_preview_section_with_non_numeric_data(self, mock_create_preview):
        """Test creating data preview section with non-numeric data."""
        # Mock the create_data_preview function
        mock_preview = html.Div("Mock preview no plot")
        mock_create_preview.return_value = mock_preview
        
        data_info = {
            'filename': 'text_data.csv',
            'rows': 100,
            'columns': 2
        }
        
        # Create DataFrame with non-numeric data
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'description': ['Description ' + str(i) for i in range(100)]
        })
        
        result = upload_module.create_data_preview_section(data_info, df, None)
        
        assert result == mock_preview
        mock_create_preview.assert_called_once()

    @patch('vitalDSP_webapp.callbacks.upload_callbacks.create_data_preview')
    def test_create_data_preview_section_empty_dataframe(self, mock_create_preview):
        """Test creating data preview section with empty DataFrame."""
        # Mock the create_data_preview function
        mock_preview = html.Div("Mock preview empty")
        mock_create_preview.return_value = mock_preview
        
        data_info = {
            'filename': 'empty.csv',
            'rows': 0,
            'columns': 0
        }
        
        # Create empty DataFrame
        df = pd.DataFrame()
        
        result = upload_module.create_data_preview_section(data_info, df, None)
        
        assert result == mock_preview
        mock_create_preview.assert_called_once()

    def test_create_success_status_edge_cases(self):
        """Test edge cases for success status creation."""
        # Test with zero rows
        data_info_zero = {
            'filename': 'empty.csv',
            'rows': 0,
            'size_mb': 0.0,
            'duration_sec': 0.0,
            'quality_status': 'Unknown'
        }
        
        status_zero = upload_module.create_success_status(data_info_zero)
        assert "0" in str(status_zero)
        assert "0.00 MB" in str(status_zero)
        assert "0.0s" in str(status_zero)
        
        # Test with very small numbers
        data_info_small = {
            'filename': 'tiny.csv',
            'rows': 1,
            'size_mb': 0.001,
            'duration_sec': 0.001,
            'quality_status': 'Poor'
        }
        
        status_small = upload_module.create_success_status(data_info_small)
        assert "1" in str(status_small)
        assert "0.00 MB" in str(status_small)
        assert "0.0s" in str(status_small)

    def test_create_error_status_edge_cases(self):
        """Test edge cases for error status creation."""
        # Test with empty message
        status_empty = upload_module.create_error_status("")
        assert isinstance(status_empty, dbc.Alert)
        
        # Test with special characters
        special_message = "Error with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        status_special = upload_module.create_error_status(special_message)
        assert special_message in str(status_special)
        
        # Test with HTML-like content
        html_message = "<script>alert('xss')</script>"
        status_html = upload_module.create_error_status(html_message)
        assert html_message in str(status_html)

    def test_create_processing_success_status_edge_cases(self):
        """Test edge cases for processing success status creation."""
        # Test with None values in column mapping
        data_store = {
            'info': {
                'rows': 100,
                'filename': 'edge_case.csv'
            }
        }
        config_store = {
            'sampling_freq': 0,
            'time_unit': ''
        }
        column_mapping = {
            'time': None,
            'signal': None,
            'red': None,
            'ir': None
        }
        
        status = upload_module.create_processing_success_status(data_store, config_store, column_mapping)
        assert isinstance(status, dbc.Alert)
        assert "0 Hz" in str(status)
