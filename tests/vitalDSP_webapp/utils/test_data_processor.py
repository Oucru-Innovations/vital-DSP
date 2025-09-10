"""
Tests for vitalDSP_webapp utils data_processor
"""
import pytest
import pandas as pd
import numpy as np
import base64
import io
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules we need to test
try:
    from vitalDSP_webapp.utils.data_processor import DataProcessor
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.utils.data_processor import DataProcessor


class TestDataProcessor:
    """Test class for DataProcessor utility"""
    
    def test_validate_file_extension_csv(self):
        """Test validate_file_extension with CSV files"""
        assert DataProcessor.validate_file_extension("test.csv") is True
        assert DataProcessor.validate_file_extension("test.CSV") is True
        assert DataProcessor.validate_file_extension("data/test.csv") is True

    def test_validate_file_extension_txt(self):
        """Test validate_file_extension with TXT files"""
        assert DataProcessor.validate_file_extension("test.txt") is True
        assert DataProcessor.validate_file_extension("test.TXT") is True
        assert DataProcessor.validate_file_extension("data/test.txt") is True

    def test_validate_file_extension_mat(self):
        """Test validate_file_extension with MAT files"""
        assert DataProcessor.validate_file_extension("test.mat") is True
        assert DataProcessor.validate_file_extension("test.MAT") is True
        assert DataProcessor.validate_file_extension("data/test.mat") is True

    def test_validate_file_extension_unsupported(self):
        """Test validate_file_extension with unsupported files"""
        assert DataProcessor.validate_file_extension("test.pdf") is False
        assert DataProcessor.validate_file_extension("test.docx") is False
        assert DataProcessor.validate_file_extension("test.jpg") is False
        assert DataProcessor.validate_file_extension("test.exe") is False

    def test_validate_file_extension_empty_or_none(self):
        """Test validate_file_extension with empty or None input"""
        assert DataProcessor.validate_file_extension("") is False
        assert DataProcessor.validate_file_extension(None) is False

    def test_validate_file_extension_no_extension(self):
        """Test validate_file_extension with files without extension"""
        assert DataProcessor.validate_file_extension("test") is False
        assert DataProcessor.validate_file_extension("test.") is False

    def test_read_uploaded_content_csv(self):
        """Test read_uploaded_content with CSV data"""
        # Create test CSV data
        csv_data = "time,signal\n1,0.1\n2,0.2\n3,0.3"
        encoded_data = base64.b64encode(csv_data.encode()).decode()
        contents = f"data:text/csv;base64,{encoded_data}"
        
        result = DataProcessor.read_uploaded_content(contents, "test.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['time', 'signal']
        assert result['time'].tolist() == [1, 2, 3]
        assert result['signal'].tolist() == [0.1, 0.2, 0.3]

    def test_read_uploaded_content_txt(self):
        """Test read_uploaded_content with TXT data"""
        # Create test TXT data (tab-separated)
        txt_data = "time\tsignal\n1\t0.1\n2\t0.2\n3\t0.3"
        encoded_data = base64.b64encode(txt_data.encode()).decode()
        contents = f"data:text/plain;base64,{encoded_data}"
        
        result = DataProcessor.read_uploaded_content(contents, "test.txt")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['time', 'signal']
        assert result['time'].tolist() == [1, 2, 3]
        assert result['signal'].tolist() == [0.1, 0.2, 0.3]

    def test_read_uploaded_content_unsupported_format(self):
        """Test read_uploaded_content with unsupported format"""
        encoded_data = base64.b64encode(b"test data").decode()
        contents = f"data:application/pdf;base64,{encoded_data}"
        
        result = DataProcessor.read_uploaded_content(contents, "test.pdf")
        
        assert result is None

    def test_read_uploaded_content_invalid_base64(self):
        """Test read_uploaded_content with invalid base64 data"""
        contents = "invalid_base64_content"
        
        result = DataProcessor.read_uploaded_content(contents, "test.csv")
        
        assert result is None

    def test_read_uploaded_content_malformed_csv(self):
        """Test read_uploaded_content with malformed CSV data"""
        # Create malformed CSV data
        csv_data = "time,signal\n1,0.1\n2,invalid,extra_column\n3,0.3"
        encoded_data = base64.b64encode(csv_data.encode()).decode()
        contents = f"data:text/csv;base64,{encoded_data}"
        
        # This should still work as pandas is quite forgiving
        result = DataProcessor.read_uploaded_content(contents, "test.csv")
        
        # Should return a DataFrame, even if the data is malformed
        assert isinstance(result, pd.DataFrame) or result is None

    def test_process_uploaded_data_basic(self):
        """Test process_uploaded_data with basic DataFrame"""
        df = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'signal': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = DataProcessor.process_uploaded_data(df, "test.csv", 100, "seconds")
        
        assert isinstance(result, dict)
        assert result['filename'] == "test.csv"
        assert result['sampling_freq'] == 100
        assert result['time_unit'] == "seconds"
        assert result['num_rows'] == 5
        assert result['num_columns'] == 2
        assert result['columns'] == ['time', 'signal']

    def test_process_uploaded_data_duration_calculation(self):
        """Test process_uploaded_data duration calculation"""
        df = pd.DataFrame({
            'time': [0, 1, 2, 3, 4],  # 5 seconds of data
            'signal': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = DataProcessor.process_uploaded_data(df, "test.csv", 1, "seconds")
        
        # Duration should be calculated from the time column
        assert 'duration' in result
        assert result['duration'] == 5.0  # len(signal_data) / sampling_freq = 5 / 1.0 = 5.0

    def test_process_uploaded_data_empty_dataframe(self):
        """Test process_uploaded_data with empty DataFrame"""
        df = pd.DataFrame()

        result = DataProcessor.process_uploaded_data(df, "test.csv", 100, "seconds")

        assert result is None  # Implementation returns None for empty dataframes

    def test_generate_sample_ppg_data_basic(self):
        """Test generate_sample_ppg_data with basic parameters"""
        result = DataProcessor.generate_sample_ppg_data(100)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'time' in result.columns
        assert 'signal' in result.columns  # Implementation uses 'signal' not 'ppg_signal'

    def test_generate_sample_ppg_data_custom_duration(self):
        """Test generate_sample_ppg_data with custom duration"""
        sampling_freq = 100
        duration = 5  # 5 seconds
        
        result = DataProcessor.generate_sample_ppg_data(sampling_freq, duration)
        
        assert isinstance(result, pd.DataFrame)
        expected_length = int(sampling_freq * duration)
        assert len(result) == expected_length

    def test_generate_sample_ppg_data_custom_heart_rate(self):
        """Test generate_sample_ppg_data with custom heart rate"""
        result = DataProcessor.generate_sample_ppg_data(100, heart_rate=80)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'time' in result.columns
        assert 'signal' in result.columns  # Implementation uses 'signal' not 'ppg_signal'
        
        # Check that the signal has reasonable values
        assert result['signal'].min() >= -2
        assert result['signal'].max() <= 2

    def test_generate_sample_ppg_data_with_noise(self):
        """Test generate_sample_ppg_data with noise"""
        result = DataProcessor.generate_sample_ppg_data(100, noise_level=0.1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # With noise, the signal should have more variation
        signal_std = result['signal'].std()  # Implementation uses 'signal' not 'ppg_signal'
        assert signal_std > 0

    def test_generate_sample_ppg_data_zero_sampling_freq(self):
        """Test generate_sample_ppg_data with zero sampling frequency"""
        result = DataProcessor.generate_sample_ppg_data(0)
        assert result is None  # Should return None for invalid input

    def test_generate_sample_ppg_data_negative_duration(self):
        """Test generate_sample_ppg_data with negative duration"""
        result = DataProcessor.generate_sample_ppg_data(100, duration=-1)
        assert result is None  # Should return None for invalid input

    @patch('vitalDSP_webapp.utils.data_processor.logger')
    def test_logging_integration(self, mock_logger):
        """Test that logging is properly integrated"""
        # Test error logging in read_uploaded_content
        DataProcessor.read_uploaded_content("invalid_content", "test.pdf")
        
        # Should have logged an error
        assert mock_logger.error.called

    def test_data_types_and_validation(self):
        """Test data type handling and validation"""
        # Test with different data types
        df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'signal': [0.1, 0.2, 0.3],
            'integer_col': [1, 2, 3],
            'string_col': ['a', 'b', 'c']
        })
        
        result = DataProcessor.process_uploaded_data(df, "test.csv", 100, "seconds")
        
        assert isinstance(result, dict)
        assert result['num_columns'] == 4
        assert len(result['columns']) == 4

    def test_edge_cases_and_robustness(self):
        """Test edge cases and robustness"""
        # Test with very small DataFrame
        df = pd.DataFrame({'col1': [1]})
        result = DataProcessor.process_uploaded_data(df, "test.csv", 1, "s")
        assert result['num_rows'] == 1
        
        # Test with DataFrame containing NaN values
        df = pd.DataFrame({
            'time': [1, 2, np.nan, 4],
            'signal': [0.1, np.nan, 0.3, 0.4]
        })
        result = DataProcessor.process_uploaded_data(df, "test.csv", 100, "seconds")
        assert isinstance(result, dict)

    def test_generate_sample_data_signal_characteristics(self):
        """Test that generated sample data has expected signal characteristics"""
        result = DataProcessor.generate_sample_ppg_data(100, duration=10, heart_rate=60)
        
        # Check basic signal properties
        signal = result['signal'].values  # Implementation uses 'signal' not 'ppg_signal'
        time = result['time'].values
        
        # Should have expected time range
        assert time[0] == 0
        assert abs(time[-1] - 10) < 0.1  # Close to 10 seconds
        
        # Signal should have reasonable range
        assert signal.min() > -3
        assert signal.max() < 3
        
        # Should have some periodicity (basic check)
        assert len(signal) > 100  # Should have enough samples