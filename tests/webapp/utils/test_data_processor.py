import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
import base64
import io

# Add the src directory to the Python path so tests can import vitalDSP_webapp modules
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from vitalDSP_webapp.utils.data_processor import DataProcessor


class TestDataProcessor:
    """Test class for DataProcessor utility functions."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.processor = DataProcessor()
        
        # Create test data
        self.test_df = pd.DataFrame({
            'time': np.arange(100),
            'signal': np.random.randn(100),
            'red': np.random.randn(100),
            'ir': np.random.randn(100)
        })

    def test_validate_file_extension_valid(self):
        """Test file extension validation with valid extensions."""
        valid_extensions = [
            'data.csv',
            'signal.xlsx',
            'ppg.xls',
            'ecg.txt',
            'vital.dat'
        ]
        
        for filename in valid_extensions:
            assert DataProcessor.validate_file_extension(filename) is True

    def test_validate_file_extension_invalid(self):
        """Test file extension validation with invalid extensions."""
        invalid_extensions = [
            'data.pdf',
            'signal.doc',
            'ppg.exe',
            'ecg.bat',
            'vital.sh'
        ]
        
        for filename in invalid_extensions:
            assert DataProcessor.validate_file_extension(filename) is False

    def test_validate_file_extension_no_extension(self):
        """Test file extension validation with no extension."""
        assert DataProcessor.validate_file_extension('filename') is False

    def test_validate_file_extension_case_sensitive(self):
        """Test file extension validation is case insensitive."""
        assert DataProcessor.validate_file_extension('data.CSV') is True
        assert DataProcessor.validate_file_extension('signal.XLSX') is True

    @patch('pandas.read_csv')
    def test_read_file_csv(self, mock_read_csv):
        """Test reading CSV file."""
        mock_read_csv.return_value = self.test_df
        
        result = DataProcessor.read_file('test.csv', 'test.csv')
        
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.test_df)

    @patch('pandas.read_excel')
    def test_read_file_excel(self, mock_read_excel):
        """Test reading Excel file."""
        mock_read_excel.return_value = self.test_df
        
        result = DataProcessor.read_file('test.xlsx', 'test.xlsx')
        
        mock_read_excel.assert_called_once()
        pd.testing.assert_frame_equal(result, self.test_df)

    @patch('pandas.read_csv')
    def test_read_file_txt(self, mock_read_csv):
        """Test reading text file."""
        mock_read_csv.return_value = self.test_df
        
        result = DataProcessor.read_file('test.txt', 'test.txt')
        
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.test_df)

    @patch('pandas.read_csv')
    def test_read_file_dat(self, mock_read_csv):
        """Test reading DAT file."""
        mock_read_csv.return_value = self.test_df
        
        result = DataProcessor.read_file('test.dat', 'test.dat')
        
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.test_df)

    @patch('pandas.read_csv')
    def test_read_file_unknown_extension(self, mock_read_csv):
        """Test reading file with unknown extension."""
        mock_read_csv.return_value = self.test_df
        
        result = DataProcessor.read_file('test.unknown', 'test.unknown')
        
        # Should fall back to CSV reading
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.test_df)

    @patch('pandas.read_csv')
    def test_read_file_error_handling(self, mock_read_csv):
        """Test error handling when reading file fails."""
        mock_read_csv.side_effect = Exception("File read error")
        
        result = DataProcessor.read_file('test.csv', 'test.csv')
        
        assert result is None

    def test_read_uploaded_content_csv(self):
        """Test reading uploaded CSV content."""
        # Create CSV content
        csv_content = "time,signal,red,ir\n0,1.0,0.5,0.3\n1,1.1,0.6,0.4"
        encoded_content = base64.b64encode(csv_content.encode()).decode()
        
        result = DataProcessor.read_uploaded_content(encoded_content, 'test.csv')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['time', 'signal', 'red', 'ir']
        assert len(result) == 2

    def test_read_uploaded_content_excel(self):
        """Test reading uploaded Excel content."""
        # Create Excel content (simplified)
        excel_content = b"dummy excel content"
        encoded_content = base64.b64encode(excel_content).decode()
        
        with patch('pandas.read_excel') as mock_read_excel:
            mock_read_excel.return_value = self.test_df
            
            result = DataProcessor.read_uploaded_content(encoded_content, 'test.xlsx')
            
            assert result is not None
            pd.testing.assert_frame_equal(result, self.test_df)

    def test_read_uploaded_content_invalid_base64(self):
        """Test reading uploaded content with invalid base64."""
        invalid_content = "invalid base64 content"
        
        result = DataProcessor.read_uploaded_content(invalid_content, 'test.csv')
        
        assert result is None

    def test_read_uploaded_content_read_error(self):
        """Test reading uploaded content when read operation fails."""
        csv_content = "time,signal\n0,1.0"
        encoded_content = base64.b64encode(csv_content.encode()).decode()
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = Exception("Read error")
            
            result = DataProcessor.read_uploaded_content(encoded_content, 'test.csv')
            
            assert result is None

    def test_generate_sample_ppg_data(self):
        """Test generating sample PPG data."""
        sampling_freq = 1000
        duration = 10  # seconds
        
        result = DataProcessor.generate_sample_ppg_data(sampling_freq, duration)
        
        assert isinstance(result, pd.DataFrame)
        assert 'time' in result.columns
        assert 'ppg_signal' in result.columns
        assert len(result) == sampling_freq * duration
        
        # Check time column
        expected_time = np.arange(0, duration, 1/sampling_freq)
        np.testing.assert_array_almost_equal(result['time'].values, expected_time, decimal=3)
        
        # Check signal column
        assert result['ppg_signal'].dtype in [np.float64, np.float32]
        assert not result['ppg_signal'].isna().any()

    def test_generate_sample_ppg_data_default_duration(self):
        """Test generating sample PPG data with default duration."""
        sampling_freq = 500
        
        result = DataProcessor.generate_sample_ppg_data(sampling_freq)
        
        assert len(result) == sampling_freq * 5  # Default 5 seconds

    def test_process_uploaded_data(self):
        """Test processing uploaded data."""
        filename = 'test.csv'
        sampling_freq = 1000
        time_unit = 'ms'
        
        result = DataProcessor.process_uploaded_data(self.test_df, filename, sampling_freq, time_unit)
        
        assert isinstance(result, dict)
        assert 'filename' in result
        assert 'rows' in result
        assert 'columns' in result
        assert 'size_mb' in result
        assert 'duration_sec' in result
        assert 'sampling_rate' in result
        assert 'min_value' in result
        assert 'max_value' in result
        assert 'snr_db' in result
        assert 'artifact_count' in result
        assert 'quality_status' in result
        
        # Check specific values
        assert result['filename'] == filename
        assert result['rows'] == 100
        assert result['columns'] == 4
        assert result['sampling_rate'] == sampling_freq
        assert result['duration_sec'] == 0.1  # 100 samples / 1000 Hz

    def test_process_uploaded_data_with_metadata(self):
        """Test processing uploaded data with additional metadata."""
        filename = 'test.csv'
        sampling_freq = 500
        time_unit = 's'
        
        # Add metadata to DataFrame
        df_with_metadata = self.test_df.copy()
        df_with_metadata.attrs['sampling_freq'] = sampling_freq
        df_with_metadata.attrs['time_unit'] = time_unit
        
        result = DataProcessor.process_uploaded_data(df_with_metadata, filename, sampling_freq, time_unit)
        
        assert result['sampling_rate'] == sampling_freq
        assert result['duration_sec'] == 0.2  # 100 samples / 500 Hz

    def test_auto_detect_columns(self):
        """Test automatic column detection."""
        # Create DataFrame with common column names
        df = pd.DataFrame({
            'Time': np.arange(100),
            'PPG_Signal': np.random.randn(100),
            'RED_Channel': np.random.randn(100),
            'IR_Channel': np.random.randn(100),
            'Quality': np.random.randn(100)
        })
        
        result = DataProcessor.auto_detect_columns(df)
        
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'signal' in result
        assert 'red' in result
        assert 'ir' in result
        
        # Check that columns were detected correctly
        assert result['time'] == 'Time'
        assert result['signal'] == 'PPG_Signal'
        assert result['red'] == 'RED_Channel'
        assert result['ir'] == 'IR_Channel'

    def test_auto_detect_columns_partial_match(self):
        """Test automatic column detection with partial matches."""
        # Create DataFrame with partial matches
        df = pd.DataFrame({
            'timestamp': np.arange(100),
            'signal_data': np.random.randn(100),
            'red_signal': np.random.randn(100),
            'ir_data': np.random.randn(100)
        })
        
        result = DataProcessor.auto_detect_columns(df)
        
        # Should still detect some columns
        assert result['time'] == 'timestamp'
        assert result['signal'] == 'signal_data'
        assert result['red'] == 'red_signal'
        assert result['ir'] == 'ir_data'

    def test_auto_detect_columns_no_matches(self):
        """Test automatic column detection with no matches."""
        # Create DataFrame with no matching column names
        df = pd.DataFrame({
            'col1': np.arange(100),
            'col2': np.random.randn(100),
            'col3': np.random.randn(100),
            'col4': np.random.randn(100)
        })
        
        result = DataProcessor.auto_detect_columns(df)
        
        # Should return first few columns as defaults
        assert result['time'] == 'col1'
        assert result['signal'] == 'col2'
        assert result['red'] == 'col3'
        assert result['ir'] == 'col4'

    def test_validate_column_mapping_valid(self):
        """Test column mapping validation with valid mapping."""
        column_mapping = {
            'time': 'time',
            'signal': 'signal',
            'red': 'red',
            'ir': 'ir'
        }
        
        result = DataProcessor.validate_column_mapping(self.test_df, column_mapping)
        
        assert result['is_valid'] is True
        assert 'errors' in result
        assert len(result['errors']) == 0

    def test_validate_column_mapping_invalid_columns(self):
        """Test column mapping validation with invalid columns."""
        column_mapping = {
            'time': 'nonexistent_col',
            'signal': 'signal',
            'red': 'red',
            'ir': 'ir'
        }
        
        result = DataProcessor.validate_column_mapping(self.test_df, column_mapping)
        
        assert result['is_valid'] is False
        assert 'errors' in result
        assert len(result['errors']) > 0
        assert any('nonexistent_col' in error for error in result['errors'])

    def test_validate_column_mapping_missing_required(self):
        """Test column mapping validation with missing required columns."""
        column_mapping = {
            'time': 'time',
            'signal': None,  # Missing required column
            'red': 'red',
            'ir': 'ir'
        }
        
        result = DataProcessor.validate_column_mapping(self.test_df, column_mapping)
        
        assert result['is_valid'] is False
        assert 'errors' in result
        assert len(result['errors']) > 0

    def test_validate_column_mapping_duplicate_columns(self):
        """Test column mapping validation with duplicate columns."""
        column_mapping = {
            'time': 'time',
            'signal': 'signal',
            'red': 'signal',  # Same as signal
            'ir': 'ir'
        }
        
        result = DataProcessor.validate_column_mapping(self.test_df, column_mapping)
        
        assert result['is_valid'] is False
        assert 'errors' in result
        assert len(result['errors']) > 0

    def test_calculate_signal_quality(self):
        """Test signal quality calculation."""
        # Create signal with known characteristics
        signal = np.random.randn(1000) + 10  # Signal with noise
        
        quality_metrics = DataProcessor.calculate_signal_quality(signal)
        
        assert isinstance(quality_metrics, dict)
        assert 'snr_db' in quality_metrics
        assert 'artifact_count' in quality_metrics
        assert 'quality_status' in quality_metrics
        
        # Check that SNR is reasonable
        assert quality_metrics['snr_db'] > 0
        assert isinstance(quality_metrics['artifact_count'], int)
        assert quality_metrics['quality_status'] in ['Poor', 'Fair', 'Good', 'Excellent']

    def test_calculate_signal_quality_constant_signal(self):
        """Test signal quality calculation with constant signal."""
        signal = np.ones(100)  # Constant signal
        
        quality_metrics = DataProcessor.calculate_signal_quality(signal)
        
        # Constant signal should have poor quality
        assert quality_metrics['quality_status'] == 'Poor'

    def test_calculate_signal_quality_noisy_signal(self):
        """Test signal quality calculation with very noisy signal."""
        signal = np.random.randn(1000) * 10  # Very noisy
        
        quality_metrics = DataProcessor.calculate_signal_quality(signal)
        
        # Noisy signal should have poor quality
        assert quality_metrics['quality_status'] in ['Poor', 'Fair']

    def test_estimate_sampling_frequency(self):
        """Test sampling frequency estimation."""
        # Create time series with known sampling frequency
        time = np.arange(0, 10, 0.001)  # 1000 Hz
        signal = np.sin(2 * np.pi * 10 * time)  # 10 Hz signal
        
        df = pd.DataFrame({'time': time, 'signal': signal})
        
        estimated_freq = DataProcessor.estimate_sampling_frequency(df, 'time')
        
        assert abs(estimated_freq - 1000) < 10  # Within 10 Hz tolerance

    def test_estimate_sampling_frequency_irregular(self):
        """Test sampling frequency estimation with irregular time intervals."""
        # Create irregular time series
        time = np.cumsum(np.random.exponential(0.001, 1000))  # Irregular intervals
        signal = np.random.randn(1000)
        
        df = pd.DataFrame({'time': time, 'signal': signal})
        
        estimated_freq = DataProcessor.estimate_sampling_frequency(df, 'time')
        
        # Should still return a reasonable estimate
        assert estimated_freq > 0
        assert estimated_freq < 10000  # Reasonable upper bound

    def test_clean_column_names(self):
        """Test column name cleaning."""
        dirty_names = ['  Time  ', 'PPG_Signal!', 'RED-Channel', 'IR Channel ']
        
        clean_names = DataProcessor.clean_column_names(dirty_names)
        
        expected_names = ['Time', 'PPG_Signal', 'RED-Channel', 'IR_Channel']
        
        assert clean_names == expected_names

    def test_clean_column_names_empty(self):
        """Test column name cleaning with empty list."""
        clean_names = DataProcessor.clean_column_names([])
        assert clean_names == []

    def test_clean_column_names_special_chars(self):
        """Test column name cleaning with special characters."""
        dirty_names = ['Col@#$%', 'Signal&*()', 'Data+=-[]']
        
        clean_names = DataProcessor.clean_column_names(dirty_names)
        
        # Should remove or replace special characters
        assert all('@' not in name for name in clean_names)
        assert all('#' not in name for name in clean_names)
        assert all('$' not in name for name in clean_names)
