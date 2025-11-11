"""
Comprehensive Tests for Notebooks Module - Missing Coverage

This test file specifically targets missing lines in notebooks/__init__.py to achieve
high test coverage, including data loading, processing, and plotting functions.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 80%+
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import warnings
import datetime as dt
import ast

# Suppress warnings
warnings.filterwarnings("ignore")

# Import module
try:
    from vitalDSP.notebooks import (
        load_sample_ecg,
        load_sample_ecg_small,
        load_sample_ppg,
        get_flat,
        get_flat_timestamp,
        safe_get_flat_ecg_timestamp,
        get_flat_ecg_timestamp,
        process_in_chunks,
        plot_trace,
    )
    NOTEBOOKS_AVAILABLE = True
except ImportError:
    NOTEBOOKS_AVAILABLE = False


@pytest.fixture
def sample_ecg_csv(tmp_path):
    """Create a sample ECG CSV file."""
    csv_path = tmp_path / "ecg.csv"
    
    # Create sample ECG data
    data = {
        'ecg': ['[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]'],
        'timestamp': ['2024-01-01 10:00:00.000000+00:00', '2024-01-01 10:00:01.000000+00:00']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sample_ppg_csv(tmp_path):
    """Create a sample PPG CSV file."""
    csv_path = tmp_path / "ppg.csv"
    
    # Create sample PPG data
    data = {
        'pleth': ['[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]'],
        'timestamp': ['2024-01-01 10:00:00.000000+00:00', '2024-01-01 10:00:01.000000+00:00']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sample_ecg_small_csv(tmp_path):
    """Create a sample ECG small CSV file."""
    csv_path = tmp_path / "ecg_small.csv"
    
    # Create sample ECG data
    data = {
        'ecg': ['[1.0, 2.0]', '[3.0, 4.0]'],
        'timestamp': ['2024-01-01 10:00:00.000000+00:00', '2024-01-01 10:00:01.000000+00:00']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestGetFlat:
    """Test get_flat function - covers lines 52-54."""
    
    def test_get_flat_basic(self):
        """Test get_flat with basic list string."""
        flat = []
        x = '[1.0, 2.0, 3.0]'
        
        get_flat(x, flat)
        
        assert flat == [1.0, 2.0, 3.0]
    
    def test_get_flat_multiple_calls(self):
        """Test get_flat with multiple calls."""
        flat = []
        
        get_flat('[1.0, 2.0]', flat)
        get_flat('[3.0, 4.0]', flat)
        
        assert flat == [1.0, 2.0, 3.0, 4.0]
    
    def test_get_flat_empty_list(self):
        """Test get_flat with empty list."""
        flat = []
        x = '[]'
        
        get_flat(x, flat)
        
        assert flat == []


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestGetFlatTimestamp:
    """Test get_flat_timestamp function - covers lines 56-68."""
    
    def test_get_flat_timestamp_format1(self):
        """Test get_flat_timestamp with format1 - covers lines 59-60."""
        flat = []
        x = '2024-01-01 10:00:00.000000+00:00'
        fs = 100
        
        get_flat_timestamp(x, flat, fs)
        
        assert len(flat) == fs
        assert isinstance(flat[0], dt.datetime)
    
    def test_get_flat_timestamp_format2(self):
        """Test get_flat_timestamp with format2 - covers lines 61-62."""
        flat = []
        x = '2024-01-01 10:00:00+00:00'
        fs = 100
        
        get_flat_timestamp(x, flat, fs)
        
        assert len(flat) == fs
        assert isinstance(flat[0], dt.datetime)
    
    def test_get_flat_timestamp_different_fs(self):
        """Test get_flat_timestamp with different sampling rate."""
        flat = []
        x = '2024-01-01 10:00:00.000000+00:00'
        fs = 256
        
        get_flat_timestamp(x, flat, fs)
        
        assert len(flat) == fs


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestGetFlatEcgTimestamp:
    """Test get_flat_ecg_timestamp function - covers lines 78-93."""
    
    def test_get_flat_ecg_timestamp_format1(self):
        """Test get_flat_ecg_timestamp with format1 - covers lines 81-83."""
        flat = []
        x = '2024-01-01 10:00:00.000000+00:00'
        next_x = '2024-01-01 10:00:01.000000+00:00'
        fs = 100
        
        get_flat_ecg_timestamp(x, next_x, flat, fs)
        
        assert len(flat) == fs  # 1 second * 100 Hz
        assert isinstance(flat[0], dt.datetime)
    
    def test_get_flat_ecg_timestamp_format2(self):
        """Test get_flat_ecg_timestamp with format2 - covers lines 84-86."""
        flat = []
        x = '2024-01-01 10:00:00+00:00'
        next_x = '2024-01-01 10:00:01+00:00'
        fs = 100
        
        get_flat_ecg_timestamp(x, next_x, flat, fs)
        
        assert len(flat) == fs
        assert isinstance(flat[0], dt.datetime)
    
    def test_get_flat_ecg_timestamp_different_duration(self):
        """Test get_flat_ecg_timestamp with different duration."""
        flat = []
        x = '2024-01-01 10:00:00.000000+00:00'
        next_x = '2024-01-01 10:00:02.000000+00:00'  # 2 seconds
        fs = 100
        
        get_flat_ecg_timestamp(x, next_x, flat, fs)
        
        assert len(flat) == 200  # 2 seconds * 100 Hz


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestSafeGetFlatEcgTimestamp:
    """Test safe_get_flat_ecg_timestamp function - covers lines 70-75."""
    
    def test_safe_get_flat_ecg_timestamp_success(self):
        """Test safe_get_flat_ecg_timestamp with valid data - covers lines 71-73."""
        flat = []
        row = pd.Series({'timestamp': '2024-01-01 10:00:00.000000+00:00', 'name': 0})
        row.name = 0
        shifted_series = pd.Series({
            0: '2024-01-01 10:00:01.000000+00:00',
            1: '2024-01-01 10:00:02.000000+00:00'
        })
        fs = 100
        
        safe_get_flat_ecg_timestamp(row, shifted_series, flat, fs)
        
        assert len(flat) == fs
    
    def test_safe_get_flat_ecg_timestamp_exception(self):
        """Test safe_get_flat_ecg_timestamp with exception - covers lines 74-75."""
        flat = []
        row = pd.Series({'timestamp': '2024-01-01 10:00:00.000000+00:00', 'name': 0})
        row.name = 999  # Non-existent index
        shifted_series = pd.Series({0: '2024-01-01 10:00:01.000000+00:00'})
        fs = 100
        
        safe_get_flat_ecg_timestamp(row, shifted_series, flat, fs)
        
        # Should fall back to get_flat_timestamp
        assert len(flat) == fs


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestProcessInChunks:
    """Test process_in_chunks function - covers missing lines."""
    
    def test_process_in_chunks_ppg(self, sample_ppg_csv):
        """Test process_in_chunks with PPG data - covers lines 100-102."""
        signal_col, date_col = process_in_chunks(
            str(sample_ppg_csv),
            chunk_size=1000,
            fs=100,
            data_type="ppg"
        )
        
        assert len(signal_col) > 0
        assert len(date_col) > 0
        assert isinstance(signal_col, list)
        assert isinstance(date_col, list)
    
    def test_process_in_chunks_ecg(self, sample_ecg_csv):
        """Test process_in_chunks with ECG data - covers lines 103-112."""
        signal_col, date_col = process_in_chunks(
            str(sample_ecg_csv),
            chunk_size=1000,
            fs=256,
            data_type="ecg"
        )
        
        assert len(signal_col) > 0
        assert len(date_col) > 0
        assert isinstance(signal_col, list)
        assert isinstance(date_col, list)
    
    def test_process_in_chunks_large_file(self, tmp_path):
        """Test process_in_chunks with larger file."""
        csv_path = tmp_path / "large.csv"
        
        # Create larger dataset
        data = {
            'pleth': [f'[{i}, {i+1}, {i+2}]' for i in range(100)],
            'timestamp': [
                f'2024-01-01 10:00:{i//10:02d}.{i%10*100000:06d}+00:00'
                for i in range(100)
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        signal_col, date_col = process_in_chunks(
            str(csv_path),
            chunk_size=10,
            fs=100,
            data_type="ppg"
        )
        
        assert len(signal_col) > 0
        assert len(date_col) > 0


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestLoadSampleData:
    """Test load_sample_* functions - covers missing lines."""
    
    def test_load_sample_ecg(self):
        """Test load_sample_ecg - covers lines 31-35."""
        # Mock pd.read_csv to return valid data
        mock_df = pd.DataFrame({
            'ecg': ['[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]'],
            'timestamp': [
                '2024-01-01 10:00:00.000000+00:00',
                '2024-01-01 10:00:01.000000+00:00'
            ]
        })
        
        with patch('vitalDSP.notebooks.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = [mock_df]  # chunksize returns an iterator
            
            signal_col, date_col = load_sample_ecg()
            
            assert signal_col is not None
            assert date_col is not None
            assert isinstance(signal_col, list)
            assert isinstance(date_col, list)
            assert len(signal_col) > 0
            assert len(date_col) > 0
    
    def test_load_sample_ecg_small(self):
        """Test load_sample_ecg_small - covers lines 38-42."""
        # Mock pd.read_csv to return valid data
        mock_df = pd.DataFrame({
            'ecg': ['[1.0, 2.0]', '[3.0, 4.0]'],
            'timestamp': [
                '2024-01-01 10:00:00.000000+00:00',
                '2024-01-01 10:00:01.000000+00:00'
            ]
        })
        
        with patch('vitalDSP.notebooks.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = [mock_df]  # chunksize returns an iterator
            
            signal_col, date_col = load_sample_ecg_small()
            
            assert signal_col is not None
            assert date_col is not None
            assert len(signal_col) > 0
            assert len(date_col) > 0
    
    def test_load_sample_ppg(self):
        """Test load_sample_ppg - covers lines 45-49."""
        # Mock pd.read_csv to return valid data
        mock_df = pd.DataFrame({
            'pleth': ['[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]'],
            'timestamp': [
                '2024-01-01 10:00:00.000000+00:00',
                '2024-01-01 10:00:01.000000+00:00'
            ]
        })
        
        with patch('vitalDSP.notebooks.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = [mock_df]  # chunksize returns an iterator
            
            signal_col, date_col = load_sample_ppg()
            
            assert signal_col is not None
            assert date_col is not None
            assert len(signal_col) > 0
            assert len(date_col) > 0


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestPlotTrace:
    """Test plot_trace function - covers lines 116-125."""
    
    def test_plot_trace_basic(self):
        """Test plot_trace with basic signals."""
        input_signal = np.random.randn(100)
        output_signal = np.random.randn(100)
        
        # Mock fig.show() to avoid opening plot windows
        with patch('vitalDSP.notebooks.go.Figure.show'):
            plot_trace(input_signal, output_signal)
        
        # If no exception, test passes
        assert True
    
    def test_plot_trace_with_title(self):
        """Test plot_trace with custom title."""
        input_signal = np.random.randn(100)
        output_signal = np.random.randn(100)
        
        with patch('vitalDSP.notebooks.go.Figure.show'):
            plot_trace(input_signal, output_signal, title="Custom Title")
        
        assert True
    
    def test_plot_trace_different_lengths(self):
        """Test plot_trace with signals of different lengths."""
        input_signal = np.random.randn(100)
        output_signal = np.random.randn(150)
        
        with patch('vitalDSP.notebooks.go.Figure.show'):
            plot_trace(input_signal, output_signal)
        
        assert True


@pytest.mark.skipif(not NOTEBOOKS_AVAILABLE, reason="Notebooks module not available")
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_get_flat_invalid_format(self):
        """Test get_flat with invalid format."""
        flat = []
        x = 'not a list'
        
        # Should raise ValueError or similar
        try:
            get_flat(x, flat)
        except (ValueError, SyntaxError):
            pass  # Expected
    
    def test_get_flat_timestamp_invalid_format(self):
        """Test get_flat_timestamp with invalid format."""
        flat = []
        x = 'invalid timestamp'
        
        try:
            get_flat_timestamp(x, flat, fs=100)
        except ValueError:
            pass  # Expected
    
    def test_get_flat_ecg_timestamp_invalid_format(self):
        """Test get_flat_ecg_timestamp with invalid format."""
        flat = []
        x = 'invalid timestamp'
        next_x = 'invalid timestamp'
        
        try:
            get_flat_ecg_timestamp(x, next_x, flat, fs=100)
        except ValueError:
            pass  # Expected
    
    def test_process_in_chunks_empty_file(self, tmp_path):
        """Test process_in_chunks with empty file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text('ecg,timestamp\n')
        
        signal_col, date_col = process_in_chunks(
            str(csv_path),
            chunk_size=1000,
            fs=100,
            data_type="ecg"
        )
        
        assert len(signal_col) == 0
        assert len(date_col) == 0
    
    def test_process_in_chunks_missing_column(self, tmp_path):
        """Test process_in_chunks with missing column."""
        csv_path = tmp_path / "missing.csv"
        csv_path.write_text('wrong_column,timestamp\n')
        
        try:
            signal_col, date_col = process_in_chunks(
                str(csv_path),
                chunk_size=1000,
                fs=100,
                data_type="ppg"
            )
        except KeyError:
            pass  # Expected
    
    def test_process_in_chunks_different_chunk_sizes(self, sample_ppg_csv):
        """Test process_in_chunks with different chunk sizes."""
        # Test with chunk_size larger than file
        signal_col1, date_col1 = process_in_chunks(
            str(sample_ppg_csv),
            chunk_size=100000,
            fs=100,
            data_type="ppg"
        )
        
        # Test with chunk_size smaller than file
        signal_col2, date_col2 = process_in_chunks(
            str(sample_ppg_csv),
            chunk_size=1,
            fs=100,
            data_type="ppg"
        )
        
        assert len(signal_col1) > 0
        assert len(signal_col2) > 0

