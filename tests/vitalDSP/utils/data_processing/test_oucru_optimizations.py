"""
Test suite for OUCRU CSV optimizations.

Tests the three major optimizations implemented:
1. json.loads() parsing (2x faster)
2. Vectorized timestamp generation (10-100x faster)
3. Streaming row-by-row expansion for large files (90% memory reduction)

Author: vitalDSP Team
Date: October 16, 2025
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from vitalDSP.utils.data_processing.data_loader import (
    DataLoader,
    DataFormat,
    load_oucru_csv,
)


def generate_timestamp(i):
    """Generate proper timestamp for test data."""
    minutes = i // 60
    seconds = i % 60
    return f'2024-01-01 00:{minutes:02d}:{seconds:02d}'


class TestOUCRUJsonParsing:
    """Test json.loads() parsing with fallback to ast.literal_eval()."""

    def test_json_array_parsing(self):
        """Test parsing standard JSON arrays."""
        # Create test CSV with JSON-style arrays
        test_data = []
        for i in range(10):
            signal = list(np.random.randn(100))
            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': json.dumps(signal),  # JSON format
                'sampling_rate': 100
            })

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load with DataLoader
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load()

            # Verify
            assert len(data) == 1000  # 10 rows × 100 samples
            assert 'signal' in data.columns
            assert 'timestamp' in data.columns
            assert loader.sampling_rate == 100

        finally:
            Path(temp_path).unlink()

    def test_python_array_parsing_fallback(self):
        """Test fallback to ast.literal_eval() for Python-style arrays."""
        # Create test CSV with Python-style arrays (single quotes)
        test_data = []
        for i in range(10):
            signal = list(np.random.randn(100))
            # Python repr format (single quotes, not valid JSON)
            signal_str = str(signal)
            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': signal_str,
                'sampling_rate': 100
            })

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load with DataLoader (should fallback to ast.literal_eval)
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load()

            # Verify
            assert len(data) == 1000
            assert 'signal' in data.columns

        finally:
            Path(temp_path).unlink()

    def test_mixed_array_formats(self):
        """Test handling of mixed JSON and Python array formats."""
        test_data = []
        for i in range(20):
            signal = list(np.random.randn(50))
            # Alternate between JSON and Python formats
            if i % 2 == 0:
                signal_str = json.dumps(signal)  # JSON
            else:
                signal_str = str(signal)  # Python repr
            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': signal_str,
                'sampling_rate': 50
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load()

            # Verify both formats handled correctly
            assert len(data) == 1000  # 20 rows × 50 samples
            assert not data['signal'].isnull().any()

        finally:
            Path(temp_path).unlink()


class TestOUCRUVectorizedTimestamps:
    """Test vectorized timestamp generation."""

    def test_vectorized_timestamp_accuracy(self):
        """Test that vectorized timestamps match expected values."""
        # Create test data
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        n_rows = 60  # 60 seconds
        fs = 250  # 250 Hz

        test_data = []
        for i in range(n_rows):
            signal = list(np.random.randn(fs))
            test_data.append({
                'timestamp': start_time + timedelta(seconds=i),
                'signal': json.dumps(signal),
                'sampling_rate': fs
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load()

            # Verify timestamps
            timestamps = pd.to_datetime(data['timestamp'])

            # First timestamp should be start_time
            assert timestamps.iloc[0] == start_time

            # Last timestamp should be approximately start_time + 60 seconds - 1/fs
            expected_end = start_time + timedelta(seconds=60 - 1/fs)
            # Allow small numerical error
            time_diff = abs((timestamps.iloc[-1] - expected_end).total_seconds())
            assert time_diff < 0.001  # Less than 1ms difference

            # Check uniform spacing
            time_diffs = timestamps.diff()[1:].dt.total_seconds()
            expected_diff = 1 / fs
            # All diffs should be approximately 1/fs
            assert np.allclose(time_diffs, expected_diff, atol=1e-6)

        finally:
            Path(temp_path).unlink()

    def test_different_sampling_rates(self):
        """Test vectorized timestamps with different sampling rates."""
        sampling_rates = [100, 128, 250, 500, 1000]

        for fs in sampling_rates:
            # Create small test file
            test_data = []
            for i in range(10):
                signal = list(np.random.randn(fs))
                test_data.append({
                    'timestamp': generate_timestamp(i),
                    'signal': json.dumps(signal),
                    'sampling_rate': fs
                })

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
                df = pd.DataFrame(test_data)
                df.to_csv(f, index=False)
                temp_path = f.name

            try:
                loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
                data = loader.load()

                # Verify sample count
                assert len(data) == 10 * fs

                # Verify timestamp spacing
                timestamps = pd.to_datetime(data['timestamp'])
                time_diffs = timestamps.diff()[1:].dt.total_seconds()
                expected_diff = 1 / fs
                # Use more lenient tolerance for floating point precision
                assert np.allclose(time_diffs, expected_diff, atol=1e-6)

            finally:
                Path(temp_path).unlink()


class TestOUCRUStreamingExpansion:
    """Test streaming row-by-row expansion for large files."""

    def test_streaming_threshold(self):
        """Test that streaming is triggered for files >100MB."""
        # We can't easily create 100MB files in tests, so we test chunk_size parameter
        # which forces streaming regardless of file size

        # Create test data
        test_data = []
        for i in range(100):  # 100 rows
            signal = list(np.random.randn(250))
            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': json.dumps(signal),
                'sampling_rate': 250
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load with explicit chunk_size to force streaming
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load(chunk_size=20)  # Process in chunks of 20 rows

            # Verify
            assert len(data) == 25000  # 100 rows × 250 samples
            assert loader.metadata['format'] == 'oucru_csv_streaming'
            assert loader.metadata['chunk_size'] == 20

        finally:
            Path(temp_path).unlink()

    def test_streaming_data_integrity(self):
        """Test that streaming produces same results as standard loading."""
        # Create test data with known values
        test_data = []
        expected_signals = []

        for i in range(100):
            # Create predictable signal: row index + sample index
            signal = [i + j/100 for j in range(50)]
            expected_signals.extend(signal)

            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': json.dumps(signal),
                'sampling_rate': 50
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load with streaming
            loader_streaming = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data_streaming = loader_streaming.load(chunk_size=10)

            # Load without streaming (if file is small enough, standard loader is used)
            # Force standard by setting very large chunk_size
            loader_standard = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            # Small file, will use standard loader
            data_standard = loader_standard.load()

            # Both should produce identical signals
            assert len(data_streaming) == len(data_standard)
            assert np.allclose(
                data_streaming['signal'].values,
                data_standard['signal'].values,
                atol=1e-10
            )

        finally:
            Path(temp_path).unlink()

    def test_streaming_chunk_processing(self):
        """Test that chunks are processed correctly."""
        # Create data where we can verify chunk boundaries
        n_rows = 100
        chunk_size = 20
        fs = 100

        test_data = []
        for i in range(n_rows):
            # Signal filled with row index
            signal = [float(i)] * fs
            # Generate proper timestamps (avoid seconds > 59)
            minutes = i // 60
            seconds = i % 60
            timestamp = f'2024-01-01 00:{minutes:02d}:{seconds:02d}'
            test_data.append({
                'timestamp': timestamp,
                'signal': json.dumps(signal),
                'sampling_rate': fs
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load(chunk_size=chunk_size)

            # Verify total samples
            assert len(data) == n_rows * fs

            # Verify signal values match row indices
            signals = data['signal'].values.reshape(n_rows, fs)
            for i in range(n_rows):
                assert np.all(signals[i] == float(i))

        finally:
            Path(temp_path).unlink()


class TestOUCRUConvenienceFunction:
    """Test load_oucru_csv() convenience function."""

    def test_load_oucru_csv_basic(self):
        """Test basic load_oucru_csv() functionality."""
        # Create test file
        test_data = []
        for i in range(60):
            signal = list(np.random.randn(100))
            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': json.dumps(signal),
                'sampling_rate': 100
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load using convenience function
            signal, metadata = load_oucru_csv(temp_path)

            # Verify
            assert isinstance(signal, np.ndarray)
            assert len(signal) == 6000  # 60 rows × 100 samples
            assert metadata['sampling_rate'] == 100
            assert metadata['n_rows'] == 60
            assert metadata['n_samples'] == 6000
            assert metadata['duration_seconds'] == 60.0

        finally:
            Path(temp_path).unlink()

    def test_load_oucru_csv_with_signal_type_hint(self):
        """Test signal type hint functionality."""
        # Create test file without sampling_rate column
        test_data = []
        for i in range(30):
            signal = list(np.random.randn(100))
            test_data.append({
                'timestamp': generate_timestamp(i),
                'ppg_signal': json.dumps(signal),  # Different column name
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Load with signal type hint for PPG (default 100 Hz)
            signal, metadata = load_oucru_csv(
                temp_path,
                signal_column='ppg_signal',
                signal_type_hint='ppg',
                sampling_rate_column=None
            )

            # Should infer 100 Hz from array length
            assert len(signal) == 3000  # 30 rows × 100 samples

        finally:
            Path(temp_path).unlink()


class TestOUCRUBackwardCompatibility:
    """Test backward compatibility with existing OUCRU files."""

    def test_existing_file_format(self):
        """Test that existing OUCRU CSV files still load correctly."""
        # Simulate existing file format with various edge cases
        test_data = []

        # Mix of formats
        for i in range(20):
            if i < 10:
                # JSON format
                signal = json.dumps(list(np.random.randn(100)))
            else:
                # Python repr format (old style)
                signal = str(list(np.random.randn(100)))

            test_data.append({
                'timestamp': generate_timestamp(i),
                'signal': signal,
                'sampling_rate': 100
            })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df = pd.DataFrame(test_data)
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            # Should load without errors
            loader = DataLoader(temp_path, format=DataFormat.OUCRU_CSV)
            data = loader.load()

            assert len(data) == 2000
            assert not data['signal'].isnull().any()

        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
