"""
Test OUCRU CSV Data Loader

Tests for the OUCRU CSV format loader which handles array-per-row format
where each row represents 1 second of data.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from vitalDSP.utils.data_loader import (
    DataLoader,
    DataFormat,
    load_oucru_csv
)


@pytest.fixture
def sample_oucru_csv():
    """Create a sample OUCRU CSV file for testing."""
    # Create temporary file
    fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

    # Write sample data
    content = """timestamp,signal,sampling_rate
2024-01-01 00:00:00,"[1.0, 1.1, 1.2, 1.3, 1.4]",5
2024-01-01 00:00:01,"[1.5, 1.6, 1.7, 1.8, 1.9]",5
2024-01-01 00:00:02,"[2.0, 2.1, 2.2, 2.3, 2.4]",5
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    yield filepath

    # Cleanup
    os.unlink(filepath)


@pytest.fixture
def oucru_csv_no_sr():
    """Create OUCRU CSV without sampling_rate column."""
    fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

    content = """timestamp,ecg_values
2024-01-01 00:00:00,"[0.5, 0.6, 0.7, 0.8]"
2024-01-01 00:00:01,"[0.9, 1.0, 1.1, 1.2]"
"""

    with os.fdopen(fd, 'w') as f:
        f.write(content)

    yield filepath

    os.unlink(filepath)


class TestOUCRUCSVLoader:
    """Test cases for OUCRU CSV data loader."""

    def test_load_oucru_csv_basic(self, sample_oucru_csv):
        """Test basic OUCRU CSV loading."""
        signal, metadata = load_oucru_csv(
            sample_oucru_csv,
            time_column='timestamp',
            signal_column='signal',
            sampling_rate_column='sampling_rate'
        )

        # Check signal data
        assert len(signal) == 15  # 3 rows * 5 samples each
        assert isinstance(signal, np.ndarray)
        assert signal[0] == 1.0
        assert signal[4] == 1.4
        assert signal[5] == 1.5
        assert signal[-1] == 2.4

        # Check metadata
        assert metadata['sampling_rate'] == 5
        assert metadata['n_samples'] == 15
        assert metadata['n_rows'] == 3
        assert metadata['samples_per_row'] == 5
        assert metadata['duration_seconds'] == 3.0
        assert metadata['format'] == 'oucru_csv'

    def test_load_with_dataloader(self, sample_oucru_csv):
        """Test loading using DataLoader class directly."""
        loader = DataLoader(
            sample_oucru_csv,
            format=DataFormat.OUCRU_CSV,
            sampling_rate=5
        )

        data = loader.load(
            time_column='timestamp',
            signal_column='signal',
            interpolate_time=True
        )

        assert isinstance(data, pd.DataFrame)
        assert 'timestamp' in data.columns
        assert 'signal' in data.columns
        assert len(data) == 15
        assert loader.sampling_rate == 5

    def test_sampling_rate_detection(self, oucru_csv_no_sr):
        """Test automatic sampling rate detection from array length."""
        signal, metadata = load_oucru_csv(
            oucru_csv_no_sr,
            time_column='timestamp',
            signal_column='ecg_values',
            sampling_rate_column=None  # No SR column
        )

        # Should auto-detect SR from array length
        assert metadata['sampling_rate'] == 4
        assert len(signal) == 8  # 2 rows * 4 samples
        assert metadata['samples_per_row'] == 4

    def test_timestamp_interpolation(self, sample_oucru_csv):
        """Test timestamp interpolation for each sample."""
        loader = DataLoader(
            sample_oucru_csv,
            format=DataFormat.OUCRU_CSV
        )

        data = loader.load(
            time_column='timestamp',
            signal_column='signal',
            interpolate_time=True
        )

        # Check timestamps are interpolated
        timestamps = pd.to_datetime(data['timestamp'])

        # First sample should be at second 0
        assert timestamps.iloc[0].second == 0

        # Samples within first second should have fractional time
        time_diffs = timestamps.diff().dt.total_seconds().dropna()
        expected_interval = 1.0 / 5  # 5 Hz = 0.2 seconds
        np.testing.assert_almost_equal(
            time_diffs.values,
            expected_interval,
            decimal=6
        )

    def test_no_timestamp_interpolation(self, sample_oucru_csv):
        """Test loading without timestamp interpolation."""
        loader = DataLoader(
            sample_oucru_csv,
            format=DataFormat.OUCRU_CSV
        )

        data = loader.load(
            time_column='timestamp',
            signal_column='signal',
            interpolate_time=False
        )

        # Should have sample_index instead of interpolated timestamps
        assert 'sample_index' in data.columns
        assert len(data) == 15

    def test_specified_sampling_rate(self, sample_oucru_csv):
        """Test with explicitly specified sampling rate."""
        # Specify SR that matches the data
        signal, metadata = load_oucru_csv(
            sample_oucru_csv,
            time_column='timestamp',
            signal_column='signal',
            sampling_rate=5
        )

        assert metadata['sampling_rate'] == 5
        assert len(signal) == 15

    def test_metadata_extraction(self, sample_oucru_csv):
        """Test comprehensive metadata extraction."""
        signal, metadata = load_oucru_csv(sample_oucru_csv)

        # Check all metadata fields
        assert 'sampling_rate' in metadata
        assert 'n_samples' in metadata
        assert 'duration_seconds' in metadata
        assert 'n_rows' in metadata
        assert 'samples_per_row' in metadata
        assert 'format' in metadata
        assert 'start_time' in metadata
        assert 'end_time' in metadata
        assert 'row_data' in metadata

        # Check start/end times
        assert '2024-01-01 00:00:00' in metadata['start_time']
        assert '2024-01-01 00:00:02' in metadata['end_time']

    def test_array_parsing(self, sample_oucru_csv):
        """Test ast.literal_eval array parsing."""
        signal, _ = load_oucru_csv(sample_oucru_csv)

        # Values should be correctly parsed from string arrays
        expected = np.array([
            1.0, 1.1, 1.2, 1.3, 1.4,  # Row 1
            1.5, 1.6, 1.7, 1.8, 1.9,  # Row 2
            2.0, 2.1, 2.2, 2.3, 2.4   # Row 3
        ])

        np.testing.assert_array_almost_equal(signal, expected)

    def test_inconsistent_array_length_handling(self):
        """Test handling of inconsistent array lengths."""
        # Create CSV with inconsistent array lengths
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        content = """timestamp,signal
2024-01-01 00:00:00,"[1.0, 1.1, 1.2]"
2024-01-01 00:00:01,"[2.0, 2.1]"
2024-01-01 00:00:02,"[3.0, 3.1, 3.2, 3.3]"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            # Should handle inconsistent lengths (pad/truncate)
            with pytest.warns(UserWarning):
                signal, metadata = load_oucru_csv(
                    filepath,
                    time_column='timestamp',
                    signal_column='signal'
                )

            # Should use first row's length as reference
            assert metadata['samples_per_row'] == 3
            # Total should be 3 rows * 3 samples (padded/truncated)
            assert len(signal) == 9

        finally:
            os.unlink(filepath)

    def test_missing_column_error(self, sample_oucru_csv):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError, match="not found in CSV"):
            load_oucru_csv(
                sample_oucru_csv,
                time_column='nonexistent_column',
                signal_column='signal'
            )

        with pytest.raises(ValueError, match="not found in CSV"):
            load_oucru_csv(
                sample_oucru_csv,
                time_column='timestamp',
                signal_column='nonexistent_column'
            )

    def test_invalid_array_format(self):
        """Test error handling for invalid array format."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        content = """timestamp,signal
2024-01-01 00:00:00,"[1.0, 1.1, invalid]"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            with pytest.raises(ValueError, match="Failed to parse signal array"):
                load_oucru_csv(
                    filepath,
                    time_column='timestamp',
                    signal_column='signal'
                )
        finally:
            os.unlink(filepath)

    def test_custom_column_names(self):
        """Test with custom column names."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        content = """datetime,ppg_data,fs
2024-01-01 12:00:00,"[100, 101, 102]",3
2024-01-01 12:00:01,"[103, 104, 105]",3
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            signal, metadata = load_oucru_csv(
                filepath,
                time_column='datetime',
                signal_column='ppg_data',
                sampling_rate_column='fs'
            )

            assert len(signal) == 6
            assert metadata['sampling_rate'] == 3
            assert signal[0] == 100

        finally:
            os.unlink(filepath)

    def test_row_data_preservation(self, sample_oucru_csv):
        """Test that original row data is preserved in metadata."""
        signal, metadata = load_oucru_csv(sample_oucru_csv)

        # Original row-based data should be in metadata
        assert 'row_data' in metadata
        row_data = metadata['row_data']

        assert isinstance(row_data, pd.DataFrame)
        assert len(row_data) == 3  # Original 3 rows
        assert 'timestamp' in row_data.columns
        assert 'signal' in row_data.columns

    def test_signal_type_hint_ppg(self):
        """Test signal type hint with PPG (default 100 Hz)."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        # Create CSV without sampling_rate column
        # Array will have 100 elements per row (matching PPG default)
        array_100 = str(list(range(100)))
        content = f"""timestamp,ppg_signal
2024-01-01 00:00:00,"{array_100}"
2024-01-01 00:00:01,"{array_100}"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            # Use PPG hint - should use default 100 Hz
            signal, metadata = load_oucru_csv(
                filepath,
                time_column='timestamp',
                signal_column='ppg_signal',
                signal_type_hint='ppg',
                sampling_rate_column=None
            )

            assert metadata['sampling_rate'] == 100
            assert len(signal) == 200  # 2 rows * 100 samples

        finally:
            os.unlink(filepath)

    def test_signal_type_hint_ecg(self):
        """Test signal type hint with ECG (default 128 Hz)."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        # Create CSV without sampling_rate column
        # Array will have 128 elements per row (matching ECG default)
        array_128 = str(list(range(128)))
        content = f"""timestamp,ecg_signal
2024-01-01 00:00:00,"{array_128}"
2024-01-01 00:00:01,"{array_128}"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            # Use ECG hint - should use default 128 Hz
            signal, metadata = load_oucru_csv(
                filepath,
                time_column='timestamp',
                signal_column='ecg_signal',
                signal_type_hint='ecg',
                sampling_rate_column=None
            )

            assert metadata['sampling_rate'] == 128
            assert len(signal) == 256  # 2 rows * 128 samples

        finally:
            os.unlink(filepath)

    def test_custom_default_rates(self):
        """Test custom default rates for signal types."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        # Create CSV with arrays that match custom ECG rate of 250
        array_250 = str(list(np.linspace(0, 1, 250)))
        content = f"""timestamp,ecg_signal
2024-01-01 00:00:00,"{array_250}"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            # Use custom ECG rate of 250 instead of default 128
            signal, metadata = load_oucru_csv(
                filepath,
                time_column='timestamp',
                signal_column='ecg_signal',
                signal_type_hint='ecg',
                default_ecg_rate=250,
                sampling_rate_column=None
            )

            assert metadata['sampling_rate'] == 250
            assert len(signal) == 250

        finally:
            os.unlink(filepath)

    def test_signal_type_hint_priority(self):
        """Test that explicit sampling_rate overrides signal_type_hint."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        array_64 = str(list(range(64)))
        content = f"""timestamp,signal
2024-01-01 00:00:00,"{array_64}"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            # Explicit sampling_rate should override signal_type_hint
            signal, metadata = load_oucru_csv(
                filepath,
                time_column='timestamp',
                signal_column='signal',
                sampling_rate=64,  # Explicit takes priority
                signal_type_hint='ecg',  # This should be ignored (would be 128)
                sampling_rate_column=None
            )

            # Should use explicit 64, not ECG default 128
            assert metadata['sampling_rate'] == 64
            assert len(signal) == 64

        finally:
            os.unlink(filepath)

    def test_unknown_signal_type_hint(self):
        """Test handling of unknown signal type hint."""
        fd, filepath = tempfile.mkstemp(suffix='.csv', text=True)

        content = """timestamp,signal
2024-01-01 00:00:00,"[1, 2, 3, 4]"
"""

        with os.fdopen(fd, 'w') as f:
            f.write(content)

        try:
            # Unknown hint should warn and fall back to array length
            with pytest.warns(UserWarning, match="Unknown signal_type_hint"):
                signal, metadata = load_oucru_csv(
                    filepath,
                    time_column='timestamp',
                    signal_column='signal',
                    signal_type_hint='unknown_type',
                    sampling_rate_column=None
                )

            # Should infer from array length (4)
            assert metadata['sampling_rate'] == 4

        finally:
            os.unlink(filepath)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
