"""
Comprehensive tests for vitalDSP DataLoader module.

Tests cover:
- All supported file formats
- Format detection
- Data validation
- Metadata extraction
- Error handling
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from vitalDSP.utils.data_processing.data_loader import (
    DataLoader,
    DataFormat,
    SignalType,
    StreamDataLoader,
    load_signal,
    load_multi_channel
)


class TestDataFormat:
    """Tests for DataFormat enum."""

    def test_data_format_values(self):
        """Test DataFormat enum values."""
        assert DataFormat.CSV.value == "csv"
        assert DataFormat.JSON.value == "json"
        assert DataFormat.EDF.value == "edf"
        assert DataFormat.WFDB.value == "wfdb"

    def test_all_formats_defined(self):
        """Test that all expected formats are defined."""
        expected_formats = [
            'csv', 'tsv', 'excel', 'json', 'hdf5', 'edf',
            'wfdb', 'numpy', 'mat', 'pickle', 'parquet', 'unknown'
        ]
        actual_formats = [fmt.value for fmt in DataFormat]
        for fmt in expected_formats:
            assert fmt in actual_formats


class TestSignalType:
    """Tests for SignalType enum."""

    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.ECG.value == "ecg"
        assert SignalType.PPG.value == "ppg"
        assert SignalType.EEG.value == "eeg"
        assert SignalType.RESP.value == "respiratory"


class TestDataLoaderInitialization:
    """Tests for DataLoader initialization."""

    def test_init_with_file_path(self, tmp_path):
        """Test initialization with file path."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3")

        loader = DataLoader(csv_file)
        assert loader.file_path == csv_file
        assert loader.format == DataFormat.CSV

    def test_init_with_format(self):
        """Test initialization with explicit format."""
        loader = DataLoader(format="csv")
        assert loader.format == DataFormat.CSV

        loader = DataLoader(format=DataFormat.JSON)
        assert loader.format == DataFormat.JSON

    def test_init_with_sampling_rate(self):
        """Test initialization with sampling rate."""
        loader = DataLoader(sampling_rate=250.0)
        assert loader.sampling_rate == 250.0

    def test_init_with_signal_type(self):
        """Test initialization with signal type."""
        loader = DataLoader(signal_type="ecg")
        assert loader.signal_type == SignalType.ECG

        loader = DataLoader(signal_type=SignalType.PPG)
        assert loader.signal_type == SignalType.PPG

    def test_format_detection(self, tmp_path):
        """Test automatic format detection from file extension."""
        test_cases = [
            ("test.csv", DataFormat.CSV),
            ("test.tsv", DataFormat.TSV),
            ("test.json", DataFormat.JSON),
            ("test.xlsx", DataFormat.EXCEL),
            ("test.npy", DataFormat.NUMPY),
            ("test.mat", DataFormat.MATLAB),
            ("test.pkl", DataFormat.PICKLE),
            ("test.h5", DataFormat.HDF5),
            ("test.edf", DataFormat.EDF),
            ("test.parquet", DataFormat.PARQUET),
            ("test.unknown", DataFormat.UNKNOWN),
        ]

        for filename, expected_format in test_cases:
            file_path = tmp_path / filename
            file_path.write_text("")  # Create empty file
            loader = DataLoader(file_path)
            assert loader.format == expected_format


class TestCSVLoading:
    """Tests for CSV file loading."""

    def test_load_simple_csv(self, tmp_path):
        """Test loading simple CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_content = "time,signal\n0.0,1.5\n0.01,2.3\n0.02,1.8"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert list(data.columns) == ['time', 'signal']
        assert loader.metadata['n_samples'] == 3

    def test_load_csv_specific_columns(self, tmp_path):
        """Test loading specific columns from CSV."""
        csv_file = tmp_path / "test.csv"
        csv_content = "time,ecg,ppg,resp\n0.0,1.5,2.0,3.0\n0.01,2.3,2.5,3.2"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load(columns=['time', 'ecg'])

        assert len(data.columns) == 2
        assert 'ecg' in data.columns
        assert 'ppg' not in data.columns

    def test_load_csv_no_header(self, tmp_path):
        """Test loading CSV without header."""
        csv_file = tmp_path / "test.csv"
        csv_content = "1.5\n2.3\n1.8"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load(header=None)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3

    def test_load_csv_custom_delimiter(self, tmp_path):
        """Test loading CSV with custom delimiter."""
        csv_file = tmp_path / "test.txt"
        csv_content = "time;signal\n0.0;1.5\n0.01;2.3"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file, format='csv')
        data = loader.load(delimiter=';')

        assert len(data) == 2
        assert 'signal' in data.columns

    def test_load_tsv(self, tmp_path):
        """Test loading TSV file."""
        tsv_file = tmp_path / "test.tsv"
        tsv_content = "time\tsignal\n0.0\t1.5\n0.01\t2.3"
        tsv_file.write_text(tsv_content)

        loader = DataLoader(tsv_file)
        data = loader.load()

        assert len(data) == 2
        assert 'signal' in data.columns

    def test_sampling_rate_extraction(self, tmp_path):
        """Test automatic sampling rate extraction from time column."""
        csv_file = tmp_path / "test.csv"
        # 100 Hz sampling rate (0.01 second intervals)
        csv_content = "time,signal\n0.0,1.5\n0.01,2.3\n0.02,1.8\n0.03,2.1"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load(time_column='time')

        assert 'computed_sampling_rate' in loader.metadata
        # Should be approximately 100 Hz
        assert 90 < loader.metadata['computed_sampling_rate'] < 110


class TestJSONLoading:
    """Tests for JSON file loading."""

    def test_load_json_array(self, tmp_path):
        """Test loading JSON array."""
        json_file = tmp_path / "test.json"
        data = [{"time": 0.0, "signal": 1.5}, {"time": 0.01, "signal": 2.3}]
        json_file.write_text(json.dumps(data))

        loader = DataLoader(json_file)
        loaded_data = loader.load()

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 2

    def test_load_json_dict(self, tmp_path):
        """Test loading JSON dictionary."""
        json_file = tmp_path / "test.json"
        data = {
            "time": [0.0, 0.01, 0.02],
            "signal": [1.5, 2.3, 1.8]
        }
        json_file.write_text(json.dumps(data))

        loader = DataLoader(json_file)
        loaded_data = loader.load()

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 3
        assert 'signal' in loaded_data.columns

    def test_load_json_with_metadata(self, tmp_path):
        """Test loading JSON with metadata."""
        json_file = tmp_path / "test.json"
        data = {
            "sampling_rate": 250,
            "signal_type": "ecg",
            "data": [
                {"time": 0.0, "signal": 1.5},
                {"time": 0.01, "signal": 2.3}
            ]
        }
        json_file.write_text(json.dumps(data))

        loader = DataLoader(json_file)
        loaded_data = loader.load()

        assert 'sampling_rate' in loader.metadata
        assert loader.metadata['sampling_rate'] == 250


class TestNumpyLoading:
    """Tests for NumPy file loading."""

    def test_load_npy_1d(self, tmp_path):
        """Test loading 1D NumPy array."""
        npy_file = tmp_path / "test.npy"
        data = np.array([1.5, 2.3, 1.8, 2.1])
        np.save(npy_file, data)

        loader = DataLoader(npy_file)
        loaded_data = loader.load()

        assert isinstance(loaded_data, np.ndarray)
        assert len(loaded_data) == 4
        assert np.allclose(loaded_data, data)

    def test_load_npy_2d(self, tmp_path):
        """Test loading 2D NumPy array."""
        npy_file = tmp_path / "test.npy"
        data = np.array([[1.5, 2.0], [2.3, 2.5], [1.8, 2.2]])
        np.save(npy_file, data)

        loader = DataLoader(npy_file)
        loaded_data = loader.load()

        assert loaded_data.shape == (3, 2)
        assert np.allclose(loaded_data, data)

    def test_load_npz(self, tmp_path):
        """Test loading NPZ file."""
        npz_file = tmp_path / "test.npz"
        ecg = np.array([1.5, 2.3, 1.8])
        ppg = np.array([2.0, 2.5, 2.2])
        np.savez(npz_file, ecg=ecg, ppg=ppg)

        loader = DataLoader(npz_file)
        loaded_data = loader.load()

        assert isinstance(loaded_data, dict)
        assert 'ecg' in loaded_data
        assert 'ppg' in loaded_data
        assert np.allclose(loaded_data['ecg'], ecg)


class TestPickleLoading:
    """Tests for Pickle file loading."""

    def test_load_pickle_dict(self, tmp_path):
        """Test loading pickled dictionary."""
        pkl_file = tmp_path / "test.pkl"
        data = {'signal': np.array([1, 2, 3]), 'fs': 250}

        import pickle
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

        loader = DataLoader(pkl_file)
        loaded_data = loader.load()

        assert isinstance(loaded_data, dict)
        assert 'signal' in loaded_data
        assert loaded_data['fs'] == 250


class TestDataValidation:
    """Tests for data validation."""

    def test_validate_with_nan(self, tmp_path):
        """Test validation with NaN values."""
        csv_file = tmp_path / "test.csv"
        csv_content = "signal\n1.5\nnan\n2.3"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file, validate=True)

        with pytest.warns(UserWarning, match="missing values"):
            data = loader.load()

    def test_validate_numpy_with_nan(self):
        """Test validation of NumPy array with NaN."""
        data = np.array([1.5, np.nan, 2.3])
        loader = DataLoader(validate=True)

        with pytest.warns(UserWarning, match="NaN values"):
            validated = loader._validate_data(data)

    def test_validate_with_inf(self):
        """Test validation with infinite values."""
        data = np.array([1.5, np.inf, 2.3])
        loader = DataLoader(validate=True)

        with pytest.warns(UserWarning, match="infinite values"):
            validated = loader._validate_data(data)

    def test_no_validation(self, tmp_path):
        """Test loading without validation."""
        csv_file = tmp_path / "test.csv"
        csv_content = "signal\n1.5\nnan\n2.3"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file, validate=False)
        data = loader.load()  # Should not raise warning

        assert len(data) == 3


class TestLoadFromArray:
    """Tests for loading from NumPy arrays."""

    def test_load_from_1d_array(self):
        """Test loading from 1D array."""
        data = np.array([1.5, 2.3, 1.8, 2.1])
        loader = DataLoader()

        df = loader.load_from_array(data, sampling_rate=250.0)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert loader.sampling_rate == 250.0
        assert 'signal' in df.columns

    def test_load_from_2d_array_with_names(self):
        """Test loading from 2D array with column names."""
        data = np.array([[1.5, 2.0], [2.3, 2.5], [1.8, 2.2]])
        loader = DataLoader()

        df = loader.load_from_array(
            data,
            sampling_rate=100.0,
            column_names=['ecg', 'ppg']
        )

        assert df.shape == (3, 2)
        assert 'ecg' in df.columns
        assert 'ppg' in df.columns

    def test_load_from_array_with_signal_type(self):
        """Test loading array with signal type."""
        data = np.array([1.5, 2.3, 1.8])
        loader = DataLoader()

        df = loader.load_from_array(data, signal_type='ecg')

        assert loader.signal_type == SignalType.ECG


class TestLoadFromDataFrame:
    """Tests for loading from DataFrames."""

    def test_load_from_dataframe(self):
        """Test loading from DataFrame."""
        df_input = pd.DataFrame({
            'time': [0.0, 0.01, 0.02],
            'signal': [1.5, 2.3, 1.8]
        })

        loader = DataLoader()
        df_output = loader.load_from_dataframe(df_input, sampling_rate=100.0)

        assert isinstance(df_output, pd.DataFrame)
        assert loader.metadata['n_samples'] == 3
        assert loader.sampling_rate == 100.0


class TestMetadataExtraction:
    """Tests for metadata extraction."""

    def test_csv_metadata(self, tmp_path):
        """Test metadata extraction from CSV."""
        csv_file = tmp_path / "test.csv"
        csv_content = "time,ecg,ppg\n0.0,1.5,2.0\n0.01,2.3,2.5"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load()

        assert 'columns' in loader.metadata
        assert 'n_samples' in loader.metadata
        assert 'shape' in loader.metadata
        assert loader.metadata['n_samples'] == 2
        assert loader.metadata['shape'] == (2, 3)

    def test_get_info(self, tmp_path):
        """Test get_info method."""
        csv_file = tmp_path / "test.csv"
        csv_content = "signal\n1.5\n2.3"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file, sampling_rate=250.0, signal_type='ecg')
        data = loader.load()

        info = loader.get_info()

        assert 'file_path' in info
        assert 'format' in info
        assert 'sampling_rate' in info
        assert 'signal_type' in info
        assert 'metadata' in info
        assert info['format'] == 'csv'
        assert info['sampling_rate'] == 250.0


class TestDataExport:
    """Tests for data export functionality."""

    def test_export_to_csv(self, tmp_path):
        """Test exporting to CSV."""
        data = pd.DataFrame({'signal': [1.5, 2.3, 1.8]})
        loader = DataLoader()

        output_file = tmp_path / "output.csv"
        loader.export(data, output_file)

        assert output_file.exists()

        # Verify exported data
        exported = pd.read_csv(output_file)
        assert len(exported) == 3
        assert 'signal' in exported.columns

    def test_export_numpy_array(self, tmp_path):
        """Test exporting NumPy array."""
        data = np.array([1.5, 2.3, 1.8])
        loader = DataLoader()

        output_file = tmp_path / "output.csv"
        loader.export(data, output_file)

        assert output_file.exists()

    def test_export_to_json(self, tmp_path):
        """Test exporting to JSON."""
        data = pd.DataFrame({'signal': [1.5, 2.3, 1.8]})
        loader = DataLoader()

        output_file = tmp_path / "output.json"
        loader.export(data, output_file)

        assert output_file.exists()

    def test_export_to_pickle(self, tmp_path):
        """Test exporting to pickle."""
        data = pd.DataFrame({'signal': [1.5, 2.3, 1.8]})
        loader = DataLoader()

        output_file = tmp_path / "output.pkl"
        loader.export(data, output_file)

        assert output_file.exists()


class TestErrorHandling:
    """Tests for error handling."""

    def test_file_not_found(self):
        """Test loading non-existent file."""
        loader = DataLoader("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_unsupported_format(self, tmp_path):
        """Test loading unsupported format."""
        file_path = tmp_path / "test.unknown"
        file_path.write_text("data")

        loader = DataLoader(file_path)

        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load()

    def test_invalid_csv(self, tmp_path):
        """Test loading invalid CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("not,valid,csv\ndata")

        loader = DataLoader(csv_file)
        # Should load but may have unexpected structure
        data = loader.load()
        assert isinstance(data, pd.DataFrame)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_signal(self, tmp_path):
        """Test load_signal convenience function."""
        csv_file = tmp_path / "test.csv"
        csv_content = "signal\n1.5\n2.3\n1.8"
        csv_file.write_text(csv_content)

        data = load_signal(csv_file, sampling_rate=250.0)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3

    def test_load_multi_channel(self, tmp_path):
        """Test load_multi_channel convenience function."""
        csv_file = tmp_path / "test.csv"
        csv_content = "ecg,ppg\n1.5,2.0\n2.3,2.5\n1.8,2.2"
        csv_file.write_text(csv_content)

        data = load_multi_channel(csv_file)

        assert isinstance(data, dict)
        assert 'ecg' in data
        assert 'ppg' in data
        assert len(data['ecg']) == 3


class TestStaticMethods:
    """Tests for static methods."""

    def test_list_supported_formats(self):
        """Test listing supported formats."""
        formats = DataLoader.list_supported_formats()

        assert isinstance(formats, list)
        assert 'csv' in formats
        assert 'json' in formats
        assert 'edf' in formats
        assert 'unknown' in formats

    def test_get_format_requirements(self):
        """Test getting format requirements."""
        req = DataLoader.get_format_requirements('csv')

        assert isinstance(req, dict)
        assert 'packages' in req
        assert 'extensions' in req
        assert 'description' in req

        # Test with DataFormat enum
        req = DataLoader.get_format_requirements(DataFormat.EDF)
        assert 'pyedflib' in req['packages']

    def test_get_format_requirements_all_formats(self):
        """Test requirements for all major formats."""
        formats_to_test = ['csv', 'excel', 'hdf5', 'edf', 'wfdb', 'mat']

        for fmt in formats_to_test:
            req = DataLoader.get_format_requirements(fmt)
            assert 'packages' in req or req == {}


class TestStreamDataLoader:
    """Tests for StreamDataLoader."""

    def test_stream_loader_init(self):
        """Test StreamDataLoader initialization."""
        loader = StreamDataLoader(
            source_type='serial',
            buffer_size=1000,
            sampling_rate=250.0,
            port='/dev/ttyUSB0'
        )

        assert loader.source_type == 'serial'
        assert loader.buffer_size == 1000
        assert loader.sampling_rate == 250.0
        assert loader.kwargs['port'] == '/dev/ttyUSB0'

    def test_stream_loader_stop(self):
        """Test stopping stream loader."""
        loader = StreamDataLoader(source_type='serial')
        loader.is_streaming = True
        loader.stop()

        assert not loader.is_streaming


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_file(self, tmp_path):
        """Test loading empty file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        loader = DataLoader(csv_file)

        with pytest.raises(Exception):  # Could be various exceptions
            loader.load()

    def test_single_value(self, tmp_path):
        """Test loading single value."""
        csv_file = tmp_path / "single.csv"
        csv_content = "signal\n1.5"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load()

        assert len(data) == 1

    def test_large_column_names(self, tmp_path):
        """Test with very long column names."""
        csv_file = tmp_path / "test.csv"
        long_name = "very_long_column_name_" * 10
        csv_content = f"{long_name}\n1.5\n2.3"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load()

        assert long_name in data.columns

    def test_special_characters_in_data(self, tmp_path):
        """Test loading data with special characters."""
        csv_file = tmp_path / "test.csv"
        csv_content = "signal\n1.5\n2.3\n1.8"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load()

        assert len(data) == 3

    def test_mixed_data_types(self, tmp_path):
        """Test loading mixed data types."""
        csv_file = tmp_path / "test.csv"
        csv_content = "id,signal,label\n1,1.5,normal\n2,2.3,abnormal"
        csv_file.write_text(csv_content)

        loader = DataLoader(csv_file)
        data = loader.load()

        assert len(data) == 2
        assert 'label' in data.columns


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: load, process, export."""
        # Create test data
        csv_file = tmp_path / "input.csv"
        csv_content = "time,ecg\n0.0,1.5\n0.01,2.3\n0.02,1.8"
        csv_file.write_text(csv_content)

        # Load
        loader = DataLoader(csv_file, sampling_rate=100.0, signal_type='ecg')
        data = loader.load()

        # Verify
        assert len(data) == 3
        info = loader.get_info()
        assert info['sampling_rate'] == 100.0

        # Export
        output_file = tmp_path / "output.json"
        loader.export(data, output_file)

        assert output_file.exists()

    def test_multi_format_conversion(self, tmp_path):
        """Test converting between formats."""
        # Create CSV
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({'signal': [1.5, 2.3, 1.8, 2.1]})
        df.to_csv(csv_file, index=False)

        # Load CSV
        loader = DataLoader(csv_file)
        data = loader.load()

        # Export to multiple formats
        formats = [
            (tmp_path / "data.json", DataFormat.JSON),
            (tmp_path / "data.pkl", DataFormat.PICKLE),
        ]

        for output_path, fmt in formats:
            loader.export(data, output_path, format=fmt)
            assert output_path.exists()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
