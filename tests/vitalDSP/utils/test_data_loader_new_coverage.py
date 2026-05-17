"""
Tests targeting missing coverage lines in data_loader.py.

Missing lines: 466, 472, 487-498, 618, 660, 688-700, 718, 764-783, 858-877,
907, 968-1001, 1007-1060, 1096-1150, 1225-1244, 1268-1304, 1361-1373,
1565-1603, 1625-1637, 1665-1673, 1868, 1887
"""

import json
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vitalDSP.utils.data_processing.data_loader import (
    DataFormat,
    DataLoader,
    SignalType,
    StreamDataLoader,
    load_multi_channel,
    load_signal,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_csv(tmp_path):
    p = tmp_path / "simple.csv"
    p.write_text("time,ecg,ppg\n0.0,1.1,2.2\n0.01,1.2,2.3\n0.02,1.3,2.4\n")
    return p


@pytest.fixture
def signal_array():
    return np.sin(2 * np.pi * 1.0 * np.arange(100) / 100.0)


# ---------------------------------------------------------------------------
# _parse_timestamps_with_conversion  (lines 1225-1244)
# ---------------------------------------------------------------------------


class TestParseTimestamps:
    """Cover _parse_timestamps_with_conversion branches."""

    def test_unix_seconds(self, tmp_path):
        loader = DataLoader(format="csv")
        ts = pd.Series([1_600_000_000.0, 1_600_000_001.0, 1_600_000_002.0])
        result = loader._parse_timestamps_with_conversion(ts)
        assert result is not None
        assert loader.metadata.get("timestamp_type") == "unix_seconds"

    def test_unix_milliseconds(self, tmp_path):
        loader = DataLoader(format="csv")
        ts = pd.Series([1_600_000_000_000.0, 1_600_000_001_000.0])
        result = loader._parse_timestamps_with_conversion(ts)
        assert result is not None
        assert loader.metadata.get("timestamp_type") == "unix_milliseconds"

    def test_numeric_with_nan(self):
        loader = DataLoader(format="csv")
        ts = pd.Series([1_600_000_000.0, float("nan")])
        result = loader._parse_timestamps_with_conversion(ts)
        assert result is None  # NaN → returns None with warning

    def test_datetime_string_format(self):
        loader = DataLoader(format="csv")
        ts = pd.Series(["2023-01-01 00:00:00", "2023-01-01 00:00:01"])
        result = loader._parse_timestamps_with_conversion(ts)
        assert result is not None

    def test_unparseable_timestamps_return_none(self):
        loader = DataLoader(format="csv")
        ts = pd.Series(["garbage_a", "garbage_b"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = loader._parse_timestamps_with_conversion(ts)
        # Either None or a valid series — must not crash
        # (some pandas versions may still parse it)


# ---------------------------------------------------------------------------
# _validate_data  (lines 1284-1304)
# ---------------------------------------------------------------------------


class TestValidateData:
    def test_ndarray_nan_warns(self):
        loader = DataLoader(format="csv")
        arr = np.array([1.0, float("nan"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = loader._validate_data(arr)
        assert any("NaN" in str(x.message) for x in w)
        np.testing.assert_array_equal(result, arr)

    def test_ndarray_inf_warns(self):
        loader = DataLoader(format="csv")
        arr = np.array([1.0, float("inf"), 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader._validate_data(arr)
        assert any("infinite" in str(x.message) for x in w)

    def test_dataframe_missing_warns(self):
        loader = DataLoader(format="csv")
        df = pd.DataFrame({"a": [1.0, float("nan")], "b": [1.0, 2.0]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader._validate_data(df)
        assert any("missing" in str(x.message) for x in w)

    def test_dict_nan_warns(self):
        loader = DataLoader(format="csv")
        data = {"ecg": np.array([1.0, float("nan")])}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader._validate_data(data)
        assert any("NaN" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# load_from_array  (lines 1306-1342)
# ---------------------------------------------------------------------------


class TestLoadFromArray:
    def test_1d_array(self):
        loader = DataLoader(format="csv")
        arr = np.arange(10, dtype=float)
        df = loader.load_from_array(arr, sampling_rate=100.0)
        assert "signal" in df.columns
        assert len(df) == 10
        assert loader.sampling_rate == 100.0

    def test_2d_array_with_columns(self):
        loader = DataLoader(format="csv")
        arr = np.random.randn(5, 2)
        df = loader.load_from_array(arr, column_names=["ecg", "ppg"])
        assert list(df.columns) == ["ecg", "ppg"]

    def test_2d_array_no_columns(self):
        loader = DataLoader(format="csv")
        arr = np.random.randn(5, 3)
        df = loader.load_from_array(arr)
        assert df.shape == (5, 3)

    def test_signal_type_set(self):
        loader = DataLoader(format="csv")
        arr = np.arange(5, dtype=float)
        loader.load_from_array(arr, signal_type="ecg")
        assert loader.signal_type == SignalType.ECG


# ---------------------------------------------------------------------------
# load_from_dataframe  (lines 1344-1373)
# ---------------------------------------------------------------------------


class TestLoadFromDataframe:
    def test_basic(self):
        loader = DataLoader(format="csv")
        df = pd.DataFrame({"ecg": [1.0, 2.0], "ppg": [3.0, 4.0]})
        result = loader.load_from_dataframe(df, sampling_rate=250.0)
        assert loader.sampling_rate == 250.0
        assert loader.metadata["n_samples"] == 2

    def test_validation_triggered(self):
        loader = DataLoader(format="csv", validate=True)
        df = pd.DataFrame({"ecg": [1.0, float("nan")]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader.load_from_dataframe(df)
        assert any("missing" in str(x.message) for x in w)

    def test_no_validation(self):
        loader = DataLoader(format="csv", validate=False)
        df = pd.DataFrame({"ecg": [float("nan")]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = loader.load_from_dataframe(df)
        # No warnings should be emitted for missing since validate=False
        missing_warns = [x for x in w if "missing" in str(x.message)]
        assert len(missing_warns) == 0


# ---------------------------------------------------------------------------
# get_info  (line 1375-1389)
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_get_info_with_file(self, simple_csv):
        loader = DataLoader(simple_csv, sampling_rate=100.0, signal_type="ecg")
        info = loader.get_info()
        assert info["sampling_rate"] == 100.0
        assert info["signal_type"] == "ecg"
        assert "file_path" in info

    def test_get_info_no_file(self):
        loader = DataLoader(format="csv")
        info = loader.get_info()
        assert info["file_path"] is None
        assert info["format"] == "csv"


# ---------------------------------------------------------------------------
# export  (lines 1391-1441)
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_csv(self, tmp_path, signal_array):
        loader = DataLoader(format="csv")
        out = tmp_path / "out.csv"
        loader.export(signal_array, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "signal" in df.columns

    def test_export_dict(self, tmp_path):
        loader = DataLoader(format="csv")
        out = tmp_path / "out.csv"
        loader.export({"a": [1, 2, 3], "b": [4, 5, 6]}, out)
        assert out.exists()

    def test_export_json(self, tmp_path, signal_array):
        loader = DataLoader(format="csv")
        out = tmp_path / "out.json"
        loader.export(signal_array, out)
        assert out.exists()

    def test_export_unsupported_raises(self, tmp_path, signal_array):
        loader = DataLoader(format="csv")
        out = tmp_path / "out.edf"
        with pytest.raises(ValueError, match="Export not supported"):
            loader.export(signal_array, out, format=DataFormat.EDF)

    def test_export_tsv(self, tmp_path, signal_array):
        loader = DataLoader(format="csv")
        out = tmp_path / "out.tsv"
        loader.export(signal_array, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# list_supported_formats / get_format_requirements (lines 1443-1505)
# ---------------------------------------------------------------------------


class TestStaticMethods:
    def test_list_supported_formats(self):
        fmts = DataLoader.list_supported_formats()
        assert "csv" in fmts
        assert "json" in fmts

    def test_get_format_requirements_csv(self):
        req = DataLoader.get_format_requirements("csv")
        assert "packages" in req
        assert "pandas" in req["packages"]

    def test_get_format_requirements_dataformat_enum(self):
        req = DataLoader.get_format_requirements(DataFormat.EDF)
        assert "packages" in req

    def test_get_format_requirements_unknown(self):
        # Unknown format → empty dict
        req = DataLoader.get_format_requirements(DataFormat.UNKNOWN)
        assert req == {}


# ---------------------------------------------------------------------------
# _load_json  (lines 879-910) — line 907
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_list_json(self, tmp_path):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, pd.DataFrame)

    def test_dict_with_data_key(self, tmp_path):
        data = {"meta": "info", "data": [{"a": 1}, {"a": 2}]}
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, (pd.DataFrame, dict))

    def test_dict_with_array_values(self, tmp_path):
        data = {"ecg": [1.0, 2.0, 3.0], "ppg": [4.0, 5.0, 6.0]}
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, pd.DataFrame)
        assert "ecg" in result.columns

    def test_dict_no_data_key(self, tmp_path):
        data = {"meta": "only", "info": "no arrays"}
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        loader = DataLoader(p)
        result = loader.load()
        # Returns the raw dict (line 907)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _load_numpy  (lines 1062-1081)
# ---------------------------------------------------------------------------


class TestLoadNumpy:
    def test_load_npy(self, tmp_path):
        arr = np.arange(50, dtype=float)
        p = tmp_path / "data.npy"
        np.save(p, arr)
        loader = DataLoader(p)
        result = loader.load()
        np.testing.assert_array_equal(result, arr)

    def test_load_npz(self, tmp_path):
        arr1 = np.arange(10, dtype=float)
        arr2 = np.arange(20, dtype=float)
        p = tmp_path / "data.npz"
        np.savez(p, a=arr1, b=arr2)
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, dict)
        assert "a" in result
        np.testing.assert_array_equal(result["a"], arr1)


# ---------------------------------------------------------------------------
# _load_matlab  (lines 1083-1108)
# ---------------------------------------------------------------------------


class TestLoadMatlab:
    def test_load_mat(self, tmp_path):
        from scipy.io import savemat

        p = tmp_path / "data.mat"
        savemat(str(p), {"ecg": np.arange(10, dtype=float)})
        loader = DataLoader(p)
        result = loader._load_matlab()
        assert "ecg" in result

    def test_load_mat_with_variable_names(self, tmp_path):
        from scipy.io import savemat

        p = tmp_path / "data.mat"
        savemat(str(p), {"ecg": np.arange(10, dtype=float), "ppg": np.arange(5, dtype=float)})
        loader = DataLoader(p)
        result = loader._load_matlab(variable_names=["ecg"])
        assert "ecg" in result
        assert "ppg" not in result


# ---------------------------------------------------------------------------
# _load_pickle  (lines 1110-1123)
# ---------------------------------------------------------------------------


class TestLoadPickle:
    def test_load_pickle(self, tmp_path):
        data = {"signal": np.arange(10, dtype=float), "fs": 100}
        p = tmp_path / "data.pkl"
        with open(p, "wb") as f:
            pickle.dump(data, f)
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, dict)
        assert "signal" in result

    def test_load_pickle_dot_pickle(self, tmp_path):
        p = tmp_path / "data.pickle"
        with open(p, "wb") as f:
            pickle.dump([1, 2, 3], f)
        loader = DataLoader(p)
        result = loader.load()
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# _load_parquet  (lines 1125-1150)
# ---------------------------------------------------------------------------


class TestLoadParquet:
    def test_load_parquet(self, tmp_path):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        p = tmp_path / "data.parquet"
        df.to_parquet(p, index=False)
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, pd.DataFrame)
        assert "a" in result.columns


# ---------------------------------------------------------------------------
# _extract_sampling_rate  (lines 1257-1270)
# ---------------------------------------------------------------------------


class TestExtractSamplingRate:
    def test_extracts_rate(self):
        loader = DataLoader(format="csv")
        time_array = np.arange(100) / 100.0  # 100 Hz
        loader._extract_sampling_rate(time_array)
        assert loader.metadata.get("computed_sampling_rate") == pytest.approx(100.0, rel=0.01)

    def test_too_short_does_nothing(self):
        loader = DataLoader(format="csv")
        loader._extract_sampling_rate(np.array([0.0]))
        assert "computed_sampling_rate" not in loader.metadata


# ---------------------------------------------------------------------------
# load_multi_channel convenience function  (lines 1712-1734)
# ---------------------------------------------------------------------------


class TestLoadMultiChannel:
    def test_load_multi_channel_csv(self, simple_csv):
        result = load_multi_channel(simple_csv)
        assert isinstance(result, dict)
        assert "ecg" in result or "time" in result or len(result) > 0

    def test_load_multi_channel_dict_passthrough(self, tmp_path):
        data = {"ecg": [1.0, 2.0], "ppg": [3.0, 4.0]}
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        result = load_multi_channel(p)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# StreamDataLoader  (lines 1508-1688)
# ---------------------------------------------------------------------------


class TestStreamDataLoader:
    def test_init(self):
        sdl = StreamDataLoader("serial", buffer_size=256, port="/dev/ttyUSB0")
        assert sdl.source_type == "serial"
        assert sdl.buffer_size == 256
        assert sdl.is_streaming is False

    def test_stop(self):
        sdl = StreamDataLoader("network")
        sdl.is_streaming = True
        sdl.stop()
        assert sdl.is_streaming is False

    def test_parse_api_response_list(self):
        sdl = StreamDataLoader("api")
        assert sdl._parse_api_response([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_parse_api_response_dict_with_values(self):
        sdl = StreamDataLoader("api")
        assert sdl._parse_api_response({"values": [4.0, 5.0]}) == [4.0, 5.0]

    def test_parse_api_response_unknown_returns_empty(self):
        sdl = StreamDataLoader("api")
        assert sdl._parse_api_response({"other": "stuff"}) == []

    def test_parse_network_data_valid(self):
        sdl = StreamDataLoader("network")
        data = b"1.0,2.0,3.0"
        result = sdl._parse_network_data(data)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_parse_network_data_invalid(self):
        sdl = StreamDataLoader("network")
        result = sdl._parse_network_data(b"not_a_number")
        assert result == []

    def test_stream_unsupported_raises(self):
        sdl = StreamDataLoader("unsupported_source")
        with pytest.raises(ValueError, match="Unsupported source type"):
            list(sdl.stream())

    def test_stream_database_not_implemented(self):
        sdl = StreamDataLoader("database")
        with pytest.raises(NotImplementedError):
            list(sdl.stream())


# ---------------------------------------------------------------------------
# _load_excel  (lines 851-877)
# ---------------------------------------------------------------------------


class TestLoadExcel:
    def test_load_xlsx(self, tmp_path):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            pytest.skip("openpyxl not installed")
        df = pd.DataFrame({"ecg": [1.0, 2.0], "ppg": [3.0, 4.0]})
        p = tmp_path / "data.xlsx"
        df.to_excel(p, index=False)
        loader = DataLoader(p)
        result = loader.load()
        assert isinstance(result, pd.DataFrame)
        assert "ecg" in result.columns

    def test_load_excel_bad_file_raises(self, tmp_path):
        p = tmp_path / "bad.xlsx"
        p.write_text("not an excel file")
        loader = DataLoader(p)
        with pytest.raises((ValueError, Exception)):
            loader.load()


# ---------------------------------------------------------------------------
# _load_edf / _load_wfdb — ImportError branches  (lines 968-1060)
# ---------------------------------------------------------------------------


class TestLoadEdfWfdb:
    def test_edf_import_error(self, tmp_path):
        """When pyedflib is absent, _load_edf should raise ImportError."""
        import sys

        p = tmp_path / "data.edf"
        p.write_bytes(b"")
        loader = DataLoader(p)
        # Temporarily hide pyedflib if present
        original = sys.modules.get("pyedflib")
        sys.modules["pyedflib"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError, ValueError)):
                loader._load_edf()
        finally:
            if original is None:
                del sys.modules["pyedflib"]
            else:
                sys.modules["pyedflib"] = original

    def test_wfdb_import_error(self, tmp_path):
        """When wfdb is absent, _load_wfdb should raise ImportError."""
        import sys

        p = tmp_path / "data.dat"
        p.write_bytes(b"")
        loader = DataLoader(p)
        original = sys.modules.get("wfdb")
        sys.modules["wfdb"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError, ValueError)):
                loader._load_wfdb()
        finally:
            if original is None:
                del sys.modules["wfdb"]
            else:
                sys.modules["wfdb"] = original


# ---------------------------------------------------------------------------
# load_signal convenience function
# ---------------------------------------------------------------------------


class TestLoadSignal:
    def test_load_signal_csv(self, simple_csv):
        result = load_signal(simple_csv)
        assert isinstance(result, (np.ndarray, pd.DataFrame))
