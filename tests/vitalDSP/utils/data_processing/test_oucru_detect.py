"""Tests for the OUCRU CSV auto-detection helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vitalDSP.utils.data_processing.oucru_detect import (
    DEFAULT_ECG_RATE,
    DEFAULT_PPG_RATE,
    default_rate_for_signal_type,
    detect_oucru_csv,
    parse_array_cell,
    pick_signal_column,
)


# ---------------------------------------------------------------------------
# parse_array_cell
# ---------------------------------------------------------------------------


class TestParseArrayCell:
    def test_json_bracketed(self):
        arr = parse_array_cell("[1.0, 2.0, 3.5]")
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.5]))

    def test_bracket_less_comma_separated(self):
        arr = parse_array_cell("-439,-446,-446")
        np.testing.assert_array_equal(arr, np.array([-439.0, -446.0, -446.0]))

    def test_empty_string(self):
        assert parse_array_cell("").size == 0

    def test_nan_returns_empty(self):
        assert parse_array_cell(float("nan")).size == 0

    def test_none_returns_empty(self):
        assert parse_array_cell(None).size == 0

    def test_list_input_passthrough(self):
        arr = parse_array_cell([1, 2, 3])
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_numpy_array_passthrough(self):
        arr = parse_array_cell(np.array([4.0, 5.0]))
        np.testing.assert_array_equal(arr, np.array([4.0, 5.0]))

    def test_unparseable_returns_empty(self):
        assert parse_array_cell("not_a_number,also_not").size == 0


# ---------------------------------------------------------------------------
# pick_signal_column
# ---------------------------------------------------------------------------


class TestPickSignalColumn:
    def test_picks_pleth_for_ppg(self):
        assert pick_signal_column(["timestamp", "pleth"], "ppg") == "pleth"

    def test_picks_ecg_for_ecg(self):
        assert pick_signal_column(["timestamp", "ecg"], "ecg") == "ecg"

    def test_case_insensitive(self):
        # Returns the original column name with its original casing
        assert pick_signal_column(["TimeStamp", "PLETH"], "ppg") == "PLETH"

    def test_prefers_ppg_for_ppg_hint_over_ecg(self):
        cols = ["timestamp", "ecg", "pleth"]
        assert pick_signal_column(cols, "ppg") == "pleth"

    def test_falls_back_when_no_hint(self):
        # With no hint, scans the full preferred list — should still find ecg
        assert pick_signal_column(["timestamp", "ecg"], None) == "ecg"

    def test_returns_none_when_no_match(self):
        assert pick_signal_column(["foo", "bar"], "ppg") is None


# ---------------------------------------------------------------------------
# detect_oucru_csv
# ---------------------------------------------------------------------------


@pytest.fixture
def ppg_oucru_csv(tmp_path: Path) -> Path:
    """Canonical OUCRU PPG CSV: pleth column with JSON arrays."""
    samples = list(range(100))
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:01"],
            "pleth": [json.dumps(samples), json.dumps(samples)],
        }
    )
    path = tmp_path / "ppg.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def ecg_oucru_bare_csv(tmp_path: Path) -> Path:
    """ECG file using bracket-less form for the signal column."""
    samples = ",".join(str(i) for i in range(128))
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:01"],
            "ecg": [samples, samples],
        }
    )
    path = tmp_path / "ecg_bare.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def flat_csv(tmp_path: Path) -> Path:
    """Plain flat CSV — one sample per row, not OUCRU."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="10ms"),
            "signal": np.sin(np.linspace(0, 10, 50)),
        }
    )
    path = tmp_path / "flat.csv"
    df.to_csv(path, index=False)
    return path


class TestDetectOucruCsv:
    def test_detects_ppg_oucru(self, ppg_oucru_csv: Path):
        result = detect_oucru_csv(ppg_oucru_csv, signal_type_hint="ppg")
        assert result is not None
        assert result["is_oucru"] is True
        assert result["signal_column"] == "pleth"
        assert result["samples_per_row"] == 100
        assert result["bracket_style"] == "json"

    def test_detects_ecg_oucru_bracket_less(self, ecg_oucru_bare_csv: Path):
        result = detect_oucru_csv(ecg_oucru_bare_csv, signal_type_hint="ecg")
        assert result is not None
        assert result["signal_column"] == "ecg"
        assert result["samples_per_row"] == 128
        assert result["bracket_style"] == "bare"

    def test_returns_none_for_flat_csv(self, flat_csv: Path):
        assert detect_oucru_csv(flat_csv) is None

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path):
        # Should not raise — detection is best-effort
        assert detect_oucru_csv(tmp_path / "does_not_exist.csv") is None

    def test_accepts_dataframe_directly(self):
        df = pd.DataFrame(
            {"timestamp": ["t1"], "pleth": ["[1,2,3,4,5]"]}
        )
        result = detect_oucru_csv(df, signal_type_hint="ppg")
        assert result is not None
        assert result["signal_column"] == "pleth"
        assert result["samples_per_row"] == 5

    def test_accepts_bytes(self):
        df = pd.DataFrame(
            {"timestamp": ["t1"], "ecg": ["[10,11,12]"]}
        )
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        result = detect_oucru_csv(buffer.getvalue(), signal_type_hint="ecg")
        assert result is not None
        assert result["signal_column"] == "ecg"

    def test_detects_unknown_column_name_by_shape(self, tmp_path: Path):
        # Column name isn't in our preferred list but the cell shape is OUCRU
        df = pd.DataFrame(
            {"timestamp": ["t1"], "custom_signal": ["[1,2,3,4,5,6,7,8]"]}
        )
        path = tmp_path / "custom.csv"
        df.to_csv(path, index=False)
        result = detect_oucru_csv(path)
        assert result is not None
        assert result["signal_column"] == "custom_signal"
        assert result["samples_per_row"] == 8

    def test_empty_csv_returns_none(self, tmp_path: Path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        assert detect_oucru_csv(path) is None

    def test_ppg_hint_skips_ecg_column_when_both_present(self, tmp_path: Path):
        """If both 'pleth' and 'ecg' columns exist, hint should pick the right one."""
        df = pd.DataFrame(
            {
                "timestamp": ["t1"],
                "ecg": ["[1,2,3,4]"],
                "pleth": ["[10,20,30,40,50]"],
            }
        )
        path = tmp_path / "both.csv"
        df.to_csv(path, index=False)

        ppg_result = detect_oucru_csv(path, signal_type_hint="ppg")
        assert ppg_result["signal_column"] == "pleth"
        assert ppg_result["samples_per_row"] == 5

        ecg_result = detect_oucru_csv(path, signal_type_hint="ecg")
        assert ecg_result["signal_column"] == "ecg"
        assert ecg_result["samples_per_row"] == 4


# ---------------------------------------------------------------------------
# default_rate_for_signal_type
# ---------------------------------------------------------------------------


class TestDefaultRate:
    def test_ppg(self):
        assert default_rate_for_signal_type("ppg") == DEFAULT_PPG_RATE

    def test_ecg(self):
        assert default_rate_for_signal_type("ecg") == DEFAULT_ECG_RATE

    def test_ecg_alias_ekg(self):
        assert default_rate_for_signal_type("ekg") == DEFAULT_ECG_RATE

    def test_unknown_returns_none(self):
        assert default_rate_for_signal_type("something_else") is None

    def test_none_returns_none(self):
        assert default_rate_for_signal_type(None) is None
