"""Tests for OUCRU auto-routing in the webapp's ``load_data_with_format``.

When the user leaves ``Data Format`` on ``Auto-detect`` and uploads an
OUCRU-shaped CSV, the callback should silently route to the OUCRU loader
and fill in the signal column — no extra clicks required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vitalDSP_webapp.callbacks.core.upload_callbacks import load_data_with_format


@pytest.fixture
def ppg_oucru_file(tmp_path: Path) -> Path:
    n = 100
    samples = list(range(n))
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
def ecg_bare_file(tmp_path: Path) -> Path:
    n = 128
    cell = ",".join(str(i) for i in range(n))
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:01"],
            "ecg": [cell, cell],
        }
    )
    path = tmp_path / "ecg.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def flat_csv_file(tmp_path: Path) -> Path:
    """Plain flat CSV; auto-route should NOT promote it to oucru_csv."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="10ms"),
            "signal": np.sin(np.linspace(0, 10, 50)),
        }
    )
    path = tmp_path / "flat.csv"
    df.to_csv(path, index=False)
    return path


class TestAutoRouteOucru:
    def test_auto_routes_ppg_oucru(self, ppg_oucru_file: Path):
        df, metadata = load_data_with_format(
            str(ppg_oucru_file),
            data_format="auto",
            signal_type="ppg",
        )
        # OUCRU branch produces an expanded DataFrame with a 'signal' column
        assert "signal" in df.columns
        # 2 rows * 100 samples
        assert len(df) == 200
        assert metadata["format"] == "oucru_csv"
        assert metadata["sampling_rate"] == 100.0

    def test_auto_routes_ecg_bracket_less(self, ecg_bare_file: Path):
        df, metadata = load_data_with_format(
            str(ecg_bare_file),
            data_format="auto",
            signal_type="ecg",
        )
        assert "signal" in df.columns
        assert len(df) == 256  # 2 rows * 128 samples
        assert metadata["format"] == "oucru_csv"
        assert metadata["sampling_rate"] == 128.0

    def test_auto_keeps_flat_csv_as_csv(self, flat_csv_file: Path):
        df, metadata = load_data_with_format(
            str(flat_csv_file),
            data_format="auto",
            signal_type="ppg",
            signal_column="signal",
            time_column="timestamp",
        )
        # Auto-detect did NOT promote it to oucru_csv (would have produced
        # 'sampling_rate' in metadata if so) — fell through to the
        # DataLoader CSV path that preserves the 50 raw rows.
        assert len(df) == 50
        assert metadata.get("format") != "oucru_csv"

    def test_explicit_csv_format_bypasses_auto_detect(self, ppg_oucru_file: Path):
        # When user explicitly picks "csv", we must NOT promote to OUCRU
        # even though the shape matches.
        df, metadata = load_data_with_format(
            str(ppg_oucru_file),
            data_format="csv",
            signal_type="ppg",
            signal_column="pleth",
            time_column="timestamp",
        )
        # Plain CSV path: rows stay as raw array-strings, not expanded
        assert len(df) == 2
        assert metadata["format"] == "csv"
