"""Tests for the smart-default behavior of ``load_oucru_csv``.

These cover the auto-detection of the signal column when it isn't given
explicitly, plus the new bracket-less array cell support in the inner
``_load_oucru_csv`` parser.  Pre-existing behavior is covered by the
older test files; this file targets the recent additions only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vitalDSP.utils.data_processing.data_loader import load_oucru_csv


@pytest.fixture
def ppg_pleth_file(tmp_path: Path) -> Path:
    """PPG file with column named 'pleth' — should be auto-picked."""
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
    """ECG file with bracket-less array cells."""
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


class TestAutoDetectSignalColumn:
    def test_ppg_auto_pick_pleth(self, ppg_pleth_file: Path):
        signal, metadata = load_oucru_csv(
            ppg_pleth_file,
            signal_type_hint="ppg",
            # signal_column omitted — should auto-detect 'pleth'
        )
        assert signal.shape == (200,)
        # sampling_rate priority: explicit hint -> default_ppg_rate (100)
        assert metadata["sampling_rate"] == 100.0

    def test_ecg_auto_pick_with_bracket_less(self, ecg_bare_file: Path):
        signal, metadata = load_oucru_csv(
            ecg_bare_file,
            signal_type_hint="ecg",
        )
        assert signal.shape == (256,)
        assert metadata["sampling_rate"] == 128.0

    def test_explicit_signal_column_still_works(self, ppg_pleth_file: Path):
        # Passing signal_column explicitly should bypass auto-detection
        signal, metadata = load_oucru_csv(
            ppg_pleth_file,
            signal_column="pleth",
            sampling_rate=100,
        )
        assert signal.shape == (200,)

    def test_auto_detect_disabled_falls_back_to_signal_default(
        self, tmp_path: Path
    ):
        # When auto_detect is False and no column given, the historical
        # default 'signal' is used — which won't exist in this file, so
        # the loader should raise.
        path = tmp_path / "ppg.csv"
        pd.DataFrame(
            {"timestamp": ["t1"], "pleth": ["[1,2,3]"]}
        ).to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_oucru_csv(
                path,
                signal_type_hint="ppg",
                auto_detect_signal_column=False,
            )

    def test_auto_detect_raises_when_no_column_matches(self, tmp_path: Path):
        # File doesn't look like OUCRU at all — auto-detect should fail
        # explicitly rather than silently pick a non-array column.
        path = tmp_path / "flat.csv"
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="10ms"),
                "signal": np.arange(10, dtype=float),
            }
        ).to_csv(path, index=False)
        with pytest.raises(ValueError, match="auto-detect"):
            load_oucru_csv(path, signal_type_hint="ppg")
