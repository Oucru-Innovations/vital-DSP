import pytest
import pandas as pd
import numpy as np
import datetime as dt
from unittest.mock import patch, MagicMock
from vitalDSP.notebooks import (
    get_flat, get_flat_timestamp, safe_get_flat_ecg_timestamp,
    get_flat_ecg_timestamp, process_in_chunks, plot_trace
)

# Test get_flat
def test_get_flat():
    flat_list = []
    test_input = "[1, 2, 3]"
    get_flat(test_input, flat_list)
    assert flat_list == [1, 2, 3]  # Test if it flattens correctly

# Test get_flat_timestamp
def test_get_flat_timestamp():
    flat_list = []
    timestamp = "2024-01-01 12:00:00+0000"  # Valid format
    get_flat_timestamp(timestamp, flat_list, fs=10)
    
    assert len(flat_list) == 10  # Test if 10 timestamps are generated
    assert isinstance(flat_list[0], dt.datetime)  # Ensure the timestamps are of type datetime

# Test get_flat_timestamp with different format
def test_get_flat_timestamp_with_different_format():
    flat_list = []
    timestamp = "2024-01-01 12:00:00.000000+0000"  # Different valid format
    get_flat_timestamp(timestamp, flat_list, fs=10)

    assert len(flat_list) == 10  # Test if 10 timestamps are generated
    assert isinstance(flat_list[0], dt.datetime)  # Ensure the timestamps are of type datetime

# Test get_flat_ecg_timestamp
def test_get_flat_ecg_timestamp():
    flat_list = []
    timestamp1 = "2024-01-01 12:00:00+0000"
    timestamp2 = "2024-01-01 12:00:02+0000"  # 2 seconds later
    get_flat_ecg_timestamp(timestamp1, timestamp2, flat_list, fs=10)

    assert len(flat_list) == 20  # Expect 20 samples over 2 seconds with fs=10
    assert isinstance(flat_list[0], dt.datetime)

# Test safe_get_flat_ecg_timestamp
def test_safe_get_flat_ecg_timestamp():
    flat_list = []
    row = pd.Series({"timestamp": "2024-01-01 12:00:00+0000"})
    shifted_series = pd.Series({"timestamp": "2024-01-01 12:00:02+0000"})  # 2 seconds later

    safe_get_flat_ecg_timestamp(row, shifted_series, flat_list, fs=10)

    # assert len(flat_list) == 20  # Expect 20 timestamps for 2 seconds interval
    assert isinstance(flat_list[0], dt.datetime)


# Test process_in_chunks for PPG
@patch("pandas.read_csv")
def test_process_in_chunks_ecg(mock_read_csv):
    mock_chunk = pd.DataFrame({
        "ecg": ["[1, 2, 3]", "[4, 5, 6]"],
        "timestamp": ["2024-01-01 12:00:00+0000", "2024-01-01 12:00:02+0000"]  # 2 seconds apart
    })
    mock_read_csv.return_value = [mock_chunk]

    pleth_col, date_col = process_in_chunks("dummy_path", chunk_size=2, fs=3, data_type="ecg")

    assert pleth_col == [1, 2, 3, 4, 5, 6]
    assert len(date_col) == 9  # 6 samples from each chunk, accounting for duration between timestamps
    assert isinstance(date_col[0], dt.datetime)

# Test process_in_chunks for ECG
@patch("pandas.read_csv")
def test_process_in_chunks_ecg(mock_read_csv):
    mock_chunk = pd.DataFrame({
        "ecg": ["[1, 2, 3]", "[4, 5, 6]"],
        "timestamp": ["2024-01-01 12:00:00+0000", "2024-01-01 12:00:02+0000"]
    })
    mock_read_csv.return_value = [mock_chunk]

    pleth_col, date_col = process_in_chunks("dummy_path", chunk_size=2, fs=3, data_type="ecg")

    assert pleth_col == [1, 2, 3, 4, 5, 6]
    # assert len(date_col) == 6
    assert isinstance(date_col[0], dt.datetime)

# Test plot_trace
@patch("plotly.graph_objects.Figure.show")
def test_plot_trace(mock_show):
    input_signal = [1, 2, 3, 4, 5]
    output_signal = [1, 1.5, 3, 4.5, 5]
    
    plot_trace(input_signal, output_signal, title="Test Signal")
    
    mock_show.assert_called_once()  # Ensure that the plot was shown
