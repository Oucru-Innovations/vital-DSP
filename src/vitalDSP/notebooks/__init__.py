# Init functions to handle the sample data
from plotly import graph_objects as go
import pandas as pd
import numpy as np
import ast
import datetime as dt
import pkg_resources

def load_sample_ecg():
    # Get the path to the file in the package resources
    data_path = pkg_resources.resource_filename('vitalDSP.notebooks', "ecg.csv")
    signal_col, date_col = process_in_chunks(data_path,data_type='ecg', fs=256)
    return signal_col, date_col

def load_sample_ecg_small():
    # Get the path to the file in the package resources
    data_path = pkg_resources.resource_filename('vitalDSP.notebooks', "ecg_small.csv")
    signal_col, date_col = process_in_chunks(data_path,data_type='ecg', fs=256)
    return signal_col, date_col

def load_sample_ppg():
    # Get the path to the file in the package resources
    data_path = pkg_resources.resource_filename('vitalDSP.notebooks', "ppg.csv")
    signal_col, date_col = process_in_chunks(data_path,data_type='ppg', fs=100)
    return signal_col, date_col

def get_flat(x, flat):
    flat.extend(ast.literal_eval(x))


def get_flat_timestamp(x, flat, fs=100):
    format1 = "%Y-%m-%d %H:%M:%S.%f%z"
    format2 = "%Y-%m-%d %H:%M:%S%z"

    try:
        start_time_converted = dt.datetime.strptime(x, format1)
    except ValueError:
        start_time_converted = dt.datetime.strptime(x, format2)

    time_deltas = np.arange(fs) * (1 / fs)
    dt_list = start_time_converted + np.array(
        [dt.timedelta(seconds=td) for td in time_deltas]
    )
    flat.extend(dt_list)


def safe_get_flat_ecg_timestamp(row, shifted_series, flat, fs=100):
    try:
        next_x = shifted_series.loc[row.name]
        get_flat_ecg_timestamp(row["timestamp"], next_x, flat, fs)
    except Exception:
        get_flat_timestamp(row["timestamp"], flat, fs)


def get_flat_ecg_timestamp(x, next_x, flat, fs=100):
    format1 = "%Y-%m-%d %H:%M:%S.%f%z"
    format2 = "%Y-%m-%d %H:%M:%S%z"

    try:
        start_time_converted = dt.datetime.strptime(x, format1)
        end_time_converted = dt.datetime.strptime(next_x, format1)
    except ValueError:
        start_time_converted = dt.datetime.strptime(x, format2)
        end_time_converted = dt.datetime.strptime(next_x, format2)

    total_duration = (end_time_converted - start_time_converted).total_seconds()
    num_samples = int(total_duration * fs)
    time_deltas = np.arange(num_samples) * (1 / fs)

    dt_list = start_time_converted + np.array(
        [dt.timedelta(seconds=td) for td in time_deltas]
    )
    flat.extend(dt_list)


def process_in_chunks(file_path, chunk_size=10000, fs=100, data_type="ppg"):
    pleth_col = []
    date_col = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if data_type == "ppg":
            chunk["pleth"].apply(get_flat, flat=pleth_col)
            chunk["timestamp"].apply(get_flat_timestamp, flat=date_col, fs=fs)
        elif data_type == "ecg":
            shifted_series = chunk["timestamp"].shift(-1)
            chunk["ecg"].apply(get_flat, flat=pleth_col)

            # Use a lambda function to pass the additional arguments
            chunk.apply(
                lambda row: safe_get_flat_ecg_timestamp(
                    row, shifted_series, flat=date_col, fs=fs
                ),
                axis=1,
            )

    return pleth_col, date_col


def plot_trace(input_signal, output_signal, title="Processed Signal"):
    trace1 = go.Scatter(y=input_signal, mode="lines", name="Original Signal")
    trace2 = go.Scatter(y=output_signal, mode="lines", name="Processed Signal")

    layout = go.Layout(
        title=title,
        xaxis=dict(title="Sample Index"),
        yaxis=dict(title="Amplitude"),
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()
