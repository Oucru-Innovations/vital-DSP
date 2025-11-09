"""
Transform Callbacks (Updated Version)
Handles signal transformation analysis with proper State handling for dynamic components.
"""

import logging
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback_context, html, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)


def register_transform_callbacks(app):
    """Register all transform-related callbacks."""

    @app.callback(
        Output("transforms-parameters-container", "children"),
        [Input("transforms-type", "value")],
    )
    def update_transform_parameters(transform_type):
        """Update parameter inputs based on selected transform type."""
        if not transform_type:
            return []

        if transform_type == "fft":
            return [
                html.H6("FFT Parameters", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Window Type", className="form-label"),
                                dbc.Select(
                                    id={
                                        "type": "transform-param",
                                        "param": "fft-window",
                                    },
                                    options=[
                                        {"label": "Rectangular", "value": "boxcar"},
                                        {"label": "Hamming", "value": "hamming"},
                                        {"label": "Hann", "value": "hann"},
                                        {"label": "Blackman", "value": "blackman"},
                                        {"label": "Kaiser", "value": "kaiser"},
                                    ],
                                    value="hann",
                                    className="mb-3",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("N Points", className="form-label"),
                                dbc.Input(
                                    id={
                                        "type": "transform-param",
                                        "param": "fft-npoints",
                                    },
                                    type="number",
                                    value=1024,
                                    min=64,
                                    step=64,
                                    placeholder="1024",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
            ]

        elif transform_type == "stft":
            return [
                html.H6("STFT Parameters", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Window Size", className="form-label"),
                                dbc.Input(
                                    id={
                                        "type": "transform-param",
                                        "param": "stft-windowsize",
                                    },
                                    type="number",
                                    value=256,
                                    min=32,
                                    step=32,
                                    placeholder="256",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Overlap (%)", className="form-label"),
                                dbc.Input(
                                    id={
                                        "type": "transform-param",
                                        "param": "stft-overlap",
                                    },
                                    type="number",
                                    value=50,
                                    min=0,
                                    max=99,
                                    step=1,
                                    placeholder="50",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Window Type", className="form-label"),
                                dbc.Select(
                                    id={
                                        "type": "transform-param",
                                        "param": "stft-window",
                                    },
                                    options=[
                                        {"label": "Hann", "value": "hann"},
                                        {"label": "Hamming", "value": "hamming"},
                                        {"label": "Blackman", "value": "blackman"},
                                    ],
                                    value="hann",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
            ]

        elif transform_type == "wavelet":
            return [
                html.H6("Wavelet Parameters", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Wavelet Type", className="form-label"),
                                dbc.Select(
                                    id={
                                        "type": "transform-param",
                                        "param": "wavelet-type",
                                    },
                                    options=[
                                        {"label": "Morlet", "value": "morl"},
                                        {"label": "Mexican Hat", "value": "mexh"},
                                        {"label": "Daubechies 4", "value": "db4"},
                                        {"label": "Symlet 4", "value": "sym4"},
                                        {"label": "Coiflet 1", "value": "coif1"},
                                    ],
                                    value="morl",
                                    className="mb-3",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Scales", className="form-label"),
                                dbc.Input(
                                    id={
                                        "type": "transform-param",
                                        "param": "wavelet-scales",
                                    },
                                    type="number",
                                    value=64,
                                    min=8,
                                    max=256,
                                    step=8,
                                    placeholder="64",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
            ]

        elif transform_type == "hilbert":
            return [
                html.H6("Hilbert Parameters", className="mb-3"),
                html.P(
                    "Hilbert transform extracts instantaneous amplitude, phase, and frequency.",
                    className="text-muted small",
                ),
            ]

        elif transform_type == "mfcc":
            return [
                html.H6("MFCC Parameters", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "N MFCC Coefficients", className="form-label"
                                ),
                                dbc.Input(
                                    id={
                                        "type": "transform-param",
                                        "param": "mfcc-ncoeffs",
                                    },
                                    type="number",
                                    value=13,
                                    min=5,
                                    max=40,
                                    step=1,
                                    placeholder="13",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("N FFT", className="form-label"),
                                dbc.Input(
                                    id={
                                        "type": "transform-param",
                                        "param": "mfcc-nfft",
                                    },
                                    type="number",
                                    value=512,
                                    min=128,
                                    step=128,
                                    placeholder="512",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
            ]

        return []

    # Store parameters when they change
    @app.callback(
        Output("store-transforms-data", "data"),
        [Input({"type": "transform-param", "param": ALL}, "value")],
        [
            State({"type": "transform-param", "param": ALL}, "id"),
            State("transforms-type", "value"),
        ],
        prevent_initial_call=True,
    )
    def store_parameters(param_values, param_ids, transform_type):
        """Store transform parameters for use in analysis."""
        params = {}
        if param_ids and param_values:
            for param_id, value in zip(param_ids, param_values):
                param_name = param_id["param"]
                params[param_name] = value

        params["transform_type"] = transform_type
        return params

    @app.callback(
        Output("transforms-start-position", "value"),
        [
            Input("transforms-btn-nudge-m10", "n_clicks"),
            Input("transforms-btn-nudge-m5", "n_clicks"),
            Input("transforms-btn-center", "n_clicks"),
            Input("transforms-btn-nudge-p5", "n_clicks"),
            Input("transforms-btn-nudge-p10", "n_clicks"),
        ],
        [State("transforms-start-position", "value")],
    )
    def update_time_position_nudge(
        nudge_m10, nudge_m5, center, nudge_p5, nudge_p10, start_position
    ):
        """Update time position based on nudge button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Use default if value is None
        start_position = start_position if start_position is not None else 0

        # Handle nudge buttons
        if trigger_id == "transforms-btn-nudge-m10":
            return max(0, start_position - 10)
        elif trigger_id == "transforms-btn-nudge-m5":
            return max(0, start_position - 5)
        elif trigger_id == "transforms-btn-center":
            return 50  # Center at 50%
        elif trigger_id == "transforms-btn-nudge-p5":
            return min(100, start_position + 5)
        elif trigger_id == "transforms-btn-nudge-p10":
            return min(100, start_position + 10)

        return start_position

    @app.callback(
        [
            Output("transforms-main-plot", "figure"),
            Output("transforms-analysis-plots", "figure"),
            Output("transforms-analysis-results", "children"),
            Output("transforms-peak-analysis", "children"),
            Output("transforms-frequency-bands", "children"),
            Output("store-transforms-results", "data"),
        ],
        [Input("transforms-analyze-btn", "n_clicks")],
        [
            State("transforms-signal-type", "value"),
            State("transforms-signal-source", "value"),
            State("transforms-start-position", "value"),
            State("transforms-duration", "value"),
            State("transforms-type", "value"),
            State("transforms-analysis-options", "value"),
            State("store-transforms-data", "data"),
            State("store-filtered-signal", "data"),
        ],
    )
    def analyze_transform(
        n_clicks,
        signal_type,
        signal_source,
        start_position,
        duration,
        transform_type,
        analysis_options,
        transform_params,
        filtered_signal_data,
    ):
        """Perform signal transform analysis."""
        # Import transform functions here to avoid circular imports
        from .transform_functions import (
            apply_fft_transform,
            apply_stft_transform,
            apply_wavelet_transform,
            apply_hilbert_transform,
            apply_mfcc_transform,
        )

        if not n_clicks:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Click 'Apply Transform' to begin analysis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return (
                empty_fig,
                empty_fig,
                html.Div("No data available"),
                html.Div("N/A"),
                html.Div("N/A"),
                None,
            )

        try:
            # Get data from the data service
            logger.info("Attempting to get data service...")
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()
            logger.info("Data service retrieved successfully")

            # Get the most recent data
            logger.info("Retrieving all data from service...")
            all_data = data_service.get_all_data()
            logger.info(
                f"All data keys: {list(all_data.keys()) if all_data else 'None'}"
            )

            if not all_data:
                raise ValueError(
                    "No data available. Please upload and process data first."
                )

            # Get the most recent data entry (data_service returns dict with IDs as keys)
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")

            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            logger.info(f"Column mapping: {column_mapping}")

            if not column_mapping:
                raise ValueError("No column mapping found. Please upload data first.")

            # Get the actual data DataFrame
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                raise ValueError("Data is empty. Please upload valid data.")

            logger.info(f"Data shape: {df.shape}, columns: {df.columns.tolist()}")

            # Determine which signal to use (filtered or original)
            signal_data = None
            time_data = None

            if signal_source == "filtered" and filtered_signal_data:
                logger.info("Attempting to use filtered signal data")
                signal_data = np.array(filtered_signal_data.get("filtered_signal", []))
                if len(signal_data) == 0:
                    logger.warning("Filtered signal is empty, falling back to original")
                    signal_data = None

            # Fall back to original signal if filtered not available
            if signal_data is None:
                logger.info("Using original signal data from DataFrame")
                # Get time column
                time_col = column_mapping.get("time")
                if not time_col or time_col not in df.columns:
                    # Try default time column names
                    for col in ["time", "Time", "timestamp"]:
                        if col in df.columns:
                            time_col = col
                            break

                if not time_col:
                    raise ValueError("No time column found in data")

                time_data = df[time_col].values

                # Get signal column based on signal type
                signal_col = None
                if signal_type == "PPG":
                    signal_col = column_mapping.get("signal") or column_mapping.get(
                        "ppg"
                    )
                elif signal_type == "ECG":
                    signal_col = column_mapping.get("ecg")

                # Try common column names if mapping didn't work
                if not signal_col or signal_col not in df.columns:
                    for col in ["signal", "waveform", "ppg", "ecg", "RED", "value"]:
                        if col in df.columns:
                            signal_col = col
                            logger.info(f"Using column: {col}")
                            break

                if not signal_col:
                    raise ValueError(f"No signal column found for type {signal_type}")

                signal_data = df[signal_col].values

            # If we have filtered signal but no time data yet, get time from df
            if time_data is None:
                time_col = column_mapping.get("time")
                if not time_col or time_col not in df.columns:
                    for col in ["time", "Time", "timestamp"]:
                        if col in df.columns:
                            time_col = col
                            break
                if time_col and time_col in df.columns:
                    time_data = df[time_col].values
                else:
                    # Generate time data based on sampling frequency
                    sampling_freq = latest_data.get("info", {}).get(
                        "sampling_freq", 100
                    )
                    time_data = np.arange(len(signal_data)) / sampling_freq

            # Ensure time and signal have same length
            min_len = min(len(time_data), len(signal_data))
            time_data = time_data[:min_len]
            signal_data = signal_data[:min_len]

            # Ensure time_data and signal_data are numeric arrays
            time_data = np.array(time_data, dtype=np.float64)
            signal_data = np.array(signal_data, dtype=np.float64)

            # Check if time data is in Unix timestamp format (very large values)
            # If so, normalize it to start from 0
            if time_data[0] > 1e9:  # Likely Unix timestamp
                logger.info(
                    f"Detected Unix timestamp format, normalizing time data (first value: {time_data[0]:.2e})"
                )

                # Normalize to start from 0 FIRST (to get relative time)
                time_data = time_data - time_data[0]

                # Calculate the time span
                time_span = time_data[-1]
                logger.info(f"Time span after normalization: {time_span:.2e}")

                # Check if we need to convert units based on the span
                # If span is very large, it's likely in smaller units (nanoseconds, microseconds, milliseconds)
                if time_span > 1e9:  # Nanoseconds
                    logger.info("Converting from nanoseconds to seconds")
                    time_data = time_data / 1e9
                elif time_span > 1e6:  # Microseconds
                    logger.info("Converting from microseconds to seconds")
                    time_data = time_data / 1e6
                elif time_span > 1e3:  # Milliseconds
                    logger.info("Converting from milliseconds to seconds")
                    time_data = time_data / 1000.0

            total_duration = float(time_data[-1] - time_data[0])
            logger.info(
                f"Loaded data: {len(signal_data)} samples, duration: {total_duration:.2f}s"
            )

            # Calculate start time from position percentage
            start_position = float(start_position) if start_position is not None else 0
            duration = float(duration) if duration is not None else 60
            start_time = (start_position / 100.0) * total_duration
            end_time = start_time + duration

            # Find indices for time window
            mask = (time_data >= start_time) & (time_data <= end_time)
            if not np.any(mask):
                raise ValueError(
                    f"No data in time window {start_time:.1f}-{end_time:.1f}s"
                )

            windowed_time = time_data[mask]
            windowed_signal = signal_data[mask]

            # Calculate sampling frequency
            if len(windowed_time) > 1:
                sampling_freq = 1 / np.mean(np.diff(windowed_time))
            else:
                sampling_freq = 100  # Default

            logger.info(
                f"Applying {transform_type} transform to {len(windowed_signal)} samples "
                f"at {sampling_freq:.1f} Hz (window: {start_time:.1f}-{end_time:.1f}s)"
            )

            # Extract parameters with defaults
            params = transform_params or {}

            # Apply transform based on type
            if transform_type == "fft":
                window = params.get("fft-window", "hann")
                n_points = params.get("fft-npoints", 1024)
                main_fig, analysis_fig, results, peaks, bands = apply_fft_transform(
                    windowed_time,
                    windowed_signal,
                    sampling_freq,
                    analysis_options,
                    window,
                    int(n_points) if n_points else 1024,
                )
            elif transform_type == "stft":
                window_size = params.get("stft-windowsize", 256)
                overlap = params.get("stft-overlap", 50)
                window = params.get("stft-window", "hann")
                main_fig, analysis_fig, results, peaks, bands = apply_stft_transform(
                    windowed_time,
                    windowed_signal,
                    sampling_freq,
                    analysis_options,
                    int(window_size) if window_size else 256,
                    int(overlap) if overlap else 50,
                    window,
                )
            elif transform_type == "wavelet":
                wavelet_type = params.get("wavelet-type", "morl")
                scales = params.get("wavelet-scales", 64)
                main_fig, analysis_fig, results, peaks, bands = apply_wavelet_transform(
                    windowed_time,
                    windowed_signal,
                    sampling_freq,
                    analysis_options,
                    wavelet_type,
                    int(scales) if scales else 64,
                )
            elif transform_type == "hilbert":
                main_fig, analysis_fig, results, peaks, bands = apply_hilbert_transform(
                    windowed_time,
                    windowed_signal,
                    sampling_freq,
                    analysis_options,
                )
            elif transform_type == "mfcc":
                n_coeffs = params.get("mfcc-ncoeffs", 13)
                n_fft = params.get("mfcc-nfft", 512)
                main_fig, analysis_fig, results, peaks, bands = apply_mfcc_transform(
                    windowed_time,
                    windowed_signal,
                    sampling_freq,
                    analysis_options,
                    int(n_coeffs) if n_coeffs else 13,
                    int(n_fft) if n_fft else 512,
                )
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")

            # Store results
            results_data = {
                "transform_type": transform_type,
                "start_time": start_time,
                "duration": duration,
                "sampling_freq": sampling_freq,
                "n_samples": len(windowed_signal),
            }

            return main_fig, analysis_fig, results, peaks, bands, results_data

        except Exception as e:
            logger.error(f"Error in transform analysis: {e}", exc_info=True)
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="red"),
            )
            error_msg = html.Div(
                [
                    html.H5("Error in Transform Analysis", className="text-danger"),
                    html.P(str(e)),
                ],
                className="alert alert-danger",
            )
            return (
                error_fig,
                error_fig,
                error_msg,
                html.Div("Error"),
                html.Div("Error"),
                None,
            )
