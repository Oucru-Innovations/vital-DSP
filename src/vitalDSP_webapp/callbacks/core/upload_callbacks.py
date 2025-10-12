"""
Core upload callbacks for vitalDSP webapp.

This module handles file uploads, data validation, and data processing.
"""

import base64
import io
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
from pathlib import Path
import tempfile
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import time

try:
    from vitalDSP_webapp.services.data.data_service import get_data_service
    from vitalDSP_webapp.utils.data_processor import DataProcessor
    from vitalDSP.utils.data_loader import DataLoader, load_oucru_csv
except ImportError:
    # Fallback imports for testing
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from vitalDSP_webapp.services.data.data_service import get_data_service
        from vitalDSP_webapp.utils.data_processor import DataProcessor
    except ImportError:
        # For testing, create mock versions
        def get_data_service():
            return None

        class DataProcessor:
            @staticmethod
            def process_uploaded_data(*args, **kwargs):
                return None

            @staticmethod
            def generate_sample_ppg_data(sampling_freq):
                # Generate sample PPG data for testing
                duration = 10  # seconds
                t = np.linspace(0, duration, int(sampling_freq * duration))
                signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
                signal += 0.1 * np.random.randn(len(signal))
                return pd.DataFrame({'time': t, 'signal': signal})


logger = logging.getLogger(__name__)


def load_data_with_format(
    file_path,
    data_format,
    sampling_freq=None,
    signal_type=None,
    oucru_sampling_rate_column=None,
    oucru_interpolate_time=None,
):
    """
    Load data using DataLoader based on the specified format.

    Args:
        file_path: Path to the data file
        data_format: Format type ('auto', 'csv', 'oucru_csv', 'excel', etc.)
        sampling_freq: Sampling frequency (optional)
        signal_type: Signal type for OUCRU format (ppg/ecg)
        oucru_sampling_rate_column: Column name for sampling rates in OUCRU format
        oucru_interpolate_time: Whether to interpolate timestamps in OUCRU format

    Returns:
        tuple: (DataFrame, metadata dict)
    """
    metadata = {}

    # Handle OUCRU CSV format specially
    if data_format == "oucru_csv":
        # Prepare OUCRU-specific parameters
        signal_type_hint = signal_type if signal_type and signal_type != "auto" else None
        interpolate = True if oucru_interpolate_time and True in oucru_interpolate_time else False

        # Load using load_oucru_csv function
        signal_data, oucru_metadata = load_oucru_csv(
            file_path,
            sampling_rate=sampling_freq,
            signal_type_hint=signal_type_hint,
            sampling_rate_column=oucru_sampling_rate_column,
            interpolate_time=interpolate,
        )

        # Convert to DataFrame if not already
        if not isinstance(signal_data, pd.DataFrame):
            # Assume signal_data is the signal array
            time = np.arange(len(signal_data)) / oucru_metadata.get("sampling_rate", 100)
            df = pd.DataFrame({"time": time, "signal": signal_data})
        else:
            df = signal_data

        # Use metadata from load_oucru_csv
        metadata = oucru_metadata

    else:
        # Use DataLoader for other formats
        loader = DataLoader(file_path, sampling_rate=sampling_freq)

        # Determine format enum
        if data_format == "auto" or not data_format:
            # Auto-detect based on file extension
            ext = Path(file_path).suffix.lower()
            if ext in [".csv", ".txt"]:
                df, metadata = loader.load(format_type="csv")
            elif ext in [".xlsx", ".xls"]:
                df, metadata = loader.load(format_type="excel")
            elif ext == ".h5" or ext == ".hdf5":
                df, metadata = loader.load(format_type="hdf5")
            elif ext == ".parquet":
                df, metadata = loader.load(format_type="parquet")
            elif ext == ".json":
                df, metadata = loader.load(format_type="json")
            elif ext == ".mat":
                df, metadata = loader.load(format_type="matlab")
            else:
                # Try CSV as default
                df, metadata = loader.load(format_type="csv")
        else:
            # Use specified format
            format_map = {
                "csv": "csv",
                "excel": "excel",
                "hdf5": "hdf5",
                "parquet": "parquet",
                "json": "json",
                "wfdb": "wfdb",
                "edf": "edf",
                "matlab": "matlab",
            }
            format_type = format_map.get(data_format, "csv")
            df, metadata = loader.load(format_type=format_type)

    # Add signal type to metadata if provided
    if signal_type and signal_type != "auto":
        metadata["signal_type"] = signal_type.upper()

    return df, metadata


def register_upload_callbacks(app):
    """Register all upload-related callbacks"""
    # Check if callbacks are already registered to prevent duplicates
    if (
        hasattr(app, "_upload_callbacks_registered")
        and getattr(app, "_upload_callbacks_registered", False) is True
    ):
        logger.info("Upload callbacks already registered, skipping")
        return

    logger.info("Registering upload callbacks...")
    app._upload_callbacks_registered = True

    # Callback to toggle OUCRU-specific configuration section
    @app.callback(
        Output("oucru-config-section", "style"),
        Input("data-format", "value"),
    )
    def toggle_oucru_config(data_format):
        """Show/hide OUCRU-specific configuration based on format selection"""
        if data_format == "oucru_csv":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        [
            Output("upload-status", "children", allow_duplicate=True),
            Output("store-uploaded-data", "data", allow_duplicate=True),
            Output("store-data-config", "data", allow_duplicate=True),
            Output("data-preview-section", "children", allow_duplicate=True),
            Output("time-column", "options"),
            Output("signal-column", "options"),
            Output("red-column", "options"),
            Output("ir-column", "options"),
            Output("waveform-column", "options"),
            Output("upload-progress-section", "children"),
            Output("upload-progress-section", "style"),
            Output("btn-load-path", "disabled"),
            Output("btn-load-sample", "disabled"),
            Output("file-path-loading", "children"),
            Output("file-path-loading", "style"),
            Output("upload-data", "className"),
        ],
        [
            Input("upload-data", "contents"),
            Input("btn-load-path", "n_clicks"),
            Input("btn-load-sample", "n_clicks"),
        ],
        [
            State("upload-data", "filename"),
            State("file-path-input", "value"),
            State("sampling-freq", "value"),
            State("time-unit", "value"),
            State("data-type", "value"),
            State("data-format", "value"),
            State("oucru-sampling-rate-column", "value"),
            State("oucru-interpolate-time", "value"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def handle_all_uploads(
        upload_contents,
        load_path_clicks,
        load_sample_clicks,
        filename,
        file_path,
        sampling_freq,
        time_unit,
        data_type,
        data_format,
        oucru_sampling_rate_column,
        oucru_interpolate_time,
    ):
        """Handle all types of data uploads"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Validate trigger before processing
        if trigger_id not in ["upload-data", "btn-load-path", "btn-load-sample"]:
            raise PreventUpdate

        # Show upload progress and disable buttons
        progress_bar = create_upload_progress_bar()
        progress_style = {"display": "block", "animation": "slideInDown 0.5s ease-out"}
        buttons_disabled = True

        # Create file path loading indicator
        file_path_loading = create_file_path_loading_indicator()
        file_path_loading_style = {
            "display": "block",
            "animation": "fadeInUp 0.4s ease-out",
        }

        # Add uploading class to upload area
        upload_area_class = "upload-area uploading"

        try:
            # Process the upload
            df = None
            metadata = {}

            if trigger_id == "upload-data" and upload_contents:
                # Handle file upload - save to temp file and use DataLoader
                content_type, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)

                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                    tmp_file.write(decoded)
                    temp_path = tmp_file.name

                try:
                    # Use DataLoader to load the file based on format
                    df, metadata = load_data_with_format(
                        temp_path,
                        data_format,
                        sampling_freq,
                        data_type,
                        oucru_sampling_rate_column,
                        oucru_interpolate_time,
                    )
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

            elif trigger_id == "btn-load-path" and file_path:
                # Handle load from file path using DataLoader
                try:
                    df, metadata = load_data_with_format(
                        file_path,
                        data_format,
                        sampling_freq,
                        data_type,
                        oucru_sampling_rate_column,
                        oucru_interpolate_time,
                    )
                except Exception as e:
                    return (
                        f"❌ Error loading file: {str(e)}",
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        False,
                        False,
                        no_update,
                        {"display": "none"},
                        "upload-area",
                    )

            elif trigger_id == "btn-load-sample":
                # Handle sample data generation
                df = DataProcessor.generate_sample_ppg_data(sampling_freq or 1000)
                filename = "sample_data.csv"
                metadata = {"sampling_rate": sampling_freq or 1000}

            # Validate that we have data
            if df is None:
                raise ValueError("Failed to load data")

            # Get column options for dropdowns
            column_options = [{"label": col, "value": col} for col in df.columns]

            # Process the data - merge metadata with config
            sampling_freq = metadata.get("sampling_rate", sampling_freq or 100)
            time_unit = time_unit or "seconds"  # Default time unit

            # Create data_info from metadata and user config
            data_info = {
                "filename": filename if trigger_id != "btn-load-sample" else "sample_data.csv",
                "sampling_freq": sampling_freq,
                "time_unit": time_unit,
                "format": data_format or "auto",
                "rows": len(df),
                "columns": len(df.columns),
                "duration": len(df) / sampling_freq if sampling_freq else None,
            }

            # Add metadata from loader
            data_info.update(metadata)

            # Add signal type to the data info
            logger.info(f"Signal type from upload: {data_type}")
            if data_type and data_type != "auto":
                data_info["signal_type"] = data_type.upper()
                logger.info(f"Set signal_type to: {data_info['signal_type']}")
            else:
                data_info["signal_type"] = metadata.get("signal_type", "AUTO")
                logger.info(f"Set signal_type to: {data_info['signal_type']}")

            # Store data temporarily (will be processed after column mapping)
            data_service = get_data_service()
            data_service.current_data = df
            data_service.update_config(data_info)

            # Generate preview
            preview = create_data_preview(df, data_info)

            format_display = data_format if data_format != "auto" else "Auto-detected"
            status = f"✅ Data loaded successfully [{format_display}]: {data_info.get('filename', filename)} ({len(df)} rows, {len(df.columns)} columns, {sampling_freq} Hz)"

            # Hide progress bar after completion and re-enable buttons
            progress_style = {"display": "none"}
            buttons_disabled = False
            file_path_loading_style = {"display": "none"}
            upload_area_class = "upload-area"

            return (
                status,
                df.to_dict("records"),
                data_info,
                preview,
                column_options,
                column_options,
                column_options,
                column_options,
                column_options,
                progress_bar,
                progress_style,
                buttons_disabled,
                buttons_disabled,
                file_path_loading,
                file_path_loading_style,
                upload_area_class,
            )

        except Exception as e:
            logging.error(f"Error in upload: {str(e)}")
            # Hide progress bar on error and re-enable buttons
            progress_style = {"display": "none"}
            buttons_disabled = False
            file_path_loading_style = {"display": "none"}
            upload_area_class = "upload-area"
            return (
                f"❌ Error loading data: {str(e)}",
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                progress_bar,
                progress_style,
                buttons_disabled,
                buttons_disabled,
                file_path_loading,
                file_path_loading_style,
                upload_area_class,
            )

    @app.callback(
        [
            Output("btn-process-data", "disabled", allow_duplicate=True),
            Output("btn-process-data", "color"),
        ],
        [Input("time-column", "value"), Input("signal-column", "value")],
        prevent_initial_call="initial_duplicate",
    )
    def update_process_button_state(time_col, signal_col):
        """Enable/disable process button based on required column selections"""
        if time_col and signal_col:
            return False, "success"
        return True, "secondary"

    @app.callback(
        [
            Output("time-column", "value", allow_duplicate=True),
            Output("signal-column", "value", allow_duplicate=True),
            Output("red-column", "value", allow_duplicate=True),
            Output("ir-column", "value", allow_duplicate=True),
            Output("waveform-column", "value", allow_duplicate=True),
            Output("btn-auto-detect", "disabled"),
            Output("btn-auto-detect", "children"),
        ],
        Input("btn-auto-detect", "n_clicks"),
        [State("store-uploaded-data", "data"), State("store-data-config", "data")],
        prevent_initial_call="initial_duplicate",
    )
    def auto_detect_columns(n_clicks, uploaded_data, data_config):
        """Auto-detect columns based on data content and column names"""
        if not n_clicks or not uploaded_data:
            raise PreventUpdate

        # Disable button and show loading state
        button_disabled = True
        button_content = [
            html.I(className="fas fa-spinner fa-spin me-2", style={"fontSize": "1rem"}),
            html.Span("Detecting...", className="fw-semibold"),
        ]

        try:
            # Process the auto-detection

            df = pd.DataFrame(uploaded_data)
            data_service = get_data_service()
            column_mapping = data_service._auto_detect_columns(df)

            # Re-enable button and restore original content
            button_disabled = False
            button_content = [html.I(className="fas fa-magic me-2"), "Auto-detect"]

            return (
                column_mapping.get("time", None),
                column_mapping.get("signal", None),
                column_mapping.get("red", None),
                column_mapping.get("ir", None),
                column_mapping.get("waveform", None),
                button_disabled,
                button_content,
            )

        except Exception as e:
            logging.error(f"Error in auto-detect: {str(e)}")
            # Re-enable button on error
            button_disabled = False
            button_content = [html.I(className="fas fa-magic me-2"), "Auto-detect"]
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                button_disabled,
                button_content,
            )

    @app.callback(
        [
            Output("upload-status", "children", allow_duplicate=True),
            Output("store-uploaded-data", "data", allow_duplicate=True),
            Output("store-data-config", "data", allow_duplicate=True),
            Output("data-preview-section", "children", allow_duplicate=True),
            Output("processing-progress-section", "children"),
            Output("processing-progress-section", "style"),
            Output("btn-process-data", "disabled", allow_duplicate=True),
            Output("btn-process-data", "children"),
        ],
        Input("btn-process-data", "n_clicks"),
        [
            State("time-column", "value"),
            State("signal-column", "value"),
            State("red-column", "value"),
            State("ir-column", "value"),
            State("waveform-column", "value"),
            State("store-uploaded-data", "data"),
            State("store-data-config", "data"),
            State("sampling-freq", "value"),
            State("time-unit", "value"),
        ],
        prevent_initial_call=True,
    )
    def process_data_with_columns(
        n_clicks,
        time_col,
        signal_col,
        red_col,
        ir_col,
        waveform_col,
        uploaded_data,
        data_config,
        sampling_freq,
        time_unit,
    ):
        """Process data with selected column mapping"""
        if not n_clicks:
            raise PreventUpdate

        # Show processing progress and disable process button
        progress_style = {"display": "block", "animation": "slideInDown 0.5s ease-out"}

        # Create initial processing progress section (Step 1: Validating)
        html.Div(
            [
                html.Div(
                    [
                        html.I(
                            className="fas fa-cogs fa-spin me-3 text-primary",
                            style={"fontSize": "2rem"},
                        ),
                        html.H3("Processing Your Data", className="text-primary mb-0"),
                    ],
                    className="d-flex align-items-center justify-content-center mb-4",
                ),
                dbc.Progress(
                    value=33,
                    animated=True,
                    striped=True,
                    color="primary",
                    className="mb-4",
                    style={"height": "25px", "fontSize": "1.1rem"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.I(
                                    className="fas fa-spinner fa-spin text-primary me-3",
                                    style={"fontSize": "1.2rem"},
                                ),
                                html.Span(
                                    "Validating column mapping...",
                                    className="fw-semibold fs-5 text-primary",
                                ),
                            ],
                            className="mb-3 progress-step active",
                        ),
                        html.Div(
                            [
                                html.I(
                                    className="fas fa-clock text-muted me-3",
                                    style={"fontSize": "1.2rem"},
                                ),
                                html.Span(
                                    "Processing signal data",
                                    className="fw-semibold fs-5 text-muted",
                                ),
                            ],
                            className="mb-3 progress-step pending",
                        ),
                        html.Div(
                            [
                                html.I(
                                    className="fas fa-clock text-muted me-3",
                                    style={"fontSize": "1.2rem"},
                                ),
                                html.Span(
                                    "Storing processed data",
                                    className="fw-semibold fs-5 text-muted",
                                ),
                            ],
                            className="mb-3 progress-step pending",
                        ),
                    ],
                    className="mb-4",
                ),
                html.Div(
                    [
                        html.Strong(
                            "Step 1 of 3: Validating column mapping",
                            className="text-primary fs-5",
                        ),
                        html.Br(),
                        html.Small(
                            "Checking column selections and data format",
                            className="text-muted",
                        ),
                    ],
                    className="text-center mb-4",
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Hide Progress",
                            id="hide-progress-btn",
                            color="outline-secondary",
                            size="md",
                            className="mt-2",
                        )
                    ],
                    className="text-center",
                ),
            ],
            className="p-4 bg-light border border-primary shadow",
            style={
                "zIndex": 1000,
                "position": "relative",
                "minHeight": "250px",
                "fontSize": "1.1rem",
            },
        )

        process_button_disabled = True
        process_button_content = [
            html.I(className="fas fa-spinner fa-spin me-2", style={"fontSize": "1rem"}),
            html.Span("Processing...", className="fw-semibold"),
        ]

        try:
            # Process the data (removed artificial delay)

            df = pd.DataFrame(uploaded_data)

            # Update data config with column mapping
            column_mapping = {
                "time": time_col,
                "signal": signal_col,
                "red": red_col,
                "ir": ir_col,
                "waveform": waveform_col,
            }

            # Debug: Log the data_config before and after
            logger.info(f"Data config before column mapping: {data_config}")
            data_config["column_mapping"] = column_mapping
            logger.info(f"Data config after column mapping: {data_config}")

            # Store the final processed data
            data_service = get_data_service()
            data_id = data_service.store_data(df, data_config)

            # Update status
            status = f"✅ Data processed and stored successfully! Data ID: {data_id}"

            # Generate final preview
            preview = create_data_preview(df, data_config)

            # Update progress section to show completion
            completed_progress = html.Div(
                [
                    html.Div(
                        [
                            html.I(
                                className="fas fa-check-circle text-success me-3",
                                style={"fontSize": "2rem"},
                            ),
                            html.H3(
                                "Processing Complete!", className="text-success mb-0"
                            ),
                        ],
                        className="d-flex align-items-center justify-content-center mb-4",
                    ),
                    dbc.Progress(
                        value=100,
                        animated=False,
                        striped=False,
                        color="success",
                        className="mb-4",
                        style={"height": "25px", "fontSize": "1.1rem"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-3",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Span(
                                        "Column mapping validated",
                                        className="fw-semibold fs-5 text-success",
                                    ),
                                ],
                                className="mb-3 progress-step completed",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-3",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Span(
                                        "Signal data processed",
                                        className="fw-semibold fs-5 text-success",
                                    ),
                                ],
                                className="mb-3 progress-step completed",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-3",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Span(
                                        "Data stored successfully",
                                        className="fw-semibold fs-5 text-success",
                                    ),
                                ],
                                className="mb-3 progress-step completed",
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            html.Strong(
                                "All steps completed successfully!",
                                className="text-success fs-5",
                            ),
                            html.Br(),
                            html.Small(f"Data ID: {data_id}", className="text-muted"),
                        ],
                        className="text-center mb-4",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Hide Progress",
                                id="hide-progress-btn",
                                color="outline-success",
                                size="md",
                                className="mt-2",
                            )
                        ],
                        className="text-center",
                    ),
                ],
                className="p-4 bg-light border border-success shadow",
                style={
                    "zIndex": 1000,
                    "position": "relative",
                    "minHeight": "250px",
                    "fontSize": "1.1rem",
                },
            )

            progress_style = {
                "display": "block",
                "animation": "slideInDown 0.5s ease-out",
            }
            process_button_disabled = False
            process_button_content = [
                html.I(className="fas fa-check me-2"),
                "Process Data",
            ]

            return (
                status,
                df.to_dict("records"),
                data_config,
                preview,
                completed_progress,
                progress_style,
                process_button_disabled,
                process_button_content,
            )

        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")

            # Create error progress section
            error_progress = html.Div(
                [
                    html.Div(
                        [
                            html.I(
                                className="fas fa-exclamation-triangle text-danger me-3",
                                style={"fontSize": "2rem"},
                            ),
                            html.H3("Processing Failed", className="text-danger mb-0"),
                        ],
                        className="d-flex align-items-center justify-content-center mb-4",
                    ),
                    dbc.Progress(
                        value=33,
                        animated=False,
                        striped=False,
                        color="danger",
                        className="mb-4",
                        style={"height": "25px", "fontSize": "1.1rem"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-3",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Span(
                                        "Column mapping validated",
                                        className="fw-semibold fs-5 text-success",
                                    ),
                                ],
                                className="mb-3 progress-step completed",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-times-circle text-danger me-3",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Span(
                                        "Processing failed",
                                        className="fw-semibold fs-5 text-danger",
                                    ),
                                ],
                                className="mb-3 progress-step error",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-clock text-muted me-3",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Span(
                                        "Data storage skipped",
                                        className="fw-semibold fs-5 text-muted",
                                    ),
                                ],
                                className="mb-3 progress-step pending",
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            html.Strong(
                                "Error occurred during processing",
                                className="text-danger fs-5",
                            ),
                            html.Br(),
                            html.Small(f"Error: {str(e)}", className="text-muted"),
                        ],
                        className="text-center mb-4",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Hide Progress",
                                id="hide-progress-btn",
                                color="outline-danger",
                                size="md",
                                className="mt-2",
                            )
                        ],
                        className="text-center",
                    ),
                ],
                className="p-4 bg-light border border-danger shadow",
                style={
                    "zIndex": 1000,
                    "position": "relative",
                    "minHeight": "250px",
                    "fontSize": "1.1rem",
                },
            )

            progress_style = {
                "display": "block",
                "animation": "slideInDown 0.5s ease-out",
            }
            process_button_disabled = False
            process_button_content = [
                html.I(className="fas fa-check me-2"),
                "Process Data",
            ]
            return (
                f"❌ Error processing data: {str(e)}",
                no_update,
                no_update,
                no_update,
                error_progress,
                progress_style,
                process_button_disabled,
                process_button_content,
            )

    # Add a callback to hide the progress section when a hide button is clicked
    @app.callback(
        Output("processing-progress-section", "style", allow_duplicate=True),
        Input("hide-progress-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def hide_progress_section(hide_clicks):
        """Hide the progress section when hide button is clicked"""
        if hide_clicks:
            return {"display": "none"}
        return no_update

    # Add a callback to automatically hide the progress section after 5 seconds
    @app.callback(
        Output("processing-progress-section", "style", allow_duplicate=True),
        Input("processing-progress-section", "children"),
        prevent_initial_call=True,
    )
    def auto_hide_progress_section(progress_children):
        """Automatically hide the progress section after it's displayed"""
        if progress_children:
            # For now, keep it visible - user can manually hide it
            # In a future version, we could add an interval component for auto-hide
            return {"display": "block"}

        return no_update


def create_file_path_loading_indicator():
    """Create a file path loading indicator."""
    return html.Div(
        [
            dbc.Alert(
                [
                    html.Div(
                        [
                            html.I(
                                className="fas fa-spinner fa-spin me-2 text-primary",
                                style={"fontSize": "1.2rem"},
                            ),
                            html.Span(
                                "Loading file from path...",
                                className="fw-semibold text-primary",
                            ),
                        ],
                        className="d-flex align-items-center justify-content-center mb-2",
                    ),
                    dbc.Progress(
                        value=100,
                        animated=True,
                        striped=True,
                        color="primary",
                        style={"height": "15px"},
                    ),
                ],
                color="primary",
                className="border-0 mb-0",
                style={"backgroundColor": "rgba(13, 110, 253, 0.1)"},
            )
        ]
    )


def create_upload_progress_bar():
    """Create an upload progress bar with spinner."""
    return html.Div(
        [
            dbc.Alert(
                [
                    html.Div(
                        [
                            html.I(
                                className="fas fa-upload fa-spin me-3 text-primary",
                                style={"fontSize": "1.5rem"},
                            ),
                            html.Span(
                                "Uploading data...",
                                className="fw-bold text-primary",
                                style={"fontSize": "1.1rem"},
                            ),
                        ],
                        className="mb-3 text-center",
                    ),
                    dbc.Progress(
                        value=100,
                        animated=True,
                        striped=True,
                        color="primary",
                        className="mb-3",
                        style={"height": "20px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-2"
                                    ),
                                    html.Span(
                                        "File selected",
                                        className="fw-semibold text-success",
                                    ),
                                ],
                                className="mb-2 progress-step completed",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-spinner fa-spin text-primary me-2"
                                    ),
                                    html.Span(
                                        "Reading file contents",
                                        className="fw-semibold text-primary",
                                    ),
                                ],
                                className="mb-2 progress-step active",
                            ),
                            html.Div(
                                [
                                    html.I(className="fas fa-clock text-muted me-2"),
                                    html.Span(
                                        "Processing data",
                                        className="fw-semibold text-muted",
                                    ),
                                ],
                                className="mb-2 progress-step pending",
                            ),
                        ]
                    ),
                ],
                color="primary",
                className="border-0",
                style={"backgroundColor": "rgba(13, 110, 253, 0.1)"},
            )
        ]
    )


def create_processing_progress_section():
    """Create a processing progress section with multiple steps."""
    return html.Div(
        [
            dbc.Alert(
                [
                    html.Div(
                        [
                            html.I(
                                className="fas fa-cogs fa-spin me-3 text-info",
                                style={"fontSize": "1.5rem"},
                            ),
                            html.Span(
                                "Processing Data",
                                className="fw-bold text-info",
                                style={"fontSize": "1.1rem"},
                            ),
                        ],
                        className="mb-3 text-center",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-check-circle text-success me-2"
                                    ),
                                    html.Span(
                                        "Validating column mapping",
                                        className="fw-semibold",
                                    ),
                                ],
                                className="mb-2 progress-step completed",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-spinner fa-spin text-primary me-2"
                                    ),
                                    html.Span(
                                        "Processing signal data",
                                        className="fw-semibold text-primary",
                                    ),
                                ],
                                className="mb-2 progress-step active",
                            ),
                            html.Div(
                                [
                                    html.I(className="fas fa-clock text-muted me-2"),
                                    html.Span(
                                        "Storing processed data",
                                        className="fw-semibold text-muted",
                                    ),
                                ],
                                className="mb-2 progress-step pending",
                            ),
                        ]
                    ),
                    dbc.Progress(
                        value=66,
                        animated=True,
                        striped=True,
                        color="info",
                        className="mt-3 mb-3",
                        style={"height": "20px"},
                    ),
                    html.Div(
                        [
                            html.Strong(
                                "Step 2 of 3: Processing signal data",
                                className="text-info",
                            ),
                            html.Br(),
                            html.Small(
                                "This may take a few moments depending on data size",
                                className="text-muted",
                            ),
                        ],
                        className="text-center",
                    ),
                ],
                color="info",
                className="border-0",
                style={"backgroundColor": "rgba(23, 162, 184, 0.1)"},
            )
        ]
    )


def create_error_status(message: str) -> html.Div:
    """Create an error status message."""
    return html.Div(
        [
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.Span(message, className="text-danger"),
        ],
        className="alert alert-danger",
    )


def create_success_status(message: str) -> html.Div:
    """Create a success status message."""
    return html.Div(
        [
            html.I(className="fas fa-check-circle text-success me-2"),
            html.Span(message, className="text-success"),
        ],
        className="alert alert-success",
    )


def create_data_preview(df: pd.DataFrame, data_info: dict) -> html.Div:
    """Create a data preview section."""
    return html.Div(
        [
            html.H4("Data Preview", className="mb-3"),
            html.Div(
                [
                    html.P(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"),
                    html.P(
                        f"Sampling Frequency: {data_info.get('sampling_freq', 'N/A')} Hz"
                    ),
                    html.P(f"Duration: {data_info.get('duration', 'N/A')} seconds"),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    html.H6("First 5 rows:"),
                    dash_table.DataTable(
                        data=df.head().to_dict("records"),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left"},
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                            "fontWeight": "bold",
                        },
                    ),
                ]
            ),
        ]
    )
