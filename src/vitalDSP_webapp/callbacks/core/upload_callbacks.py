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
    from vitalDSP.utils.data_processing.data_loader import DataLoader, load_oucru_csv
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
                return pd.DataFrame({"time": t, "signal": signal})


logger = logging.getLogger(__name__)


def load_data_headers_only(
    file_path, data_format, sampling_freq=None, data_type="auto"
):
    """
    Load only the headers/columns from a data file without parsing the actual data.
    This is used to show available columns to the user for selection.

    Args:
        file_path: Path to the data file
        data_format: Format of the data file
        sampling_freq: Sampling frequency (optional)
        data_type: Type of data (auto, ppg, ecg, etc.)

    Returns:
        tuple: (available_columns, metadata_dict)
    """
    logger.info(f"Loading headers only from: {file_path}")

    try:
        # Always load as normal CSV first to get headers
        if data_format in ["oucru_csv", "csv", "auto"] or not data_format:
            # Load as normal CSV to get column names
            df_preview = pd.read_csv(file_path, nrows=0)
            available_columns = list(df_preview.columns)

            # Create basic metadata
            metadata = {
                "format": "csv",  # Always treat as CSV for header loading
                "original_format": data_format,  # Keep track of original format
                "available_columns": available_columns,
                "file_path": file_path,
                "sampling_freq": sampling_freq,
                "data_type": data_type,
            }

            logger.info(f"CSV headers loaded: {available_columns}")
            return available_columns, metadata

        else:
            # For other formats, use DataLoader to get headers
            if data_format == "auto" or not data_format:
                ext = Path(file_path).suffix.lower()
                if ext in [".csv", ".txt"]:
                    format_type = "csv"
                elif ext in [".xlsx", ".xls"]:
                    format_type = "excel"
                elif ext == ".h5" or ext == ".hdf5":
                    format_type = "hdf5"
                elif ext == ".parquet":
                    format_type = "parquet"
                elif ext == ".json":
                    format_type = "json"
                elif ext == ".mat":
                    format_type = "matlab"
                else:
                    format_type = "csv"
            else:
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

            # Use DataLoader to get headers without loading full data
            loader = DataLoader(
                file_path, format=format_type, sampling_rate=sampling_freq
            )

            # For most formats, we can read just the headers
            if format_type in ["csv", "excel", "parquet"]:
                if format_type == "csv":
                    df_preview = pd.read_csv(file_path, nrows=0)
                elif format_type == "excel":
                    df_preview = pd.read_excel(file_path, nrows=0)
                elif format_type == "parquet":
                    df_preview = pd.read_parquet(file_path)

                available_columns = list(df_preview.columns)
            else:
                # For other formats, we might need to load a small sample
                try:
                    df_sample = loader.load()
                    if isinstance(df_sample, pd.DataFrame):
                        available_columns = list(df_sample.columns)
                    else:
                        available_columns = ["signal"]  # Fallback
                except Exception as e:
                    logger.warning(
                        f"Could not load sample for format {format_type}: {e}"
                    )
                    available_columns = ["signal"]  # Fallback

            metadata = {
                "format": format_type,
                "available_columns": available_columns,
                "file_path": file_path,
                "sampling_freq": sampling_freq,
                "data_type": data_type,
            }

            logger.info(f"Headers loaded for format {format_type}: {available_columns}")
            return available_columns, metadata

    except Exception as e:
        logger.error(f"Error loading headers: {str(e)}")
        raise ValueError(f"Error reading file headers: {str(e)}")


def load_data_with_format(
    file_path,
    data_format,
    sampling_freq=None,
    signal_type=None,
    signal_column=None,
    time_column=None,
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
        signal_column: User-selected signal column name
        time_column: User-selected time column name
        oucru_sampling_rate_column: Column name for sampling rates in OUCRU format
        oucru_interpolate_time: Whether to interpolate timestamps in OUCRU format

    Returns:
        tuple: (DataFrame, metadata dict)
    """
    metadata = {}

    # Handle OUCRU CSV format specially
    if data_format == "oucru_csv":
        # Validate user's column selections
        if not signal_column:
            raise ValueError("Signal column must be specified for OUCRU CSV format")

        logger.info(f"Using user-selected signal column: {signal_column}")
        logger.info(f"Using user-selected time column: {time_column}")

        # Check if the signal column contains array strings (OUCRU format)
        try:
            # Read a few rows to check the signal column format
            df_sample = pd.read_csv(file_path, nrows=3)
            signal_sample = df_sample[signal_column].iloc[0]

            # Check if it's an array string (starts with [ and ends with ])
            if (
                isinstance(signal_sample, str)
                and signal_sample.strip().startswith("[")
                and signal_sample.strip().endswith("]")
            ):
                logger.info(
                    f"Detected OUCRU format: signal column contains array strings"
                )
                is_oucru_format = True
            else:
                logger.info(
                    f"Detected normal CSV format: signal column contains individual values"
                )
                is_oucru_format = False

        except Exception as e:
            logger.warning(f"Could not detect format from signal column: {e}")
            is_oucru_format = False

        if is_oucru_format:
            # Process as OUCRU CSV format
            logger.info("Processing as OUCRU CSV format")
            logger.info(
                f"Signal column '{signal_column}' contains array strings - will expand"
            )
        else:
            # Process as normal CSV format
            logger.info("Processing as normal CSV format")
            logger.info(f"Signal column '{signal_column}' contains individual values")
            # Fall through to normal CSV processing below
            data_format = "csv"

        # Prepare OUCRU-specific parameters
        signal_type_hint = (
            signal_type if signal_type and signal_type != "auto" else None
        )
        interpolate = (
            True if oucru_interpolate_time and True in oucru_interpolate_time else False
        )

        # Load using load_oucru_csv function with user's column selections
        signal_data, oucru_metadata = load_oucru_csv(
            file_path,
            time_column=time_column or "timestamp",  # Use user's time column selection
            signal_column=signal_column,  # Use user's signal column selection
            sampling_rate=sampling_freq,
            signal_type_hint=signal_type_hint,
            sampling_rate_column=oucru_sampling_rate_column,
            interpolate_time=interpolate,
        )

        logger.info(f"OUCRU CSV processing completed:")
        logger.info(f"  Signal data shape: {signal_data.shape}")
        logger.info(f"  Signal data type: {type(signal_data)}")
        logger.info(f"  Metadata keys: {list(oucru_metadata.keys())}")
        if "timestamps" in oucru_metadata:
            logger.info(f"  Timestamps type: {type(oucru_metadata['timestamps'])}")
            if isinstance(oucru_metadata["timestamps"], pd.DataFrame):
                logger.info(
                    f"  Timestamps DataFrame shape: {oucru_metadata['timestamps'].shape}"
                )
                logger.info(
                    f"  Timestamps DataFrame columns: {list(oucru_metadata['timestamps'].columns)}"
                )

        # Convert to DataFrame if not already
        if not isinstance(signal_data, pd.DataFrame):
            # signal_data is a numpy array, create DataFrame with timestamps
            if interpolate and "timestamps" in oucru_metadata:
                # Use the timestamps DataFrame from metadata if available
                timestamps_df = oucru_metadata["timestamps"]
                if isinstance(timestamps_df, pd.DataFrame):
                    # The timestamps DataFrame already contains the expanded signal data
                    df = timestamps_df.copy()
                    logger.info(f"Using timestamps DataFrame from metadata: {df.shape}")
                    logger.info(f"Timestamps DataFrame columns: {list(df.columns)}")
                else:
                    # Fallback: create timestamps
                    timestamps = np.arange(len(signal_data)) / oucru_metadata.get(
                        "sampling_rate", sampling_freq
                    )
                    df = pd.DataFrame({"time": timestamps, "signal": signal_data})
            else:
                # Create simple DataFrame without interpolated timestamps
                timestamps = np.arange(len(signal_data)) / oucru_metadata.get(
                    "sampling_rate", sampling_freq
                )
                df = pd.DataFrame({"time": timestamps, "signal": signal_data})
        else:
            df = signal_data

        # Use metadata from load_oucru_csv
        metadata = oucru_metadata.copy()
        # Add user's column selections to metadata
        metadata["detected_signal_column"] = signal_column
        metadata["detected_time_column"] = time_column

        # Convert any DataFrame objects to JSON-serializable format
        if "timestamps" in metadata and isinstance(
            metadata["timestamps"], pd.DataFrame
        ):
            # Convert timestamps DataFrame to dict, ensuring timestamps are strings
            timestamps_df = metadata["timestamps"].copy()
            for col in timestamps_df.columns:
                if pd.api.types.is_datetime64_any_dtype(timestamps_df[col]):
                    timestamps_df[col] = timestamps_df[col].astype(str)
            metadata["timestamps"] = timestamps_df.to_dict("records")
        if "row_data" in metadata and isinstance(metadata["row_data"], pd.DataFrame):
            # Convert row_data DataFrame to dict, ensuring timestamps are strings
            row_data_df = metadata["row_data"].copy()
            for col in row_data_df.columns:
                if pd.api.types.is_datetime64_any_dtype(row_data_df[col]):
                    row_data_df[col] = row_data_df[col].astype(str)
            metadata["row_data"] = row_data_df.to_dict("records")

    elif data_format == "csv":
        # Handle normal CSV format
        logger.info(f"Processing normal CSV with signal column: {signal_column}")

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Validate that the selected columns exist
        if signal_column and signal_column not in df.columns:
            raise ValueError(
                f"Signal column '{signal_column}' not found in CSV. Available columns: {list(df.columns)}"
            )

        if time_column and time_column not in df.columns:
            raise ValueError(
                f"Time column '{time_column}' not found in CSV. Available columns: {list(df.columns)}"
            )

        # Create metadata
        metadata = {
            "format": "csv",
            "file_path": file_path,
            "sampling_freq": sampling_freq,
            "data_type": signal_type,
            "signal_column": signal_column,
            "time_column": time_column,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
        }

        logger.info(f"Normal CSV loaded: {df.shape}, columns: {list(df.columns)}")

    else:
        # Use DataLoader for other formats
        # Determine format enum first
        if data_format == "auto" or not data_format:
            # Auto-detect based on file extension
            ext = Path(file_path).suffix.lower()
            if ext in [".csv", ".txt"]:
                format_type = "csv"
            elif ext in [".xlsx", ".xls"]:
                format_type = "excel"
            elif ext == ".h5" or ext == ".hdf5":
                format_type = "hdf5"
            elif ext == ".parquet":
                format_type = "parquet"
            elif ext == ".json":
                format_type = "json"
            elif ext == ".mat":
                format_type = "matlab"
            else:
                # Try CSV as default
                format_type = "csv"
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

        # Initialize DataLoader with the correct format
        loader = DataLoader(file_path, format=format_type, sampling_rate=sampling_freq)

        # Load the data
        df = loader.load()

        # Get metadata from loader
        metadata = loader.metadata.copy()

        # Convert any DataFrame objects to JSON-serializable format
        for key, value in metadata.items():
            if isinstance(value, pd.DataFrame):
                # Convert DataFrame to dict, ensuring timestamps are strings
                df_copy = value.copy()
                for col in df_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].astype(str)
                metadata[key] = df_copy.to_dict("records")

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

    # Callback to set default column values based on data config
    @app.callback(
        [
            Output("time-column", "value"),
            Output("signal-column", "value"),
            Output("red-column", "value"),
            Output("ir-column", "value"),
            Output("waveform-column", "value"),
        ],
        Input("store-data-config", "data"),
        prevent_initial_call=True,
    )
    def set_default_column_values(data_config):
        """Set default values for column dropdowns based on detected columns"""
        if not data_config:
            return no_update, no_update, no_update, no_update, no_update

        # Set default signal column for OUCRU CSV
        default_signal_column = data_config.get("default_signal_column")
        if default_signal_column:
            logger.info(
                f"Setting default signal column value to: {default_signal_column}"
            )
            return no_update, default_signal_column, no_update, no_update, no_update

        return no_update, no_update, no_update, no_update, no_update

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
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(filename).suffix
                ) as tmp_file:
                    tmp_file.write(decoded)
                    temp_path = tmp_file.name

                try:
                    # Load headers only for file uploads too
                    available_columns, metadata = load_data_headers_only(
                        temp_path,
                        data_format,
                        sampling_freq,
                        data_type,
                    )

                    # Store the temp file path for later processing
                    metadata["file_path"] = temp_path

                except Exception as e:
                    # Clean up temp file on error
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    raise e

                # Store headers and metadata for later processing
                data_service = get_data_service()
                data_service.current_headers = available_columns
                data_service.current_metadata = metadata

                # Create column options for dropdowns
                column_options = [
                    {"label": col, "value": col} for col in available_columns
                ]

                # Auto-detect potential signal columns
                potential_signal_columns = [
                    "signal",
                    "ecg",
                    "ppg",
                    "rri",
                    "rr",
                    "hr",
                    "waveform",
                    "pleth",
                ]
                detected_signal_column = None
                for col in potential_signal_columns:
                    if col in available_columns:
                        detected_signal_column = col
                        break

                # Auto-detect potential time columns
                potential_time_columns = ["timestamp", "time", "ts", "datetime"]
                detected_time_column = None
                for col in potential_time_columns:
                    if col in available_columns:
                        detected_time_column = col
                        break

                # Update metadata with detected columns
                metadata["detected_signal_column"] = detected_signal_column
                metadata["detected_time_column"] = detected_time_column

                # Create preview message
                preview_message = f"✅ File uploaded successfully!\n\n"
                preview_message += f"📁 File: {filename}\n"
                preview_message += f"📊 Format: {metadata['format']}\n"
                preview_message += (
                    f"📋 Available columns: {', '.join(available_columns)}\n"
                )
                if detected_signal_column:
                    preview_message += (
                        f"🎯 Auto-detected signal column: {detected_signal_column}\n"
                    )
                if detected_time_column:
                    preview_message += (
                        f"⏰ Auto-detected time column: {detected_time_column}\n"
                    )
                preview_message += f"\n💡 Please select your columns below and click 'Process Data' to continue."

                return (
                    preview_message,
                    no_update,  # store-uploaded-data.data
                    metadata,  # store-data-config.data
                    html.Div(
                        [
                            html.H5("📋 Available Columns", className="mb-3"),
                            html.P(
                                f"Found {len(available_columns)} columns in the file:",
                                className="text-muted mb-3",
                            ),
                            html.Ul(
                                [html.Li(col) for col in available_columns],
                                className="list-unstyled",
                            ),
                            html.Hr(),
                            html.P(
                                "Please configure your column mapping below and click 'Process Data' to continue.",
                                className="text-info",
                            ),
                        ]
                    ),
                    column_options,  # time-column.options
                    column_options,  # signal-column.options
                    column_options,  # red-column.options
                    column_options,  # ir-column.options
                    column_options,  # waveform-column.options
                    no_update,  # upload-progress-section.children
                    {"display": "none"},  # upload-progress-section.style
                    False,  # btn-load-path.disabled
                    False,  # btn-load-sample.disabled
                    no_update,  # file-path-loading.children
                    {"display": "none"},  # file-path-loading.style
                    "upload-area",  # upload-data.className
                )

            elif trigger_id == "btn-load-path" and file_path:
                # Handle load from file path - load headers only for column selection
                try:
                    available_columns, metadata = load_data_headers_only(
                        file_path,
                        data_format,
                        sampling_freq,
                        data_type,
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
                    )

                # Store headers and metadata for later processing
                data_service = get_data_service()
                data_service.current_headers = available_columns
                data_service.current_metadata = metadata

                # Create column options for dropdowns
                column_options = [
                    {"label": col, "value": col} for col in available_columns
                ]

                # Auto-detect potential signal columns
                potential_signal_columns = [
                    "signal",
                    "ecg",
                    "ppg",
                    "rri",
                    "rr",
                    "hr",
                    "waveform",
                    "pleth",
                ]
                detected_signal_column = None
                for col in potential_signal_columns:
                    if col in available_columns:
                        detected_signal_column = col
                        break

                # Auto-detect potential time columns
                potential_time_columns = ["timestamp", "time", "ts", "datetime"]
                detected_time_column = None
                for col in potential_time_columns:
                    if col in available_columns:
                        detected_time_column = col
                        break

                # Update metadata with detected columns
                metadata["detected_signal_column"] = detected_signal_column
                metadata["detected_time_column"] = detected_time_column

                # Create preview message
                preview_message = f"✅ File loaded successfully!\n\n"
                preview_message += f"📁 File: {Path(file_path).name}\n"
                preview_message += f"📊 Format: {metadata['format']}\n"
                preview_message += (
                    f"📋 Available columns: {', '.join(available_columns)}\n"
                )
                if detected_signal_column:
                    preview_message += (
                        f"🎯 Auto-detected signal column: {detected_signal_column}\n"
                    )
                if detected_time_column:
                    preview_message += (
                        f"⏰ Auto-detected time column: {detected_time_column}\n"
                    )
                preview_message += f"\n💡 Please select your columns below and click 'Process Data' to continue."

                return (
                    preview_message,
                    no_update,  # store-uploaded-data.data
                    metadata,  # store-data-config.data
                    html.Div(
                        [
                            html.H5("📋 Available Columns", className="mb-3"),
                            html.P(
                                f"Found {len(available_columns)} columns in the file:",
                                className="text-muted mb-3",
                            ),
                            html.Ul(
                                [html.Li(col) for col in available_columns],
                                className="list-unstyled",
                            ),
                            html.Hr(),
                            html.P(
                                "Please configure your column mapping below and click 'Process Data' to continue.",
                                className="text-info",
                            ),
                        ]
                    ),
                    column_options,  # time-column.options
                    column_options,  # signal-column.options
                    column_options,  # red-column.options
                    column_options,  # ir-column.options
                    column_options,  # waveform-column.options
                    no_update,  # upload-progress-section.children
                    {"display": "none"},  # upload-progress-section.style
                    False,  # btn-load-path.disabled
                    False,  # btn-load-sample.disabled
                    no_update,  # file-path-loading.children
                    {"display": "none"},  # file-path-loading.style
                    "upload-area",  # upload-data.className
                )

            elif trigger_id == "btn-load-sample":
                # Handle sample data generation
                df = DataProcessor.generate_sample_ppg_data(sampling_freq or 1000)
                filename = "sample_data.csv"
                metadata = {"sampling_rate": sampling_freq or 1000}

                # Get column options for dropdowns
                column_options = [{"label": col, "value": col} for col in df.columns]

                # Process the data - merge metadata with config
                sampling_freq = metadata.get("sampling_rate", sampling_freq)
                time_unit = time_unit or "seconds"  # Default time unit

                # Create data_info from metadata and user config
                data_info = {
                    "filename": "sample_data.csv",
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

                format_display = (
                    data_format if data_format != "auto" else "Auto-detected"
                )
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
                    "upload-area",  # upload-data.className
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

            # Parse the actual signal data based on user's column selection
            data_service = get_data_service()

            # Get the stored metadata and headers
            metadata = data_service.current_metadata
            if not metadata:
                raise ValueError("No file metadata found. Please upload a file first.")

            file_path = metadata.get("file_path")
            if not file_path:
                raise ValueError("No file path found in metadata.")

            logger.info(f"Processing signal data from: {file_path}")
            logger.info(f"User selected columns: time={time_col}, signal={signal_col}")

            # Parse the actual signal data using the user's column selection
            # Determine format based on original format and signal column content
            original_format = metadata.get("original_format", "csv")

            df, processed_metadata = load_data_with_format(
                file_path,
                original_format,  # Use original format (oucru_csv, csv, etc.)
                metadata.get("sampling_freq"),
                metadata.get("data_type"),
                signal_column=signal_col,  # Use user's signal column selection
                time_column=time_col,  # Use user's time column selection
                oucru_interpolate_time=[
                    True
                ],  # Enable timestamp interpolation for OUCRU CSV
            )

            logger.info(f"Parsed signal data shape: {df.shape}")
            logger.info(f"Parsed signal data columns: {list(df.columns)}")

            # Update data config with column mapping
            # Use the user's actual column selections from the frontend UI
            column_mapping = {
                "time": time_col,
                "signal": signal_col,
                "red": red_col,
                "ir": ir_col,
                "waveform": waveform_col,
            }

            # Log the user's selections and available columns for debugging
            logger.info(f"User selected columns: {column_mapping}")
            logger.info(f"Available columns in processed data: {list(df.columns)}")

            # For OUCRU CSV, we need to ensure the user's selected columns exist in the processed data
            if original_format == "oucru_csv":
                logger.info("OUCRU CSV format detected - validating column selections")
                logger.info(f"Processed OUCRU data columns: {list(df.columns)}")
                logger.info(f"User selected signal column: {signal_col}")
                logger.info(f"User selected time column: {time_col}")

                # For OUCRU CSV, the processed data should have standardized column names
                # Map user selections to the actual processed column names
                if "signal" in df.columns:
                    logger.info("Using 'signal' column from processed OUCRU data")
                    column_mapping["signal"] = "signal"
                else:
                    logger.warning("No 'signal' column found in processed OUCRU data")

                if "timestamp" in df.columns:
                    logger.info("Using 'timestamp' column from processed OUCRU data")
                    column_mapping["time"] = "timestamp"
                elif "time" in df.columns:
                    logger.info("Using 'time' column from processed OUCRU data")
                    column_mapping["time"] = "time"
                else:
                    logger.warning("No time column found in processed OUCRU data")

            # Debug: Log the data_config before and after
            logger.info(f"Data config before column mapping: {data_config}")
            data_config["column_mapping"] = column_mapping
            logger.info(f"Data config after column mapping: {data_config}")

            # Log essential signal data info (optimized for performance)
            logger.info("=== SIGNAL DATA SUMMARY ===")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")

            # Log signal column summary (not full data)
            signal_col = column_mapping.get("signal")
            if signal_col and signal_col in df.columns:
                signal_data = df[signal_col]

                # Check if signal data is numeric before trying to format min/max
                if pd.api.types.is_numeric_dtype(signal_data):
                    logger.info(
                        f"Signal column '{signal_col}': dtype={signal_data.dtype}, range={signal_data.min():.3f} to {signal_data.max():.3f}, count={signal_data.count()}"
                    )
                else:
                    logger.info(
                        f"Signal column '{signal_col}': dtype={signal_data.dtype}, count={signal_data.count()}, unique_values={signal_data.nunique()}"
                    )

                # Check for non-numeric values (only log if there are issues)
                try:
                    numeric_data = pd.to_numeric(signal_data, errors="coerce")
                    non_numeric_count = (
                        numeric_data.isnull().sum() - signal_data.isnull().sum()
                    )
                    if non_numeric_count > 0:
                        logger.warning(
                            f"Signal column '{signal_col}' contains {non_numeric_count} non-numeric values!"
                        )
                except Exception as e:
                    logger.error(f"Error checking numeric values: {str(e)}")
            else:
                logger.error(f"Signal column '{signal_col}' not found in DataFrame!")

            # Log time column summary
            time_col = column_mapping.get("time")
            if time_col and time_col in df.columns:
                time_data = df[time_col]
                logger.info(
                    f"Time column '{time_col}': dtype={time_data.dtype}, count={time_data.count()}"
                )
            else:
                logger.error(f"Time column '{time_col}' not found in DataFrame!")

            logger.info("=== END SIGNAL DATA SUMMARY ===")

            # Store the final processed data
            data_service = get_data_service()
            logger.info(
                f"Storing processed data: shape={df.shape}, columns={list(df.columns)}"
            )
            data_id = data_service.store_data(df, data_config)
            logger.info(f"Data stored successfully with ID: {data_id}")

            # Clean up temp file if it exists
            temp_file_path = metadata.get("file_path")
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temp file: {temp_file_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not clean up temp file {temp_file_path}: {str(e)}"
                    )

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
    """Create a data preview section with pagination for large datasets."""
    # Determine if we need pagination based on data size
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    
    # For large datasets, use pagination
    if total_rows > 1000:
        # Use first 100 rows for preview with pagination
        preview_data = df.head(100).to_dict("records")
        page_size = 25
        page_action = "native"
        virtualization = True
        show_pagination_info = True
    else:
        # For smaller datasets, show all data
        preview_data = df.to_dict("records")
        page_size = 10
        page_action = "native"
        virtualization = False
        show_pagination_info = False
    
    return html.Div(
        [
            html.H4("Data Preview", className="mb-3"),
            html.Div(
                [
                    html.P(f"Shape: {total_rows:,} rows × {total_cols} columns"),
                    html.P(
                        f"Sampling Frequency: {data_info.get('sampling_freq', 'N/A')} Hz"
                    ),
                    html.P(f"Duration: {data_info.get('duration', 'N/A')} seconds"),
                    html.P(
                        f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                        className="text-muted small"
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    html.H6("Data Table:"),
                    html.P(
                        f"Showing {min(100, total_rows)} of {total_rows:,} rows" if show_pagination_info else "",
                        className="text-muted small mb-2"
                    ),
                    dash_table.DataTable(
                        id="data-preview-table",
                        data=preview_data,
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "fontSize": "12px",
                            "fontFamily": "monospace"
                        },
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                            "fontWeight": "bold",
                        },
                        style_data={
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                        # Pagination settings
                        page_size=page_size,
                        page_action=page_action,
                        virtualization=virtualization,
                        # Sorting and filtering
                        sort_action="native",
                        filter_action="native",
                        # Performance optimizations
                        fixed_rows={"headers": True},
                        # Export options
                        export_format="csv",
                        export_headers="display",
                    ),
                ]
            ),
        ]
    )
