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
    # Migration to enhanced data service complete (2025-10-31)
    # OLD DATA SERVICE - Commented out, no longer used:
    # from vitalDSP_webapp.services.data.data_service import get_data_service
    from vitalDSP_webapp.services.data.enhanced_data_service import (
        get_enhanced_data_service,
        EnhancedDataService,
    )
    from vitalDSP_webapp.utils.data_processor import DataProcessor
    from vitalDSP_webapp.utils.plot_utils import limit_plot_data, check_plot_data_size
    from vitalDSP.utils.data_processing.data_loader import DataLoader, load_oucru_csv
    from vitalDSP_webapp.services.progress_tracker import get_progress_tracker
    from vitalDSP_webapp.utils.column_introspect import (
        candidate_to_option,
        introspect_columns,
    )

    ENHANCED_SERVICE_AVAILABLE = True
except ImportError as e:
    # Logger not yet defined, will log later
    ENHANCED_SERVICE_AVAILABLE = False
    # Fallback imports for testing
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from vitalDSP_webapp.services.data.enhanced_data_service import (
            get_enhanced_data_service,
        )
        from vitalDSP_webapp.utils.data_processor import DataProcessor
    except ImportError:
        # For testing, create mock versions
        def get_enhanced_data_service():
            return None

        ENHANCED_SERVICE_AVAILABLE = False

        class DataProcessor:
            @staticmethod
            def process_uploaded_data(*args, **kwargs):
                return None

            @staticmethod
            def generate_sample_ppg_data(sampling_freq):
                # Generate sample PPG data for testing using vitalDSP
                duration = 10  # seconds
                try:
                    from vitalDSP.utils.data_processing.synthesize_data import (
                        generate_synthetic_ppg,
                    )

                    signal_data = generate_synthetic_ppg(
                        duration=duration, fs=sampling_freq
                    )
                    t = np.linspace(0, duration, len(signal_data))
                    return pd.DataFrame({"time": t, "signal": signal_data})
                except ImportError:
                    # Fallback to numpy implementation if vitalDSP not available
                    t = np.linspace(0, duration, int(sampling_freq * duration))
                    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(
                        2 * np.pi * 2.4 * t
                    )
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

    # Auto-detect OUCRU shape when the user left the format dropdown on
    # "Auto-detect" (or didn't set one).  We only peek when the file
    # extension is CSV-like; any positive hit promotes the format to
    # 'oucru_csv' and fills in the detected signal column, so the rest
    # of the function can take the OUCRU branch below.
    if (not data_format or data_format == "auto") and file_path:
        ext = Path(file_path).suffix.lower()
        if ext in (".csv", ".txt", ".tsv"):
            try:
                from vitalDSP.utils.data_processing.oucru_detect import (
                    detect_oucru_csv,
                )

                hint = signal_type if signal_type and signal_type != "auto" else None
                detection = detect_oucru_csv(file_path, signal_type_hint=hint)
            except Exception as exc:
                logger.debug("OUCRU auto-detection skipped: %s", exc)
                detection = None
            if detection is not None:
                logger.info(
                    "Auto-detected OUCRU CSV: column=%r, samples/row=%d, style=%s",
                    detection["signal_column"],
                    detection["samples_per_row"],
                    detection["bracket_style"],
                )
                data_format = "oucru_csv"
                # Don't override an explicit user pick — only fill the gap.
                if not signal_column:
                    signal_column = detection["signal_column"]

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

        # ``metadata["timestamps"]`` is the SAME parsed (timestamp,signal)
        # DataFrame that we already returned as ``df``; serialising it
        # would double the dcc.Store payload.  ``row_data`` is the
        # unexpanded source DataFrame - one row per second, each holding
        # the full JSON-array-string, also huge.  Both are dropped here
        # because the parsed signal in ``df`` is all downstream pages
        # actually use.
        metadata.pop("timestamps", None)
        metadata.pop("row_data", None)

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

        # Get metadata from loader, but drop any DataFrame entries -
        # they're either duplicates of the parsed ``df`` we already
        # return, or large source frames that bloat the dcc.Store and
        # don't get used downstream.
        metadata = {
            k: v for k, v in loader.metadata.items()
            if not isinstance(v, pd.DataFrame)
        }

    # Add signal type to metadata if provided
    if signal_type and signal_type != "auto":
        metadata["signal_type"] = signal_type.upper()

    return df, metadata


def _spill_upload_to_tempfile(upload_contents: str, filename: str) -> str:
    """Decode a ``dcc.Upload`` data URL into a NamedTemporaryFile and return its path.

    The temp file is left on disk; the caller is responsible for cleanup
    (typically once the load + process pipeline has consumed it).
    """
    _, content_string = upload_contents.split(",")
    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix if filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(decoded)
        return fh.name


def _slim_progress_alert(message: str, color: str = "primary") -> html.Div:
    """A one-line in-flight indicator that replaces the old multi-step card.

    ``dbc.Progress`` keeps it lightweight and Plotly-friendly (no nested
    icon trees re-rendered on every callback fire).
    """
    return html.Div(
        [
            html.Small(message, className=f"text-{color}"),
            dbc.Progress(
                value=100,
                animated=True,
                striped=True,
                color=color,
                style={"height": "8px"},
                className="mt-1",
            ),
        ]
    )


def _staged_tuple(
    summary: str,
    metadata: dict,
    options: list,
    default_col,
    row_style: dict,
):
    """Build the success-path 11-tuple for ``handle_all_uploads``.

    The stores stay untouched at staging time - only headers were read.
    Process Data is the callback that actually parses the file and
    fills ``store-uploaded-data`` / ``data-preview-section``.
    """
    return (
        summary,                              # upload-status.children
        no_update,                            # store-uploaded-data.data
        metadata,                             # store-data-config.data
        no_update,                            # data-preview-section.children
        options,                              # signal-column.options
        default_col,                          # signal-column.value
        row_style,                            # signal-column-row.style
        no_update,                            # upload-progress-section.children
        {"display": "none"},                  # upload-progress-section.style
        False,                                # btn-load-sample-ppg.disabled
        False,                                # btn-load-sample-ecg.disabled
    )


def _error_tuple(message: str):
    """Build the error-path 11-tuple for ``handle_all_uploads``."""
    return (
        message,                # upload-status.children
        no_update,
        no_update,
        no_update,
        no_update,              # signal-column.options
        no_update,              # signal-column.value
        no_update,              # signal-column-row.style
        no_update,              # upload-progress-section.children
        {"display": "none"},    # upload-progress-section.style
        False,                  # btn-load-sample-ppg.disabled
        False,                  # btn-load-sample-ecg.disabled
    )


#: Suffix marker on datetime column names emitted by
#: :func:`_df_to_compact_payload`.  The column values are int64
#: nanoseconds since the epoch; consumers that need real timestamps
#: call :func:`rehydrate_payload` (or just ``pd.to_datetime(arr,
#: unit='ns')`` on the column).  A name suffix is used instead of a
#: top-level sidecar so the payload remains a flat dict-of-lists that
#: pandas can ingest directly with ``pd.DataFrame(payload)``.
_NS_SUFFIX = "__ns__"


def _df_to_compact_payload(df: pd.DataFrame) -> dict:
    """Serialise a DataFrame for ``store-uploaded-data`` compactly.

    Returns a dict-of-lists (one entry per column).  Datetime columns
    are emitted as **int64 nanoseconds** under a name suffixed with
    ``__ns__`` - benchmarked at ~20x faster than ``Series.tolist()``
    for a 1 M-row datetime column.  Downstream consumers that just
    need the signal can read ``payload['signal']`` directly; consumers
    that need real timestamps use :func:`rehydrate_payload` (one
    ``pd.to_datetime`` call per datetime column, also fast).
    """
    payload: dict = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            # Strip timezone before viewing as int64 (datetime64[ns] only).
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            payload[f"{col}{_NS_SUFFIX}"] = (
                s.values.astype("datetime64[ns]").astype("int64").tolist()
            )
        else:
            payload[col] = s.tolist()
    return payload


def rehydrate_payload(payload: dict) -> pd.DataFrame:
    """Inverse of :func:`_df_to_compact_payload`.

    Detects ``__ns__``-suffixed columns and converts them back to
    ``datetime64[ns]`` with their original names.  Other columns pass
    through unchanged.  Returns an empty DataFrame on an empty payload.
    """
    if not isinstance(payload, dict) or not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    for col in list(df.columns):
        if col.endswith(_NS_SUFFIX):
            real_name = col[: -len(_NS_SUFFIX)]
            df[real_name] = pd.to_datetime(df[col], unit="ns")
            df = df.drop(columns=[col])
    return df


def _column_options_for(file_path: str, data_type: str):
    """Build the signal-column dropdown options for a freshly staged file.

    Returns ``(options, default_value, row_style)``.  Falls back to a
    plain alphabetical list of all columns if introspection fails (e.g.
    non-CSV format) so the dropdown still works.
    """
    try:
        candidates = introspect_columns(file_path, signal_type=data_type)
    except Exception as exc:
        logger.debug("introspect_columns failed: %s", exc)
        candidates = []
    if candidates:
        options = [candidate_to_option(c) for c in candidates]
        default = candidates[0].name
        return options, default, {"display": "block"}
    # Fall back to header read + plain options
    try:
        df_head = pd.read_csv(file_path, nrows=0)
        columns = list(df_head.columns)
    except Exception:
        columns = []
    if not columns:
        return [], None, {"display": "none"}
    options = [{"label": col, "value": col} for col in columns]
    return options, columns[0], {"display": "block"}


def register_upload_callbacks(app):
    """Register the slim upload-page callback set.

    The page exposes one file dropzone, a signal-type select, a sampling
    rate field, a single signal-column dropdown (populated post-upload),
    and a Process Data button.  Everything else (OUCRU vs flat CSV
    detection, signal-column auto-pick, sampling-rate inference) happens
    behind the scenes in :func:`load_data_with_format`.
    """
    if (
        hasattr(app, "_upload_callbacks_registered")
        and getattr(app, "_upload_callbacks_registered", False) is True
    ):
        logger.info("Upload callbacks already registered, skipping")
        return

    logger.info("Registering upload callbacks...")
    app._upload_callbacks_registered = True

    @app.callback(
        [
            Output("upload-status", "children", allow_duplicate=True),
            Output("store-uploaded-data", "data", allow_duplicate=True),
            Output("store-data-config", "data", allow_duplicate=True),
            Output("data-preview-section", "children", allow_duplicate=True),
            Output("signal-column", "options"),
            Output("signal-column", "value"),
            Output("signal-column-row", "style"),
            Output("upload-progress-section", "children"),
            Output("upload-progress-section", "style"),
            Output("btn-load-sample-ppg", "disabled"),
            Output("btn-load-sample-ecg", "disabled"),
        ],
        [
            Input("upload-data", "contents"),
            Input("btn-load-sample-ppg", "n_clicks"),
            Input("btn-load-sample-ecg", "n_clicks"),
        ],
        [
            State("upload-data", "filename"),
            State("sampling-freq", "value"),
            State("data-type", "value"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def handle_all_uploads(
        upload_contents,
        load_ppg_clicks,
        load_ecg_clicks,
        filename,
        sampling_freq,
        data_type,
    ):
        """Stage a freshly uploaded or synthetic recording.

        Three trigger sources share the same handler: the drop-zone,
        and the two synthetic-data buttons (PPG / ECG).  Each produces
        the same 11-tuple of outputs.
        """
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id not in (
            "upload-data", "btn-load-sample-ppg", "btn-load-sample-ecg",
        ):
            raise PreventUpdate

        try:
            if trigger_id == "upload-data" and upload_contents:
                temp_path = _spill_upload_to_tempfile(upload_contents, filename)
                try:
                    available_columns, metadata = load_data_headers_only(
                        temp_path, "auto", sampling_freq, data_type
                    )
                    metadata["file_path"] = temp_path
                except Exception:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise

                # Metadata + headers travel through store-data-config now;
                # no per-process state on the data service is needed (which
                # also makes background callbacks usable).
                options, default_col, row_style = _column_options_for(
                    temp_path, data_type
                )
                # NB: don't parse the whole file here.  The original
                # design did, but for OUCRU recordings the heavy parse
                # blocked the signal-column dropdown from showing for
                # seconds.  Vital-sqi shows the dropdown as soon as the
                # headers are read; full parsing happens on Process Data.
                summary = (
                    f"{filename} loaded - "
                    f"{len(available_columns)} columns, "
                    f"format: {metadata.get('format', 'auto')}"
                )
                return _staged_tuple(
                    summary, metadata, options, default_col, row_style,
                )

            elif trigger_id in ("btn-load-sample-ppg", "btn-load-sample-ecg"):
                fs = sampling_freq or 1000
                if trigger_id == "btn-load-sample-ecg":
                    df = DataProcessor.generate_sample_ecg_data(fs)
                    signal_kind = "ECG"
                    filename_synth = "sample_ecg.csv"
                else:
                    df = DataProcessor.generate_sample_ppg_data(fs)
                    signal_kind = "PPG"
                    filename_synth = "sample_ppg.csv"

                data_info = {
                    "filename": filename_synth,
                    "sampling_freq": fs,
                    "format": "synthetic",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "duration": len(df) / fs,
                    # User can override the kind via the radio; default to
                    # whichever button was pressed.
                    "signal_type": (
                        data_type.upper()
                        if data_type and data_type != "auto"
                        else signal_kind
                    ),
                }
                data_service = get_enhanced_data_service()
                data_id = data_service.store_data(df, data_info)
                if data_id:
                    data_info["data_id"] = data_id

                preview = create_data_preview(df, data_info)
                status = (
                    f"Synthetic {signal_kind} loaded: {len(df)} rows, "
                    f"{len(df.columns)} columns @ {fs} Hz"
                )
                options = [{"label": col, "value": col} for col in df.columns]
                default_col = "signal" if "signal" in df.columns else df.columns[0]

                return (
                    status,
                    _df_to_compact_payload(df),
                    data_info,
                    preview,
                    options,                # signal-column.options
                    default_col,            # signal-column.value
                    {"display": "block"},   # signal-column-row.style
                    no_update,              # upload-progress-section.children
                    {"display": "none"},    # upload-progress-section.style
                    False,                  # btn-load-sample-ppg.disabled
                    False,                  # btn-load-sample-ecg.disabled
                )

        except Exception as exc:
            logger.exception("Error in upload: %s", exc)
            return _error_tuple(f"Error loading data: {exc}")

    @app.callback(
        [
            Output("btn-process-data", "disabled", allow_duplicate=True),
            Output("btn-process-data", "color"),
        ],
        Input("signal-column", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def update_process_button_state(signal_col):
        """Enable Process Data once the user has a signal column picked."""
        if signal_col:
            return False, "primary"
        return True, "secondary"

    @app.callback(
        Output("store-data-config", "data", allow_duplicate=True),
        Input("signal-column", "value"),
        State("store-data-config", "data"),
        prevent_initial_call=True,
    )
    def restage_on_column_change(signal_col, data_config):
        """Record the user's column pick in the config without re-parsing.

        Heavy work (full file parse) is deferred to the Process Data
        click - so changing the dropdown stays instant.  The selected
        column travels with ``store-data-config`` so Process Data
        knows which column to expand.
        """
        if not signal_col or not data_config:
            raise PreventUpdate
        if data_config.get("signal_column") == signal_col:
            raise PreventUpdate
        new_config = dict(data_config)
        new_config["signal_column"] = signal_col
        return new_config

    # NOTE: Process Data runs SYNCHRONOUSLY in the main Dash process.
    # An earlier version wired this through Dash's DiskcacheManager
    # (background=True) for non-blocking UI + cancellation, but the
    # background worker is a separate subprocess and so doesn't share
    # the in-memory ``enhanced_data_service`` singleton with the main
    # UI - the data analysis pages on /filtering, /features etc. would
    # see an empty data registry.  After the parse-once / vectorise /
    # compact-payload work this callback typically runs in 50-200 ms
    # for normal recordings, so synchronous is fine.  If we later want
    # the responsiveness back, the data service needs a filesystem
    # backing (see dev_docs/webapp_perf_followup.md).
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
            State("signal-column", "value"),
            State("store-uploaded-data", "data"),
            State("store-data-config", "data"),
            State("sampling-freq", "value"),
        ],
        prevent_initial_call=True,
    )
    def process_data_with_columns(
        n_clicks,
        signal_col,
        uploaded_data,
        data_config,
        sampling_freq,
    ):
        """Persist the already-parsed data and render the final preview.

        Parses the staged file using the chosen signal column, persists
        the result via the data service, and renders the preview.  This
        is where the heavy work happens - the upload + column-change
        callbacks above only read headers, so the UI stays responsive
        until the user explicitly asks to process.
        """
        if not n_clicks:
            raise PreventUpdate
        if not signal_col:
            return (
                "Pick a signal column before pressing Process Data.",
                no_update,
                no_update,
                no_update,
                no_update,
                {"display": "none"},
                False,
                "Process data",
            )

        tracker = get_progress_tracker()
        task_id = tracker.start_task(
            operation_name="Data Processing",
            total_steps=1,
            metadata={"chunks_processed": 0, "total_chunks": 1},
        )

        progress_style = {"display": "block"}

        try:
            metadata = data_config or {}

            # Synthetic-data short-circuit.  When the user clicks one
            # of the synthetic-data buttons we already parsed the
            # DataFrame, stored it on the data service, and wrote it
            # to ``store-uploaded-data``.  There is no file on disk,
            # so the file-path branch below would fail with
            # "No staged file found".  Just re-render the preview
            # against the chosen signal column and confirm.
            if metadata.get("format") == "synthetic":
                df = rehydrate_payload(uploaded_data) if uploaded_data else None
                if df is None or df.empty:
                    raise ValueError(
                        "Synthetic data missing from store; please regenerate."
                    )
                data_config = {**metadata, "signal_column": signal_col}
                # Make sure sampling_freq is numeric and stays put.
                fs = (
                    data_config.get("sampling_freq")
                    or data_config.get("sampling_rate")
                    or sampling_freq
                    or 1000
                )
                try:
                    fs = float(fs)
                except (TypeError, ValueError):
                    fs = 1000.0
                data_config["sampling_freq"] = fs
                data_config["sampling_rate"] = fs

                # Refresh the data-service entry with the up-to-date
                # config (in case the user changed signal_type or
                # sampling rate after the synthetic load).
                data_service = get_enhanced_data_service()
                data_id = metadata.get("data_id") or data_service.store_data(
                    df, data_config
                )
                data_config["data_id"] = data_id

                tracker.complete_task(
                    task_id=task_id,
                    metadata={"data_id": data_id, "chunks_processed": 1},
                )

                preview = create_data_preview(df, data_config)
                kind = metadata.get("signal_type", "synthetic")
                status = (
                    f"Processed synthetic {kind}: "
                    f"{df.shape[0]:,} rows × {df.shape[1]} columns @ {fs:g} Hz."
                )
                done_indicator = _slim_progress_alert("Done", color="success")
                return (
                    status,
                    uploaded_data,           # already correct, just echo it
                    data_config,
                    preview,
                    done_indicator,
                    progress_style,
                    False,
                    "Process data",
                )

            file_path = metadata.get("file_path")
            if not file_path or not os.path.exists(file_path):
                raise ValueError(
                    "No staged file found. Please upload a file first."
                )
            original_format = metadata.get("original_format", "auto")
            data_type = metadata.get("data_type", "auto")
            df, processed_metadata = load_data_with_format(
                file_path,
                original_format,
                metadata.get("sampling_freq") or sampling_freq,
                data_type,
                signal_column=signal_col,
                time_column=None,
                oucru_interpolate_time=[True],
            )

            data_config = {**metadata, **processed_metadata}
            data_config["signal_column"] = signal_col

            # Normalise the sampling-rate key so downstream consumers always
            # see a numeric ``sampling_freq``.  OUCRU loaders write
            # ``sampling_rate`` but the analysis callbacks (filtering,
            # features, ...) read ``sampling_freq``; without this they
            # crashed with TypeError on ``len(df) / sampling_freq``.
            #
            # ``data_config`` may already carry ``sampling_freq=None``
            # from the staged-header pass (when the user left the field
            # blank); we must *overwrite* that None, not fall through to
            # a missing-key default.  Try, in order: the rate inferred
            # by the OUCRU loader, the user's input, the existing
            # config value, and finally a 1000 Hz fallback so the
            # downstream maths never see None.
            candidates = [
                processed_metadata.get("sampling_rate"),
                sampling_freq,
                data_config.get("sampling_freq"),
                data_config.get("sampling_rate"),
            ]
            inferred_fs = None
            for c in candidates:
                if c is None:
                    continue
                try:
                    val = float(c)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    inferred_fs = val
                    break
            if inferred_fs is None:
                # Last-resort fallback so the analysis pages don't crash
                # on ``len(df) / None``.  Logged so it's not silent.
                logger.warning(
                    "Could not infer sampling frequency from metadata or "
                    "user input; falling back to 1000 Hz.  Tried: %s",
                    candidates,
                )
                inferred_fs = 1000.0
            data_config["sampling_freq"] = inferred_fs
            data_config["sampling_rate"] = inferred_fs

            data_service = get_enhanced_data_service()
            data_id = data_service.store_data(df, data_config)
            data_config["data_id"] = data_id

            tracker.complete_task(
                task_id=task_id,
                metadata={"data_id": data_id, "chunks_processed": 1},
            )

            preview = create_data_preview(df, data_config)
            status = (
                f"Processed {df.shape[0]:,} rows x {df.shape[1]} columns "
                f"@ {processed_metadata.get('sampling_rate', sampling_freq)} Hz."
            )
            done_indicator = _slim_progress_alert("Done", color="success")
            return (
                status,
                _df_to_compact_payload(df),
                data_config,
                preview,
                done_indicator,
                progress_style,
                False,
                "Process data",
            )
        except Exception as exc:
            logger.exception("Process Data failed: %s", exc)
            err_indicator = html.Div(
                [
                    html.Small(f"Error: {exc}", className="text-danger"),
                ],
                className="alert alert-danger py-2 mb-0 small",
            )
            return (
                f"Error: {exc}",
                no_update,
                no_update,
                no_update,
                err_indicator,
                progress_style,
                False,
                "Process data",
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


def create_data_preview(df: pd.DataFrame, data_info: dict) -> html.Div:
    """Render a modern Data Preview block: stat chips + styled DataTable.

    Stat chips replace the four bulleted ``html.P`` lines (rows, fs,
    duration, memory) so the most-used numbers are scannable at a
    glance.  The table picks up the page-scoped styles defined in
    ``upload_page.css`` via the ``data-preview-table-wrap`` class on
    the outer wrapper.
    """
    total_rows = df.shape[0]
    total_cols = df.shape[1]

    if total_rows > 1000:
        preview_data = df.head(100).to_dict("records")
        page_size = 25
        page_action = "native"
        virtualization = True
        show_pagination_info = True
    else:
        preview_data = df.to_dict("records")
        page_size = 10
        page_action = "native"
        virtualization = False
        show_pagination_info = False

    fs_val = data_info.get("sampling_freq")
    fs_text = f"{fs_val} Hz" if fs_val not in (None, "N/A") else "—"

    dur_val = data_info.get("duration")
    try:
        dur_text = f"{float(dur_val):.1f} s" if dur_val not in (None, "N/A") else "—"
    except (TypeError, ValueError):
        dur_text = str(dur_val)

    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    def _chip(icon: str, label: str, value: str) -> html.Span:
        return html.Span(
            [
                html.I(className=f"{icon} me-2 text-primary"),
                html.Span(label, className="me-1 text-muted small"),
                html.Span(value, className="fw-semibold"),
            ],
            className="preview-chip",
        )

    chips_row = html.Div(
        [
            _chip("fas fa-table", "Shape", f"{total_rows:,} × {total_cols}"),
            _chip("fas fa-clock", "Sampling", fs_text),
            _chip("fas fa-stopwatch", "Duration", dur_text),
            _chip("fas fa-memory", "Memory", f"{mem_mb:.2f} MB"),
        ],
        className="preview-chip-row mb-3",
    )

    pagination_hint = (
        f"Showing first 100 of {total_rows:,} rows  ·  use pagination below for more"
        if show_pagination_info
        else f"Showing all {total_rows:,} rows"
    )

    return html.Div(
        [
            # Keep the literal "Data Preview" heading visible so screen
            # readers and tests both have an anchor; the visual weight
            # comes from the chip row below it.
            html.H6(
                "Data Preview",
                className="fw-semibold text-uppercase text-muted small mb-2",
                style={"letterSpacing": "0.04em"},
            ),
            chips_row,
            html.Small(pagination_hint, className="text-muted d-block mb-2"),
            html.Div(
                dash_table.DataTable(
                    id="data-preview-table",
                    data=preview_data,
                    columns=[{"name": c, "id": c} for c in df.columns],
                    style_table={"overflowX": "auto", "borderRadius": "10px"},
                    style_cell={
                        "textAlign": "left",
                        "fontSize": "0.825rem",
                        "fontFamily": (
                            "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"
                        ),
                        "padding": "0.55rem 0.75rem",
                        "border": "0",
                        "borderBottom": "1px solid #f1f3f5",
                        "color": "#212529",
                    },
                    style_header={
                        "backgroundColor": "#f8f9fa",
                        "color": "#495057",
                        "fontWeight": "600",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.04em",
                        "fontSize": "0.72rem",
                        "border": "0",
                        "borderBottom": "2px solid #dee2e6",
                    },
                    style_data={
                        "whiteSpace": "normal",
                        "height": "auto",
                        "backgroundColor": "#ffffff",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#fbfbfd",
                        },
                        {
                            "if": {"state": "active"},
                            "backgroundColor": "#e7f1ff",
                            "border": "0",
                        },
                        {
                            "if": {"state": "selected"},
                            "backgroundColor": "#cfe2ff",
                            "border": "0",
                        },
                    ],
                    style_filter={
                        "backgroundColor": "#ffffff",
                        "color": "#495057",
                        "fontSize": "0.78rem",
                        "border": "0",
                        "borderBottom": "1px solid #dee2e6",
                    },
                    css=[
                        # Round the wrapper so style_table's radius is visible.
                        {"selector": ".dash-spreadsheet", "rule": "border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04);"},
                    ],
                    page_size=page_size,
                    page_action=page_action,
                    virtualization=virtualization,
                    sort_action="native",
                    filter_action="native",
                    fixed_rows={"headers": True},
                    export_format="csv",
                    export_headers="display",
                ),
                className="data-preview-table-wrap",
            ),
        ]
    )

