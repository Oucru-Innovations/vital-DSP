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


logger = logging.getLogger(__name__)


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
    ):
        """Handle all types of data uploads"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

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

            if trigger_id == "upload-data" and upload_contents:
                # Handle file upload
                content_type, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)

                if filename.endswith(".csv"):
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                elif filename.endswith(".txt"):
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")
                else:
                    return (
                        "❌ Unsupported file format. Please upload CSV or TXT files.",
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

            elif trigger_id == "btn-load-path" and file_path:
                # Handle load from file path
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".txt"):
                    df = pd.read_csv(file_path, sep="\t")
                else:
                    return (
                        "❌ Unsupported file format. Please use CSV or TXT files.",
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

            else:
                raise PreventUpdate

            # Get column options for dropdowns
            column_options = [{"label": col, "value": col} for col in df.columns]

            # Process the data
            sampling_freq = sampling_freq or 1000  # Default sampling freq
            time_unit = time_unit or "seconds"  # Default time unit

            data_info = DataProcessor.process_uploaded_data(
                df, filename, sampling_freq, time_unit
            )

            # Store data temporarily (will be processed after column mapping)
            data_service = get_data_service()
            data_service.current_data = df
            data_service.update_config(data_info)

            # Generate preview
            preview = create_data_preview(df, data_info)

            status = f"✅ Data loaded successfully: {filename} ({len(df)} rows, {len(df.columns)} columns)"

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

            data_config["column_mapping"] = column_mapping

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
