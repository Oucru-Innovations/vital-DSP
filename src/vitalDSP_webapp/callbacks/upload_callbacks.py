"""
Upload callbacks for vitalDSP webapp.
Handles file uploads, data validation, column mapping, and data preview.
"""

import base64
import io
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
from pathlib import Path
import tempfile
import os
import logging
from datetime import datetime

from ..services.data_service import get_data_service
from ..utils.data_processor import DataProcessor
from ..config.settings import app_config, ui_styles

logger = logging.getLogger(__name__)


def register_upload_callbacks(app):
    """Register all upload-related callbacks."""
    
    @app.callback(
        [Output("upload-status", "children"),
         Output("store-uploaded-data", "data"),
         Output("store-data-config", "data"),
         Output("data-preview-section", "children")],
        [Input("upload-data", "contents"),
         Input("btn-load-path", "n_clicks"),
         Input("btn-load-sample", "n_clicks")],
        [State("upload-data", "filename"),
         State("file-path-input", "value"),
         State("sampling-freq", "value"),
         State("time-unit", "value")]
    )
    def handle_all_uploads(upload_contents, load_path_clicks, load_sample_clicks, filename, file_path, sampling_freq, time_unit):
        """Handle all upload-related actions: file upload, path loading, and sample data."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        data_service = get_data_service()
        
        try:
            df = None
            
            if trigger_id == "upload-data" and upload_contents:
                # Handle file upload
                if not DataProcessor.validate_file_extension(filename):
                    return create_error_status(f"Unsupported file format. Supported formats: {', '.join(app_config.ALLOWED_EXTENSIONS)}"), no_update, no_update, no_update
                
                df = DataProcessor.read_uploaded_content(upload_contents, filename)
                if df is None:
                    return create_error_status("Failed to read uploaded file content"), no_update, no_update, no_update
                    
                upload_method = "file_upload"
                    
            elif trigger_id == "btn-load-path" and file_path and load_path_clicks:
                # Handle file path loading
                if not os.path.exists(file_path):
                    return create_error_status("File not found"), no_update, no_update, no_update
                
                if not DataProcessor.validate_file_extension(file_path):
                    return create_error_status(f"Unsupported file format. Supported formats: {', '.join(app_config.ALLOWED_EXTENSIONS)}"), no_update, no_update, no_update
                
                df = DataProcessor.read_file(file_path, Path(file_path).name)
                if df is None:
                    return create_error_status("Failed to read file from path"), no_update, no_update, no_update
                    
                filename = Path(file_path).name
                upload_method = "path_loading"
                
            elif trigger_id == "btn-load-sample" and load_sample_clicks:
                # Handle sample data loading
                sampling_freq = sampling_freq or app_config.DEFAULT_SAMPLING_FREQ
                time_unit = time_unit or app_config.DEFAULT_TIME_UNIT
                
                df = DataProcessor.generate_sample_ppg_data(sampling_freq)
                filename = "sample_ppg_data.csv"
                upload_method = "sample_data"
                
            else:
                raise PreventUpdate
            
            # Process the data
            sampling_freq = sampling_freq or app_config.DEFAULT_SAMPLING_FREQ
            time_unit = time_unit or app_config.DEFAULT_TIME_UNIT
            
            data_info = DataProcessor.process_uploaded_data(df, filename, sampling_freq, time_unit)
            
            # Store data in service
            metadata = {
                'filename': filename,
                'sampling_freq': sampling_freq,
                'time_unit': time_unit,
                'upload_method': upload_method
            }
            
            data_id = data_service.store_data(df, metadata)
            
            # Create success status and data preview
            status = create_success_status(data_info)
            data_store = {
                "data_id": data_id,
                "dataframe": df.to_dict("records"),
                "columns": df.columns.tolist(),
                "info": data_info
            }
            config_store = {
                "data_id": data_id,
                "sampling_freq": sampling_freq,
                "time_unit": time_unit,
                "filename": data_info["filename"]
            }
            preview = create_data_preview_section(data_info, df)
            
            return status, data_store, config_store, preview
            
        except Exception as e:
            logger.error(f"Error in upload handling: {str(e)}")
            error_status = create_error_status(f"Error processing upload: {str(e)}")
            return error_status, no_update, no_update, no_update
    
    @app.callback(
        [Output("time-column", "options"),
         Output("signal-column", "options"),
         Output("red-column", "options"),
         Output("ir-column", "options")],
        [Input("store-uploaded-data", "data")]
    )
    def update_column_options(data_store):
        """Update column dropdown options based on uploaded data."""
        if not data_store or "columns" not in data_store:
            return [], [], [], []
            
        columns = data_store.get("columns", [])
        options = [{"label": col, "value": col} for col in columns]
        
        return options, options, options, options
    
    @app.callback(
        [Output("btn-process-data", "disabled"),
         Output("btn-auto-detect", "disabled")],
        [Input("store-uploaded-data", "data"),
         Input("time-column", "value"),
         Input("signal-column", "value")]
    )
    def update_button_states(data_store, time_col, signal_col):
        """Enable/disable buttons based on data availability and column selection."""
        has_data = data_store is not None
        has_required_cols = time_col and signal_col
        
        return not (has_data and has_required_cols), not has_data
    
    @app.callback(
        [Output("time-column", "value"),
         Output("signal-column", "value"),
         Output("red-column", "value"),
         Output("ir-column", "value")],
        [Input("btn-auto-detect", "n_clicks")],
        [State("store-uploaded-data", "data")]
    )
    def auto_detect_columns(n_clicks, data_store):
        """Auto-detect columns based on common naming patterns."""
        if not n_clicks or not data_store:
            raise PreventUpdate
            
        try:
            df = pd.DataFrame(data_store.get("dataframe", []))
            mapping = DataProcessor.auto_detect_columns(df)
            
            return mapping['time'], mapping['signal'], mapping['red'], mapping['ir']
            
        except Exception as e:
            logger.error(f"Error in auto-detection: {str(e)}")
            return no_update, no_update, no_update, no_update
    
    @app.callback(
        Output("processing-status", "children"),
        [Input("btn-process-data", "n_clicks")],
        [State("store-uploaded-data", "data"),
         State("store-data-config", "data"),
         State("time-column", "value"),
         State("signal-column", "value"),
         State("red-column", "value"),
         State("ir-column", "value")]
    )
    def process_data(n_clicks, data_store, config_store, time_col, signal_col, red_col, ir_col):
        """Process the uploaded data with selected column mapping."""
        logger.info("=== PROCESS DATA CALLBACK TRIGGERED ===")
        logger.info(f"n_clicks: {n_clicks}")
        logger.info(f"data_store keys: {list(data_store.keys()) if data_store else 'None'}")
        logger.info(f"config_store keys: {list(config_store.keys()) if config_store else 'None'}")
        logger.info(f"time_col: {time_col}")
        logger.info(f"signal_col: {signal_col}")
        logger.info(f"red_col: {red_col}")
        logger.info(f"ir_col: {ir_col}")
        
        if not n_clicks or not data_store:
            logger.warning("No n_clicks or no data_store - raising PreventUpdate")
            raise PreventUpdate
            
        try:
            data_service = get_data_service()
            data_id = data_store.get("data_id")
            logger.info(f"Data ID: {data_id}")
            
            if not data_id:
                logger.error("No data ID found")
                return create_error_status("No data ID found")
            
            # Create column mapping
            column_mapping = {
                "time": time_col,
                "signal": signal_col,
                "red": red_col,
                "ir": ir_col
            }
            logger.info(f"Column mapping created: {column_mapping}")
            
            # Validate column mapping
            df = data_service.get_data(data_id)
            if df is None:
                logger.error("Data not found in service")
                return create_error_status("Data not found")
            
            logger.info(f"Data retrieved from service, shape: {df.shape}")
            logger.info(f"Data columns: {df.columns.tolist()}")
            logger.info(f"Data preview (first 3 rows):")
            logger.info(df.head(3).to_string())
            
            validation_result = DataProcessor.validate_column_mapping(df, column_mapping)
            logger.info(f"Validation result: {validation_result}")
            
            if not validation_result['is_valid']:
                error_messages = "\n".join(validation_result['errors'])
                logger.error(f"Validation failed: {error_messages}")
                return create_error_status(f"Column mapping validation failed:\n{error_messages}")
            
            # Store the mapping in the data service
            logger.info("Storing column mapping in data service...")
            data_service.store_column_mapping(data_id, column_mapping)
            logger.info("Column mapping stored successfully!")
            
            # Verify the mapping was stored
            stored_mapping = data_service.get_column_mapping(data_id)
            logger.info(f"Verification - stored mapping: {stored_mapping}")
            
            # Also store the config info
            config_info = {
                "sampling_freq": config_store.get("sampling_freq", 1000),
                "time_unit": config_store.get("time_unit", "ms"),
                "filename": config_store.get("filename", "unknown")
            }
            logger.info(f"Config info to store: {config_info}")
            
            # Store config in data info
            data_service.update_data_info(data_id, config_info)
            logger.info("Config info stored in data service!")
            
            # Create success message
            success_message = create_processing_success_status(data_store, config_store, column_mapping)
            logger.info("Success message created, returning...")
            return success_message
            
        except Exception as e:
            logger.error(f"Exception in process_data: {str(e)}")
            logger.error(f"Error in data processing: {str(e)}")
            return create_error_status(f"An error occurred while processing the data: {str(e)}")


def create_success_status(data_info):
    """Create success status message."""
    return dbc.Alert([
        html.H5("✅ File Upload Successful!", className="alert-heading"),
        html.P([
            f"Successfully uploaded {data_info['filename']} with ",
            f"{data_info['rows']:,} data points."
        ]),
        html.Hr(),
        html.P([
            html.Strong("File Info: "),
            f"Size: {data_info['size_mb']:.2f} MB, ",
            f"Duration: {data_info['duration_sec']:.1f}s, ",
            f"Quality: {data_info['quality_status']}"
        ])
    ], color="success")


def create_error_status(message):
    """Create error status message."""
    return dbc.Alert([
        html.H5("❌ Upload Error", className="alert-heading"),
        html.P(message)
    ], color="danger")


def create_processing_success_status(data_store, config_store, column_mapping):
    """Create processing success status message."""
    return dbc.Alert([
        html.H5("✅ Data Processing Complete!", className="alert-heading"),
        html.P([
            f"Successfully processed {data_store['info']['rows']:,} data points. ",
            "Your data is now ready for analysis in the Time Domain page."
        ]),
        html.Hr(),
        html.P([
            html.Strong("Column Mapping: "),
            f"Time: {column_mapping['time']}, Signal: {column_mapping['signal']}"
        ] + ([] if not column_mapping['red'] else [f", RED: {column_mapping['red']}"]) + 
        ([] if not column_mapping['ir'] else [f", IR: {column_mapping['ir']}"])),
        html.P([
            html.Strong("Sampling Rate: "),
            f"{config_store['sampling_freq']} Hz"
        ])
    ], color="success", className="mt-3")


def create_data_preview_section(data_info, df):
    """Create the data preview section."""
    from ..layout.upload_section import create_data_preview
    
    # Create data table preview
    preview_df = df.head(app_config.MAX_PREVIEW_ROWS)
    table = dbc.Table.from_dataframe(
        preview_df, 
        striped=ui_styles.TABLE_STRIPED,
        bordered=ui_styles.TABLE_BORDERED,
        hover=ui_styles.TABLE_HOVER,
        className=f"table-{ui_styles.TABLE_SIZE}"
    )
    
    # Create signal preview plot
    if len(df.columns) > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            signal_col = numeric_cols[0]
            # Sample data for plot (limit points to avoid performance issues)
            plot_data = df.head(app_config.MAX_PLOT_POINTS)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data[signal_col],
                mode='lines',
                name=signal_col,
                line=dict(color=ui_styles.PRIMARY_COLOR, width=1)
            ))
            
            fig.update_layout(
                title=f"Signal Preview: {signal_col}",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude",
                template="plotly_white",
                height=300,
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            
            # Update the data info with the plot
            data_info["preview_plot"] = fig
            data_info["preview_table"] = table
            
            return create_data_preview(data_info)
    
    # Fallback if no numeric data
    return create_data_preview(data_info)
