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

from vitalDSP_webapp.services.data.data_service import get_data_service
from vitalDSP_webapp.utils.data_processor import DataProcessor
from vitalDSP_webapp.config.settings import app_config, ui_styles

logger = logging.getLogger(__name__)


def register_upload_callbacks(app):
    """Register all upload-related callbacks"""
    
    @app.callback(
        [Output("upload-status", "children"),
         Output("store-uploaded-data", "data"),
         Output("store-data-config", "data"),
         Output("data-preview-section", "children"),
         Output("time-column", "options"),
         Output("signal-column", "options"),
         Output("red-column", "options"),
         Output("ir-column", "options"),
         Output("waveform-column", "options")],
        [Input("upload-data", "contents"),
         Input("btn-load-path", "n_clicks"),
         Input("btn-load-sample", "n_clicks")],
        [State("upload-data", "filename"),
         State("file-path-input", "value"),
         State("sampling-freq", "value"),
         State("time-unit", "value")]
    )
    def handle_all_uploads(upload_contents, load_path_clicks, load_sample_clicks, 
                          filename, file_path, sampling_freq, time_unit):
        """Handle all types of data uploads"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        try:
            if trigger_id == "upload-data" and upload_contents:
                # Handle file upload
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif filename.endswith('.txt'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')
                else:
                    return "❌ Unsupported file format. Please upload CSV or TXT files.", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
                    
            elif trigger_id == "btn-load-path" and file_path:
                # Handle load from file path
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.txt'):
                    df = pd.read_csv(file_path, sep='\t')
                else:
                    return "❌ Unsupported file format. Please use CSV or TXT files.", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
                    
            elif trigger_id == "btn-load-sample":
                # Handle sample data generation
                df = DataProcessor.generate_sample_ppg_data(sampling_freq or 1000)
                filename = "sample_data.csv"
                
            else:
                raise PreventUpdate
                
            # Get column options for dropdowns
            column_options = [{"label": col, "value": col} for col in df.columns]
            
            # Process the data
            sampling_freq = sampling_freq or app_config.DEFAULT_SAMPLING_FREQ
            time_unit = time_unit or app_config.DEFAULT_TIME_UNIT
            
            data_info = DataProcessor.process_uploaded_data(df, filename, sampling_freq, time_unit)
            
            # Store data temporarily (will be processed after column mapping)
            data_service = get_data_service()
            data_service.current_data = df
            data_service.update_config(data_info)
            
            # Generate preview
            preview = create_data_preview(df, data_info)
            
            status = f"✅ Data loaded successfully: {filename} ({len(df)} rows, {len(df.columns)} columns)"
            
            return (status, df.to_dict('records'), data_info, preview, 
                   column_options, column_options, column_options, column_options, column_options)
                   
        except Exception as e:
            logging.error(f"Error in upload: {str(e)}")
            return f"❌ Error loading data: {str(e)}", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        [Output("btn-process-data", "disabled"),
         Output("btn-process-data", "color")],
        [Input("time-column", "value"),
         Input("signal-column", "value")]
    )
    def update_process_button_state(time_col, signal_col):
        """Enable/disable process button based on required column selections"""
        if time_col and signal_col:
            return False, "success"
        return True, "secondary"

    @app.callback(
        [Output("time-column", "value"),
         Output("signal-column", "value"),
         Output("red-column", "value"),
         Output("ir-column", "value"),
         Output("waveform-column", "value")],
        Input("btn-auto-detect", "n_clicks"),
        [State("store-uploaded-data", "data"),
         State("store-data-config", "data")]
    )
    def auto_detect_columns(n_clicks, uploaded_data, data_config):
        """Auto-detect columns based on data content and column names"""
        if not n_clicks or not uploaded_data:
            raise PreventUpdate
            
        try:
            df = pd.DataFrame(uploaded_data)
            data_service = get_data_service()
            column_mapping = data_service._auto_detect_columns(df)
            
            return (column_mapping.get('time', None),
                   column_mapping.get('signal', None),
                   column_mapping.get('red', None),
                   column_mapping.get('ir', None),
                   column_mapping.get('waveform', None))
                   
        except Exception as e:
            logging.error(f"Error in auto-detect: {str(e)}")
            return no_update, no_update, no_update, no_update, no_update

    @app.callback(
        [Output("upload-status", "children", allow_duplicate=True),
         Output("store-uploaded-data", "data", allow_duplicate=True),
         Output("store-data-config", "data", allow_duplicate=True),
         Output("data-preview-section", "children", allow_duplicate=True)],
        Input("btn-process-data", "n_clicks"),
        [State("time-column", "value"),
         State("signal-column", "value"),
         State("red-column", "value"),
         State("ir-column", "value"),
         State("waveform-column", "value"),
         State("store-uploaded-data", "data"),
         State("store-data-config", "data"),
         State("sampling-freq", "value"),
         State("time-unit", "value")],
        prevent_initial_call=True
    )
    def process_data_with_columns(n_clicks, time_col, signal_col, red_col, ir_col, 
                                waveform_col, uploaded_data, data_config, sampling_freq, time_unit):
        """Process data with selected column mapping"""
        if not n_clicks:
            raise PreventUpdate
            
        try:
            df = pd.DataFrame(uploaded_data)
            
            # Debug logging
            logging.info(f"=== PROCESSING DATA WITH COLUMNS ===")
            logging.info(f"Time column: {time_col}")
            logging.info(f"Signal column: {signal_col}")
            logging.info(f"Red column: {red_col}")
            logging.info(f"IR column: {ir_col}")
            logging.info(f"Waveform column: {waveform_col}")
            logging.info(f"Data columns: {list(df.columns)}")
            logging.info(f"Data shape: {df.shape}")
            
            # Update data config with column mapping
            column_mapping = {
                'time': time_col,
                'signal': signal_col,
                'red': red_col,
                'ir': ir_col,
                'waveform': waveform_col
            }
            
            logging.info(f"Constructed column mapping: {column_mapping}")
            
            data_config['column_mapping'] = column_mapping
            
            # Store the final processed data
            data_service = get_data_service()
            data_id = data_service.store_data(df, data_config)
            
            logging.info(f"Data stored with ID: {data_id}")
            
            # Update status
            status = f"✅ Data processed and stored successfully! Data ID: {data_id}"
            
            # Generate final preview
            preview = create_data_preview(df, data_config)
            
            return status, df.to_dict('records'), data_config, preview
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            return f"❌ Error processing data: {str(e)}", no_update, no_update, no_update


def create_error_status(message: str) -> html.Div:
    """Create an error status message."""
    return html.Div([
        html.I(className="fas fa-exclamation-triangle text-danger me-2"),
        html.Span(message, className="text-danger")
    ], className="alert alert-danger")


def create_success_status(message: str) -> html.Div:
    """Create a success status message."""
    return html.Div([
        html.I(className="fas fa-check-circle text-success me-2"),
        html.Span(message, className="text-success")
    ], className="alert alert-success")


def create_data_preview(df: pd.DataFrame, data_info: dict) -> html.Div:
    """Create a data preview section."""
    return html.Div([
        html.H4("Data Preview", className="mb-3"),
        html.Div([
            html.P(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"),
            html.P(f"Sampling Frequency: {data_info.get('sampling_freq', 'N/A')} Hz"),
            html.P(f"Duration: {data_info.get('duration', 'N/A')} seconds"),
        ], className="mb-3"),
        html.Div([
            html.H6("First 5 rows:"),
            dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ])
    ])
