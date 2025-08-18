"""
Upload page layout for vitalDSP webapp.

This module provides the upload page layout with file upload and data configuration.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def upload_layout():
    """Create the upload page layout."""
    return html.Div([
        html.Div([
            html.H2("Data Upload", className="mb-4"),
            html.P("Upload your PPG/ECG data or load from file path. Supported formats: CSV, TXT, MAT"),
            
            # File upload section
            html.Div([
                html.H4("File Upload", className="mb-3"),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                        html.Br(),
                        "Drag and Drop or Click to Select Files"
                    ]),
                    style={
                        "width": "100%",
                        "height": "120px",
                        "lineHeight": "60px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px"
                    },
                    multiple=False
                )
            ], className="mb-4"),
            
            # File path section
            html.Div([
                html.H4("Load from File Path", className="mb-3"),
                html.Div([
                    dbc.Input(
                        id="file-path-input",
                        placeholder="Enter file path...",
                        type="text",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Load File",
                        id="btn-load-path",
                        color="primary",
                        className="me-2"
                    )
                ], className="d-flex")
            ], className="mb-4"),
            
            # Sample data section
            html.Div([
                html.H4("Load Sample Data", className="mb-3"),
                html.P("Generate sample PPG data for testing purposes"),
                dbc.Button(
                    "Load Sample Data",
                    id="btn-load-sample",
                    color="success"
                )
            ], className="mb-4"),
            
            # Data configuration
            html.Div([
                html.H4("Data Configuration", className="mb-3"),
                html.Div([
                    html.Div([
                        html.Label("Sampling Frequency (Hz):", className="form-label"),
                        dbc.Input(
                            id="sampling-freq",
                            type="number",
                            placeholder="1000",
                            min=1,
                            step=1
                        )
                    ], className="me-3"),
                    html.Div([
                        html.Label("Time Unit:", className="form-label"),
                        dbc.Select(
                            id="time-unit",
                            options=[
                                {"label": "Seconds", "value": "seconds"},
                                {"label": "Milliseconds", "value": "milliseconds"},
                                {"label": "Minutes", "value": "minutes"}
                            ],
                            value="seconds"
                        )
                    ], className="me-3")
                ], className="d-flex")
            ], className="mb-4"),
            
            # Column Mapping Section
            html.Div([
                html.H4("Column Mapping", className="mb-3"),
                html.P("Configure which columns represent different data types", className="text-muted mb-3"),
                
                # Time and Signal columns
                dbc.Row([
                    dbc.Col([
                        html.Label("Time Column", className="form-label"),
                        dcc.Dropdown(
                            id="time-column",
                            placeholder="Select time column...",
                            clearable=True
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("Signal Column", className="form-label"),
                        dcc.Dropdown(
                            id="signal-column",
                            placeholder="Select signal column...",
                            clearable=True
                        )
                    ], md=6)
                ], className="mb-3"),
                
                # RED and IR channels
                dbc.Row([
                    dbc.Col([
                        html.Label("RED Channel (if applicable)", className="form-label"),
                        dcc.Dropdown(
                            id="red-column",
                            placeholder="Select RED column...",
                            clearable=True
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label("IR Channel (if applicable)", className="form-label"),
                        dcc.Dropdown(
                            id="ir-column",
                            placeholder="Select IR column...",
                            clearable=True
                        )
                    ], md=6)
                ], className="mb-3"),
                
                # Waveform column
                dbc.Row([
                    dbc.Col([
                        html.Label("Waveform (PLETH)", className="form-label"),
                        dcc.Dropdown(
                            id="waveform-column",
                            placeholder="Select PLETH column...",
                            clearable=True
                        )
                    ], md=6)
                ], className="mb-4"),
                
                # Action Buttons
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "🔍 Auto-detect Columns",
                            id="btn-auto-detect",
                            color="info",
                            outline=True,
                            className="w-100"
                        )
                    ], md=6),
                    dbc.Col([
                        dbc.Button(
                            "✅ Process Data",
                            id="btn-process-data",
                            color="success",
                            className="w-100",
                            disabled=True
                        )
                    ], md=6)
                ])
            ], className="mb-4"),
            
            # Status section
            html.Div([
                html.H4("Upload Status", className="mb-3"),
                html.Div(id="upload-status", className="mb-3")
            ], className="mb-4"),
            
            # Data preview section
            html.Div([
                html.H4("Data Preview", className="mb-3"),
                html.Div(id="data-preview-section")
            ])
            
        ], className="container"),
        
        # Stores for data management
        dcc.Store(id="store-uploaded-data"),
        dcc.Store(id="store-data-config"),
        dcc.Store(id="store-column-mapping"),
        dcc.Store(id="store-preview-window", data={"start": 0, "end": 1000})
    ])
