"""
Upload section layout for vitalDSP webapp.
Provides comprehensive data upload functionality with drag & drop, validation, and preview.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def upload_layout():
    """Create the upload page layout with comprehensive data upload functionality."""
    return html.Div([
        # Page Header
        html.Div([
            html.H1("Data Upload & Management", className="text-center mb-4"),
            html.P([
                "Upload your PPG/ECG data files and configure data parameters for analysis. ",
                "Supported formats: CSV, Excel, and other common data formats."
            ], className="text-center text-muted mb-5")
        ], className="mb-4"),
        
        # Main Upload Section
        dbc.Row([
            # Left Panel - Upload Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📁 Data Upload", className="mb-0"),
                        html.Small("Drag & drop or click to upload your data files", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # File Upload Area
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-3x mb-3", style={"color": "#6c757d"}),
                                html.H5("Drag and Drop Files Here", className="mb-2"),
                                html.P("or click to browse files", className="text-muted mb-0"),
                                html.Small("Supports CSV, Excel, and other formats", className="text-muted")
                            ]),
                            style={
                                "width": "100%",
                                "height": "200px",
                                "lineHeight": "60px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "backgroundColor": "#f8f9fa",
                                "borderColor": "#dee2e6",
                                "cursor": "pointer"
                            },
                            multiple=False,
                            accept=".csv,.xlsx,.xls,.txt,.dat"
                        ),
                        
                        html.Hr(),
                        
                        # File Path Input (Alternative)
                        html.H6("Or specify file path:", className="mb-3"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="file-path-input",
                                placeholder="Enter file path (e.g., /path/to/data.csv)",
                                type="text"
                            ),
                            dbc.Button("Load", id="btn-load-path", color="primary", n_clicks=0)
                        ], className="mb-3"),
                        
                        # Upload Status
                        html.Div(id="upload-status", className="mt-3"),
                        
                        # Sample Data Button
                        html.Hr(),
                        html.H6("Quick Start", className="mb-3"),
                        dbc.Button(
                            "📊 Load Sample PPG Data",
                            id="btn-load-sample",
                            color="info",
                            outline=True,
                            className="w-100 mb-3"
                        ),
                        html.Small(
                            "Load sample data to test the application without uploading files",
                            className="text-muted d-block"
                        )
                    ])
                ], className="h-100")
            ], md=4),
            
            # Right Panel - Data Configuration & Preview
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("⚙️ Data Configuration", className="mb-0"),
                        html.Small("Configure data parameters and column mapping", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Data Parameters
                        html.H6("Data Parameters", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Sampling Frequency (Hz)", className="form-label"),
                                dbc.Input(
                                    id="sampling-freq",
                                    type="number",
                                    value=1000,
                                    min=1,
                                    step=0.1,
                                    placeholder="1000"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Time Unit", className="form-label"),
                                dbc.Select(
                                    id="time-unit",
                                    options=[
                                        {"label": "Milliseconds", "value": "ms"},
                                        {"label": "Seconds", "value": "s"},
                                        {"label": "Minutes", "value": "min"}
                                    ],
                                    value="ms"
                                )
                            ], md=6)
                        ], className="mb-4"),
                        
                        # Column Mapping
                        html.H6("Column Mapping", className="mb-3"),
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
                    ])
                ], className="h-100")
            ], md=8)
        ], className="mb-4"),
        
        # Data Preview Section
        html.Div(id="data-preview-section", className="mt-4"),
        
        # Processing Status
        html.Div(id="processing-status", className="mt-4"),
        
        # Stores for data management
        dcc.Store(id="store-uploaded-data"),
        dcc.Store(id="store-data-config"),
        dcc.Store(id="store-column-mapping"),
        dcc.Store(id="store-preview-window", data={"start": 0, "end": 1000})
    ])


def create_data_preview(data_info):
    """Create data preview section after successful upload."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📊 Data Preview", className="mb-0"),
            html.Small("Preview of uploaded data and basic statistics", className="text-muted")
        ]),
        dbc.CardBody([
            # Data Info Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📁 File Info", className="card-title"),
                            html.P(f"Filename: {data_info.get('filename', 'N/A')}", className="mb-1"),
                            html.P(f"Size: {data_info.get('size_mb', 0):.2f} MB", className="mb-1"),
                            html.P(f"Format: {data_info.get('format', 'N/A')}", className="mb-0")
                        ])
                    ], className="h-100")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📈 Data Structure", className="card-title"),
                            html.P(f"Rows: {data_info.get('rows', 0):,}", className="mb-1"),
                            html.P(f"Columns: {data_info.get('columns', 0)}", className="mb-1"),
                            html.P(f"Duration: {data_info.get('duration_sec', 0):.1f}s", className="mb-0")
                        ])
                    ], className="h-100")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("⚡ Signal Info", className="card-title"),
                            html.P(f"Sampling Rate: {data_info.get('sampling_rate', 0)} Hz", className="mb-1"),
                            html.P(f"Min Value: {data_info.get('min_value', 0):.2f}", className="mb-1"),
                            html.P(f"Max Value: {data_info.get('max_value', 0):.2f}", className="mb-0")
                        ])
                    ], className="h-100")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🎯 Quality", className="card-title"),
                            html.P(f"SNR: {data_info.get('snr_db', 0):.1f} dB", className="mb-1"),
                            html.P(f"Artifacts: {data_info.get('artifact_count', 0)}", className="mb-1"),
                            html.P(f"Status: {data_info.get('quality_status', 'Unknown')}", className="mb-0")
                        ])
                    ], className="h-100")
                ], md=3)
            ], className="mb-4"),
            
            # Data Table Preview
            html.H6("Data Preview (First 10 rows)", className="mb-3"),
            html.Div(id="data-table-preview", className="table-responsive", children=[
                data_info.get("preview_table", html.P("No table data available"))
            ]),
            
            # Advanced Preview Controls (like sample_tool)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("🎛️ Filter Controls", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter Family", className="form-label"),
                                    dcc.Dropdown(
                                        id="preview-filter-family",
                                        value="butter",
                                        options=[
                                            {"label": "Butterworth", "value": "butter"},
                                            {"label": "Chebyshev I", "value": "cheby1"},
                                            {"label": "Chebyshev II", "value": "cheby2"},
                                            {"label": "Elliptic", "value": "ellip"},
                                            {"label": "Bessel", "value": "bessel"},
                                        ],
                                        clearable=False,
                                        style={"width": "100%"}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Response Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="preview-filter-response",
                                        value="bandpass",
                                        options=[
                                            {"label": "Bandpass", "value": "bandpass"},
                                            {"label": "Bandstop (Notch)", "value": "bandstop"},
                                            {"label": "Lowpass", "value": "lowpass"},
                                            {"label": "Highpass", "value": "highpass"},
                                        ],
                                        clearable=False,
                                        style={"width": "100%"}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Order", className="form-label"),
                                    dcc.Input(
                                        id="preview-filter-order",
                                        type="number",
                                        value=4,
                                        min=1,
                                        max=10,
                                        step=1,
                                        style={"width": "100%"}
                                    )
                                ], md=4)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Low Freq (Hz)", className="form-label"),
                                    dcc.Input(
                                        id="preview-filter-low",
                                        type="number",
                                        value=0.5,
                                        min=0.1,
                                        step=0.1,
                                        style={"width": "100%"}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("High Freq (Hz)", className="form-label"),
                                    dcc.Input(
                                        id="preview-filter-high",
                                        type="number",
                                        value=8.0,
                                        min=0.1,
                                        step=0.1,
                                        style={"width": "100%"}
                                    )
                                ], md=6)
                            ])
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("📊 Data Range Selection", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Start Row", className="form-label"),
                                    dcc.Input(
                                        id="preview-start-row",
                                        type="number",
                                        value=0,
                                        min=0,
                                        step=100,
                                        style={"width": "100%"}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("End Row", className="form-label"),
                                    dcc.Input(
                                        id="preview-end-row",
                                        type="number",
                                        value=1000,
                                        min=0,
                                        step=100,
                                        style={"width": "100%"}
                                    )
                                ], md=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Range Slider", className="form-label"),
                                    dcc.RangeSlider(
                                        id="preview-range-slider",
                                        min=0,
                                        max=data_info.get('rows', 1000),
                                        step=100,
                                        value=[0, min(1000, data_info.get('rows', 1000))],
                                        marks={i: str(i) for i in range(0, data_info.get('rows', 1000) + 1, 500)},
                                        pushable=100,
                                        updatemode="mouseup"
                                    )
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Apply Range", id="preview-apply-range", color="primary", className="w-100")
                                ])
                            ], className="mt-2")
                        ])
                    ])
                ], md=6)
            ], className="mb-4"),
            
            # Signal Preview with Tabs (like sample_tool)
            dbc.Tabs([
                dbc.Tab([
                    html.Div(id="signal-preview-plot-container", children=[
                        dcc.Graph(
                            id="signal-preview-plot",
                            figure=data_info.get("preview_plot", {}),
                            style={"height": "600px"},
                            config={"displayModeBar": True}
                        )
                    ])
                ], label="Raw Signals", tab_id="raw-signals"),
                dbc.Tab([
                    html.Div(id="filtered-signals-plot-container", children=[
                        dcc.Graph(
                            id="filtered-signals-plot",
                            style={"height": "600px"},
                            config={"displayModeBar": True}
                        )
                    ])
                ], label="Filtered Signals", tab_id="filtered-signals"),
                dbc.Tab([
                    html.Div(id="frequency-domain-plot-container", children=[
                        dcc.Graph(
                            id="frequency-domain-plot",
                            style={"height": "600px"},
                            config={"displayModeBar": True}
                        )
                    ])
                ], label="Frequency Domain", tab_id="frequency-domain")
            ], id="preview-tabs", className="mb-4"),
            
            # Insights Panel (like sample_tool)
            dbc.Card([
                dbc.CardHeader([
                    html.H6("🧠 Insights & Analysis", className="mb-0")
                ]),
                dbc.CardBody([
                    html.Div(id="preview-insights", className="d-flex flex-wrap gap-2")
                ])
            ])
        ])
    ])
