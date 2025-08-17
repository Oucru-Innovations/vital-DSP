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
            
            # Advanced Analysis Tabs (like sample_tool)
            dbc.Tabs([
                # Raw Signals Tab
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
                
                # Filtered Signals Tab
                dbc.Tab([
                    html.Div(id="filtered-signals-plot-container", children=[
                        dcc.Graph(
                            id="filtered-signals-plot",
                            style={"height": "600px"},
                            config={"displayModeBar": True}
                        )
                    ])
                ], label="Filtered Signals", tab_id="filtered-signals"),
                
                # Frequency Domain Tab with Enhanced PSD & Spectrogram Analysis
                dbc.Tab([
                    html.Div([
                        # PSD (Power Spectral Density) Controls
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("📊 Power Spectral Density (PSD) Controls", className="mb-0")
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("PSD Window (s)", className="form-label"),
                                        dcc.Input(
                                            id="psd-window",
                                            type="number",
                                            value=2.0,
                                            min=0.5,
                                            max=10.0,
                                            step=0.5,
                                            style={"width": "100%"}
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Label("PSD Overlap (0-0.95)", className="form-label"),
                                        dcc.Input(
                                            id="psd-overlap",
                                            type="number",
                                            value=0.5,
                                            min=0.0,
                                            max=0.95,
                                            step=0.05,
                                            style={"width": "100%"}
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Label("Max Frequency (Hz)", className="form-label"),
                                        dcc.Input(
                                            id="psd-freq-max",
                                            type="number",
                                            value=25.0,
                                            min=1.0,
                                            max=100.0,
                                            step=1.0,
                                            style={"width": "100%"}
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Label("Log Scale", className="form-label"),
                                        dcc.Checklist(
                                            id="psd-log-scale",
                                            options=[{"label": "dB Scale", "value": "on"}],
                                            value=["on"],
                                            style={"marginTop": "8px"}
                                        )
                                    ], md=3)
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Show PSD", className="form-label"),
                                        dcc.Checklist(
                                            id="show-psd",
                                            options=[{"label": "Enable PSD Analysis", "value": "on"}],
                                            value=["on"],
                                            style={"marginTop": "8px"}
                                        )
                                    ], md=4),
                                    dbc.Col([
                                        html.Label("Channel Selection", className="form-label"),
                                        dcc.Checklist(
                                            id="psd-channels",
                                            options=[
                                                {"label": "RED Channel", "value": "red"},
                                                {"label": "IR Channel", "value": "ir"},
                                                {"label": "Waveform", "value": "waveform"}
                                            ],
                                            value=["red", "ir", "waveform"],
                                            style={"marginTop": "8px"}
                                        )
                                    ], md=4),
                                    dbc.Col([
                                        html.Label("Normalize", className="form-label"),
                                        dcc.Checklist(
                                            id="psd-normalize",
                                            options=[{"label": "Normalize PSD", "value": "on"}],
                                            value=[],
                                            style={"marginTop": "8px"}
                                        )
                                    ], md=4)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Spectrogram Controls
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("🎵 Spectrogram Controls", className="mb-0")
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Spectrogram Window (s)", className="form-label"),
                                        dcc.Input(
                                            id="spectrogram-window",
                                            type="number",
                                            value=2.0,
                                            min=0.5,
                                            max=10.0,
                                            step=0.5,
                                            style={"width": "100%"}
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Label("Overlap (0-0.95)", className="form-label"),
                                        dcc.Input(
                                            id="spectrogram-overlap",
                                            type="number",
                                            value=0.5,
                                            min=0.0,
                                            max=0.95,
                                            step=0.05,
                                            style={"width": "100%"}
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Label("Frequency Range (Hz)", className="form-label"),
                                        dcc.Input(
                                            id="spectrogram-freq-max",
                                            type="number",
                                            value=20.0,
                                            min=1.0,
                                            max=50.0,
                                            step=1.0,
                                            style={"width": "100%"}
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Label("Show Spectrogram", className="form-label"),
                                        dcc.Checklist(
                                            id="show-spectrogram",
                                            options=[{"label": "Enable", "value": "on"}],
                                            value=["on"],
                                            style={"marginTop": "8px"}
                                        )
                                    ], md=3)
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Colormap", className="form-label"),
                                        dcc.Dropdown(
                                            id="spectrogram-colormap",
                                            options=[
                                                {"label": "Viridis", "value": "Viridis"},
                                                {"label": "Plasma", "value": "Plasma"},
                                                {"label": "Inferno", "value": "Inferno"},
                                                {"label": "Magma", "value": "Magma"},
                                                {"label": "Jet", "value": "Jet"},
                                                {"label": "Hot", "value": "Hot"}
                                            ],
                                            value="Viridis",
                                            clearable=False,
                                            style={"width": "100%"}
                                        )
                                    ], md=4),
                                    dbc.Col([
                                        html.Label("Scaling", className="form-label"),
                                        dcc.Dropdown(
                                            id="spectrogram-scaling",
                                            options=[
                                                {"label": "Density", "value": "density"},
                                                {"label": "Spectrum", "value": "spectrum"}
                                            ],
                                            value="density",
                                            clearable=False,
                                            style={"width": "100%"}
                                        )
                                    ], md=4),
                                    dbc.Col([
                                        html.Label("Channel", className="form-label"),
                                        dcc.Dropdown(
                                            id="spectrogram-channel",
                                            options=[
                                                {"label": "IR Channel (Recommended)", "value": "ir"},
                                                {"label": "RED Channel", "value": "red"},
                                                {"label": "Waveform", "value": "waveform"}
                                            ],
                                            value="ir",
                                            clearable=False,
                                            style={"width": "100%"}
                                        )
                                    ], md=4)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Analysis Results
                        dbc.Row([
                            dbc.Col([
                                # Frequency Domain Plot (PSD)
                                html.Div(id="frequency-domain-plot-container", children=[
                                    dcc.Graph(
                                        id="frequency-domain-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], md=6),
                            dbc.Col([
                                # Spectrogram Plot
                                html.Div(id="spectrogram-plot-container", children=[
                                    dcc.Graph(
                                        id="spectrogram-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Frequency Analysis Summary
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6("📈 Frequency Analysis Summary", className="mb-0")
                            ]),
                            dbc.CardBody([
                                html.Div(id="frequency-analysis-summary", className="d-flex flex-wrap gap-2")
                            ])
                        ])
                    ])
                ], label="Frequency Domain", tab_id="frequency-domain"),
                
                # Dual-source Analytics Tab
                dbc.Tab([
                    html.Div([
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id="r-trend-spo2-container", children=[
                                    dcc.Graph(
                                        id="r-trend-spo2-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="R-trend & SpO₂", tab_id="r-trend"),
                            dbc.Tab([
                                html.Div(id="coherence-container", children=[
                                    dcc.Graph(
                                        id="coherence-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="Coherence", tab_id="coherence"),
                            dbc.Tab([
                                html.Div(id="lissajous-container", children=[
                                    dcc.Graph(
                                        id="lissajous-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="Lissajous", tab_id="lissajous"),
                            dbc.Tab([
                                html.Div(id="average-beat-container", children=[
                                    dcc.Graph(
                                        id="average-beat-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="Average Beat", tab_id="average-beat"),
                            dbc.Tab([
                                html.Div(id="sdppg-container", children=[
                                    dcc.Graph(
                                        id="sdppg-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="SDPPG", tab_id="sdppg")
                        ], id="dual-source-tabs")
                    ])
                ], label="Dual-source Analytics", tab_id="dual-source"),
                
                # Dynamics (HR/IBI) Tab
                dbc.Tab([
                    html.Div([
                        # HR Analysis Controls
                        dbc.Row([
                            dbc.Col([
                                html.Label("HR Source", className="form-label"),
                                dcc.Dropdown(
                                    id="hr-source",
                                    options=[
                                        {"label": "IR (default)", "value": "ir"},
                                        {"label": "RED", "value": "red"}
                                    ],
                                    value="ir",
                                    clearable=False,
                                    style={"width": "100%"}
                                )
                            ], md=3),
                            dbc.Col([
                                html.Label("HR Min (bpm)", className="form-label"),
                                dcc.Input(
                                    id="hr-min",
                                    type="number",
                                    value=40,
                                    min=20,
                                    max=200,
                                    step=5,
                                    style={"width": "100%"}
                                )
                            ], md=3),
                            dbc.Col([
                                html.Label("HR Max (bpm)", className="form-label"),
                                dcc.Input(
                                    id="hr-max",
                                    type="number",
                                    value=200,
                                    min=60,
                                    max=300,
                                    step=5,
                                    style={"width": "100%"}
                                )
                            ], md=3),
                            dbc.Col([
                                html.Label("Peak Prominence", className="form-label"),
                                dcc.Input(
                                    id="peak-prominence",
                                    type="number",
                                    value=2.0,
                                    min=0.1,
                                    max=10.0,
                                    step=0.1,
                                    style={"width": "100%"}
                                )
                            ], md=3)
                        ], className="mb-3"),
                        
                        # Dynamics Analysis Tabs
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id="hr-trend-container", children=[
                                    dcc.Graph(
                                        id="hr-trend-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="HR Trend", tab_id="hr-trend"),
                            dbc.Tab([
                                html.Div(id="ibi-histogram-container", children=[
                                    dcc.Graph(
                                        id="ibi-histogram-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="IBI Histogram", tab_id="ibi-histogram"),
                            dbc.Tab([
                                html.Div(id="poincare-container", children=[
                                    dcc.Graph(
                                        id="poincare-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="Poincaré", tab_id="poincare"),
                            dbc.Tab([
                                html.Div(id="cross-correlation-container", children=[
                                    dcc.Graph(
                                        id="cross-correlation-plot",
                                        style={"height": "400px"},
                                        config={"displayModeBar": True}
                                    )
                                ])
                            ], label="Cross-correlation", tab_id="cross-correlation")
                        ], id="dynamics-tabs")
                    ])
                ], label="Dynamics (HR/IBI)", tab_id="dynamics")
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
