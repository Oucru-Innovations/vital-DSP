"""
Upload page layout for vitalDSP webapp.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def upload_layout():
    """Create the modern, elegant upload page layout."""
    return html.Div(
        [
            # Header Section
            html.Div(
                [
                    html.H2("📊 Data Upload", className="text-primary mb-2 fw-bold"),
                    html.P(
                        "Upload your PPG/ECG data or load from file path. Supported formats: CSV, TXT, MAT",
                        className="text-muted mb-0",
                    ),
                ],
                className="text-center mb-4",
            ),
            # Main Content Container
            dbc.Row(
                [
                    # File Upload Section - Left Column
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(
                                                className="fas fa-cloud-upload-alt me-2 text-primary"
                                            ),
                                            html.Span(
                                                "File Upload", className="fw-bold"
                                            ),
                                        ],
                                        className="bg-light border-0",
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Upload(
                                                id="upload-data",
                                                children=html.Div(
                                                    [
                                                        html.I(
                                                            className="fas fa-cloud-upload-alt fa-3x text-primary mb-3"
                                                        ),
                                                        html.Br(),
                                                        html.Span(
                                                            "Drag and Drop or Click to Select Files",
                                                            className="fw-semibold text-muted",
                                                        ),
                                                        html.Br(),
                                                        html.Small(
                                                            "CSV, TXT, MAT files supported",
                                                            className="text-muted",
                                                        ),
                                                        html.Br(),
                                                        html.Small(
                                                            "Maximum file size: 50MB",
                                                            className="text-muted",
                                                        ),
                                                    ],
                                                    className="text-center py-4",
                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "160px",
                                                    "borderWidth": "2px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "12px",
                                                    "textAlign": "center",
                                                    "cursor": "pointer",
                                                    "transition": "all 0.3s ease",
                                                    "backgroundColor": "#f8f9fa",
                                                },
                                                className="upload-area",
                                                multiple=False,
                                            ),
                                            html.Hr(className="my-3"),
                                            # Upload Progress Section
                                            html.Div(
                                                id="upload-progress-section",
                                                className="mb-3",
                                                style={"display": "none"},
                                            ),
                                            # Upload Status
                                            html.Div(
                                                id="upload-status", className="py-2"
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm border-0",
                            )
                        ],
                        md=6,
                        className="mb-4",
                    ),
                    # Quick Actions Section - Right Column
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.I(
                                                className="fas fa-bolt me-2 text-success"
                                            ),
                                            html.Span(
                                                "Quick Actions", className="fw-bold"
                                            ),
                                        ],
                                        className="bg-light border-0",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Load from File Path",
                                                        className="form-label fw-semibold mb-2",
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.Input(
                                                                id="file-path-input",
                                                                placeholder="Enter file path...",
                                                                type="text",
                                                                className="border-0",
                                                            ),
                                                            dbc.Button(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-folder-open me-2"
                                                                    ),
                                                                    "Load",
                                                                ],
                                                                id="btn-load-path",
                                                                color="primary",
                                                                size="sm",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        id="file-path-loading",
                                                        className="mt-2",
                                                        style={"display": "none"},
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.Hr(className="my-3"),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Generate Sample Data",
                                                        className="form-label fw-semibold mb-2",
                                                    ),
                                                    html.P(
                                                        "Create synthetic PPG data for testing",
                                                        className="text-muted small mb-2",
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="fas fa-database me-2"
                                                            ),
                                                            "Load Sample Data",
                                                        ],
                                                        id="btn-load-sample",
                                                        color="success",
                                                        className="w-100",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm border-0",
                            )
                        ],
                        md=6,
                        className="mb-4",
                    ),
                ]
            ),
            # Configuration Section
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.I(className="fas fa-cog me-2 text-info"),
                            html.Span("Data Configuration", className="fw-bold"),
                        ],
                        className="bg-light border-0",
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Sampling Frequency (Hz)",
                                                className="form-label fw-semibold",
                                            ),
                                            dbc.Input(
                                                id="sampling-freq",
                                                type="number",
                                                value=100,
                                                min=1,
                                                step=1,
                                                className="border-0 bg-light",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Time Unit",
                                                className="form-label fw-semibold",
                                            ),
                                            dbc.Select(
                                                id="time-unit",
                                                options=[
                                                    {
                                                        "label": "Seconds",
                                                        "value": "seconds",
                                                    },
                                                    {
                                                        "label": "Milliseconds",
                                                        "value": "milliseconds",
                                                    },
                                                    {
                                                        "label": "Minutes",
                                                        "value": "minutes",
                                                    },
                                                ],
                                                value="seconds",
                                                className="border-0 bg-light",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Data Type",
                                                className="form-label fw-semibold",
                                            ),
                                            dbc.Select(
                                                id="data-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {"label": "PPG", "value": "ppg"},
                                                    {"label": "ECG", "value": "ecg"},
                                                    {
                                                        "label": "Other",
                                                        "value": "other",
                                                    },
                                                ],
                                                value="ppg",
                                                className="border-0 bg-light",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="mb-3",
                            )
                        ]
                    ),
                ],
                className="mb-4 shadow-sm border-0",
            ),
            # Column Mapping Section
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.I(className="fas fa-columns me-2 text-warning"),
                            html.Span("Column Mapping", className="fw-bold"),
                            html.Small(
                                " Configure which columns represent different data types",
                                className="text-muted ms-2 fw-normal",
                            ),
                        ],
                        className="bg-light border-0",
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Time Column",
                                                className="form-label fw-semibold",
                                            ),
                                            dcc.Dropdown(
                                                id="time-column",
                                                placeholder="Select time column...",
                                                clearable=True,
                                                className="border-0",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Signal Column",
                                                className="form-label fw-semibold",
                                            ),
                                            dcc.Dropdown(
                                                id="signal-column",
                                                placeholder="Select signal column...",
                                                clearable=True,
                                                className="border-0",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "RED Channel (if applicable)",
                                                className="form-label fw-semibold",
                                            ),
                                            dcc.Dropdown(
                                                id="red-column",
                                                placeholder="Select RED column...",
                                                clearable=True,
                                                className="border-0",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "IR Channel (if applicable)",
                                                className="form-label fw-semibold",
                                            ),
                                            dcc.Dropdown(
                                                id="ir-column",
                                                placeholder="Select IR column...",
                                                clearable=True,
                                                className="border-0",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Waveform (PLETH)",
                                                className="form-label fw-semibold",
                                            ),
                                            dcc.Dropdown(
                                                id="waveform-column",
                                                placeholder="Select PLETH column...",
                                                clearable=True,
                                                className="border-0",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("", className="form-label"),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="fas fa-magic me-2"
                                                            ),
                                                            "Auto-detect",
                                                        ],
                                                        id="btn-auto-detect",
                                                        color="info",
                                                        outline=True,
                                                        size="sm",
                                                        className="me-2",
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="fas fa-check me-2"
                                                            ),
                                                            "Process Data",
                                                        ],
                                                        id="btn-process-data",
                                                        color="success",
                                                        size="sm",
                                                        disabled=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                className="mb-4 shadow-sm border-0",
            ),
            # Processing Progress Section
            html.Div(
                id="processing-progress-section",
                className="mb-4",
                style={"display": "none"},
            ),
            # Data Preview Section - Full width
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.I(className="fas fa-table me-2 text-success"),
                            html.Span("Data Preview", className="fw-bold"),
                        ],
                        className="bg-light border-0",
                    ),
                    dbc.CardBody(
                        [
                            dcc.Loading(
                                id="data-preview-loading",
                                type="default",
                                children=html.Div(
                                    id="data-preview-section", className="min-vh-100"
                                ),
                            )
                        ]
                    ),
                ],
                className="mb-4 shadow-sm border-0",
            ),
            # Stores for data management
            dcc.Store(id="store-uploaded-data"),
            dcc.Store(id="store-data-config"),
            dcc.Store(id="store-column-mapping"),
            dcc.Store(id="store-preview-window", data={"start": 0, "end": 1000}),
            # Loading states store
            dcc.Store(
                id="store-loading-states",
                data={
                    "uploading": False,
                    "processing": False,
                    "upload_progress": 0,
                    "processing_progress": 0,
                },
            ),
        ],
        className="container-fluid px-4 py-3",
    )
