"""
Upload page layout for vitalDSP webapp.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from vitalDSP_webapp.layout.common.progress_components import (
    create_progress_interval,
    create_progress_store,
)


def upload_layout():
    """Create the modern, elegant upload page layout."""
    return html.Div(
        [
            # Header Section
            html.Div(
                [
                    html.H2("📊 Data Upload", className="text-primary mb-2 fw-bold"),
                    html.P(
                        "Upload your physiological signal data in various formats including standard CSV, OUCRU CSV, Excel, WFDB, EDF, and more",
                        className="text-muted mb-0",
                    ),
                ],
                className="text-center mb-4",
            ),
            # Data Configuration Section - MOVED TO TOP
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
                            # Row 1: Format, Signal Type, Sampling Frequency
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Data Format",
                                                className="form-label fw-semibold",
                                            ),
                                            dbc.Select(
                                                id="data-format",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "Standard CSV/TXT",
                                                        "value": "csv",
                                                    },
                                                    {
                                                        "label": "OUCRU CSV Format",
                                                        "value": "oucru_csv",
                                                    },
                                                    {
                                                        "label": "Excel (XLSX)",
                                                        "value": "excel",
                                                    },
                                                    {"label": "HDF5", "value": "hdf5"},
                                                    {
                                                        "label": "Parquet",
                                                        "value": "parquet",
                                                    },
                                                    {"label": "JSON", "value": "json"},
                                                    {"label": "WFDB", "value": "wfdb"},
                                                    {
                                                        "label": "EDF/EDF+",
                                                        "value": "edf",
                                                    },
                                                    {
                                                        "label": "MATLAB (.mat)",
                                                        "value": "matlab",
                                                    },
                                                ],
                                                value="auto",
                                                className="border-0 bg-light",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Signal Type",
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
                                                    {"label": "EEG", "value": "eeg"},
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "resp",
                                                    },
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
                                                placeholder="Auto if available",
                                                className="border-0 bg-light",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Row 2: OUCRU-specific options (conditionally displayed)
                            html.Div(
                                id="oucru-config-section",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Sampling Rate Column",
                                                        className="form-label fw-semibold",
                                                    ),
                                                    dbc.Input(
                                                        id="oucru-sampling-rate-column",
                                                        type="text",
                                                        placeholder="e.g., 'sampling_rate' (optional)",
                                                        className="border-0 bg-light",
                                                    ),
                                                    html.Small(
                                                        "Column containing sampling rates per row",
                                                        className="text-muted",
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Interpolate Timestamps",
                                                        className="form-label fw-semibold",
                                                    ),
                                                    dbc.Checklist(
                                                        id="oucru-interpolate-time",
                                                        options=[
                                                            {
                                                                "label": " Generate sub-second timestamps",
                                                                "value": True,
                                                            }
                                                        ],
                                                        value=[True],
                                                        switch=True,
                                                    ),
                                                    html.Small(
                                                        "Create precise timestamps for each sample within the second",
                                                        className="text-muted",
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                ],
                                style={"display": "none"},  # Hidden by default
                            ),
                            # Row 3: Time Unit
                            dbc.Row(
                                [
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
                                ],
                                className="mb-3",
                            ),
                        ]
                    ),
                ],
                className="mb-4 shadow-sm border-0",
            ),
            # Main Content Container - File Upload and Quick Actions
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
                                                            "CSV, TXT, Excel, HDF5, Parquet, JSON, WFDB, EDF, MATLAB and more",
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
            # Progress tracking components for real-time updates
            create_progress_store(
                store_id="upload-progress-store",
                initial_data={
                    "task_id": None,
                    "progress_percent": 0,
                    "bytes_processed": 0,
                    "total_bytes": 0,
                    "chunks_processed": 0,
                    "total_chunks": 0,
                    "elapsed_time": 0,
                    "estimated_remaining": 0,
                    "status": "idle",
                    "message": "Ready to upload",
                },
            ),
            create_progress_interval(
                interval_id="upload-progress-interval",
                interval_ms=500,  # Update every 500ms
                disabled=True,  # Start disabled, enable during upload
            ),
        ],
        className="container-fluid px-4 py-3",
    )
