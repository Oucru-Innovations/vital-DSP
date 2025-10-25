"""
Time Domain Analysis page layout for vitalDSP webapp.

This module provides the layout for the Time Domain Analysis page,
which allows users to analyze PPG/ECG signals in the time domain with
interactive plots, filtering, and comprehensive analysis tools.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def time_domain_layout():
    """Create the time domain analysis page layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("⏱️ Time Domain Analysis", className="text-center mb-4"),
                    html.P(
                        [
                            "Analyze your PPG/ECG signals in the time domain with interactive plots, ",
                            "filtering, and comprehensive analysis tools.",
                        ],
                        className="text-center text-muted mb-5",
                    ),
                ],
                className="mb-4",
            ),
            # Action Buttons - Moved to top for easy access
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "🔄 Update Analysis",
                                                                id="btn-update-analysis",
                                                                color="primary",
                                                                size="lg",
                                                                className="w-100",
                                                            )
                                                        ],
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.ButtonGroup(
                                                                [
                                                                    dbc.Button(
                                                                        [
                                                                            html.I(
                                                                                className="fas fa-file-csv me-2"
                                                                            ),
                                                                            "Export CSV",
                                                                        ],
                                                                        id="btn-export-time-domain-csv",
                                                                        color="success",
                                                                        outline=True,
                                                                        size="lg",
                                                                    ),
                                                                    dbc.Button(
                                                                        [
                                                                            html.I(
                                                                                className="fas fa-file-code me-2"
                                                                            ),
                                                                            "Export JSON",
                                                                        ],
                                                                        id="btn-export-time-domain-json",
                                                                        color="info",
                                                                        outline=True,
                                                                        size="lg",
                                                                    ),
                                                                ],
                                                                className="w-100",
                                                            )
                                                        ],
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "🎯 Comprehensive Dashboard",
                                                                id="btn-comprehensive-dashboard",
                                                                color="info",
                                                                outline=True,
                                                                size="lg",
                                                                className="w-100",
                                                            )
                                                        ],
                                                        md=4,
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            # Main Analysis Section
            dbc.Row(
                [
                    # Left Panel - Controls & Parameters (Reduced width)
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🎛️ Analysis Controls", className="mb-0"
                                            ),
                                            html.Small(
                                                "Configure analysis parameters and filters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Data Selection
                                            html.H6("Data Selection", className="mb-3"),
                                            dbc.Select(
                                                id="data-source-select",
                                                options=[
                                                    {
                                                        "label": "Uploaded Data",
                                                        "value": "uploaded",
                                                    },
                                                    {
                                                        "label": "Sample Data",
                                                        "value": "sample",
                                                    },
                                                ],
                                                value="uploaded",
                                                className="mb-3",
                                            ),
                                            # Signal Type Selection for Critical Points Detection
                                            html.H6("Signal Type", className="mb-3"),
                                            dbc.Select(
                                                id="signal-type-select",
                                                options=[
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "PPG",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ECG",
                                                    },
                                                ],
                                                value="PPG",
                                                className="mb-3",
                                            ),
                                            # Time Window Controls
                                            html.H6("Time Window", className="mb-3"),
                                            # Modern Time Range Controls
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Position (%)",
                                                                className="form-label fw-bold",
                                                            ),
                                                            html.Small(
                                                                "Position in data (0% = start, 100% = end)",
                                                                className="text-muted",
                                                            ),
                                                            dcc.Slider(
                                                                id="start-position-slider",
                                                                min=0,
                                                                max=100,
                                                                step=1,
                                                                value=0,
                                                                marks={
                                                                    0: "0%",
                                                                    25: "25%",
                                                                    50: "50%",
                                                                    75: "75%",
                                                                    100: "100%",
                                                                },
                                                                tooltip={"placement": "bottom", "always_visible": True},
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                        md=8,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Duration",
                                                                className="form-label fw-bold",
                                                            ),
                                                            html.Small(
                                                                "Analysis window size",
                                                                className="text-muted",
                                                            ),
                                                            dbc.Select(
                                                                id="duration-select",
                                                                options=[
                                                                    {"label": "30 seconds", "value": 30},
                                                                    {"label": "1 minute", "value": 60},
                                                                    {"label": "2 minutes", "value": 120},
                                                                    {"label": "5 minutes", "value": 300},
                                                                ],
                                                                value=60,  # Default to 1 minute
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                        md=4,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Quick Navigation Buttons
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Quick Navigation",
                                                        className="form-label fw-bold mb-2",
                                                    ),
                                                    dbc.ButtonGroup(
                                                        [
                                                            dbc.Button(
                                                                "⏪ -10%",
                                                                id="btn-nudge-m10",
                                                                color="secondary",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "⏪ -5%",
                                                                id="btn-nudge-m5",
                                                                color="secondary",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "Center",
                                                                id="btn-center",
                                                                color="info",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "+5% ⏩",
                                                                id="btn-nudge-p5",
                                                                color="secondary",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "+10% ⏩",
                                                                id="btn-nudge-p10",
                                                                color="secondary",
                                                                size="sm",
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                            ),
                                            # Signal Source Selection
                                            html.H6("Signal Source", className="mb-3"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Select Signal Source",
                                                                className="form-label",
                                                            ),
                                                            dbc.Select(
                                                                id="signal-source-select",
                                                                options=[
                                                                    {
                                                                        "label": "Original Signal",
                                                                        "value": "original",
                                                                    },
                                                                    {
                                                                        "label": "Filtered Signal",
                                                                        "value": "filtered",
                                                                    },
                                                                ],
                                                                value="filtered",  # Default to filtered
                                                                className="mb-3",
                                                            ),
                                                            html.Small(
                                                                "Filtered signal will be used if available from the filtering screen. Falls back to original signal if no filtering has been performed.",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        md=12,
                                                    )
                                                ],
                                                className="mb-3",
                                            ),
                                            # Analysis Options
                                            html.H6(
                                                "Analysis Options", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="analysis-options",
                                                options=[
                                                    {
                                                        "label": "Peak Detection",
                                                        "value": "peaks",
                                                    },
                                                    {
                                                        "label": "Critical Points Detection",
                                                        "value": "critical_points",
                                                    },
                                                    {
                                                        "label": "Heart Rate Calculation",
                                                        "value": "hr",
                                                    },
                                                    {
                                                        "label": "Signal Quality Assessment",
                                                        "value": "quality",
                                                    },
                                                    {
                                                        "label": "Artifact Detection",
                                                        "value": "artifacts",
                                                    },
                                                    {
                                                        "label": "Trend Analysis",
                                                        "value": "trend",
                                                    },
                                                ],
                                                value=[
                                                    "peaks",
                                                    "critical_points",
                                                    "hr",
                                                    "quality",
                                                ],
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100",
                            )
                        ],
                        md=3,
                    ),  # Reduced from md=4 to md=3
                    # Right Panel - Plots & Results (Increased width)
                    dbc.Col(
                        [
                            # Main Signal Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📈 Raw Signal with Critical Points",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Time domain representation with detected morphological features",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="main-signal-plot",
                                                    style={"height": "400px"},
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToRemove": [
                                                            "pan2d",
                                                            "lasso2d",
                                                            "select2d",
                                                        ],
                                                        "displaylogo": False,
                                                    },
                                                ),
                                                type="default",
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Signal Comparison Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🔍 Signal Comparison",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Side-by-side comparison of original and filtered signals",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="signal-comparison-plot",
                                                    style={"height": "400px"},
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToRemove": [
                                                            "pan2d",
                                                            "lasso2d",
                                                            "select2d",
                                                        ],
                                                        "displaylogo": False,
                                                    },
                                                ),
                                                type="default",
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Analysis Results - Reorganized for Better Display
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📊 Analysis Results", className="mb-0"
                                            ),
                                            html.Small(
                                                "Key metrics and insights from your signal",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Main analysis results
                                            html.Div(
                                                id="analysis-results", className="mb-4"
                                            ),
                                            # Peak Analysis Table
                                            html.Div(
                                                id="peak-analysis-table",
                                                className="mb-4",
                                            ),
                                            # Signal Quality Table
                                            html.Div(
                                                id="signal-quality-table",
                                                className="mb-4",
                                            ),
                                            # Signal Source Information Table
                                            html.Div(
                                                id="signal-source-table",
                                                className="mb-4",
                                            ),
                                            # Additional Metrics Table
                                            html.Div(
                                                id="additional-metrics-table",
                                                className="mb-4",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        md=9,
                    ),  # Increased from md=8 to md=9
                ]
            ),
            # Bottom Section - Additional Analysis
            html.Div(id="additional-analysis-section", className="mt-4"),
            # Hidden components for compatibility with signal filtering callbacks
            # These are needed because filtering page callbacks may output to these IDs
            html.Div(id="filter-btn-apply", style={"display": "none"}),
            dcc.Dropdown(id="filter-type-select", style={"display": "none"}),
            dcc.Dropdown(id="filter-family-advanced", style={"display": "none"}),
            dcc.Dropdown(id="filter-response-advanced", style={"display": "none"}),
            dcc.Input(id="filter-low-freq-advanced", style={"display": "none"}),
            dcc.Input(id="filter-high-freq-advanced", style={"display": "none"}),
            dcc.Input(id="filter-order-advanced", style={"display": "none"}),
            dcc.Dropdown(id="advanced-filter-method", style={"display": "none"}),
            dcc.Input(id="ensemble-n-filters", style={"display": "none"}),
            dcc.Checklist(id="filter-quality-options", style={"display": "none"}),
            dcc.Dropdown(id="filter-signal-type-select", style={"display": "none"}),
            # Advanced filter parameters
            dcc.Input(id="advanced-noise-level", style={"display": "none"}),
            dcc.Input(id="advanced-iterations", style={"display": "none"}),
            dcc.Input(id="advanced-learning-rate", style={"display": "none"}),
            # Artifact removal parameters
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dcc.Input(id="artifact-removal-strength", style={"display": "none"}),
            # Neural network parameters
            dcc.Dropdown(id="neural-network-type", style={"display": "none"}),
            dcc.Input(id="neural-model-complexity", style={"display": "none"}),
            # Ensemble parameters
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            # Detrending parameters
            dcc.Checklist(id="detrend-option", style={"display": "none"}),
            # Savitzky-Golay parameters
            dcc.Input(id="savgol-window", style={"display": "none"}),
            dcc.Input(id="savgol-polyorder", style={"display": "none"}),
            # Moving average parameters
            dcc.Input(id="moving-avg-window", style={"display": "none"}),
            dcc.Input(id="gaussian-sigma", style={"display": "none"}),
            # Wavelet parameters
            dcc.Dropdown(id="wavelet-type", style={"display": "none"}),
            dcc.Input(id="wavelet-level", style={"display": "none"}),
            # Threshold parameters
            dcc.Dropdown(id="threshold-type", style={"display": "none"}),
            dcc.Input(id="threshold-value", style={"display": "none"}),
            # Reference signal and fusion parameters
            dcc.Dropdown(id="reference-signal", style={"display": "none"}),
            dcc.Dropdown(id="fusion-method", style={"display": "none"}),
            dcc.Graph(id="filter-original-plot", style={"display": "none"}),
            dcc.Graph(id="filter-filtered-plot", style={"display": "none"}),
            dcc.Graph(id="filter-comparison-plot", style={"display": "none"}),
            html.Div(id="filter-quality-metrics", style={"display": "none"}),
            dcc.Graph(id="filter-quality-plots", style={"display": "none"}),
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            # Stores for data management
            dcc.Store(id="store-time-domain-data"),
            dcc.Store(id="store-filtered-data"),
            dcc.Store(id="store-filtered-signal"),  # Access to filtered signal from filtering page
            dcc.Store(id="store-analysis-results"),
            dcc.Store(id="store-time-domain-features"),  # For export
            # Download components for export
            dcc.Download(id="download-time-domain-csv"),
            dcc.Download(id="download-time-domain-json"),
        ]
    )
