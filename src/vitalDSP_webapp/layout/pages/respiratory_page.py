"""
Respiratory Analysis Page Layout

This module contains the layout for the respiratory analysis page,
providing comprehensive respiratory rate estimation and breathing pattern analysis.

Features:
- Multiple RR estimation methods (Peak Detection, Zero Crossing, FFT, etc.)
- Sleep apnea detection
- Breathing pattern analysis
- Multimodal fusion (PPG-ECG, Respiratory-Cardiac)
- Breath duration constraints (respiratory-specific validation)
- Time window navigation with start position (%) + duration pattern
- Uses filtered signal from filtering page (signal preprocessing done there)
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def respiratory_layout():
    """
    Create the comprehensive respiratory analysis page layout.

    Returns:
        html.Div: Complete respiratory analysis page layout

    Layout Structure:
        - Page Header
        - Main Analysis Section (2 columns):
            - Left Panel (Controls & Parameters)
            - Right Panel (Plots & Results)
        - Additional Analysis Section (dynamic)
        - Data Stores
        - Hidden Components (for cross-page compatibility)
    """
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("🫁 Respiratory Analysis", className="text-center mb-2"),
                    html.P(
                        [
                            "Comprehensive respiratory rate estimation and breathing pattern analysis using vitalDSP. ",
                            "Analyze respiratory signals with multiple estimation methods, sleep apnea detection, and multimodal fusion.",
                        ],
                        className="text-center text-muted mb-3",
                    ),
                ],
                className="mb-2",
            ),
            # Main Analysis Section
            dbc.Row(
                [
                    # Left Panel - Controls & Parameters
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🎛️ Respiratory Analysis Controls",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure respiratory analysis parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Data Selection
                                            html.H6("Data Selection", className="mb-2"),
                                            dbc.Select(
                                                id="resp-data-source-select",
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
                                            # Time Window Controls - NEW PATTERN
                                            html.H6("Time Window", className="mb-2 fw-bold"),
                                            html.Small(
                                                "Select analysis window position and duration",
                                                className="text-muted mb-2 d-block",
                                            ),
                                            # Start Position Slider
                                            html.Label(
                                                "Start Position (%)",
                                                className="form-label fw-bold",
                                            ),
                                            html.Small(
                                                "Position in data (0% = start, 100% = end)",
                                                className="text-muted d-block mb-1",
                                            ),
                                            dcc.Slider(
                                                id="resp-start-position-slider",
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
                                            # Duration Dropdown
                                            html.Label(
                                                "Duration",
                                                className="form-label fw-bold",
                                            ),
                                            html.Small(
                                                "Analysis window size",
                                                className="text-muted d-block mb-1",
                                            ),
                                            dbc.Select(
                                                id="resp-duration-select",
                                                options=[
                                                    {"label": "30 seconds", "value": 30},
                                                    {"label": "1 minute", "value": 60},
                                                    {"label": "2 minutes", "value": 120},
                                                    {"label": "5 minutes", "value": 300},
                                                    {"label": "10 minutes", "value": 600},
                                                ],
                                                value=60,  # Default to 1 minute
                                                className="mb-3",
                                            ),
                                            # Quick Navigation Buttons
                                            html.Label(
                                                "Quick Navigation",
                                                className="form-label fw-bold",
                                            ),
                                            html.Small(
                                                "Adjust start position",
                                                className="text-muted d-block mb-1",
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10%",
                                                        id="resp-btn-nudge-m10",
                                                        size="sm",
                                                        color="secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -5%",
                                                        id="resp-btn-nudge-m1",
                                                        size="sm",
                                                        color="secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+5% ⏩",
                                                        id="resp-btn-nudge-p1",
                                                        size="sm",
                                                        color="secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10% ⏩",
                                                        id="resp-btn-nudge-p10",
                                                        size="sm",
                                                        color="secondary",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-2 mt-3 fw-bold"),
                                            html.Small(
                                                "Select respiratory signal type or auto-detect",
                                                className="text-muted d-block mb-2",
                                            ),
                                            dbc.Select(
                                                id="resp-signal-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {"label": "PPG", "value": "ppg"},
                                                    {"label": "ECG", "value": "ecg"},
                                                    {
                                                        "label": "Respiratory Belt",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "Nasal Cannula",
                                                        "value": "nasal",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Signal Source Selection
                                            html.H6("Signal Source", className="mb-2 fw-bold"),
                                            dbc.Select(
                                                id="resp-signal-source-select",
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
                                                className="mb-2",
                                            ),
                                            html.Small(
                                                "Filtered signal will be used if available from the filtering screen. "
                                                "Falls back to original signal if no filtering has been performed.",
                                                className="text-muted d-block mb-3",
                                            ),
                                            # Respiratory Rate Estimation Methods
                                            html.H6(
                                                "Estimation Methods", className="mb-2 mt-3 fw-bold"
                                            ),
                                            html.Small(
                                                "Select one or more respiratory rate estimation methods",
                                                className="text-muted d-block mb-2",
                                            ),
                                            dbc.Checklist(
                                                id="resp-estimation-methods",
                                                options=[
                                                    {
                                                        "label": "Peak Detection",
                                                        "value": "peak_detection",
                                                    },
                                                    {
                                                        "label": "Zero Crossing",
                                                        "value": "zero_crossing",
                                                    },
                                                    {
                                                        "label": "Time Domain",
                                                        "value": "time_domain",
                                                    },
                                                    {
                                                        "label": "Frequency Domain",
                                                        "value": "frequency_domain",
                                                    },
                                                    {
                                                        "label": "FFT-based",
                                                        "value": "fft_based",
                                                    },
                                                    {
                                                        "label": "Counting",
                                                        "value": "counting",
                                                    },
                                                    {
                                                        "label": "Ensemble (All Methods)",
                                                        "value": "ensemble",
                                                    },
                                                ],
                                                value=["peak_detection", "fft_based"],
                                                className="mb-3",
                                            ),
                                            # Ensemble Output Options (shown when ensemble is selected)
                                            html.Div(
                                                id="resp-ensemble-options",
                                                style={"display": "none"},
                                                children=[
                                                    html.H6(
                                                        "Ensemble Output Method",
                                                        className="mb-2 fw-bold",
                                                    ),
                                                    dbc.Select(
                                                        id="resp-ensemble-method",
                                                        options=[
                                                            {
                                                                "label": "Mean (Simple Average)",
                                                                "value": "mean",
                                                            },
                                                            {
                                                                "label": "Weighted Mean",
                                                                "value": "weighted_mean",
                                                            },
                                                            {
                                                                "label": "Bagging (Bootstrap Aggregation)",
                                                                "value": "bagging",
                                                            },
                                                            {
                                                                "label": "Boosting (Sequential Learning)",
                                                                "value": "boosting",
                                                            },
                                                        ],
                                                        value="mean",
                                                        className="mb-2",
                                                    ),
                                                    html.Small(
                                                        "Mean: Simple average of all methods. Weighted Mean: Weighted average based on method reliability. "
                                                        "Bagging: Bootstrap aggregation for robust estimates. Boosting: Sequential learning to improve accuracy.",
                                                        className="text-muted d-block",
                                                    ),
                                                ],
                                            ),
                                            # Advanced Analysis Options
                                            html.H6(
                                                "Advanced Analysis", className="mb-2 mt-3 fw-bold"
                                            ),
                                            html.Small(
                                                "Select advanced analysis features",
                                                className="text-muted d-block mb-2",
                                            ),
                                            dbc.Checklist(
                                                id="resp-advanced-options",
                                                options=[
                                                    {
                                                        "label": "Sleep Apnea Detection",
                                                        "value": "sleep_apnea",
                                                    },
                                                    {
                                                        "label": "Breathing Pattern Analysis",
                                                        "value": "breathing_pattern",
                                                    },
                                                    {
                                                        "label": "Respiratory Variability",
                                                        "value": "respiratory_variability",
                                                    },
                                                    {
                                                        "label": "Multimodal Fusion",
                                                        "value": "multimodal",
                                                    },
                                                    {
                                                        "label": "PPG-ECG Fusion",
                                                        "value": "ppg_ecg_fusion",
                                                    },
                                                    {
                                                        "label": "Respiratory-Cardiac Fusion",
                                                        "value": "resp_cardiac_fusion",
                                                    },
                                                    {
                                                        "label": "Quality Assessment",
                                                        "value": "quality_assessment",
                                                    },
                                                ],
                                                value=[
                                                    "sleep_apnea",
                                                    "breathing_pattern",
                                                ],
                                                className="mb-3",
                                            ),
                                            # Breath Duration Constraints (Respiratory-Specific)
                                            html.H6(
                                                "Breath Duration Constraints",
                                                className="mb-2 mt-3 fw-bold",
                                            ),
                                            html.Small(
                                                "Min and max duration for valid breath cycles",
                                                className="text-muted d-block mb-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Min Duration (s)",
                                                                size="sm",
                                                            ),
                                                            dbc.Input(
                                                                id="resp-min-breath-duration",
                                                                type="number",
                                                                value=0.1,
                                                                min=0.1,
                                                                max=2.0,
                                                                step=0.1,
                                                                className="mb-1",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Max Duration (s)",
                                                                size="sm",
                                                            ),
                                                            dbc.Input(
                                                                id="resp-max-breath-duration",
                                                                type="number",
                                                                value=6.0,
                                                                min=2.0,
                                                                max=20.0,
                                                                step=0.5,
                                                                className="mb-1",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Action Buttons
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "🔄 Update Analysis",
                                                        id="resp-analyze-btn",
                                                        color="primary",
                                                        className="w-100 mb-1",
                                                    ),
                                                    dbc.Button(
                                                        "📊 Export Results",
                                                        id="resp-btn-export-results",
                                                        color="success",
                                                        className="w-100",
                                                    ),
                                                ],
                                                className="mt-3",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm",
                            )
                        ],
                        md=3,
                    ),
                    # Right Panel - Plots & Results
                    dbc.Col(
                        [
                            # Main Respiratory Signal Display & Analysis
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📈 Respiratory Signal Analysis & Results",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Signal display, breathing pattern detection, and comprehensive analysis",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Main signal plot
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="resp-main-plot",
                                                    style={"height": "350px"},
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
                                            ),
                                            # Analysis results text
                                            html.Div(
                                                id="resp-analysis-results",
                                                className="my-3",
                                            ),
                                            # Comprehensive analysis plots
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="resp-analysis-plots",
                                                    style={"height": "600px"},
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
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow-sm",
                            ),
                        ],
                        md=9,
                    ),
                ]
            ),
            # Bottom Section - Additional Analysis (dynamically populated)
            html.Div(id="resp-additional-analysis-section", className="mt-3"),
            # Stores for data management
            dcc.Store(id="resp-data-store"),
            dcc.Store(id="resp-features-store"),
            # Hidden components for cross-page compatibility (signal_filtering_callbacks)
            html.Div(id="filter-btn-apply", style={"display": "none"}),
            dcc.Dropdown(id="filter-type-select", style={"display": "none"}),
            dcc.Dropdown(id="filter-family-advanced", style={"display": "none"}),
            dcc.Dropdown(id="filter-response-advanced", style={"display": "none"}),
            dcc.Input(id="filter-low-freq-advanced", type="number", style={"display": "none"}),
            dcc.Input(id="filter-high-freq-advanced", type="number", style={"display": "none"}),
            dcc.Input(id="filter-order-advanced", type="number", style={"display": "none"}),
            dcc.Checklist(id="filter-quality-options", style={"display": "none"}),
            dcc.Dropdown(id="filter-signal-type-select", style={"display": "none"}),
            # Additional filter parameters
            dcc.Dropdown(id="advanced-filter-method", style={"display": "none"}),
            dcc.Input(id="advanced-iterations", type="number", style={"display": "none"}),
            dcc.Input(id="advanced-learning-rate", type="number", style={"display": "none"}),
            dcc.Input(id="advanced-noise-level", type="number", style={"display": "none"}),
            dcc.Input(id="artifact-removal-strength", type="number", style={"display": "none"}),
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dcc.Dropdown(id="detrend-option", style={"display": "none"}),
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            dcc.Input(id="ensemble-n-filters", type="number", style={"display": "none"}),
            dcc.Dropdown(id="fusion-method", style={"display": "none"}),
            dcc.Input(id="gaussian-sigma", type="number", style={"display": "none"}),
            dcc.Input(id="moving-avg-window", type="number", style={"display": "none"}),
            dcc.Dropdown(id="neural-model-complexity", style={"display": "none"}),
            dcc.Dropdown(id="neural-network-type", style={"display": "none"}),
            dcc.Input(id="reference-signal", style={"display": "none"}),
            dcc.Input(id="savgol-polyorder", type="number", style={"display": "none"}),
            dcc.Input(id="savgol-window", type="number", style={"display": "none"}),
            dcc.Dropdown(id="threshold-type", style={"display": "none"}),
            dcc.Input(id="threshold-value", type="number", style={"display": "none"}),
            dcc.Input(id="wavelet-level", type="number", style={"display": "none"}),
            dcc.Dropdown(id="wavelet-type", style={"display": "none"}),
            # Stores
            dcc.Store(id="store-filtered-signal"),
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            dcc.Graph(id="filter-original-plot", style={"display": "none"}),
            dcc.Graph(id="filter-filtered-plot", style={"display": "none"}),
            dcc.Graph(id="filter-comparison-plot", style={"display": "none"}),
            html.Div(id="filter-quality-metrics", style={"display": "none"}),
            dcc.Graph(id="filter-quality-plots", style={"display": "none"}),
            # Hidden navigation buttons for signal_filtering_callbacks compatibility
            html.Button(id="btn-nudge-m10", style={"display": "none"}),
            html.Button(id="btn-center", style={"display": "none"}),
            html.Button(id="btn-nudge-p10", style={"display": "none"}),
            # Hidden time window controls for signal_filtering_callbacks compatibility
            html.Div(
                children=[dcc.Slider(id="start-position-slider", min=0, max=100, value=0)],
                style={"display": "none"}
            ),
            dcc.Dropdown(id="duration-select", style={"display": "none"}),
        ]
    )
