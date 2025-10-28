"""
Signal Transforms Page Layout
Provides comprehensive signal transformation visualization and analysis.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def transforms_layout():
    """Create the comprehensive signal transforms layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("🔄 Signal Transforms", className="text-center mb-4"),
                    html.P(
                        [
                            "Apply various signal transformations including FFT, wavelet, Hilbert, STFT, ",
                            "and other advanced transforms for comprehensive signal analysis.",
                        ],
                        className="text-center text-muted mb-5",
                    ),
                ],
                className="mb-4",
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
                                                "🎛️ Transform Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure signal transformation parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-3"),
                                            dbc.Select(
                                                id="transforms-signal-type",
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
                                            # Signal Source Selection
                                            html.H6("Signal Source", className="mb-3"),
                                            dbc.Select(
                                                id="transforms-signal-source",
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
                                                value="filtered",
                                                className="mb-2",
                                            ),
                                            html.Small(
                                                "Filtered signal will be used if available from the filtering screen. Falls back to original signal if no filtering has been performed.",
                                                className="text-muted d-block mb-3",
                                            ),
                                            # Time Window Configuration (START POSITION + DURATION)
                                            html.H6("Time Window", className="mb-3"),
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
                                                                id="transforms-start-position",
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
                                                                tooltip={
                                                                    "placement": "bottom",
                                                                    "always_visible": True,
                                                                },
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
                                                                id="transforms-duration",
                                                                options=[
                                                                    {
                                                                        "label": "30 seconds",
                                                                        "value": 30,
                                                                    },
                                                                    {
                                                                        "label": "1 minute",
                                                                        "value": 60,
                                                                    },
                                                                    {
                                                                        "label": "2 minutes",
                                                                        "value": 120,
                                                                    },
                                                                    {
                                                                        "label": "5 minutes",
                                                                        "value": 300,
                                                                    },
                                                                ],
                                                                value=60,
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
                                                                id="transforms-btn-nudge-m10",
                                                                color="secondary",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "⏪ -5%",
                                                                id="transforms-btn-nudge-m5",
                                                                color="secondary",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "Center",
                                                                id="transforms-btn-center",
                                                                color="info",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "+5% ⏩",
                                                                id="transforms-btn-nudge-p5",
                                                                color="secondary",
                                                                size="sm",
                                                                className="me-1",
                                                            ),
                                                            dbc.Button(
                                                                "+10% ⏩",
                                                                id="transforms-btn-nudge-p10",
                                                                color="secondary",
                                                                size="sm",
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                            ),
                                            html.Hr(),
                                            # Transform Type Selection
                                            html.H6("Transform Type", className="mb-3"),
                                            dcc.Dropdown(
                                                id="transforms-type",
                                                options=[
                                                    {
                                                        "label": "Fast Fourier Transform (FFT)",
                                                        "value": "fft",
                                                    },
                                                    {
                                                        "label": "Short-Time Fourier Transform (STFT)",
                                                        "value": "stft",
                                                    },
                                                    {
                                                        "label": "Wavelet Transform",
                                                        "value": "wavelet",
                                                    },
                                                    {
                                                        "label": "Hilbert Transform",
                                                        "value": "hilbert",
                                                    },
                                                    {
                                                        "label": "Mel-Frequency Cepstral Coefficients (MFCC)",
                                                        "value": "mfcc",
                                                    },
                                                ],
                                                value="fft",
                                                className="mb-3",
                                            ),
                                            # Transform-Specific Parameters (Dynamic)
                                            html.Div(
                                                id="transforms-parameters-container",
                                                children=[],
                                                className="mb-4",
                                            ),
                                            # Analysis Options
                                            html.H6(
                                                "Analysis Options", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="transforms-analysis-options",
                                                options=[
                                                    {
                                                        "label": "Magnitude Spectrum",
                                                        "value": "magnitude",
                                                    },
                                                    {
                                                        "label": "Phase Spectrum",
                                                        "value": "phase",
                                                    },
                                                    {
                                                        "label": "Power Spectrum",
                                                        "value": "power",
                                                    },
                                                    {
                                                        "label": "Log Scale",
                                                        "value": "log_scale",
                                                    },
                                                    {
                                                        "label": "Peak Detection",
                                                        "value": "peak_detection",
                                                    },
                                                    {
                                                        "label": "Frequency Bands",
                                                        "value": "frequency_bands",
                                                    },
                                                ],
                                                value=["magnitude", "power"],
                                                className="mb-4",
                                            ),
                                            # Main Analysis Button
                                            dbc.Button(
                                                "🔄 Apply Transform",
                                                id="transforms-analyze-btn",
                                                color="primary",
                                                size="lg",
                                                className="w-100 mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100",
                            )
                        ],
                        md=4,
                    ),
                    # Right Panel - Results & Plots
                    dbc.Col(
                        [
                            # Main Transform Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📈 Main Transform Result",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="transforms-main-plot",
                                                style={"height": "400px"},
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Additional Analysis Plots
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Additional Analysis",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="transforms-analysis-plots",
                                                style={"height": "400px"},
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        md=8,
                    ),
                ]
            ),
            # Analysis Results Section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📋 Transform Results", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="transforms-analysis-results")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("🔍 Peak Analysis", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="transforms-peak-analysis")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Frequency Band Analysis
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Frequency Band Analysis",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="transforms-frequency-bands")]
                                    ),
                                ]
                            )
                        ],
                        md=12,
                    )
                ],
                className="mb-4",
            ),
            # Data Storage
            dcc.Store(id="store-transforms-data"),
            dcc.Store(id="store-transforms-results"),
            dcc.Store(id="store-filtered-signal"),  # Access to filtered signal from filtering page
            # Hidden components for compatibility with filtering/time-domain pages
            # These are needed because filtering page callbacks may reference these IDs
            html.Div(id="filter-btn-apply", style={"display": "none"}),
            dcc.Dropdown(id="filter-type-select", style={"display": "none"}),
            dcc.Dropdown(id="filter-signal-type-select", style={"display": "none"}),
            dcc.Checklist(id="filter-quality-options", style={"display": "none"}),
            dcc.Checklist(id="detrend-option", style={"display": "none"}),
            html.Div(id="filter-type-callback", style={"display": "none"}),
            # Traditional filter parameters
            dcc.Dropdown(id="filter-family-advanced", style={"display": "none"}),
            dcc.Dropdown(id="filter-response-advanced", style={"display": "none"}),
            dbc.Input(id="filter-low-freq-advanced", style={"display": "none"}),
            dbc.Input(id="filter-high-freq-advanced", style={"display": "none"}),
            dbc.Input(id="filter-order-advanced", style={"display": "none"}),
            # Savitzky-Golay parameters
            dbc.Input(id="savgol-window", style={"display": "none"}),
            dbc.Input(id="savgol-polyorder", style={"display": "none"}),
            # Moving average and Gaussian parameters
            dbc.Input(id="moving-avg-window", style={"display": "none"}),
            dbc.Input(id="gaussian-sigma", style={"display": "none"}),
            html.Div(id="traditional-filter-params", style={"display": "none"}),
            # Advanced filter parameters
            dcc.Dropdown(id="advanced-filter-method", style={"display": "none"}),
            # Kalman filter parameters
            dbc.Input(id="kalman-r", style={"display": "none"}),
            dbc.Input(id="kalman-q", style={"display": "none"}),
            html.Div(id="kalman-params", style={"display": "none"}),
            # Optimization-based filter parameters
            dcc.Dropdown(id="optimization-loss-type", style={"display": "none"}),
            dbc.Input(id="optimization-initial-guess", style={"display": "none"}),
            dbc.Input(id="optimization-learning-rate", style={"display": "none"}),
            dbc.Input(id="optimization-iterations", style={"display": "none"}),
            html.Div(id="optimization-params", style={"display": "none"}),
            # Gradient-based filter parameters
            dbc.Input(id="gradient-learning-rate", style={"display": "none"}),
            dbc.Input(id="gradient-iterations", style={"display": "none"}),
            html.Div(id="gradient-params", style={"display": "none"}),
            # Convolution filter parameters
            dcc.Dropdown(id="convolution-kernel-type", style={"display": "none"}),
            dbc.Input(id="convolution-kernel-size", style={"display": "none"}),
            html.Div(id="convolution-params", style={"display": "none"}),
            # Attention mechanism parameters
            dcc.Dropdown(id="attention-type", style={"display": "none"}),
            dbc.Input(id="attention-size", style={"display": "none"}),
            dbc.Input(id="attention-sigma", style={"display": "none"}),
            html.Div(id="attention-gaussian-params", style={"display": "none"}),
            dcc.Checklist(id="attention-ascending", style={"display": "none"}),
            html.Div(id="attention-linear-params", style={"display": "none"}),
            dbc.Input(id="attention-base", style={"display": "none"}),
            html.Div(id="attention-exponential-params", style={"display": "none"}),
            html.Div(id="attention-params", style={"display": "none"}),
            # Adaptive filter parameters
            dbc.Input(id="adaptive-mu", style={"display": "none"}),
            dbc.Input(id="adaptive-order", style={"display": "none"}),
            html.Div(id="adaptive-params", style={"display": "none"}),
            html.Div(id="advanced-filter-params", style={"display": "none"}),
            # Artifact removal parameters
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dbc.Input(id="artifact-removal-strength", style={"display": "none"}),
            dcc.Dropdown(id="wavelet-type", style={"display": "none"}),
            dbc.Input(id="wavelet-level", style={"display": "none"}),
            dcc.Dropdown(id="threshold-type", style={"display": "none"}),
            dbc.Input(id="threshold-value", style={"display": "none"}),
            html.Div(id="artifact-removal-params", style={"display": "none"}),
            # Neural network parameters
            dcc.Dropdown(id="neural-network-type", style={"display": "none"}),
            dbc.Input(id="neural-model-complexity", style={"display": "none"}),
            html.Div(id="neural-network-params", style={"display": "none"}),
            # Ensemble parameters
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            dbc.Input(id="ensemble-n-filters", style={"display": "none"}),
            html.Div(id="ensemble-params", style={"display": "none"}),
            # Reference signal and fusion parameters
            dcc.Dropdown(id="reference-signal", style={"display": "none"}),
            dcc.Dropdown(id="fusion-method", style={"display": "none"}),
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            dcc.Graph(id="filter-original-plot", style={"display": "none"}),
            dcc.Graph(id="filter-filtered-plot", style={"display": "none"}),
            dcc.Graph(id="filter-comparison-plot", style={"display": "none"}),
            dcc.Graph(id="filter-quality-plots", style={"display": "none"}),
            html.Div(id="filter-quality-metrics", style={"display": "none"}),
            dcc.Graph(id="frequency-filtered-signal-plot", style={"display": "none"}),
            dcc.Graph(id="filter-response-plot", style={"display": "none"}),
            html.Div(id="filter-stats", style={"display": "none"}),
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