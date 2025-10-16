"""
Analysis pages layout for vitalDSP webapp.

This module provides layouts for various analysis pages.
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
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "End Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="10",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Quick Window Navigation
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10s",
                                                        id="btn-nudge-m10",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -1s",
                                                        id="btn-nudge-m1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+1s ⏩",
                                                        id="btn-nudge-p1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10s ⏩",
                                                        id="btn-nudge-p10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Range Slider for Time Window
                                            html.Label(
                                                "Time Range Slider",
                                                className="form-label",
                                            ),
                                            dcc.RangeSlider(
                                                id="time-range-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 10],
                                                allowCross=False,
                                                pushable=1,
                                                updatemode="mouseup",
                                                className="mb-4",
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
            # Stores for data management
            dcc.Store(id="store-time-domain-data"),
            dcc.Store(id="store-filtered-data"),
            dcc.Store(id="store-analysis-results"),
            dcc.Store(id="store-time-domain-features"),  # For export
            # Download components for export
            dcc.Download(id="download-time-domain-csv"),
            dcc.Download(id="download-time-domain-json"),
        ]
    )


def frequency_layout():
    """Create the frequency domain analysis layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1(
                        "📊 Frequency Domain Analysis", className="text-center mb-4"
                    ),
                    html.P(
                        [
                            "Analyze your PPG/ECG signals in the frequency domain with FFT, PSD, STFT, ",
                            "and wavelet analysis. Explore signal characteristics, harmonics, and frequency content.",
                        ],
                        className="text-center text-muted mb-5",
                    ),
                ],
                className="mb-4",
            ),
            # Action Button and Key Controls - Always at the top for easy access
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
                                                                "📊 Analyze Frequency Domain",
                                                                id="freq-btn-update-analysis",
                                                                color="primary",
                                                                size="lg",
                                                                className="w-100",
                                                            )
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            # Signal Source Selection
                                                            html.Label(
                                                                "Signal Source",
                                                                className="form-label",
                                                            ),
                                                            dbc.Select(
                                                                id="freq-signal-source-select",
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
                                                                "Filtered signal will be used if available from the filtering screen. Falls back to original signal if no filtering has been performed.",
                                                                className="text-muted",
                                                            ),
                                                            # Analysis Type Selection
                                                            html.Label(
                                                                "Analysis Type",
                                                                className="form-label",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="freq-analysis-type",
                                                                options=[
                                                                    {
                                                                        "label": "FFT (Fast Fourier Transform)",
                                                                        "value": "fft",
                                                                    },
                                                                    {
                                                                        "label": "Power Spectral Density (PSD)",
                                                                        "value": "psd",
                                                                    },
                                                                    {
                                                                        "label": "Short-Time Fourier Transform (STFT)",
                                                                        "value": "stft",
                                                                    },
                                                                    {
                                                                        "label": "Wavelet Analysis",
                                                                        "value": "wavelet",
                                                                    },
                                                                ],
                                                                value="fft",
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            # Time Window Controls
                                                            html.Label(
                                                                "Start Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="freq-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "End Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="freq-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="10",
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            # Quick Window Navigation
                                                            html.Label(
                                                                "Quick Navigation",
                                                                className="form-label",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dbc.Button(
                                                                        "⏪ -10s",
                                                                        id="freq-btn-nudge-m10",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="me-1 mb-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "⏪ -1s",
                                                                        id="freq-btn-nudge-m1",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="me-1 mb-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "+1s ⏩",
                                                                        id="freq-btn-nudge-p1",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="me-1 mb-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "+10s ⏩",
                                                                        id="freq-btn-nudge-p10",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="mb-1",
                                                                    ),
                                                                ],
                                                                className="mb-2 quick-nav-buttons",
                                                            ),
                                                            # Range Slider for Time Window
                                                            html.Label(
                                                                "Time Range Slider",
                                                                className="form-label",
                                                            ),
                                                            dcc.RangeSlider(
                                                                id="freq-time-range-slider",
                                                                min=0,
                                                                max=100,
                                                                step=0.1,
                                                                value=[0, 10],
                                                                allowCross=False,
                                                                pushable=1,
                                                                updatemode="mouseup",
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                className="mb-4 freq-top-container analysis-controls",
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            # Main Analysis Section - Two Panel Layout
            dbc.Row(
                [
                    # Left Panel - Controls & Parameters (Compact)
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
                                                "Configure frequency analysis parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # FFT Parameters
                                            html.Div(
                                                id="fft-params",
                                                children=[
                                                    html.H6(
                                                        "FFT Parameters",
                                                        className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Window Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="fft-window-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Hann",
                                                                                "value": "hann",
                                                                            },
                                                                            {
                                                                                "label": "Hamming",
                                                                                "value": "hamming",
                                                                            },
                                                                            {
                                                                                "label": "Blackman",
                                                                                "value": "blackman",
                                                                            },
                                                                            {
                                                                                "label": "None",
                                                                                "value": "none",
                                                                            },
                                                                        ],
                                                                        value="hann",
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "N Points",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="fft-n-points",
                                                                        type="number",
                                                                        value=1024,
                                                                        min=64,
                                                                        step=64,
                                                                        placeholder="1024",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                            ),
                                            # PSD Parameters
                                            html.Div(
                                                id="psd-params",
                                                children=[
                                                    html.H6(
                                                        "PSD Parameters",
                                                        className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Window (s)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="psd-window",
                                                                        type="number",
                                                                        value=2.0,
                                                                        min=0.5,
                                                                        max=10.0,
                                                                        step=0.5,
                                                                        placeholder="2.0",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Overlap (%)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="psd-overlap",
                                                                        type="number",
                                                                        value=50,
                                                                        min=0,
                                                                        max=95,
                                                                        step=5,
                                                                        placeholder="50",
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
                                                                        "Max Freq (Hz)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="psd-freq-max",
                                                                        type="number",
                                                                        value=25.0,
                                                                        min=1.0,
                                                                        max=100.0,
                                                                        step=1.0,
                                                                        placeholder="25.0",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Log Scale",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Checklist(
                                                                        id="psd-log-scale",
                                                                        options=[
                                                                            {
                                                                                "label": "dB Scale",
                                                                                "value": "on",
                                                                            }
                                                                        ],
                                                                        value=["on"],
                                                                        className="mb-3",
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
                                                                        "Normalize",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Checklist(
                                                                        id="psd-normalize",
                                                                        options=[
                                                                            {
                                                                                "label": "Normalize PSD",
                                                                                "value": "on",
                                                                            }
                                                                        ],
                                                                        value=[],
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Channel",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="psd-channel",
                                                                        options=[
                                                                            {
                                                                                "label": "Signal",
                                                                                "value": "signal",
                                                                            },
                                                                            {
                                                                                "label": "RED",
                                                                                "value": "red",
                                                                            },
                                                                            {
                                                                                "label": "IR",
                                                                                "value": "ir",
                                                                            },
                                                                            {
                                                                                "label": "Waveform",
                                                                                "value": "waveform",
                                                                            },
                                                                        ],
                                                                        value="signal",
                                                                        clearable=False,
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                            ),
                                            # STFT Parameters
                                            html.Div(
                                                id="stft-params",
                                                children=[
                                                    html.H6(
                                                        "STFT Parameters",
                                                        className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Window Size",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="stft-window-size",
                                                                        type="number",
                                                                        value=256,
                                                                        min=64,
                                                                        step=64,
                                                                        placeholder="256",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Hop Size",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="stft-hop-size",
                                                                        type="number",
                                                                        value=128,
                                                                        min=16,
                                                                        step=16,
                                                                        placeholder="128",
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
                                                                        "Window Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="stft-window-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Hann",
                                                                                "value": "hann",
                                                                            },
                                                                            {
                                                                                "label": "Hamming",
                                                                                "value": "hamming",
                                                                            },
                                                                            {
                                                                                "label": "Blackman",
                                                                                "value": "blackman",
                                                                            },
                                                                            {
                                                                                "label": "Kaiser",
                                                                                "value": "kaiser",
                                                                            },
                                                                            {
                                                                                "label": "Gaussian",
                                                                                "value": "gaussian",
                                                                            },
                                                                            {
                                                                                "label": "Rectangular",
                                                                                "value": "rectangular",
                                                                            },
                                                                        ],
                                                                        value="hann",
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Overlap (%)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="stft-overlap",
                                                                        type="number",
                                                                        value=50,
                                                                        min=0,
                                                                        max=95,
                                                                        step=5,
                                                                        placeholder="50",
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
                                                                        "Scaling",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="stft-scaling",
                                                                        options=[
                                                                            {
                                                                                "label": "Density",
                                                                                "value": "density",
                                                                            },
                                                                            {
                                                                                "label": "Spectrum",
                                                                                "value": "spectrum",
                                                                            },
                                                                        ],
                                                                        value="density",
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Max Freq (Hz)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="stft-freq-max",
                                                                        type="number",
                                                                        value=50,
                                                                        min=0.1,
                                                                        step=1,
                                                                        placeholder="50",
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
                                                                        "Colormap",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="stft-colormap",
                                                                        options=[
                                                                            {
                                                                                "label": "Viridis",
                                                                                "value": "viridis",
                                                                            },
                                                                            {
                                                                                "label": "Plasma",
                                                                                "value": "plasma",
                                                                            },
                                                                            {
                                                                                "label": "Inferno",
                                                                                "value": "inferno",
                                                                            },
                                                                            {
                                                                                "label": "Magma",
                                                                                "value": "magma",
                                                                            },
                                                                            {
                                                                                "label": "Jet",
                                                                                "value": "jet",
                                                                            },
                                                                            {
                                                                                "label": "Hot",
                                                                                "value": "hot",
                                                                            },
                                                                            {
                                                                                "label": "Cool",
                                                                                "value": "cool",
                                                                            },
                                                                        ],
                                                                        value="viridis",
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=12,
                                                            )
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                            ),
                                            # Wavelet Parameters
                                            html.Div(
                                                id="wavelet-params",
                                                children=[
                                                    html.H6(
                                                        "Wavelet Parameters",
                                                        className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Wavelet Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="wavelet-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Daubechies (db4)",
                                                                                "value": "db4",
                                                                            },
                                                                            {
                                                                                "label": "Symlets (sym4)",
                                                                                "value": "sym4",
                                                                            },
                                                                            {
                                                                                "label": "Coiflets (coif4)",
                                                                                "value": "coif4",
                                                                            },
                                                                            {
                                                                                "label": "Haar",
                                                                                "value": "haar",
                                                                            },
                                                                            {
                                                                                "label": "Mexican Hat",
                                                                                "value": "mexh",
                                                                            },
                                                                            {
                                                                                "label": "Morlet",
                                                                                "value": "morl",
                                                                            },
                                                                        ],
                                                                        value="db4",
                                                                        className="mb-3",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Decomposition Levels",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="wavelet-levels",
                                                                        type="number",
                                                                        value=4,
                                                                        min=1,
                                                                        max=10,
                                                                        step=1,
                                                                        placeholder="4",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                            ),
                                            # Frequency Range
                                            html.H6(
                                                "Frequency Range", className="mb-3"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Min Freq (Hz)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="freq-min",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Max Freq (Hz)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="freq-max",
                                                                type="number",
                                                                value=100,
                                                                min=0.1,
                                                                step=1,
                                                                placeholder="100",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Analysis Options
                                            html.H6(
                                                "Analysis Options", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="freq-analysis-options",
                                                options=[
                                                    {
                                                        "label": "Peak Detection",
                                                        "value": "peak_detection",
                                                    },
                                                    {
                                                        "label": "Harmonic Analysis",
                                                        "value": "harmonic_analysis",
                                                    },
                                                    {
                                                        "label": "Band Power Analysis",
                                                        "value": "band_power",
                                                    },
                                                    {
                                                        "label": "Stability Analysis",
                                                        "value": "stability",
                                                    },
                                                ],
                                                value=[
                                                    "peak_detection",
                                                    "harmonic_analysis",
                                                ],
                                                className="mb-4",
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100",
                            )
                        ],
                        md=3,
                    ),
                    # Right Panel - Results & Plots (Full width)
                    dbc.Col(
                        [
                            # Main Frequency Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📈 Main Frequency Analysis",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="freq-main-plot",
                                                style={
                                                    "height": "600px",
                                                    "minHeight": "550px",
                                                    "overflow": "visible",
                                                },
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ],
                                        style={"overflow": "visible"},
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # PSD Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Power Spectral Density",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="freq-psd-plot",
                                                style={
                                                    "height": "400px",
                                                    "minHeight": "350px",
                                                    "overflow": "visible",
                                                },
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ],
                                        style={"overflow": "visible"},
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Spectrogram Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🌊 Time-Frequency Analysis (Spectrogram)",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="freq-spectrogram-plot",
                                                style={
                                                    "height": "400px",
                                                    "minHeight": "350px",
                                                    "overflow": "visible",
                                                },
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ],
                                        style={"overflow": "visible"},
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        md=9,
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
                                                "📋 Analysis Results", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="freq-analysis-results")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                    # Peak Analysis Table
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("🔍 Peak Analysis", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="freq-peak-analysis-table")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Additional Analysis Tables
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Band Power Analysis",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="freq-band-power-table")]
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
                                        [
                                            html.H5(
                                                "🔬 Frequency Stability",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="freq-stability-table")]),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Harmonic Analysis Table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🎵 Harmonic Analysis", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="freq-harmonics-table")]),
                                ]
                            )
                        ],
                        md=12,
                    )
                ],
                className="mb-4",
            ),
            # Stores for data management
            dcc.Store(id="store-frequency-data"),
            dcc.Store(id="store-time-freq-data"),
            dcc.Store(id="store-freq-analysis-results"),
        ]
    )


def filtering_layout():
    """Create the comprehensive signal filtering page layout with optimized space usage."""
    return html.Div(
        [
            # Page Header - Compact
            html.Div(
                [
                    html.H1(
                        "🔧 Advanced Signal Filtering", className="text-center mb-3"
                    ),
                    html.P(
                        [
                            "Apply advanced filtering techniques including traditional filters, Kalman filters, ",
                            "neural network filtering, and artifact removal for optimal signal quality.",
                        ],
                        className="text-center text-muted mb-3",
                    ),
                ],
                className="mb-3",
            ),
            # Main Controller Bar - Top
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    # Filter Type Selection
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Filter Type:",
                                                className="form-label mb-1",
                                            ),
                                            dbc.Select(
                                                id="filter-type-select",
                                                options=[
                                                    {
                                                        "label": "Traditional Filters",
                                                        "value": "traditional",
                                                    },
                                                    {
                                                        "label": "Advanced Filters",
                                                        "value": "advanced",
                                                    },
                                                    {
                                                        "label": "Artifact Removal",
                                                        "value": "artifact",
                                                    },
                                                    {
                                                        "label": "Neural Network",
                                                        "value": "neural",
                                                    },
                                                    {
                                                        "label": "Ensemble Methods",
                                                        "value": "ensemble",
                                                    },
                                                ],
                                                value="traditional",
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    # Time Window Controls - Compact
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Time Range (s):",
                                                className="form-label mb-1",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Input(
                                                                id="filter-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                size="sm",
                                                                placeholder="Start",
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Input(
                                                                id="filter-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                size="sm",
                                                                placeholder="End",
                                                            )
                                                        ],
                                                        width=6,
                                                    ),
                                                ]
                                            ),
                                            # Time Range Slider
                                            html.Label(
                                                "Time Range Slider:",
                                                className="form-label mb-1 mt-2",
                                            ),
                                            dcc.RangeSlider(
                                                id="filter-time-range-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 10],
                                                marks={},
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                                className="mt-1",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    # Quick Navigation
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Navigation:",
                                                className="form-label mb-1",
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "←10s",
                                                        id="filter-btn-nudge-m10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "←1s",
                                                        id="filter-btn-nudge-m1",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "+1s",
                                                        id="filter-btn-nudge-p1",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "+10s",
                                                        id="filter-btn-nudge-p10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                size="sm",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    # Apply Filter Button
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Action:", className="form-label mb-1"
                                            ),
                                            dbc.Button(
                                                "Apply Filter",
                                                id="filter-btn-apply",
                                                color="primary",
                                                size="sm",
                                                className="w-100",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    # Export Buttons
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Export:", className="form-label mb-1"
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="fas fa-file-csv me-1"
                                                            ),
                                                            "CSV",
                                                        ],
                                                        id="btn-export-filtered-csv",
                                                        color="success",
                                                        outline=True,
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="fas fa-file-code me-1"
                                                            ),
                                                            "JSON",
                                                        ],
                                                        id="btn-export-filtered-json",
                                                        color="info",
                                                        outline=True,
                                                        size="sm",
                                                    ),
                                                ],
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    # Signal Type Selection
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Signal Type:",
                                                className="form-label mb-1",
                                            ),
                                            dbc.Select(
                                                id="filter-signal-type-select",
                                                options=[
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "PPG",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ECG",
                                                    },
                                                    {
                                                        "label": "Other",
                                                        "value": "Other",
                                                    },
                                                ],
                                                value="PPG",
                                                className="mb-1",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    # Quality Assessment Toggle
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Quality:", className="form-label mb-1"
                                            ),
                                            dbc.Checklist(
                                                id="filter-quality-options",
                                                options=[
                                                    {"label": "SNR", "value": "snr"},
                                                    {
                                                        "label": "Metrics",
                                                        "value": "metrics",
                                                    },
                                                ],
                                                value=["snr"],
                                                inline=True,
                                            ),
                                        ],
                                        md=2,
                                    ),
                                ],
                                className="g-2",
                            )
                        ]
                    )
                ],
                className="mb-3",
            ),
            # Main Analysis Section
            dbc.Row(
                [
                    # Left Panel - Side Controller (Compact)
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H6(
                                                "🎛️ Filter Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Customize filter parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Preprocessing Options
                                            html.H6("Preprocessing", className="mb-2"),
                                            dbc.Checklist(
                                                id="detrend-option",
                                                options=[
                                                    {
                                                        "label": "Detrending",
                                                        "value": "detrend",
                                                    }
                                                ],
                                                value=["detrend"],
                                                className="mb-3",
                                            ),
                                            # Filter Type Change Callback
                                            html.Div(
                                                id="filter-type-callback",
                                                style={"display": "none"},
                                            ),
                                            # Traditional Filter Parameters
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Traditional Filters",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Family:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="filter-family-advanced",
                                                                        options=[
                                                                            {
                                                                                "label": "Butterworth",
                                                                                "value": "butter",
                                                                            },
                                                                            {
                                                                                "label": "Chebyshev I",
                                                                                "value": "cheby1",
                                                                            },
                                                                            {
                                                                                "label": "Chebyshev II",
                                                                                "value": "cheby2",
                                                                            },
                                                                            {
                                                                                "label": "Elliptic",
                                                                                "value": "ellip",
                                                                            },
                                                                            {
                                                                                "label": "Bessel",
                                                                                "value": "bessel",
                                                                            },
                                                                        ],
                                                                        value="butter",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Response:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="filter-response-advanced",
                                                                        options=[
                                                                            {
                                                                                "label": "Low Pass",
                                                                                "value": "low",
                                                                            },
                                                                            {
                                                                                "label": "High Pass",
                                                                                "value": "high",
                                                                            },
                                                                            {
                                                                                "label": "Band Pass",
                                                                                "value": "bandpass",
                                                                            },
                                                                        ],
                                                                        value="bandpass",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Low Freq (Hz):",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="filter-low-freq-advanced",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0,
                                                                        step=0.1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "High Freq (Hz):",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="filter-high-freq-advanced",
                                                                        type="number",
                                                                        value=5,
                                                                        min=0,
                                                                        step=0.1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Order:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="filter-order-advanced",
                                                                        type="number",
                                                                        value=4,
                                                                        min=1,
                                                                        max=10,
                                                                        step=1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            )
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    # Additional Traditional Filters
                                                    html.H6(
                                                        "Additional Filters",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Savitzky-Golay:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="savgol-window",
                                                                        type="number",
                                                                        placeholder="Window",
                                                                        min=3,
                                                                        step=2,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Poly Order:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="savgol-polyorder",
                                                                        type="number",
                                                                        placeholder="Order",
                                                                        min=1,
                                                                        max=5,
                                                                        step=1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Moving Avg:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="moving-avg-window",
                                                                        type="number",
                                                                        placeholder="Window",
                                                                        min=3,
                                                                        step=1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Gaussian σ:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="gaussian-sigma",
                                                                        type="number",
                                                                        placeholder="Sigma",
                                                                        min=0.1,
                                                                        max=10.0,
                                                                        step=0.1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                                id="traditional-filter-params",
                                            ),
                                            # Advanced Filter Parameters
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Advanced Filters",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Method:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="advanced-filter-method",
                                                                        options=[
                                                                            {
                                                                                "label": "Kalman",
                                                                                "value": "kalman",
                                                                            },
                                                                            {
                                                                                "label": "Optimization",
                                                                                "value": "optimization",
                                                                            },
                                                                            {
                                                                                "label": "Gradient Descent",
                                                                                "value": "gradient_descent",
                                                                            },
                                                                            {
                                                                                "label": "Convolution",
                                                                                "value": "convolution",
                                                                            },
                                                                            {
                                                                                "label": "Attention",
                                                                                "value": "attention",
                                                                            },
                                                                        ],
                                                                        value="kalman",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Noise Level:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="advanced-noise-level",
                                                                        type="number",
                                                                        value=0.1,
                                                                        min=0.01,
                                                                        max=1.0,
                                                                        step=0.01,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Iterations:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="advanced-iterations",
                                                                        type="number",
                                                                        value=100,
                                                                        min=10,
                                                                        max=1000,
                                                                        step=10,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Learning Rate:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="advanced-learning-rate",
                                                                        type="number",
                                                                        value=0.01,
                                                                        min=0.001,
                                                                        max=0.1,
                                                                        step=0.001,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                                id="advanced-filter-params",
                                                style={"display": "none"},
                                            ),
                                            # Artifact Removal Parameters
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Artifact Removal",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Type:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="artifact-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Baseline Drift",
                                                                                "value": "baseline",
                                                                            },
                                                                            {
                                                                                "label": "Spike Artifacts",
                                                                                "value": "spike",
                                                                            },
                                                                            {
                                                                                "label": "Noise",
                                                                                "value": "noise",
                                                                            },
                                                                            {
                                                                                "label": "Powerline",
                                                                                "value": "powerline",
                                                                            },
                                                                            {
                                                                                "label": "PCA Removal",
                                                                                "value": "pca",
                                                                            },
                                                                            {
                                                                                "label": "ICA Removal",
                                                                                "value": "ica",
                                                                            },
                                                                        ],
                                                                        value="baseline",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Strength:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="artifact-removal-strength",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0.1,
                                                                        max=1.0,
                                                                        step=0.1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    # Enhanced Artifact Removal Options
                                                    html.H6(
                                                        "Enhanced Options",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Wavelet:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="wavelet-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Daubechies 4",
                                                                                "value": "db4",
                                                                            },
                                                                            {
                                                                                "label": "Daubechies 8",
                                                                                "value": "db8",
                                                                            },
                                                                            {
                                                                                "label": "Haar",
                                                                                "value": "haar",
                                                                            },
                                                                            {
                                                                                "label": "Symlets 4",
                                                                                "value": "sym4",
                                                                            },
                                                                            {
                                                                                "label": "Coiflets 4",
                                                                                "value": "coif4",
                                                                            },
                                                                        ],
                                                                        value="db4",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Level:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="wavelet-level",
                                                                        type="number",
                                                                        value=3,
                                                                        min=1,
                                                                        max=8,
                                                                        step=1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Threshold:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="threshold-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Soft",
                                                                                "value": "soft",
                                                                            },
                                                                            {
                                                                                "label": "Hard",
                                                                                "value": "hard",
                                                                            },
                                                                            {
                                                                                "label": "Universal",
                                                                                "value": "universal",
                                                                            },
                                                                            {
                                                                                "label": "Minimax",
                                                                                "value": "minimax",
                                                                            },
                                                                        ],
                                                                        value="soft",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Value:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="threshold-value",
                                                                        type="number",
                                                                        value=0.1,
                                                                        min=0.01,
                                                                        max=1.0,
                                                                        step=0.01,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                                id="artifact-removal-params",
                                                style={"display": "none"},
                                            ),
                                            # Neural Network Parameters
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Neural Network",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Type:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="neural-network-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Autoencoder",
                                                                                "value": "autoencoder",
                                                                            },
                                                                            {
                                                                                "label": "LSTM",
                                                                                "value": "lstm",
                                                                            },
                                                                            {
                                                                                "label": "CNN",
                                                                                "value": "cnn",
                                                                            },
                                                                        ],
                                                                        value="autoencoder",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Complexity:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="neural-model-complexity",
                                                                        options=[
                                                                            {
                                                                                "label": "Low",
                                                                                "value": "low",
                                                                            },
                                                                            {
                                                                                "label": "Medium",
                                                                                "value": "medium",
                                                                            },
                                                                            {
                                                                                "label": "High",
                                                                                "value": "high",
                                                                            },
                                                                        ],
                                                                        value="medium",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                                id="neural-network-params",
                                                style={"display": "none"},
                                            ),
                                            # Ensemble Parameters
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Ensemble Methods",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Method:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="ensemble-method",
                                                                        options=[
                                                                            {
                                                                                "label": "Mean",
                                                                                "value": "mean",
                                                                            },
                                                                            {
                                                                                "label": "Median",
                                                                                "value": "median",
                                                                            },
                                                                            {
                                                                                "label": "Weighted",
                                                                                "value": "weighted",
                                                                            },
                                                                            {
                                                                                "label": "Bagging",
                                                                                "value": "bagging",
                                                                            },
                                                                            {
                                                                                "label": "Boosting",
                                                                                "value": "boosting",
                                                                            },
                                                                            {
                                                                                "label": "Stacking",
                                                                                "value": "stacking",
                                                                            },
                                                                        ],
                                                                        value="mean",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "N Filters:",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="ensemble-n-filters",
                                                                        type="number",
                                                                        value=3,
                                                                        min=2,
                                                                        max=10,
                                                                        step=1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                                id="ensemble-params",
                                                style={"display": "none"},
                                            ),
                                            # Multi-modal Filtering Options
                                            html.H6("Multi-modal", className="mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Reference:",
                                                                className="form-label",
                                                            ),
                                                            dbc.Select(
                                                                id="reference-signal",
                                                                options=[
                                                                    {
                                                                        "label": "None",
                                                                        "value": "none",
                                                                    },
                                                                    {
                                                                        "label": "ECG",
                                                                        "value": "ecg",
                                                                    },
                                                                    {
                                                                        "label": "PPG",
                                                                        "value": "ppg",
                                                                    },
                                                                    {
                                                                        "label": "Respiration",
                                                                        "value": "respiration",
                                                                    },
                                                                    {
                                                                        "label": "Motion",
                                                                        "value": "motion",
                                                                    },
                                                                ],
                                                                value="none",
                                                                size="sm",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Fusion:",
                                                                className="form-label",
                                                            ),
                                                            dbc.Select(
                                                                id="fusion-method",
                                                                options=[
                                                                    {
                                                                        "label": "Weighted",
                                                                        "value": "weighted",
                                                                    },
                                                                    {
                                                                        "label": "Kalman",
                                                                        "value": "kalman",
                                                                    },
                                                                    {
                                                                        "label": "Bayesian",
                                                                        "value": "bayesian",
                                                                    },
                                                                    {
                                                                        "label": "Deep Learning",
                                                                        "value": "deep_learning",
                                                                    },
                                                                ],
                                                                value="weighted",
                                                                size="sm",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
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
                    ),
                    # Right Panel - Results & Plots (Larger)
                    dbc.Col(
                        [
                            # Top Row - Original and Filtered Signals (Increased height)
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        [
                                                            html.H6(
                                                                "📈 Original Signal",
                                                                className="mb-0",
                                                            ),
                                                            html.Small(
                                                                "Input signal",
                                                                className="text-muted",
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dcc.Loading(
                                                                dcc.Graph(
                                                                    id="filter-original-plot",
                                                                    style={
                                                                        "height": "450px"
                                                                    },
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
                                                        [
                                                            html.H6(
                                                                "🔧 Filtered Signal",
                                                                className="mb-0",
                                                            ),
                                                            html.Small(
                                                                "After filtering",
                                                                className="text-muted",
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dcc.Loading(
                                                                dcc.Graph(
                                                                    id="filter-filtered-plot",
                                                                    style={
                                                                        "height": "450px"
                                                                    },
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
                                                ]
                                            )
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Filter Comparison - Full Width
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        [
                                                            html.H6(
                                                                "⚖️ Filter Comparison",
                                                                className="mb-0",
                                                            ),
                                                            html.Small(
                                                                "Compare different filtering approaches",
                                                                className="text-muted",
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dcc.Loading(
                                                                dcc.Graph(
                                                                    id="filter-comparison-plot",
                                                                    style={
                                                                        "height": "350px"
                                                                    },
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
                                                ]
                                            )
                                        ],
                                        md=12,
                                    )
                                ],
                                className="mb-3",
                            ),
                            # Quality Metrics - Full Width
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        [
                                                            html.H6(
                                                                "📊 Quality Metrics",
                                                                className="mb-0",
                                                            ),
                                                            html.Small(
                                                                "Quantitative assessment of filtering performance",
                                                                className="text-muted",
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                id="filter-quality-metrics",
                                                                className="mb-3",
                                                            ),
                                                            dcc.Loading(
                                                                dcc.Graph(
                                                                    id="filter-quality-plots",
                                                                    style={
                                                                        "height": "350px"
                                                                    },
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
                                                ]
                                            )
                                        ],
                                        md=12,
                                    )
                                ]
                            ),
                        ],
                        md=9,
                    ),
                ]
            ),
            # Bottom Section - Additional Analysis
            html.Div(id="filter-additional-analysis-section", className="mt-3"),
            # Stores for data management
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            dcc.Store(id="store-filtered-signal"),  # For export
            # Download components for export
            dcc.Download(id="download-filtered-csv"),
            dcc.Download(id="download-filtered-json"),
        ]
    )


def physiological_layout():
    """Create the comprehensive physiological analysis layout."""
    return html.Div(
        [
            # Main Analysis Section
            dbc.Row(
                [
                    # Left Panel - Modern Analysis Controls
                    dbc.Col(
                        [
                            # Quick Actions Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "⚡ Quick Actions",
                                                className="mb-0 text-primary",
                                            ),
                                            html.Small(
                                                "Essential controls for immediate analysis",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="bg-primary bg-opacity-10 border-primary",
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "🔄 Update Analysis",
                                                                id="physio-btn-update-analysis",
                                                                color="primary",
                                                                size="lg",
                                                                className="w-100 mb-2",
                                                            )
                                                        ],
                                                        md=12,
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
                                                                        id="btn-export-physio-csv",
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
                                                                        id="btn-export-physio-json",
                                                                        color="info",
                                                                        outline=True,
                                                                        size="lg",
                                                                    ),
                                                                ],
                                                                className="w-100",
                                                            )
                                                        ],
                                                        md=12,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Data & Signal Configuration Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📁 Data & Signal",
                                                className="mb-0 text-info",
                                            ),
                                            html.Small(
                                                "Configure data source and signal parameters",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="bg-info bg-opacity-10 border-info",
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Data Source Selection
                                            html.Label(
                                                "Data Source",
                                                className="form-label fw-bold small",
                                            ),
                                            dbc.Select(
                                                id="physio-data-source-select",
                                                options=[
                                                    {
                                                        "label": "📤 Uploaded Data",
                                                        "value": "uploaded",
                                                    },
                                                    {
                                                        "label": "📋 Sample Data",
                                                        "value": "sample",
                                                    },
                                                ],
                                                value="uploaded",
                                                className="mb-3",
                                            ),
                                            # Signal Type Selection
                                            html.Label(
                                                "Signal Type",
                                                className="form-label fw-bold small",
                                            ),
                                            dbc.Select(
                                                id="physio-signal-type",
                                                options=[
                                                    {
                                                        "label": "💓 PPG (Photoplethysmography)",
                                                        "value": "ppg",
                                                    },
                                                    {
                                                        "label": "❤️ ECG (Electrocardiogram)",
                                                        "value": "ecg",
                                                    },
                                                    {
                                                        "label": "🧠 EEG (Electroencephalogram)",
                                                        "value": "eeg",
                                                    },
                                                    {
                                                        "label": "🔍 Auto-detect",
                                                        "value": "auto",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Signal Source Selection
                                            html.Label(
                                                "Signal Source",
                                                className="form-label fw-bold small",
                                            ),
                                            dbc.Select(
                                                id="physio-signal-source-select",
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
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Time Window Control Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "⏰ Time Window",
                                                className="mb-0 text-warning",
                                            ),
                                            html.Small(
                                                "Set analysis time range and navigation",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="bg-warning bg-opacity-10 border-warning",
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Time Inputs
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Time (s)",
                                                                className="form-label fw-bold small",
                                                            ),
                                                            dbc.Input(
                                                                id="physio-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                                className="form-control-sm",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "End Time (s)",
                                                                className="form-label fw-bold small",
                                                            ),
                                                            dbc.Input(
                                                                id="physio-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="10",
                                                                className="form-control-sm",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Quick Navigation Buttons
                                            html.Label(
                                                "Quick Navigation",
                                                className="form-label fw-bold small",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "⏪ -10s",
                                                                id="physio-btn-nudge-m10",
                                                                color="secondary",
                                                                size="sm",
                                                                className="w-100 mb-1",
                                                            )
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "⏪ -1s",
                                                                id="physio-btn-nudge-m1",
                                                                color="secondary",
                                                                size="sm",
                                                                className="w-100 mb-1",
                                                            )
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "+1s ⏩",
                                                                id="physio-btn-nudge-p1",
                                                                color="secondary",
                                                                size="sm",
                                                                className="w-100 mb-1",
                                                            )
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "+10s ⏩",
                                                                id="physio-btn-nudge-p10",
                                                                color="secondary",
                                                                size="sm",
                                                                className="w-100",
                                                            )
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Time Range Slider
                                            html.Label(
                                                "Time Range Slider",
                                                className="form-label fw-bold small",
                                            ),
                                            dcc.RangeSlider(
                                                id="physio-time-range-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 10],
                                                allowCross=False,
                                                pushable=1,
                                                updatemode="mouseup",
                                                className="mb-2",
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Analysis Configuration Accordion
                            dbc.Accordion(
                                [
                                    # Core Analysis Categories
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-analysis-categories",
                                                options=[
                                                    {
                                                        "label": "💓 Heart Rate & Variability (HRV)",
                                                        "value": "hrv",
                                                    },
                                                    {
                                                        "label": "📊 Morphological Features",
                                                        "value": "morphology",
                                                    },
                                                    {
                                                        "label": "🫀 Beat-to-Beat Analysis",
                                                        "value": "beat2beat",
                                                    },
                                                    {
                                                        "label": "⚡ Energy Analysis",
                                                        "value": "energy",
                                                    },
                                                    {
                                                        "label": "📦 Envelope Detection",
                                                        "value": "envelope",
                                                    },
                                                    {
                                                        "label": "✂️ Signal Segmentation",
                                                        "value": "segmentation",
                                                    },
                                                    {
                                                        "label": "📈 Trend Analysis",
                                                        "value": "trend",
                                                    },
                                                    {
                                                        "label": "🌊 Waveform Analysis",
                                                        "value": "waveform",
                                                    },
                                                    {
                                                        "label": "📊 Statistical Analysis",
                                                        "value": "statistical",
                                                    },
                                                    {
                                                        "label": "🔊 Frequency Analysis",
                                                        "value": "frequency",
                                                    },
                                                    {
                                                        "label": "🔄 Signal Transforms",
                                                        "value": "transforms",
                                                    },
                                                ],
                                                value=[
                                                    "hrv",
                                                    "morphology",
                                                    "beat2beat",
                                                    "energy",
                                                    "envelope",
                                                    "segmentation",
                                                    "trend",
                                                    "waveform",
                                                    "statistical",
                                                    "frequency",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="🎯 Core Analysis Categories",
                                        item_id="core-analysis",
                                    ),
                                    # HRV Options
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-hrv-options",
                                                options=[
                                                    {
                                                        "label": "⏱️ Time Domain Features",
                                                        "value": "time_domain",
                                                    },
                                                    {
                                                        "label": "🔊 Frequency Domain Features",
                                                        "value": "freq_domain",
                                                    },
                                                    {
                                                        "label": "🌀 Nonlinear Features",
                                                        "value": "nonlinear",
                                                    },
                                                    {
                                                        "label": "💜 Poincaré Plot",
                                                        "value": "poincare",
                                                    },
                                                    {
                                                        "label": "📏 Detrended Fluctuation",
                                                        "value": "dfa",
                                                    },
                                                ],
                                                value=[
                                                    "time_domain",
                                                    "freq_domain",
                                                    "nonlinear",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="💓 HRV Analysis Options",
                                        item_id="hrv-options",
                                    ),
                                    # Morphology Options
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-morphology-options",
                                                options=[
                                                    {
                                                        "label": "🔝 Peak Detection",
                                                        "value": "peaks",
                                                    },
                                                    {
                                                        "label": "⏱️ Duration Analysis",
                                                        "value": "duration",
                                                    },
                                                    {
                                                        "label": "📐 Area Calculations",
                                                        "value": "area",
                                                    },
                                                    {
                                                        "label": "📊 Amplitude Variability",
                                                        "value": "amplitude",
                                                    },
                                                    {
                                                        "label": "📈 Slope Analysis",
                                                        "value": "slope",
                                                    },
                                                    {
                                                        "label": "🔄 Dicrotic Notch (PPG)",
                                                        "value": "dicrotic",
                                                    },
                                                ],
                                                value=["peaks", "duration", "area"],
                                                className="small",
                                            )
                                        ],
                                        title="📊 Morphology Analysis",
                                        item_id="morphology-options",
                                    ),
                                    # Advanced Features
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-advanced-features",
                                                options=[
                                                    {
                                                        "label": "🔗 Cross-Signal Analysis",
                                                        "value": "cross_signal",
                                                    },
                                                    {
                                                        "label": "👥 Ensemble Methods",
                                                        "value": "ensemble",
                                                    },
                                                    {
                                                        "label": "🔍 Change Detection",
                                                        "value": "change_detection",
                                                    },
                                                    {
                                                        "label": "⚡ Power Analysis",
                                                        "value": "power_analysis",
                                                    },
                                                    {
                                                        "label": "🔗 Coherence Analysis",
                                                        "value": "coherence",
                                                    },
                                                    {
                                                        "label": "🔄 Cross-Correlation",
                                                        "value": "cross_correlation",
                                                    },
                                                ],
                                                value=[
                                                    "cross_signal",
                                                    "ensemble",
                                                    "change_detection",
                                                    "power_analysis",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="🚀 Advanced Features",
                                        item_id="advanced-features",
                                    ),
                                    # Signal Quality
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-quality-options",
                                                options=[
                                                    {
                                                        "label": "⚖️ Signal Quality Index",
                                                        "value": "quality_index",
                                                    },
                                                    {
                                                        "label": "📊 SNR Estimation",
                                                        "value": "snr_estimation",
                                                    },
                                                    {
                                                        "label": "🚫 Artifact Detection",
                                                        "value": "artifact_detection",
                                                    },
                                                    {
                                                        "label": "🔍 Blind Source Separation",
                                                        "value": "blind_source",
                                                    },
                                                    {
                                                        "label": "🔄 Multi-modal Artifact Detection",
                                                        "value": "multimodal_artifacts",
                                                    },
                                                ],
                                                value=[
                                                    "quality_index",
                                                    "artifact_detection",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="⚖️ Signal Quality",
                                        item_id="signal-quality",
                                    ),
                                    # Signal Transforms
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-transform-options",
                                                options=[
                                                    {
                                                        "label": "🌊 Wavelet Transform",
                                                        "value": "wavelet",
                                                    },
                                                    {
                                                        "label": "🔊 Fourier Transform",
                                                        "value": "fourier",
                                                    },
                                                    {
                                                        "label": "🔄 Hilbert Transform",
                                                        "value": "hilbert",
                                                    },
                                                    {
                                                        "label": "⏱️ STFT Analysis",
                                                        "value": "stft",
                                                    },
                                                    {
                                                        "label": "📊 PCA/ICA Decomposition",
                                                        "value": "pca_ica",
                                                    },
                                                    {
                                                        "label": "🎵 MFCC Features",
                                                        "value": "mfcc",
                                                    },
                                                    {
                                                        "label": "🎨 Chroma Features",
                                                        "value": "chroma",
                                                    },
                                                ],
                                                value=["wavelet", "fourier", "hilbert"],
                                                className="small",
                                            )
                                        ],
                                        title="🔄 Signal Transforms",
                                        item_id="signal-transforms",
                                    ),
                                    # Advanced Computation
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-advanced-computation",
                                                options=[
                                                    {
                                                        "label": "🚨 Anomaly Detection",
                                                        "value": "anomaly_detection",
                                                    },
                                                    {
                                                        "label": "📊 Bayesian Analysis",
                                                        "value": "bayesian",
                                                    },
                                                    {
                                                        "label": "🧠 Neural Network Filtering",
                                                        "value": "neural_network",
                                                    },
                                                    {
                                                        "label": "🔍 Kalman Filtering",
                                                        "value": "kalman",
                                                    },
                                                    {
                                                        "label": "🌀 EMD Analysis",
                                                        "value": "emd",
                                                    },
                                                    {
                                                        "label": "🔗 Multimodal Fusion",
                                                        "value": "multimodal_fusion",
                                                    },
                                                    {
                                                        "label": "⚡ Real-time Processing",
                                                        "value": "realtime",
                                                    },
                                                ],
                                                value=[
                                                    "anomaly_detection",
                                                    "bayesian",
                                                    "kalman",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="🧠 Advanced Computation",
                                        item_id="advanced-computation",
                                    ),
                                    # Feature Engineering
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-feature-engineering",
                                                options=[
                                                    {
                                                        "label": "💡 PPG Light Features",
                                                        "value": "ppg_light",
                                                    },
                                                    {
                                                        "label": "🫀 PPG Autonomic Features",
                                                        "value": "ppg_autonomic",
                                                    },
                                                    {
                                                        "label": "❤️ ECG Autonomic Features",
                                                        "value": "ecg_autonomic",
                                                    },
                                                    {
                                                        "label": "📊 Morphology Features",
                                                        "value": "morphology_eng",
                                                    },
                                                    {
                                                        "label": "🔗 ECG-PPG Synchronization",
                                                        "value": "ecg_ppg_sync",
                                                    },
                                                ],
                                                value=[
                                                    "ppg_light",
                                                    "ppg_autonomic",
                                                    "ecg_autonomic",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="🔧 Feature Engineering",
                                        item_id="feature-engineering",
                                    ),
                                    # Preprocessing
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checklist(
                                                id="physio-preprocessing",
                                                options=[
                                                    {
                                                        "label": "🔇 Noise Reduction",
                                                        "value": "noise_reduction",
                                                    },
                                                    {
                                                        "label": "📏 Baseline Correction",
                                                        "value": "baseline_correction",
                                                    },
                                                    {
                                                        "label": "🔍 Filtering",
                                                        "value": "filtering",
                                                    },
                                                    {
                                                        "label": "📊 Normalization",
                                                        "value": "normalization",
                                                    },
                                                    {
                                                        "label": "📈 Interpolation",
                                                        "value": "interpolation",
                                                    },
                                                ],
                                                value=[
                                                    "noise_reduction",
                                                    "baseline_correction",
                                                    "filtering",
                                                ],
                                                className="small",
                                            )
                                        ],
                                        title="🔧 Preprocessing",
                                        item_id="preprocessing",
                                    ),
                                ],
                                start_collapsed=True,
                                className="mb-3",
                            ),
                            # Analysis Summary Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📋 Analysis Summary",
                                                className="mb-0 text-success",
                                            ),
                                            html.Small(
                                                "Quick overview of selected options",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="bg-success bg-opacity-10 border-success",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="physio-analysis-summary",
                                                className="small text-muted",
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        md=3,
                    ),
                    # Right Panel - Plots & Results
                    dbc.Col(
                        [
                            # Main Signal Display
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📈 Signal Overview", className="mb-0"
                                            ),
                                            html.Small(
                                                "Raw signal with annotations and detected features",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="physio-main-signal-plot",
                                                    style={"height": "400px"},
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToRemove": [
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
                            # Feature Analysis Plots
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📊 Analysis Plots", className="mb-0"
                                            ),
                                            html.Small(
                                                "Visual representation of physiological features and analysis",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="physio-analysis-plots",
                                                    style={"height": "800px"},
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToRemove": [
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
                            # Feature Analysis Statistics
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📈 Analysis Statistics",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Comprehensive physiological feature extraction metrics and results",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="physio-analysis-results")]
                                    ),
                                ]
                            ),
                        ],
                        md=9,
                    ),
                ]
            ),
            # Bottom Section - Additional Analysis
            html.Div(id="physio-additional-analysis-section", className="mt-4"),
            # Stores for data management
            dcc.Store(id="store-physio-data"),
            dcc.Store(id="store-physio-features"),
            dcc.Store(id="store-physio-analysis"),
            # Download components for export
            dcc.Download(id="download-physio-csv"),
            dcc.Download(id="download-physio-json"),
        ]
    )


def respiratory_layout():
    """Create the comprehensive respiratory analysis page layout matching legacy design."""
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
                                                className="mb-2",
                                            ),
                                            # Time Window Controls
                                            html.H6("Time Window", className="mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Start Time (s)",
                                                                size="sm",
                                                            ),
                                                            dbc.Input(
                                                                id="resp-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "End Time (s)",
                                                                size="sm",
                                                            ),
                                                            dbc.Input(
                                                                id="resp-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                            # Time Range Slider
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Time Range Slider", size="sm"
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="resp-time-range-slider",
                                                        min=0,
                                                        max=100,
                                                        step=0.1,
                                                        value=[0, 10],
                                                        marks={},
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            # Nudge Buttons
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "-10s",
                                                        id="resp-btn-nudge-m10",
                                                        size="sm",
                                                        color="outline-secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "-1s",
                                                        id="resp-btn-nudge-m1",
                                                        size="sm",
                                                        color="outline-secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+1s",
                                                        id="resp-btn-nudge-p1",
                                                        size="sm",
                                                        color="outline-secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10s",
                                                        id="resp-btn-nudge-p10",
                                                        size="sm",
                                                        color="outline-secondary",
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-2"),
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
                                                className="mb-2",
                                            ),
                                            # Signal Source Selection
                                            html.H6("Signal Source", className="mb-2"),
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
                                                "Filtered signal will be used if available from the filtering screen. Falls back to original signal if no filtering has been performed.",
                                                className="text-muted",
                                            ),
                                            # Respiratory Rate Estimation Methods
                                            html.H6(
                                                "Estimation Methods", className="mb-2"
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
                                                className="mb-2",
                                            ),
                                            # Ensemble Output Options (shown when ensemble is selected)
                                            html.Div(
                                                id="resp-ensemble-options",
                                                style={"display": "none"},
                                                children=[
                                                    html.H6(
                                                        "Ensemble Output Method",
                                                        className="mb-2",
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
                                                        "Mean: Simple average of all methods. Weighted Mean: Weighted average based on method reliability. Bagging: Bootstrap aggregation for robust estimates. Boosting: Sequential learning to improve accuracy.",
                                                        className="text-muted",
                                                    ),
                                                ],
                                            ),
                                            # Advanced Analysis Options
                                            html.H6(
                                                "Advanced Analysis", className="mb-2"
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
                                                className="mb-2",
                                            ),
                                            # Preprocessing Options
                                            html.H6("Preprocessing", className="mb-2"),
                                            dbc.Checklist(
                                                id="resp-preprocessing-options",
                                                options=[
                                                    {
                                                        "label": "Bandpass Filter",
                                                        "value": "filter",
                                                    },
                                                    {
                                                        "label": "Wavelet Denoising",
                                                        "value": "wavelet",
                                                    },
                                                    {
                                                        "label": "Moving Average",
                                                        "value": "smooth",
                                                    },
                                                    {
                                                        "label": "Baseline Correction",
                                                        "value": "detrend",
                                                    },
                                                    {
                                                        "label": "Normalization",
                                                        "value": "normalize",
                                                    },
                                                    {
                                                        "label": "Artifact Removal",
                                                        "value": "artifact_removal",
                                                    },
                                                ],
                                                value=["filter", "detrend"],
                                                className="mb-2",
                                            ),
                                            # Filter Parameters
                                            html.H6(
                                                "Filter Parameters", className="mb-2"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Low Cut (Hz)",
                                                                size="sm",
                                                            ),
                                                            dbc.Input(
                                                                id="resp-low-cut",
                                                                type="number",
                                                                value=0.1,
                                                                min=0.01,
                                                                max=1.0,
                                                                step=0.01,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "High Cut (Hz)",
                                                                size="sm",
                                                            ),
                                                            dbc.Input(
                                                                id="resp-high-cut",
                                                                type="number",
                                                                value=0.8,
                                                                min=0.1,
                                                                max=2.0,
                                                                step=0.01,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                            # Breath Duration Constraints
                                            html.H6(
                                                "Breath Duration Constraints",
                                                className="mb-2",
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
                                                className="mb-2",
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
                            # Main Respiratory Signal Display
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📈 Respiratory Signal Analysis",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Raw signal with breathing pattern detection and annotations",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
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
                                            )
                                        ]
                                    ),
                                ],
                                className="mb-3 shadow-sm",
                            ),
                            # Respiratory Analysis Results
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🔍 Respiratory Analysis Results",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Comprehensive respiratory rate estimation and pattern analysis",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="resp-analysis-results",
                                                className="mb-2",
                                            ),
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="resp-analysis-plots",
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
            # Bottom Section - Additional Analysis
            html.Div(id="resp-additional-analysis-section", className="mt-3"),
            # Stores for data management
            dcc.Store(id="resp-data-store"),
            dcc.Store(id="resp-features-store"),
            dcc.Store(id="store-resp-analysis"),
        ]
    )


def features_layout():
    """Create the advanced features layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1(
                        "🚀 Advanced Feature Engineering", className="text-center mb-4"
                    ),
                    html.P(
                        [
                            "Extract comprehensive signal processing features including statistical, spectral, ",
                            "temporal, and morphological characteristics for advanced signal analysis.",
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
                                                "🎛️ Feature Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure feature extraction parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-3"),
                                            dcc.Dropdown(
                                                id="features-signal-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "ppg",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ecg",
                                                    },
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "General Signal",
                                                        "value": "general",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Signal Source Selection
                                            html.H6("Signal Source", className="mb-3"),
                                            dcc.Dropdown(
                                                id="features-signal-source-select",
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
                                            # Preprocessing Options
                                            html.H6(
                                                "Preprocessing Options",
                                                className="mb-3",
                                            ),
                                            dcc.Checklist(
                                                id="features-preprocessing",
                                                options=[
                                                    {
                                                        "label": "Detrending",
                                                        "value": "detrend",
                                                    },
                                                    {
                                                        "label": "Normalization",
                                                        "value": "normalize",
                                                    },
                                                    {
                                                        "label": "Filtering",
                                                        "value": "filter",
                                                    },
                                                    {
                                                        "label": "Outlier Removal",
                                                        "value": "outlier_removal",
                                                    },
                                                    {
                                                        "label": "Smoothing",
                                                        "value": "smoothing",
                                                    },
                                                    {
                                                        "label": "Baseline Correction",
                                                        "value": "baseline_correction",
                                                    },
                                                    {
                                                        "label": "Noise Reduction",
                                                        "value": "noise_reduction",
                                                    },
                                                    {
                                                        "label": "Artifact Removal",
                                                        "value": "artifact_removal",
                                                    },
                                                ],
                                                value=["detrend", "normalize"],
                                                className="mb-3",
                                            ),
                                            # Feature Categories
                                            html.H6(
                                                "Feature Categories", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="features-categories",
                                                options=[
                                                    {
                                                        "label": "Statistical Features",
                                                        "value": "statistical",
                                                    },
                                                    {
                                                        "label": "Spectral Features",
                                                        "value": "spectral",
                                                    },
                                                    {
                                                        "label": "Temporal Features",
                                                        "value": "temporal",
                                                    },
                                                    {
                                                        "label": "Morphological Features",
                                                        "value": "morphological",
                                                    },
                                                    {
                                                        "label": "Entropy Features",
                                                        "value": "entropy",
                                                    },
                                                    {
                                                        "label": "Fractal Features",
                                                        "value": "fractal",
                                                    },
                                                ],
                                                value=["statistical", "spectral"],
                                                className="mb-3",
                                            ),
                                            # Advanced Options
                                            html.H6(
                                                "Advanced Options", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="features-advanced-options",
                                                options=[
                                                    {
                                                        "label": "Cross-correlation",
                                                        "value": "cross_correlation",
                                                    },
                                                    {
                                                        "label": "Phase Analysis",
                                                        "value": "phase_analysis",
                                                    },
                                                    {
                                                        "label": "Non-linear Features",
                                                        "value": "nonlinear",
                                                    },
                                                    {
                                                        "label": "Wavelet Features",
                                                        "value": "wavelet",
                                                    },
                                                    {
                                                        "label": "Machine Learning Features",
                                                        "value": "ml_features",
                                                    },
                                                ],
                                                value=["cross_correlation"],
                                                className="mb-3",
                                            ),
                                            # Main Analysis Button
                                            dbc.Button(
                                                "🚀 Analyze Features",
                                                id="features-analyze-btn",
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
                            # Analysis Results
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📋 Feature Analysis Results",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="features-analysis-results")]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Analysis Plots
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Feature Analysis Plots",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="features-analysis-plots",
                                                style={"height": "500px"},
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
            # Data Storage
            dcc.Store(id="store-features-data"),
            dcc.Store(id="store-features-features"),
        ]
    )


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
                                            dcc.Dropdown(
                                                id="transforms-signal-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "ppg",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ecg",
                                                    },
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "General Signal",
                                                        "value": "general",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Time Window Configuration
                                            html.H6("Time Window", className="mb-3"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="transforms-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "End Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="transforms-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="10",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Quick Window Navigation
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10s",
                                                        id="transforms-btn-nudge-m10",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -1s",
                                                        id="transforms-btn-nudge-m1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+1s ⏩",
                                                        id="transforms-btn-nudge-p1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10s ⏩",
                                                        id="transforms-btn-nudge-p10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Range Slider for Time Window
                                            html.Label(
                                                "Time Range Slider",
                                                className="form-label",
                                            ),
                                            dcc.RangeSlider(
                                                id="transforms-time-range-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 10],
                                                allowCross=False,
                                                pushable=1,
                                                updatemode="mouseup",
                                                className="mb-4",
                                            ),
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
                                                    {
                                                        "label": "Principal Component Analysis (PCA)",
                                                        "value": "pca",
                                                    },
                                                    {
                                                        "label": "Independent Component Analysis (ICA)",
                                                        "value": "ica",
                                                    },
                                                    {
                                                        "label": "Z-Transform",
                                                        "value": "z_transform",
                                                    },
                                                    {
                                                        "label": "Laplace Transform",
                                                        "value": "laplace",
                                                    },
                                                ],
                                                value="fft",
                                                className="mb-3",
                                            ),
                                            # Transform-Specific Parameters
                                            html.Div(
                                                id="transforms-parameters",
                                                children=[
                                                    # FFT Parameters
                                                    html.Div(
                                                        id="fft-params",
                                                        children=[
                                                            html.H6(
                                                                "FFT Parameters",
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Window Type",
                                                                                className="form-label",
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="fft-window-type",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Rectangular",
                                                                                        "value": "rectangular",
                                                                                    },
                                                                                    {
                                                                                        "label": "Hamming",
                                                                                        "value": "hamming",
                                                                                    },
                                                                                    {
                                                                                        "label": "Hann",
                                                                                        "value": "hann",
                                                                                    },
                                                                                    {
                                                                                        "label": "Blackman",
                                                                                        "value": "blackman",
                                                                                    },
                                                                                    {
                                                                                        "label": "Kaiser",
                                                                                        "value": "kaiser",
                                                                                    },
                                                                                ],
                                                                                value="hann",
                                                                                className="mb-3",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "N Points",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="fft-n-points",
                                                                                type="number",
                                                                                value=1024,
                                                                                min=64,
                                                                                step=64,
                                                                                placeholder="1024",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                    )
                                                ],
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
        ]
    )


def quality_layout():
    """Create the comprehensive signal quality assessment layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1(
                        "🎯 Signal Quality Assessment", className="text-center mb-4"
                    ),
                    html.P(
                        [
                            "Comprehensive signal quality assessment including SNR, artifact detection, ",
                            "baseline wander, and other quality metrics for optimal signal analysis.",
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
                                                "🎛️ Quality Assessment Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure quality assessment parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-3"),
                                            dcc.Dropdown(
                                                id="quality-signal-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "ppg",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ecg",
                                                    },
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "General Signal",
                                                        "value": "general",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Time Window Configuration
                                            html.H6("Time Window", className="mb-3"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="quality-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "End Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="quality-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="10",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Quick Window Navigation
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10s",
                                                        id="quality-btn-nudge-m10",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -1s",
                                                        id="quality-btn-nudge-m1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+1s ⏩",
                                                        id="quality-btn-nudge-p1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10s ⏩",
                                                        id="quality-btn-nudge-p10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Range Slider for Time Window
                                            html.Label(
                                                "Time Range Slider",
                                                className="form-label",
                                            ),
                                            dcc.RangeSlider(
                                                id="quality-time-range-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 10],
                                                allowCross=False,
                                                pushable=1,
                                                updatemode="mouseup",
                                                className="mb-4",
                                            ),
                                            # Quality Metrics Selection
                                            html.H6(
                                                "Quality Metrics", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="quality-metrics",
                                                options=[
                                                    {
                                                        "label": "Signal-to-Noise Ratio (SNR)",
                                                        "value": "snr",
                                                    },
                                                    {
                                                        "label": "Artifact Detection",
                                                        "value": "artifacts",
                                                    },
                                                    {
                                                        "label": "Baseline Wander",
                                                        "value": "baseline_wander",
                                                    },
                                                    {
                                                        "label": "Motion Artifacts",
                                                        "value": "motion_artifacts",
                                                    },
                                                    {
                                                        "label": "Signal Amplitude",
                                                        "value": "amplitude",
                                                    },
                                                    {
                                                        "label": "Signal Stability",
                                                        "value": "stability",
                                                    },
                                                    {
                                                        "label": "Frequency Content",
                                                        "value": "frequency_content",
                                                    },
                                                    {
                                                        "label": "Peak Detection Quality",
                                                        "value": "peak_quality",
                                                    },
                                                    {
                                                        "label": "Signal Continuity",
                                                        "value": "continuity",
                                                    },
                                                    {
                                                        "label": "Outlier Detection",
                                                        "value": "outliers",
                                                    },
                                                ],
                                                value=[
                                                    "snr",
                                                    "artifacts",
                                                    "baseline_wander",
                                                ],
                                                className="mb-3",
                                            ),
                                            # Assessment Parameters
                                            html.H6(
                                                "Assessment Parameters",
                                                className="mb-3",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "SNR Threshold (dB)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="quality-snr-threshold",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=1,
                                                                placeholder="10",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Artifact Threshold",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="quality-artifact-threshold",
                                                                type="number",
                                                                value=0.1,
                                                                min=0,
                                                                step=0.01,
                                                                placeholder="0.1",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Advanced Options
                                            html.H6(
                                                "Advanced Options", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="quality-advanced-options",
                                                options=[
                                                    {
                                                        "label": "Adaptive Thresholding",
                                                        "value": "adaptive_threshold",
                                                    },
                                                    {
                                                        "label": "Multi-scale Analysis",
                                                        "value": "multiscale",
                                                    },
                                                    {
                                                        "label": "Statistical Analysis",
                                                        "value": "statistical",
                                                    },
                                                    {
                                                        "label": "Machine Learning Assessment",
                                                        "value": "ml_assessment",
                                                    },
                                                    {
                                                        "label": "Real-time Monitoring",
                                                        "value": "realtime",
                                                    },
                                                ],
                                                value=[
                                                    "adaptive_threshold",
                                                    "multiscale",
                                                ],
                                                className="mb-4",
                                            ),
                                            # Main Analysis Button
                                            dbc.Button(
                                                "🎯 Assess Signal Quality",
                                                id="quality-analyze-btn",
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
                            # Main Quality Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📈 Signal Quality Overview",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="quality-main-plot",
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
                            # Quality Metrics Visualization
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Quality Metrics Visualization",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="quality-metrics-plot",
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
            # Quality Assessment Results
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📋 Quality Assessment Results",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="quality-assessment-results")]
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
                                        [
                                            html.H5(
                                                "⚠️ Quality Issues & Recommendations",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="quality-issues-recommendations")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Detailed Quality Analysis
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🔍 Detailed Quality Analysis",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="quality-detailed-analysis")]
                                    ),
                                ]
                            )
                        ],
                        md=12,
                    )
                ],
                className="mb-4",
            ),
            # Quality Score Dashboard
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Quality Score Dashboard",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="quality-score-dashboard")]
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
            dcc.Store(id="store-quality-data"),
            dcc.Store(id="store-quality-results"),
        ]
    )


def advanced_layout():
    """Create the comprehensive advanced analysis layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("🧠 Advanced Analysis", className="text-center mb-4"),
                    html.P(
                        [
                            "Advanced signal processing methods including machine learning, deep learning, ",
                            "ensemble methods, and cutting-edge analysis techniques for research applications.",
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
                                                "🎛️ Advanced Analysis Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure advanced analysis parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Signal Type Selection
                                            html.H6("Signal Type", className="mb-3"),
                                            dcc.Dropdown(
                                                id="advanced-signal-type",
                                                options=[
                                                    {
                                                        "label": "Auto-detect",
                                                        "value": "auto",
                                                    },
                                                    {
                                                        "label": "PPG (Photoplethysmography)",
                                                        "value": "ppg",
                                                    },
                                                    {
                                                        "label": "ECG (Electrocardiography)",
                                                        "value": "ecg",
                                                    },
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "General Signal",
                                                        "value": "general",
                                                    },
                                                    {
                                                        "label": "Multi-modal",
                                                        "value": "multimodal",
                                                    },
                                                ],
                                                value="auto",
                                                className="mb-3",
                                            ),
                                            # Time Window Configuration
                                            html.H6("Time Window", className="mb-3"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="advanced-start-time",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="0",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "End Time (s)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="advanced-end-time",
                                                                type="number",
                                                                value=10,
                                                                min=0,
                                                                step=0.1,
                                                                placeholder="10",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Quick Window Navigation
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10s",
                                                        id="advanced-btn-nudge-m10",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -1s",
                                                        id="advanced-btn-nudge-m1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+1s ⏩",
                                                        id="advanced-btn-nudge-p1",
                                                        color="secondary",
                                                        size="sm",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10s ⏩",
                                                        id="advanced-btn-nudge-p10",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Range Slider for Time Window
                                            html.Label(
                                                "Time Range Slider",
                                                className="form-label",
                                            ),
                                            dcc.RangeSlider(
                                                id="advanced-time-range-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 10],
                                                allowCross=False,
                                                pushable=1,
                                                updatemode="mouseup",
                                                className="mb-4",
                                            ),
                                            # Analysis Categories
                                            html.H6(
                                                "Analysis Categories", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="advanced-analysis-categories",
                                                options=[
                                                    {
                                                        "label": "Machine Learning Analysis",
                                                        "value": "ml_analysis",
                                                    },
                                                    {
                                                        "label": "Deep Learning Models",
                                                        "value": "deep_learning",
                                                    },
                                                    {
                                                        "label": "Ensemble Methods",
                                                        "value": "ensemble",
                                                    },
                                                    {
                                                        "label": "Pattern Recognition",
                                                        "value": "pattern_recognition",
                                                    },
                                                    {
                                                        "label": "Anomaly Detection",
                                                        "value": "anomaly_detection",
                                                    },
                                                    {
                                                        "label": "Classification",
                                                        "value": "classification",
                                                    },
                                                    {
                                                        "label": "Regression Analysis",
                                                        "value": "regression",
                                                    },
                                                    {
                                                        "label": "Clustering",
                                                        "value": "clustering",
                                                    },
                                                    {
                                                        "label": "Dimensionality Reduction",
                                                        "value": "dimensionality_reduction",
                                                    },
                                                    {
                                                        "label": "Time Series Forecasting",
                                                        "value": "forecasting",
                                                    },
                                                ],
                                                value=[
                                                    "ml_analysis",
                                                    "pattern_recognition",
                                                ],
                                                className="mb-3",
                                            ),
                                            # Machine Learning Options
                                            html.H6(
                                                "Machine Learning Options",
                                                className="mb-3",
                                            ),
                                            dcc.Checklist(
                                                id="advanced-ml-options",
                                                options=[
                                                    {
                                                        "label": "Support Vector Machines (SVM)",
                                                        "value": "svm",
                                                    },
                                                    {
                                                        "label": "Random Forest",
                                                        "value": "random_forest",
                                                    },
                                                    {
                                                        "label": "Neural Networks",
                                                        "value": "neural_networks",
                                                    },
                                                    {
                                                        "label": "Gradient Boosting",
                                                        "value": "gradient_boosting",
                                                    },
                                                    {
                                                        "label": "K-Means Clustering",
                                                        "value": "kmeans",
                                                    },
                                                    {
                                                        "label": "Principal Component Analysis (PCA)",
                                                        "value": "pca",
                                                    },
                                                    {
                                                        "label": "Independent Component Analysis (ICA)",
                                                        "value": "ica",
                                                    },
                                                    {
                                                        "label": "Hidden Markov Models (HMM)",
                                                        "value": "hmm",
                                                    },
                                                    {
                                                        "label": "Gaussian Mixture Models (GMM)",
                                                        "value": "gmm",
                                                    },
                                                    {
                                                        "label": "Autoencoders",
                                                        "value": "autoencoders",
                                                    },
                                                ],
                                                value=["svm", "random_forest"],
                                                className="mb-3",
                                            ),
                                            # Deep Learning Options
                                            html.H6(
                                                "Deep Learning Options",
                                                className="mb-3",
                                            ),
                                            dcc.Checklist(
                                                id="advanced-deep-learning-options",
                                                options=[
                                                    {
                                                        "label": "Convolutional Neural Networks (CNN)",
                                                        "value": "cnn",
                                                    },
                                                    {
                                                        "label": "Recurrent Neural Networks (RNN)",
                                                        "value": "rnn",
                                                    },
                                                    {
                                                        "label": "Long Short-Term Memory (LSTM)",
                                                        "value": "lstm",
                                                    },
                                                    {
                                                        "label": "Gated Recurrent Units (GRU)",
                                                        "value": "gru",
                                                    },
                                                    {
                                                        "label": "Transformer Models",
                                                        "value": "transformer",
                                                    },
                                                    {
                                                        "label": "Autoencoders",
                                                        "value": "autoencoders",
                                                    },
                                                    {
                                                        "label": "Generative Adversarial Networks (GAN)",
                                                        "value": "gan",
                                                    },
                                                    {
                                                        "label": "Attention Mechanisms",
                                                        "value": "attention",
                                                    },
                                                    {
                                                        "label": "Transfer Learning",
                                                        "value": "transfer_learning",
                                                    },
                                                    {
                                                        "label": "Federated Learning",
                                                        "value": "federated_learning",
                                                    },
                                                ],
                                                value=["cnn", "lstm"],
                                                className="mb-3",
                                            ),
                                            # Advanced Parameters
                                            html.H6(
                                                "Advanced Parameters", className="mb-3"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Cross-validation Folds",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="advanced-cv-folds",
                                                                type="number",
                                                                value=5,
                                                                min=2,
                                                                max=10,
                                                                step=1,
                                                                placeholder="5",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Random State",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="advanced-random-state",
                                                                type="number",
                                                                value=42,
                                                                min=0,
                                                                step=1,
                                                                placeholder="42",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Model Configuration
                                            html.H6(
                                                "Model Configuration", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="advanced-model-config",
                                                options=[
                                                    {
                                                        "label": "Hyperparameter Tuning",
                                                        "value": "hyperparameter_tuning",
                                                    },
                                                    {
                                                        "label": "Feature Selection",
                                                        "value": "feature_selection",
                                                    },
                                                    {
                                                        "label": "Model Interpretability",
                                                        "value": "interpretability",
                                                    },
                                                    {
                                                        "label": "Model Validation",
                                                        "value": "validation",
                                                    },
                                                    {
                                                        "label": "Performance Metrics",
                                                        "value": "performance_metrics",
                                                    },
                                                    {
                                                        "label": "Confusion Matrix",
                                                        "value": "confusion_matrix",
                                                    },
                                                    {
                                                        "label": "ROC Analysis",
                                                        "value": "roc_analysis",
                                                    },
                                                    {
                                                        "label": "Learning Curves",
                                                        "value": "learning_curves",
                                                    },
                                                ],
                                                value=[
                                                    "hyperparameter_tuning",
                                                    "feature_selection",
                                                ],
                                                className="mb-4",
                                            ),
                                            # Main Analysis Button
                                            dbc.Button(
                                                "🧠 Run Advanced Analysis",
                                                id="advanced-analyze-btn",
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
                            # Main Analysis Plot
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📈 Advanced Analysis Results",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="advanced-main-plot",
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
                            # Model Performance Visualization
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Model Performance & Metrics",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="advanced-performance-plot",
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
                                                "📋 Analysis Summary", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-analysis-summary")]
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
                                        [html.H5("🔍 Model Details", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-model-details")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Performance Metrics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📊 Performance Metrics",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-performance-metrics")]
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
                                        [
                                            html.H5(
                                                "🎯 Feature Importance",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="advanced-feature-importance")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Advanced Visualizations
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🌊 Advanced Visualizations",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="advanced-visualizations",
                                                style={"height": "500px"},
                                                config={
                                                    "displayModeBar": True,
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
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
            dcc.Store(id="store-advanced-data"),
            dcc.Store(id="store-advanced-results"),
        ]
    )


def health_report_layout():
    """Create the comprehensive health report generation layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("📋 Health Report Generator", className="text-center mb-4"),
                    html.P(
                        [
                            "Generate comprehensive health reports with customizable templates, ",
                            "professional formatting, and multiple export formats for clinical and research use.",
                        ],
                        className="text-center text-muted mb-5",
                    ),
                ],
                className="mb-4",
            ),
            # Main Configuration Section
            dbc.Row(
                [
                    # Left Panel - Report Configuration
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "⚙️ Report Configuration",
                                                className="mb-0",
                                            ),
                                            html.Small(
                                                "Configure report generation parameters",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Report Type Selection
                                            html.H6("Report Type", className="mb-3"),
                                            dcc.Dropdown(
                                                id="health-report-type",
                                                options=[
                                                    {
                                                        "label": "Comprehensive Health Assessment",
                                                        "value": "comprehensive",
                                                    },
                                                    {
                                                        "label": "Cardiovascular Health Report",
                                                        "value": "cardiovascular",
                                                    },
                                                    {
                                                        "label": "Respiratory Health Report",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "General Wellness Report",
                                                        "value": "wellness",
                                                    },
                                                    {
                                                        "label": "Research Summary Report",
                                                        "value": "research",
                                                    },
                                                    {
                                                        "label": "Clinical Assessment Report",
                                                        "value": "clinical",
                                                    },
                                                    {
                                                        "label": "Fitness & Performance Report",
                                                        "value": "fitness",
                                                    },
                                                    {
                                                        "label": "Custom Report",
                                                        "value": "custom",
                                                    },
                                                ],
                                                value="comprehensive",
                                                className="mb-3",
                                            ),
                                            # Data Selection
                                            html.H6("Data Selection", className="mb-3"),
                                            dcc.Dropdown(
                                                id="health-report-data-selection",
                                                options=[
                                                    {
                                                        "label": "All Available Data",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "Most Recent Session",
                                                        "value": "recent",
                                                    },
                                                    {
                                                        "label": "Specific Time Range",
                                                        "value": "time_range",
                                                    },
                                                    {
                                                        "label": "Selected Analysis Results",
                                                        "value": "selected",
                                                    },
                                                ],
                                                value="recent",
                                                className="mb-3",
                                            ),
                                            # Time Range Configuration (if applicable)
                                            html.Div(
                                                id="health-report-time-config",
                                                children=[
                                                    html.H6(
                                                        "Time Range", className="mb-3"
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Start Date",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.DatePickerSingle(
                                                                        id="health-report-start-date",
                                                                        date=None,
                                                                        display_format="DD/MM/YYYY",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "End Date",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.DatePickerSingle(
                                                                        id="health-report-end-date",
                                                                        date=None,
                                                                        display_format="DD/MM/YYYY",
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            # Report Sections
                                            html.H6(
                                                "Report Sections", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="health-report-sections",
                                                options=[
                                                    {
                                                        "label": "Executive Summary",
                                                        "value": "executive_summary",
                                                    },
                                                    {
                                                        "label": "Data Overview",
                                                        "value": "data_overview",
                                                    },
                                                    {
                                                        "label": "Vital Signs Analysis",
                                                        "value": "vital_signs",
                                                    },
                                                    {
                                                        "label": "Physiological Features",
                                                        "value": "physiological_features",
                                                    },
                                                    {
                                                        "label": "Signal Quality Assessment",
                                                        "value": "signal_quality",
                                                    },
                                                    {
                                                        "label": "Trend Analysis",
                                                        "value": "trend_analysis",
                                                    },
                                                    {
                                                        "label": "Risk Assessment",
                                                        "value": "risk_assessment",
                                                    },
                                                    {
                                                        "label": "Recommendations",
                                                        "value": "recommendations",
                                                    },
                                                    {
                                                        "label": "Technical Details",
                                                        "value": "technical_details",
                                                    },
                                                    {
                                                        "label": "References & Methodology",
                                                        "value": "references",
                                                    },
                                                ],
                                                value=[
                                                    "executive_summary",
                                                    "vital_signs",
                                                    "physiological_features",
                                                    "recommendations",
                                                ],
                                                className="mb-3",
                                            ),
                                            # Report Customization
                                            html.H6(
                                                "Report Customization", className="mb-3"
                                            ),
                                            dcc.Checklist(
                                                id="health-report-customization",
                                                options=[
                                                    {
                                                        "label": "Include Charts & Graphs",
                                                        "value": "include_charts",
                                                    },
                                                    {
                                                        "label": "Include Statistical Tables",
                                                        "value": "include_tables",
                                                    },
                                                    {
                                                        "label": "Include Raw Data Summary",
                                                        "value": "include_raw_data",
                                                    },
                                                    {
                                                        "label": "Include Comparison Analysis",
                                                        "value": "include_comparison",
                                                    },
                                                    {
                                                        "label": "Include Annotations",
                                                        "value": "include_annotations",
                                                    },
                                                    {
                                                        "label": "Include Quality Metrics",
                                                        "value": "include_quality",
                                                    },
                                                    {
                                                        "label": "Include Confidence Intervals",
                                                        "value": "include_confidence",
                                                    },
                                                    {
                                                        "label": "Include Trend Predictions",
                                                        "value": "include_predictions",
                                                    },
                                                ],
                                                value=[
                                                    "include_charts",
                                                    "include_tables",
                                                    "include_quality",
                                                ],
                                                className="mb-3",
                                            ),
                                            # Report Format
                                            html.H6("Report Format", className="mb-3"),
                                            dcc.RadioItems(
                                                id="health-report-format",
                                                options=[
                                                    {
                                                        "label": "PDF Document",
                                                        "value": "pdf",
                                                    },
                                                    {
                                                        "label": "HTML Report",
                                                        "value": "html",
                                                    },
                                                    {
                                                        "label": "Word Document",
                                                        "value": "docx",
                                                    },
                                                    {
                                                        "label": "LaTeX Document",
                                                        "value": "latex",
                                                    },
                                                    {
                                                        "label": "Markdown",
                                                        "value": "markdown",
                                                    },
                                                ],
                                                value="pdf",
                                                className="mb-3",
                                            ),
                                            # Report Style
                                            html.H6("Report Style", className="mb-3"),
                                            dcc.Dropdown(
                                                id="health-report-style",
                                                options=[
                                                    {
                                                        "label": "Professional Medical",
                                                        "value": "medical",
                                                    },
                                                    {
                                                        "label": "Research Paper",
                                                        "value": "research",
                                                    },
                                                    {
                                                        "label": "Executive Summary",
                                                        "value": "executive",
                                                    },
                                                    {
                                                        "label": "Technical Report",
                                                        "value": "technical",
                                                    },
                                                    {
                                                        "label": "Patient-Friendly",
                                                        "value": "patient",
                                                    },
                                                    {
                                                        "label": "Custom Style",
                                                        "value": "custom",
                                                    },
                                                ],
                                                value="medical",
                                                className="mb-4",
                                            ),
                                            # Generate Report Button
                                            dbc.Button(
                                                "📋 Generate Health Report",
                                                id="health-report-generate-btn",
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
                    # Right Panel - Report Preview & Export
                    dbc.Col(
                        [
                            # Report Preview
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("👁️ Report Preview", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="health-report-preview")]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Report Actions
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("📤 Report Actions", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "💾 Save Report",
                                                        id="health-report-save-btn",
                                                        color="success",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "📧 Email Report",
                                                        id="health-report-email-btn",
                                                        color="info",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "🖨️ Print Report",
                                                        id="health-report-print-btn",
                                                        color="secondary",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "📱 Mobile View",
                                                        id="health-report-mobile-btn",
                                                        color="warning",
                                                        className="me-2 mb-2",
                                                    ),
                                                ],
                                                className="text-center",
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
            # Report Content Sections
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("📊 Report Content", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="health-report-content")]
                                    ),
                                ]
                            )
                        ],
                        md=12,
                    )
                ],
                className="mb-4",
            ),
            # Report Templates
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "📋 Report Templates", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="health-report-templates")]
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
                                        [
                                            html.H5(
                                                "⚙️ Template Settings", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="health-report-template-settings")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Report History
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("📚 Report History", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="health-report-history")]
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
            dcc.Store(id="store-health-report-data"),
            dcc.Store(id="store-health-report-config"),
        ]
    )


def settings_layout():
    """Create the comprehensive settings layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("⚙️ Settings", className="text-center mb-4"),
                    html.P(
                        [
                            "Configure application settings, user preferences, analysis parameters, ",
                            "and system configuration for optimal performance and user experience.",
                        ],
                        className="text-center text-muted mb-5",
                    ),
                ],
                className="mb-4",
            ),
            # Settings Navigation Tabs
            dbc.Tabs(
                [
                    # General Settings Tab
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🌐 General Settings", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Application Settings",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Theme",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="settings-theme",
                                                                        options=[
                                                                            {
                                                                                "label": "Light Theme",
                                                                                "value": "light",
                                                                            },
                                                                            {
                                                                                "label": "Dark Theme",
                                                                                "value": "dark",
                                                                            },
                                                                            {
                                                                                "label": "Auto (System)",
                                                                                "value": "auto",
                                                                            },
                                                                        ],
                                                                        value="light",
                                                                        className="mb-3",
                                                                    ),
                                                                    # Debug info
                                                                    html.Div(
                                                                        id="theme-debug",
                                                                        style={"font-size": "12px", "color": "gray", "margin-top": "10px"},
                                                                    ),
                                                                    html.Label(
                                                                        "Time Zone",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="settings-timezone",
                                                                        options=[
                                                                            {
                                                                                "label": "UTC",
                                                                                "value": "UTC",
                                                                            },
                                                                            {
                                                                                "label": "EST (UTC-5)",
                                                                                "value": "EST",
                                                                            },
                                                                            {
                                                                                "label": "PST (UTC-8)",
                                                                                "value": "PST",
                                                                            },
                                                                            {
                                                                                "label": "GMT (UTC+0)",
                                                                                "value": "GMT",
                                                                            },
                                                                            {
                                                                                "label": "CET (UTC+1)",
                                                                                "value": "CET",
                                                                            },
                                                                        ],
                                                                        value="UTC",
                                                                        className="mb-3",
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Display Settings",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Default Page Size",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="settings-page-size",
                                                                        options=[
                                                                            {
                                                                                "label": "10 items",
                                                                                "value": 10,
                                                                            },
                                                                            {
                                                                                "label": "25 items",
                                                                                "value": 25,
                                                                            },
                                                                            {
                                                                                "label": "50 items",
                                                                                "value": 50,
                                                                            },
                                                                            {
                                                                                "label": "100 items",
                                                                                "value": 100,
                                                                            },
                                                                        ],
                                                                        value=25,
                                                                        className="mb-3",
                                                                    ),
                                                                    html.Label(
                                                                        "Auto-refresh Interval (seconds)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-auto-refresh",
                                                                        type="number",
                                                                        value=30,
                                                                        min=0,
                                                                        step=5,
                                                                        placeholder="30",
                                                                    ),
                                                                    html.Div(
                                                                        [
                                                                            dcc.Checklist(
                                                                                id="settings-display-options",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Show Tooltips",
                                                                                        "value": "tooltips",
                                                                                    },
                                                                                    {
                                                                                        "label": "Show Loading Indicators",
                                                                                        "value": "loading",
                                                                                    },
                                                                                    {
                                                                                        "label": "Show Debug Information",
                                                                                        "value": "debug",
                                                                                    },
                                                                                    {
                                                                                        "label": "Compact Mode",
                                                                                        "value": "compact",
                                                                                    },
                                                                                ],
                                                                                value=[
                                                                                    "tooltips",
                                                                                    "loading",
                                                                                ],
                                                                                className="mt-3",
                                                                            )
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        label="General",
                        tab_id="general-settings",
                    ),
                    # Analysis Settings Tab
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "📊 Analysis Settings", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Default Analysis Parameters",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Default Sampling Frequency (Hz)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-default-sampling-freq",
                                                                        type="number",
                                                                        value=1000,
                                                                        min=100,
                                                                        step=100,
                                                                        placeholder="1000",
                                                                    ),
                                                                    html.Label(
                                                                        "Default FFT Points",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-default-fft-points",
                                                                        type="number",
                                                                        value=1024,
                                                                        min=256,
                                                                        step=256,
                                                                        placeholder="1024",
                                                                    ),
                                                                    html.Label(
                                                                        "Default Window Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="settings-default-window",
                                                                        options=[
                                                                            {
                                                                                "label": "Hann",
                                                                                "value": "hann",
                                                                            },
                                                                            {
                                                                                "label": "Hamming",
                                                                                "value": "hamming",
                                                                            },
                                                                            {
                                                                                "label": "Blackman",
                                                                                "value": "blackman",
                                                                            },
                                                                            {
                                                                                "label": "Rectangular",
                                                                                "value": "rectangular",
                                                                            },
                                                                        ],
                                                                        value="hann",
                                                                        className="mb-3",
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Analysis Options",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Peak Detection Threshold",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-peak-threshold",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0.1,
                                                                        max=1.0,
                                                                        step=0.1,
                                                                        placeholder="0.5",
                                                                    ),
                                                                    html.Label(
                                                                        "Quality Assessment Threshold",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-quality-threshold",
                                                                        type="number",
                                                                        value=0.7,
                                                                        min=0.1,
                                                                        max=1.0,
                                                                        step=0.1,
                                                                        placeholder="0.7",
                                                                    ),
                                                                    html.Div(
                                                                        [
                                                                            dcc.Checklist(
                                                                                id="settings-analysis-options",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Auto-detect Signal Type",
                                                                                        "value": "auto_detect",
                                                                                    },
                                                                                    {
                                                                                        "label": "Enable Advanced Features",
                                                                                        "value": "advanced_features",
                                                                                    },
                                                                                    {
                                                                                        "label": "Enable Real-time Processing",
                                                                                        "value": "realtime",
                                                                                    },
                                                                                    {
                                                                                        "label": "Enable Batch Processing",
                                                                                        "value": "batch_processing",
                                                                                    },
                                                                                ],
                                                                                value=[
                                                                                    "auto_detect",
                                                                                    "advanced_features",
                                                                                ],
                                                                                className="mt-3",
                                                                            )
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        label="Analysis",
                        tab_id="analysis-settings",
                    ),
                    # Data Settings Tab
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H4("💾 Data Settings", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Data Management",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Maximum File Size (MB)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-max-file-size",
                                                                        type="number",
                                                                        value=100,
                                                                        min=10,
                                                                        max=1000,
                                                                        step=10,
                                                                        placeholder="100",
                                                                    ),
                                                                    html.Label(
                                                                        "Auto-save Interval (minutes)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-auto-save",
                                                                        type="number",
                                                                        value=5,
                                                                        min=1,
                                                                        max=60,
                                                                        step=1,
                                                                        placeholder="5",
                                                                    ),
                                                                    html.Label(
                                                                        "Data Retention Period (days)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-data-retention",
                                                                        type="number",
                                                                        value=30,
                                                                        min=1,
                                                                        max=365,
                                                                        step=1,
                                                                        placeholder="30",
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Export Settings",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Default Export Format",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="settings-export-format",
                                                                        options=[
                                                                            {
                                                                                "label": "CSV",
                                                                                "value": "csv",
                                                                            },
                                                                            {
                                                                                "label": "Excel",
                                                                                "value": "excel",
                                                                            },
                                                                            {
                                                                                "label": "JSON",
                                                                                "value": "json",
                                                                            },
                                                                            {
                                                                                "label": "MATLAB",
                                                                                "value": "matlab",
                                                                            },
                                                                            {
                                                                                "label": "Python Pickle",
                                                                                "value": "pickle",
                                                                            },
                                                                        ],
                                                                        value="csv",
                                                                        className="mb-3",
                                                                    ),
                                                                    html.Label(
                                                                        "Default Image Format",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="settings-image-format",
                                                                        options=[
                                                                            {
                                                                                "label": "PNG",
                                                                                "value": "png",
                                                                            },
                                                                            {
                                                                                "label": "JPEG",
                                                                                "value": "jpeg",
                                                                            },
                                                                            {
                                                                                "label": "SVG",
                                                                                "value": "svg",
                                                                            },
                                                                            {
                                                                                "label": "PDF",
                                                                                "value": "pdf",
                                                                            },
                                                                        ],
                                                                        value="png",
                                                                        className="mb-3",
                                                                    ),
                                                                    html.Div(
                                                                        [
                                                                            dcc.Checklist(
                                                                                id="settings-export-options",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Include Metadata",
                                                                                        "value": "metadata",
                                                                                    },
                                                                                    {
                                                                                        "label": "High Quality Images",
                                                                                        "value": "high_quality",
                                                                                    },
                                                                                    {
                                                                                        "label": "Compress Exports",
                                                                                        "value": "compress",
                                                                                    },
                                                                                    {
                                                                                        "label": "Auto-export Results",
                                                                                        "value": "auto_export",
                                                                                    },
                                                                                ],
                                                                                value=[
                                                                                    "metadata",
                                                                                    "high_quality",
                                                                                ],
                                                                                className="mt-3",
                                                                            )
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        label="Data",
                        tab_id="data-settings",
                    ),
                    # System Settings Tab
                    dbc.Tab(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H4(
                                                "🔧 System Settings", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Performance Settings",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Maximum CPU Usage (%)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-cpu-usage",
                                                                        type="number",
                                                                        value=80,
                                                                        min=10,
                                                                        max=100,
                                                                        step=10,
                                                                        placeholder="80",
                                                                    ),
                                                                    html.Label(
                                                                        "Memory Limit (GB)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-memory-limit",
                                                                        type="number",
                                                                        value=4,
                                                                        min=1,
                                                                        max=32,
                                                                        step=1,
                                                                        placeholder="4",
                                                                    ),
                                                                    html.Label(
                                                                        "Parallel Processing Threads",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-parallel-threads",
                                                                        type="number",
                                                                        value=4,
                                                                        min=1,
                                                                        max=16,
                                                                        step=1,
                                                                        placeholder="4",
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Security & Privacy",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Session Timeout (minutes)",
                                                                        className="form-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="settings-session-timeout",
                                                                        type="number",
                                                                        value=60,
                                                                        min=15,
                                                                        max=480,
                                                                        step=15,
                                                                        placeholder="60",
                                                                    ),
                                                                    html.Div(
                                                                        [
                                                                            dcc.Checklist(
                                                                                id="settings-security-options",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Enable HTTPS",
                                                                                        "value": "https",
                                                                                    },
                                                                                    {
                                                                                        "label": "Data Encryption",
                                                                                        "value": "encryption",
                                                                                    },
                                                                                    {
                                                                                        "label": "Audit Logging",
                                                                                        "value": "audit_log",
                                                                                    },
                                                                                    {
                                                                                        "label": "Two-Factor Authentication",
                                                                                        "value": "2fa",
                                                                                    },
                                                                                ],
                                                                                value=[
                                                                                    "https",
                                                                                    "encryption",
                                                                                ],
                                                                                className="mt-3",
                                                                            )
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        label="System",
                        tab_id="system-settings",
                    ),
                ],
                id="settings-tabs",
                className="mb-4",
            ),
            # Settings Actions and Status
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "💾 Settings Actions", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "💾 Save Settings",
                                                        id="settings-save-btn",
                                                        color="success",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "🔄 Reset to Defaults",
                                                        id="settings-reset-btn",
                                                        color="warning",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "📤 Export Settings",
                                                        id="settings-export-btn",
                                                        color="info",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "📥 Import Settings",
                                                        id="settings-import-btn",
                                                        color="secondary",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "✅ Validate Settings",
                                                        id="settings-validate-btn",
                                                        color="primary",
                                                        className="me-2 mb-2",
                                                    ),
                                                    dbc.Button(
                                                        "💡 Get Recommendations",
                                                        id="settings-recommendations-btn",
                                                        color="info",
                                                        className="me-2 mb-2",
                                                    ),
                                                ],
                                                className="text-center",
                                            )
                                        ]
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
                                        [
                                            html.H5(
                                                "📊 Settings Status", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody([html.Div(id="settings-status")]),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Additional Settings Features
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [html.H5("🎨 Theme Preview", className="mb-0")]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="theme-preview-display")]
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
                                        [
                                            html.H5(
                                                "✅ Settings Validation",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [html.Div(id="settings-validation-display")]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # System Monitoring and Recommendations
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5(
                                                "🖥️ System Monitoring", className="mb-0"
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Interval(
                                                id="system-monitor-interval",
                                                interval=5000,
                                                n_intervals=0,
                                            ),
                                            html.Div(id="system-monitor-display"),
                                        ]
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
                                        [
                                            html.H5(
                                                "💡 Smart Recommendations",
                                                className="mb-0",
                                            )
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="settings-recommendations-display"
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ],
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            # Data Storage
            dcc.Store(id="store-settings-data"),
            dcc.Store(id="store-settings-config"),
        ]
    )
