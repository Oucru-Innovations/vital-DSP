"""
Frequency Domain Analysis page layout for vitalDSP webapp.

This module provides the layout for the Frequency Domain Analysis page,
which allows users to analyze PPG/ECG signals using FFT, PSD, STFT, and
wavelet transforms with comprehensive frequency domain metrics.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc




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
                                                            html.Label(
                                                                "Start Position (%)",
                                                                className="form-label fw-bold",
                                                            ),
                                                            html.Small(
                                                                "Position in data (0% = start, 100% = end)",
                                                                className="text-muted",
                                                            ),
                                                            dcc.Slider(
                                                                id="freq-start-position-slider",
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
                                                        md=4,
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
                                                                id="freq-duration-select",
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
                                                        md=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Quick Navigation",
                                                                className="form-label fw-bold",
                                                            ),
                                                            html.Small(
                                                                "Adjust start position",
                                                                className="text-muted",
                                                            ),
                                                            dbc.ButtonGroup(
                                                                [
                                                                    dbc.Button(
                                                                        "⏪ -10%",
                                                                        id="freq-btn-nudge-m10",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="me-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "⏪ -5%",
                                                                        id="freq-btn-nudge-m1",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="me-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "+5% ⏩",
                                                                        id="freq-btn-nudge-p1",
                                                                        color="secondary",
                                                                        size="sm",
                                                                        className="me-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "+10% ⏩",
                                                                        id="freq-btn-nudge-p10",
                                                                        color="secondary",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                className="mb-3",
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
            # Hidden components for compatibility with filtering/time-domain pages
            html.Div(id="filter-btn-apply", style={"display": "none"}),
            html.Button(id="btn-apply-filter", style={"display": "none"}),
            dcc.Dropdown(id="filter-type-select", style={"display": "none"}),
            dcc.Dropdown(id="filter-family-advanced", style={"display": "none"}),
            dcc.Dropdown(id="filter-response-advanced", style={"display": "none"}),
            dbc.Input(id="filter-type", style={"display": "none"}),
            dbc.Input(id="cutoff-freq", style={"display": "none"}),
            dbc.Input(id="filter-order", style={"display": "none"}),
            dbc.Input(id="filter-low-freq-advanced", style={"display": "none"}),
            dbc.Input(id="filter-high-freq-advanced", style={"display": "none"}),
            dbc.Input(id="filter-order-advanced", style={"display": "none"}),
            dcc.Dropdown(id="advanced-filter-method", style={"display": "none"}),
            dbc.Input(id="advanced-noise-level", style={"display": "none"}),
            dbc.Input(id="advanced-iterations", style={"display": "none"}),
            dbc.Input(id="advanced-learning-rate", style={"display": "none"}),
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dbc.Input(id="artifact-removal-strength", style={"display": "none"}),
            dcc.Dropdown(id="neural-network-type", style={"display": "none"}),
            dbc.Input(id="neural-model-complexity", style={"display": "none"}),
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            dbc.Input(id="ensemble-n-filters", style={"display": "none"}),
            dcc.Checklist(id="filter-quality-options", style={"display": "none"}),
            dcc.Checklist(id="detrend-option", style={"display": "none"}),
            dcc.Dropdown(id="filter-signal-type-select", style={"display": "none"}),
            dbc.Input(id="savgol-window", style={"display": "none"}),
            dbc.Input(id="savgol-polyorder", style={"display": "none"}),
            dbc.Input(id="moving-avg-window", style={"display": "none"}),
            dbc.Input(id="gaussian-sigma", style={"display": "none"}),
            dbc.Input(id="wavelet-level", style={"display": "none"}),
            dcc.Dropdown(id="threshold-type", style={"display": "none"}),
            dbc.Input(id="threshold-value", style={"display": "none"}),
            dcc.Dropdown(id="reference-signal", style={"display": "none"}),
            dcc.Dropdown(id="fusion-method", style={"display": "none"}),
            dcc.Store(id="store-filtered-signal"),
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
