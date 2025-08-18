"""
VitalDSP layout components for different analysis pages.
Provides comprehensive layouts for time domain, frequency, filtering, and other analyses.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def time_domain_layout():
    """Create the time domain analysis page layout."""
    return html.Div([
        # Page Header
        html.Div([
            html.H1("⏱️ Time Domain Analysis", className="text-center mb-4"),
            html.P([
                "Analyze your PPG/ECG signals in the time domain with interactive plots, ",
                "filtering, and comprehensive analysis tools."
            ], className="text-center text-muted mb-5")
        ], className="mb-4"),
        
        # Main Analysis Section
        dbc.Row([
            # Left Panel - Controls & Parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🎛️ Analysis Controls", className="mb-0"),
                        html.Small("Configure analysis parameters and filters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Data Selection
                        html.H6("Data Selection", className="mb-3"),
                        dbc.Select(
                            id="data-source-select",
                            options=[
                                {"label": "Uploaded Data", "value": "uploaded"},
                                {"label": "Sample Data", "value": "sample"}
                            ],
                            value="uploaded",
                            className="mb-3"
                        ),
                        
                        # Time Window Controls
                        html.H6("Time Window", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Start Time (s)", className="form-label"),
                                dbc.Input(
                                    id="start-time",
                                    type="number",
                                    value=0,
                                    min=0,
                                    step=0.1,
                                    placeholder="0"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("End Time (s)", className="form-label"),
                                dbc.Input(
                                    id="end-time",
                                    type="number",
                                    value=10,
                                    min=0,
                                    step=0.1,
                                    placeholder="10"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Quick Window Navigation
                        html.Div([
                            dbc.Button("⏪ -10s", id="btn-nudge-m10", color="secondary", size="sm", className="me-1"),
                            dbc.Button("⏪ -1s", id="btn-nudge-m1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+1s ⏩", id="btn-nudge-p1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+10s ⏩", id="btn-nudge-p10", color="secondary", size="sm")
                        ], className="mb-3"),
                        
                        # Range Slider for Time Window
                        html.Label("Time Range Slider", className="form-label"),
                        dcc.RangeSlider(
                            id="time-range-slider",
                            min=0,
                            max=100,
                            step=0.1,
                            value=[0, 10],
                            allowCross=False,
                            pushable=1,
                            updatemode="mouseup",
                            className="mb-4"
                        ),
                        
                        # Filtering Controls
                        html.H6("Signal Filtering", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Filter Type", className="form-label"),
                                dcc.Dropdown(
                                    id="filter-family",
                                    options=[
                                        {"label": "Butterworth", "value": "butter"},
                                        {"label": "Chebyshev I", "value": "cheby1"},
                                        {"label": "Chebyshev II", "value": "cheby2"},
                                        {"label": "Elliptic", "value": "ellip"},
                                        {"label": "Bessel", "value": "bessel"}
                                    ],
                                    value="butter",
                                    clearable=False
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Response", className="form-label"),
                                dcc.Dropdown(
                                    id="filter-response",
                                    options=[
                                        {"label": "Bandpass", "value": "bandpass"},
                                        {"label": "Bandstop (Notch)", "value": "bandstop"},
                                        {"label": "Lowpass", "value": "lowpass"},
                                        {"label": "Highpass", "value": "highpass"}
                                    ],
                                    value="bandpass",
                                    clearable=False
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Low Freq (Hz)", className="form-label"),
                                dbc.Input(
                                    id="filter-low-freq",
                                    type="number",
                                    value=0.5,
                                    min=0,
                                    step=0.1,
                                    placeholder="0.5"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("High Freq (Hz)", className="form-label"),
                                dbc.Input(
                                    id="filter-high-freq",
                                    type="number",
                                    value=40,
                                    min=0,
                                    step=0.1,
                                    placeholder="40"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Order", className="form-label"),
                                dcc.Slider(
                                    id="filter-order",
                                    min=2,
                                    max=10,
                                    step=1,
                                    value=4,
                                    marks={i: str(i) for i in [2, 4, 6, 8, 10]},
                                    className="mb-3"
                                )
                            ], md=12)
                        ]),
                        
                        # Analysis Options
                        html.H6("Analysis Options", className="mb-3"),
                        dbc.Checklist(
                            id="analysis-options",
                            options=[
                                {"label": "Peak Detection", "value": "peaks"},
                                {"label": "Heart Rate Calculation", "value": "hr"},
                                {"label": "Signal Quality Assessment", "value": "quality"},
                                {"label": "Artifact Detection", "value": "artifacts"},
                                {"label": "Trend Analysis", "value": "trend"}
                            ],
                            value=["peaks", "hr", "quality"],
                            className="mb-3"
                        ),
                        
                        # Action Buttons
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🔄 Update Analysis",
                                    id="btn-update-analysis",
                                    color="primary",
                                    className="w-100"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Button(
                                    "📊 Export Results",
                                    id="btn-export-results",
                                    color="success",
                                    outline=True,
                                    className="w-100"
                                )
                            ], md=6)
                        ])
                    ])
                ], className="h-100")
            ], md=3),
            
            # Right Panel - Plots & Results
            dbc.Col([
                # Main Signal Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📈 Raw Signal", className="mb-0"),
                        html.Small("Time domain representation of your signal", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="main-signal-plot",
                                style={"height": "400px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Filtered Signal Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔧 Filtered Signal", className="mb-0"),
                        html.Small("Signal after applying selected filters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="filtered-signal-plot",
                                style={"height": "400px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Analysis Results - Reorganized for Better Display
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📊 Analysis Results", className="mb-0"),
                        html.Small("Key metrics and insights from your signal", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Main analysis results
                        html.Div(id="analysis-results", className="mb-4"),
                        
                        # Peak Analysis Table
                        html.Div(id="peak-analysis-table", className="mb-4"),
                        
                        # Signal Quality Table
                        html.Div(id="signal-quality-table", className="mb-4"),
                        
                        # Filtering Results Table
                        html.Div(id="filtering-results-table", className="mb-4"),
                        
                        # Additional Metrics Table
                        html.Div(id="additional-metrics-table", className="mb-4")
                    ])
                ])
            ], md=9)
        ]),
        
        # Bottom Section - Additional Analysis
        html.Div(id="additional-analysis-section", className="mt-4"),
        
        # Stores for data management
        dcc.Store(id="store-time-domain-data"),
        dcc.Store(id="store-filtered-data"),
        dcc.Store(id="store-analysis-results")
    ])


def frequency_layout():
    """Create the frequency domain analysis page layout."""
    return html.Div([
        # Page Header
        html.Div([
        html.H1("🌊 Frequency Domain Analysis", className="text-center mb-4"),
            html.P([
                "Analyze your PPG/ECG signals in the frequency domain with FFT, STFT, wavelet transforms, ",
                "and comprehensive spectral analysis tools."
            ], className="text-center text-muted mb-5")
        ], className="mb-4"),
        
        # Main Analysis Section
        dbc.Row([
            # Left Panel - Controls & Parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🎛️ Analysis Controls", className="mb-0"),
                        html.Small("Configure frequency domain analysis parameters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Data Selection
                        html.H6("Data Selection", className="mb-3"),
                        dbc.Select(
                            id="freq-data-source-select",
                            options=[
                                {"label": "Uploaded Data", "value": "uploaded"},
                                {"label": "Sample Data", "value": "sample"}
                            ],
                            value="uploaded",
                            className="mb-3"
                        ),
                        
                        # Time Window Controls
                        html.H6("Time Window", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Start Time (s)", className="form-label"),
                                dbc.Input(
                                    id="freq-start-time",
                                    type="number",
                                    value=0,
                                    min=0,
                                    step=0.1,
                                    placeholder="0"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("End Time (s)", className="form-label"),
                                dbc.Input(
                                    id="freq-end-time",
                                    type="number",
                                    value=10,
                                    min=0,
                                    step=0.1,
                                    placeholder="10"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Quick Window Navigation
                        html.Div([
                            dbc.Button("⏪ -10s", id="freq-btn-nudge-m10", color="secondary", size="sm", className="me-1"),
                            dbc.Button("⏪ -1s", id="freq-btn-nudge-m1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+1s ⏩", id="freq-btn-nudge-p1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+10s ⏩", id="freq-btn-nudge-p10", color="secondary", size="sm")
                        ], className="mb-3"),
                        
                        # Range Slider for Time Window
                        html.Label("Time Range Slider", className="form-label"),
                        dcc.RangeSlider(
                            id="freq-time-range-slider",
                            min=0,
                            max=100,
                            step=0.1,
                            value=[0, 10],
                            allowCross=False,
                            pushable=1,
                            updatemode="mouseup",
                            className="mb-4"
                        ),
                        
                        # Frequency Analysis Type
                        html.H6("Analysis Type", className="mb-3"),
                        dcc.Dropdown(
                            id="freq-analysis-type",
                            options=[
                                {"label": "FFT (Fast Fourier Transform)", "value": "fft"},
                                {"label": "STFT (Short-Time Fourier Transform)", "value": "stft"},
                                {"label": "Wavelet Transform", "value": "wavelet"},
                                {"label": "Power Spectral Density", "value": "psd"},
                                {"label": "Spectrogram", "value": "spectrogram"}
                            ],
                            value="fft",
                            clearable=False,
                            className="mb-3"
                        ),
                        
                        # FFT Parameters
                        html.Div(id="fft-params", children=[
                            html.H6("FFT Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Window Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="fft-window-type",
                                        options=[
                                            {"label": "Hamming", "value": "hamming"},
                                            {"label": "Hanning", "value": "hanning"},
                                            {"label": "Blackman", "value": "blackman"},
                                            {"label": "Rectangular", "value": "rectangular"},
                                            {"label": "Kaiser", "value": "kaiser"}
                                        ],
                                        value="hamming",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("N FFT Points", className="form-label"),
                                    dcc.Dropdown(
                                        id="fft-n-points",
                                        options=[
                                            {"label": "256", "value": 256},
                                            {"label": "512", "value": 512},
                                            {"label": "1024", "value": 1024},
                                            {"label": "2048", "value": 2048},
                                            {"label": "4096", "value": 4096}
                                        ],
                                        value=1024,
                                        clearable=False
                                    )
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # PSD Parameters
                        html.Div(id="psd-params", children=[
                            html.H6("Power Spectral Density Parameters", className="mb-3"),
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
                                        placeholder="2.0",
                                        style={"width": "100%"}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("PSD Overlap (%)", className="form-label"),
                                    dcc.Slider(
                                        id="psd-overlap",
                                        min=0,
                                        max=95,
                                        step=5,
                                        value=50,
                                        marks={i: str(i) for i in [0, 25, 50, 75, 95]},
                                        className="mb-3"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Max Frequency (Hz)", className="form-label"),
                                    dcc.Input(
                                        id="psd-freq-max",
                                        type="number",
                                        value=25.0,
                                        min=1.0,
                                        max=100.0,
                                        step=1.0,
                                        placeholder="25.0",
                                        style={"width": "100%"}
                                    )
                                ], md=4)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Log Scale", className="form-label"),
                                    dcc.Checklist(
                                        id="psd-log-scale",
                                        options=[{"label": "dB Scale", "value": "on"}],
                                        value=["on"],
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
                                ], md=4),
                                dbc.Col([
                                    html.Label("Channel", className="form-label"),
                                    dcc.Dropdown(
                                        id="psd-channel",
                                        options=[
                                            {"label": "Signal", "value": "signal"},
                                            {"label": "RED", "value": "red"},
                                            {"label": "IR", "value": "ir"},
                                            {"label": "Waveform", "value": "waveform"}
                                        ],
                                        value="signal",
                                        clearable=False,
                                        style={"width": "100%"}
                                    )
                                ], md=4)
                            ], className="mb-3")
                        ], className="mb-4"),
                        
                        # STFT Parameters
                        html.Div(id="stft-params", children=[
                            html.H6("STFT Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Window Size", className="form-label"),
                                    dcc.Input(
                                        id="stft-window-size",
                                        type="number",
                                        value=256,
                                        min=32,
                                        max=2048,
                                        step=32,
                                        placeholder="256",
                                        style={"width": "100%"}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Hop Size", className="form-label"),
                                    dcc.Input(
                                        id="stft-hop-size",
                                        type="number",
                                        value=128,
                                        min=16,
                                        max=1024,
                                        step=16,
                                        placeholder="128",
                                        style={"width": "100%"}
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Window Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="stft-window-type",
                                        options=[
                                            {"label": "Hann", "value": "hann"},
                                            {"label": "Hamming", "value": "hamming"},
                                            {"label": "Blackman", "value": "blackman"},
                                            {"label": "Kaiser", "value": "kaiser"},
                                            {"label": "Gaussian", "value": "gaussian"},
                                            {"label": "Rectangular", "value": "rectangular"}
                                        ],
                                        value="hann",
                                        clearable=False,
                                        style={"width": "100%"}
                                    )
                                ], md=4)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Overlap (%)", className="form-label"),
                                    dcc.Slider(
                                        id="stft-overlap",
                                        min=0,
                                        max=95,
                                        step=5,
                                        value=50,
                                        marks={i: str(i) for i in [0, 25, 50, 75, 95]},
                                        className="mb-3"
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Scaling", className="form-label"),
                                    dcc.Dropdown(
                                        id="stft-scaling",
                                        options=[
                                            {"label": "Density", "value": "density"},
                                            {"label": "Spectrum", "value": "spectrum"}
                                        ],
                                        value="density",
                                        clearable=False,
                                        style={"width": "100%"}
                                    )
                                ], md=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Frequency Range (Hz)", className="form-label"),
                                    dcc.Input(
                                        id="stft-freq-max",
                                        type="number",
                                        value=50,
                                        min=1,
                                        max=200,
                                        step=1,
                                        placeholder="50",
                                        style={"width": "100%"}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Colormap", className="form-label"),
                                    dcc.Dropdown(
                                        id="stft-colormap",
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
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # Wavelet Parameters
                        html.Div(id="wavelet-params", children=[
                            html.H6("Wavelet Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Wavelet Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="wavelet-type",
                                        options=[
                                            {"label": "Haar", "value": "haar"},
                                            {"label": "Daubechies 4", "value": "db4"},
                                            {"label": "Daubechies 8", "value": "db8"},
                                            {"label": "Symlets 4", "value": "sym4"},
                                            {"label": "Coiflets 4", "value": "coif4"}
                                        ],
                                        value="haar",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Decomposition Levels", className="form-label"),
                                    dcc.Slider(
                                        id="wavelet-levels",
                                        min=1,
                                        max=8,
                                        step=1,
                                        value=4,
                                        marks={i: str(i) for i in [1, 2, 4, 6, 8]},
                                        className="mb-3"
                                    )
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # Frequency Range Controls
                        html.H6("Frequency Range", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Min Freq (Hz)", className="form-label"),
                                dbc.Input(
                                    id="freq-min",
                                    type="number",
                                    value=0,
                                    min=0,
                                    step=0.1,
                                    placeholder="0"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Max Freq (Hz)", className="form-label"),
                                dbc.Input(
                                    id="freq-max",
                                    type="number",
                                    value=50,
                                    min=0,
                                    step=0.1,
                                    placeholder="50"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Analysis Options
                        html.H6("Analysis Options", className="mb-3"),
                        dbc.Checklist(
                            id="freq-analysis-options",
                            options=[
                                {"label": "Peak Frequency Detection", "value": "peak_freq"},
                                {"label": "Dominant Frequency", "value": "dominant_freq"},
                                {"label": "Band Power Analysis", "value": "band_power"},
                                {"label": "Frequency Stability", "value": "freq_stability"},
                                {"label": "Harmonic Analysis", "value": "harmonics"}
                            ],
                            value=["peak_freq", "dominant_freq", "band_power"],
                            className="mb-3"
                        ),
                        
                        # Action Buttons
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🔄 Update Analysis",
                                    id="freq-btn-update-analysis",
                                    color="primary",
                                    className="w-100"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Button(
                                    "📊 Export Results",
                                    id="freq-btn-export-results",
                                    color="success",
                                    outline=True,
                                    className="w-100"
                                )
                            ], md=6)
                        ])
                    ])
                ], className="h-100")
            ], md=3),
            
            # Right Panel - Plots & Results
            dbc.Col([
                # Main Frequency Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📊 Frequency Spectrum", className="mb-0"),
                        html.Small("Frequency domain representation of your signal", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="freq-main-plot",
                                style={"height": "650px", "minHeight": "600px", "overflow": "visible"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False,
                                    "responsive": True
                                }
                            ),
                            type="default"
                        )
                    ], style={"overflow": "visible"})
                ], className="mb-4"),
                
                # Power Spectral Density Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📊 Power Spectral Density", className="mb-0"),
                        html.Small("Enhanced PSD analysis with configurable parameters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="freq-psd-plot",
                                style={"height": "400px", "minHeight": "350px", "overflow": "visible"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False,
                                    "responsive": True
                                }
                            ),
                            type="default"
                        )
                    ], style={"overflow": "visible"})
                ], className="mb-4"),
                
                # Spectrogram Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🎵 Spectrogram Analysis", className="mb-0"),
                        html.Small("Time-frequency representation with advanced controls", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="freq-spectrogram-plot",
                                style={"height": "400px", "minHeight": "350px", "overflow": "visible"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False,
                                    "responsive": True
                                }
                            ),
                            type="default"
                        )
                    ], style={"overflow": "visible"})
                ], className="mb-4"),
                
                # Analysis Results - Reorganized for Better Display
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📈 Frequency Analysis Results", className="mb-0"),
                        html.Small("Key frequency metrics and insights", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Main analysis results
                        html.Div(id="freq-analysis-results", className="mb-4"),
                        
                        # Peak Frequency Analysis Table
                        html.Div(id="freq-peak-analysis-table", className="mb-4"),
                        
                        # Band Power Analysis Table
                        html.Div(id="freq-band-power-table", className="mb-4"),
                        
                        # Frequency Stability Table
                        html.Div(id="freq-stability-table", className="mb-4"),
                        
                        # Harmonic Analysis Table
                        html.Div(id="freq-harmonics-table", className="mb-4")
                    ])
                ])
            ], md=9)
        ]),
        
        # Bottom Section - Additional Analysis
        html.Div(id="freq-additional-analysis-section", className="mt-4"),
        
        # Stores for data management
        dcc.Store(id="store-frequency-data"),
        dcc.Store(id="store-time-freq-data"),
        dcc.Store(id="store-freq-analysis-results")
    ])


def filtering_layout():
    """Create the signal filtering page layout."""
    return html.Div([
        # Page Header
        html.Div([
            html.H1("🔧 Advanced Signal Filtering", className="text-center mb-4"),
            html.P([
                "Apply advanced filtering techniques including traditional filters, Kalman filters, ",
                "neural network filtering, and artifact removal for optimal signal quality."
            ], className="text-center text-muted mb-5")
        ], className="mb-4"),
        
        # Main Analysis Section
        html.Div(className="plot-container-wrapper", style={"overflow": "visible", "padding": "20px"}),
        dbc.Row([
            # Left Panel - Controls & Parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🎛️ Filtering Controls", className="mb-0"),
                        html.Small("Configure advanced filtering parameters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Data Selection
                        html.H6("Data Selection", className="mb-3"),
                        dbc.Select(
                            id="filter-data-source-select",
                            options=[
                                {"label": "Uploaded Data", "value": "uploaded"},
                                {"label": "Sample Data", "value": "sample"}
                            ],
                            value="uploaded",
                            className="mb-3"
                        ),
                        
                        # Time Window Controls
                        html.H6("Time Window", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Start Time (s)", className="form-label"),
                                dbc.Input(
                                    id="filter-start-time",
                                    type="number",
                                    value=0,
                                    min=0,
                                    step=0.1,
                                    placeholder="0"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("End Time (s)", className="form-label"),
                                dbc.Input(
                                    id="filter-end-time",
                                    type="number",
                                    value=10,
                                    min=0,
                                    step=0.1,
                                    placeholder="10"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Quick Window Navigation
                        html.Div([
                            dbc.Button("⏪ -10s", id="filter-btn-nudge-m10", color="secondary", size="sm", className="me-1"),
                            dbc.Button("⏪ -1s", id="filter-btn-nudge-m1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+1s ⏩", id="filter-btn-nudge-p1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+10s ⏩", id="filter-btn-nudge-p10", color="secondary", size="sm")
                        ], className="mb-3"),
                        
                        # Range Slider for Time Window
                        html.Label("Time Range Slider", className="form-label"),
                        dcc.RangeSlider(
                            id="filter-time-range-slider",
                            min=0,
                            max=100,
                            step=0.1,
                            value=[0, 10],
                            allowCross=False,
                            pushable=1,
                            updatemode="mouseup",
                            className="mb-4"
                        ),
                        
                        # Filter Type Selection
                        html.H6("Filter Type", className="mb-3"),
                        dcc.Dropdown(
                            id="filter-type-select",
                            options=[
                                {"label": "Traditional Filters", "value": "traditional"},
                                {"label": "Advanced Filters", "value": "advanced"},
                                {"label": "Artifact Removal", "value": "artifact"},
                                {"label": "Neural Network Filtering", "value": "neural"},
                                {"label": "Ensemble Filtering", "value": "ensemble"}
                            ],
                            value="traditional",
                            clearable=False,
                            className="mb-3"
                        ),
                        
                        # Traditional Filter Parameters
                        html.Div(id="traditional-filter-params", children=[
                            html.H6("Traditional Filter Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter Family", className="form-label"),
                                    dcc.Dropdown(
                                        id="filter-family-advanced",
                                        options=[
                                            {"label": "Butterworth", "value": "butter"},
                                            {"label": "Chebyshev I", "value": "cheby1"},
                                            {"label": "Chebyshev II", "value": "cheby2"},
                                            {"label": "Elliptic", "value": "ellip"},
                                            {"label": "Bessel", "value": "bessel"}
                                        ],
                                        value="butter",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Response Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="filter-response-advanced",
                                        options=[
                                            {"label": "Bandpass", "value": "bandpass"},
                                            {"label": "Bandstop (Notch)", "value": "bandstop"},
                                            {"label": "Lowpass", "value": "lowpass"},
                                            {"label": "Highpass", "value": "highpass"}
                                        ],
                                        value="bandpass",
                                        clearable=False
                                    )
                                ], md=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Low Freq (Hz)", className="form-label"),
                                    dbc.Input(
                                        id="filter-low-freq-advanced",
                                        type="number",
                                        value=0.5,
                                        min=0,
                                        step=0.1,
                                        placeholder="0.5"
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("High Freq (Hz)", className="form-label"),
                                    dbc.Input(
                                        id="filter-high-freq-advanced",
                                        type="number",
                                        value=40,
                                        min=0,
                                        step=0.1,
                                        placeholder="40"
                                    )
                                ], md=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter Order", className="form-label"),
                                    dcc.Slider(
                                        id="filter-order-advanced",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=4,
                                        marks={i: str(i) for i in [2, 4, 6, 8, 10]},
                                        className="mb-3"
                                    )
                                ], md=12)
                            ])
                        ]),
                        
                        # Advanced Filter Parameters
                        html.Div(id="advanced-filter-params", children=[
                            html.H6("Advanced Filter Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter Method", className="form-label"),
                                    dcc.Dropdown(
                                        id="advanced-filter-method",
                                        options=[
                                            {"label": "Kalman Filter", "value": "kalman"},
                                            {"label": "Optimization Based", "value": "optimization"},
                                            {"label": "Gradient Descent", "value": "gradient"},
                                            {"label": "Convolution Based", "value": "convolution"},
                                            {"label": "Attention Based", "value": "attention"}
                                        ],
                                        value="kalman",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Noise Level", className="form-label"),
                                    dcc.Slider(
                                        id="advanced-noise-level",
                                        min=0.01,
                                        max=10,
                                        step=0.01,
                                        value=1,
                                        marks={0.01: "0.01", 1: "1", 5: "5", 10: "10"},
                                        className="mb-3"
                                    )
                                ], md=6)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Iterations", className="form-label"),
                                    dcc.Slider(
                                        id="advanced-iterations",
                                        min=10,
                                        max=500,
                                        step=10,
                                        value=100,
                                        marks={10: "10", 100: "100", 250: "250", 500: "500"},
                                        className="mb-3"
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Learning Rate", className="form-label"),
                                    dcc.Slider(
                                        id="advanced-learning-rate",
                                        min=0.001,
                                        max=0.1,
                                        step=0.001,
                                        value=0.01,
                                        marks={0.001: "0.001", 0.01: "0.01", 0.05: "0.05", 0.1: "0.1"},
                                        className="mb-3"
                                    )
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # Artifact Removal Parameters
                        html.Div(id="artifact-removal-params", children=[
                            html.H6("Artifact Removal Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Artifact Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="artifact-type",
                                        options=[
                                            {"label": "Baseline Wander", "value": "baseline"},
                                            {"label": "Motion Artifacts", "value": "motion"},
                                            {"label": "Power Line Interference", "value": "powerline"},
                                            {"label": "Muscle Noise", "value": "muscle"},
                                            {"label": "Electrode Noise", "value": "electrode"}
                                        ],
                                        value="baseline",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Removal Strength", className="form-label"),
                                    dcc.Slider(
                                        id="artifact-removal-strength",
                                        min=0.1,
                                        max=1.0,
                                        step=0.1,
                                        value=0.5,
                                        marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 1.0: "1.0"},
                                        className="mb-3"
                                    )
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # Neural Network Filter Parameters
                        html.Div(id="neural-filter-params", children=[
                            html.H6("Neural Network Filter Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Network Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="neural-network-type",
                                        options=[
                                            {"label": "Autoencoder", "value": "autoencoder"},
                                            {"label": "CNN", "value": "cnn"},
                                            {"label": "LSTM", "value": "lstm"},
                                            {"label": "Transformer", "value": "transformer"}
                                        ],
                                        value="autoencoder",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Model Complexity", className="form-label"),
                                    dcc.Slider(
                                        id="neural-model-complexity",
                                        min=1,
                                        max=5,
                                        step=1,
                                        value=3,
                                        marks={i: str(i) for i in [1, 2, 3, 4, 5]},
                                        className="mb-3"
                                    )
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # Ensemble Filter Parameters
                        html.Div(id="ensemble-filter-params", children=[
                            html.H6("Ensemble Filter Parameters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Ensemble Method", className="form-label"),
                                    dcc.Dropdown(
                                        id="ensemble-method",
                                        options=[
                                            {"label": "Mean", "value": "mean"},
                                            {"label": "Weighted Mean", "value": "weighted"},
                                            {"label": "Bagging", "value": "bagging"},
                                            {"label": "Boosting", "value": "boosting"}
                                        ],
                                        value="mean",
                                        clearable=False
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Number of Filters", className="form-label"),
                                    dcc.Slider(
                                        id="ensemble-n-filters",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=3,
                                        marks={i: str(i) for i in [2, 3, 5, 7, 10]},
                                        className="mb-3"
                                    )
                                ], md=6)
                            ], className="mb-3")
                        ]),
                        
                        # Filter Quality Assessment
                        html.H6("Quality Assessment", className="mb-3"),
                        dbc.Checklist(
                            id="filter-quality-options",
                            options=[
                                {"label": "SNR Improvement", "value": "snr"},
                                {"label": "Artifact Reduction", "value": "artifact_reduction"},
                                {"label": "Signal Distortion", "value": "distortion"},
                                {"label": "Computational Cost", "value": "computational_cost"}
                            ],
                            value=["snr", "artifact_reduction"],
                            className="mb-3"
                        ),
                        
                        # Action Buttons
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🔄 Apply Filter",
                                    id="filter-btn-apply",
                                    color="primary",
                                    className="w-100"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Button(
                                    "📊 Export Results",
                                    id="filter-btn-export",
                                    color="success",
                                    outline=True,
                                    className="w-100"
                                )
                            ], md=6)
                        ])
                    ])
                ], className="h-100")
            ], md=3),
            
            # Right Panel - Plots & Results
            dbc.Col([
                # Original Signal Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📈 Original Signal", className="mb-0"),
                        html.Small("Raw signal before filtering", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="filter-original-plot",
                                style={"height": "750px", "minHeight": "700px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Filtered Signal Plot
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔧 Filtered Signal", className="mb-0"),
                        html.Small("Signal after applying selected filters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="filter-filtered-plot",
                                style={"height": "750px", "minHeight": "700px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Filter Comparison
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("⚖️ Filter Comparison", className="mb-0"),
                        html.Small("Compare different filtering approaches", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="filter-comparison-plot",
                                style={"height": "500px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Filter Quality Metrics
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📊 Quality Metrics", className="mb-0"),
                        html.Small("Quantitative assessment of filtering performance", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.Div(id="filter-quality-metrics", className="mb-3"),
                        dcc.Loading(
                            dcc.Graph(
                                id="filter-quality-plots",
                                style={"height": "400px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ])
            ], md=9)
        ]),
        
        # Bottom Section - Additional Analysis
        html.Div(id="filter-additional-analysis-section", className="mt-4"),
        
        # Stores for data management
        dcc.Store(id="store-filtering-data"),
        dcc.Store(id="store-filter-comparison"),
        dcc.Store(id="store-filter-quality-metrics")
    ])


def physiological_layout():
    """Create the physiological features page layout."""
    return html.Div([
        # Page Header
        html.Div([
            html.H1("❤️ Physiological Features Analysis", className="text-center mb-4"),
            html.P([
                "Comprehensive physiological feature extraction and analysis for ECG, PPG, and other vital signs. ",
                "Utilize advanced vitalDSP algorithms for heart rate variability, morphological analysis, and health insights."
            ], className="text-center text-muted mb-5")
        ], className="mb-4"),
        
        # Main Analysis Section
        dbc.Row([
            # Left Panel - Controls & Parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🎛️ Analysis Controls", className="mb-0"),
                        html.Small("Configure physiological analysis parameters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Data Selection
                        html.H6("Data Selection", className="mb-3"),
                        dbc.Select(
                            id="physio-data-source-select",
                            options=[
                                {"label": "Uploaded Data", "value": "uploaded"},
                                {"label": "Sample Data", "value": "sample"}
                            ],
                            value="uploaded",
                            className="mb-3"
                        ),
                        
                        # Time Window Controls
                        html.H6("Time Window", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Start Time (s)", className="form-label"),
                                dbc.Input(
                                    id="physio-start-time",
                                    type="number",
                                    value=0,
                                    min=0,
                                    step=0.1,
                                    placeholder="0"
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("End Time (s)", className="form-label"),
                                dbc.Input(
                                    id="physio-end-time",
                                    type="number",
                                    value=10,
                                    min=0,
                                    step=0.1,
                                    placeholder="10"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Quick Window Navigation
                        html.Div([
                            dbc.Button("⏪ -10s", id="physio-btn-nudge-m10", color="secondary", size="sm", className="me-1"),
                            dbc.Button("⏪ -1s", id="physio-btn-nudge-m1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+1s ⏩", id="physio-btn-nudge-p1", color="secondary", size="sm", className="me-1"),
                            dbc.Button("+10s ⏩", id="physio-btn-nudge-p10", color="secondary", size="sm")
                        ], className="mb-3"),
                        
                        # Range Slider for Time Window
                        html.Label("Time Range Slider", className="form-label"),
                        dcc.RangeSlider(
                            id="physio-time-range-slider",
                            min=0,
                            max=100,
                            step=0.1,
                            value=[0, 10],
                            allowCross=False,
                            pushable=1,
                            updatemode="mouseup",
                            className="mb-4"
                        ),
                        
                        # Signal Type Selection
                        html.H6("Signal Type", className="mb-3"),
                        dbc.Select(
                            id="physio-signal-type",
                            options=[
                                {"label": "PPG (Photoplethysmography)", "value": "ppg"},
                                {"label": "ECG (Electrocardiogram)", "value": "ecg"},
                                {"label": "EEG (Electroencephalogram)", "value": "eeg"},
                                {"label": "Auto-detect", "value": "auto"}
                            ],
                            value="auto",
                            className="mb-3"
                        ),
                        
                        # Analysis Categories
                        html.H6("Analysis Categories", className="mb-3"),
                        dbc.Checklist(
                            id="physio-analysis-categories",
                            options=[
                                {"label": "Heart Rate & Variability", "value": "hrv"},
                                {"label": "Morphological Features", "value": "morphology"},
                                {"label": "Beat-to-Beat Analysis", "value": "beat2beat"},
                                {"label": "Energy Analysis", "value": "energy"},
                                {"label": "Envelope Detection", "value": "envelope"},
                                {"label": "Signal Segmentation", "value": "segmentation"},
                                {"label": "Trend Analysis", "value": "trend"},
                                {"label": "Waveform Analysis", "value": "waveform"},
                                {"label": "Statistical Analysis", "value": "statistical"},
                                {"label": "Frequency Analysis", "value": "frequency"},
                                {"label": "Signal Transforms", "value": "transforms"}
                            ],
                            value=["hrv", "morphology", "beat2beat", "energy", "envelope", "segmentation", "trend", "waveform", "statistical", "frequency"],
                            className="mb-3"
                        ),
                        
                        # HRV Specific Options
                        html.Div([
                            html.H6("HRV Analysis Options", className="mb-2"),
                            dbc.Checklist(
                                id="physio-hrv-options",
                                options=[
                                    {"label": "Time Domain Features", "value": "time_domain"},
                                    {"label": "Frequency Domain Features", "value": "freq_domain"},
                                    {"label": "Nonlinear Features", "value": "nonlinear"},
                                    {"label": "Poincaré Plot", "value": "poincare"},
                                    {"label": "Detrended Fluctuation", "value": "dfa"}
                                ],
                                value=["time_domain", "freq_domain", "nonlinear"],
                                className="mb-2"
                            )
                        ], id="physio-hrv-options-container", className="mb-3"),
                        
                        # Morphology Options
                        html.Div([
                            html.H6("Morphology Analysis Options", className="mb-2"),
                            dbc.Checklist(
                                id="physio-morphology-options",
                                options=[
                                    {"label": "Peak Detection", "value": "peaks"},
                                    {"label": "Duration Analysis", "value": "duration"},
                                    {"label": "Area Calculations", "value": "area"},
                                    {"label": "Amplitude Variability", "value": "amplitude"},
                                    {"label": "Slope Analysis", "value": "slope"},
                                    {"label": "Dicrotic Notch (PPG)", "value": "dicrotic"}
                                ],
                                value=["peaks", "duration", "area"],
                                className="mb-2"
                            )
                        ], id="physio-morphology-options-container", className="mb-3"),
                        
                        # Advanced Options
                        html.H6("Advanced Options", className="mb-3"),
                        dbc.Checklist(
                            id="physio-advanced-options",
                            options=[
                                {"label": "Cross-Signal Analysis", "value": "cross_signal"},
                                {"label": "Ensemble Methods", "value": "ensemble"},
                                {"label": "Change Detection", "value": "change_detection"},
                                {"label": "Power Analysis", "value": "power_analysis"},
                                {"label": "Signal Transforms", "value": "transforms"}
                            ],
                            value=["cross_signal", "ensemble", "change_detection", "power_analysis", "transforms"],
                            className="mb-3"
                        ),
                        
                        # Action Buttons
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🔄 Update Analysis",
                                    id="physio-btn-update-analysis",
                                    color="primary",
                                    className="w-100"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Button(
                                    "📊 Export Results",
                                    id="physio-btn-export-results",
                                    color="success",
                                    outline=True,
                                    className="w-100"
                                )
                            ], md=6)
                        ])
                    ])
                ], className="h-100")
            ], md=3),
            
            # Right Panel - Plots & Results
            dbc.Col([
                # Main Signal Display
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📈 Signal Overview", className="mb-0"),
                        html.Small("Raw signal with annotations and detected features", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="physio-main-signal-plot",
                                style={"height": "400px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Feature Analysis Results
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔍 Feature Analysis Results", className="mb-0"),
                        html.Small("Comprehensive physiological feature extraction results", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.Div(id="physio-analysis-results", className="mb-3"),
                        dcc.Loading(
                            dcc.Graph(
                                id="physio-analysis-plots",
                                style={"height": "500px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ])
            ], md=9)
        ]),
        
        # Bottom Section - Additional Analysis
        html.Div(id="physio-additional-analysis-section", className="mt-4"),
        
        # Stores for data management
        dcc.Store(id="store-physio-data"),
        dcc.Store(id="store-physio-features"),
        dcc.Store(id="store-physio-analysis")
    ])


def respiratory_layout():
    """Create the respiratory analysis page layout."""
    return html.Div([
        # Page Header
        html.Div([
            html.H1("🫁 Respiratory Analysis", className="text-center mb-4"),
            html.P([
                "Comprehensive respiratory rate estimation and breathing pattern analysis using vitalDSP. ",
                "Analyze respiratory signals with multiple estimation methods, sleep apnea detection, and multimodal fusion."
            ], className="text-center text-muted mb-5")
        ], className="mb-4"),
        
        # Main Analysis Section
        dbc.Row([
            # Left Panel - Controls & Parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🎛️ Respiratory Analysis Controls", className="mb-0"),
                        html.Small("Configure respiratory analysis parameters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Data Selection
                        html.H6("Data Selection", className="mb-3"),
                        dbc.Select(
                            id="resp-data-source-select",
                            options=[
                                {"label": "Uploaded Data", "value": "uploaded"},
                                {"label": "Sample Data", "value": "sample"}
                            ],
                            value="uploaded",
                            className="mb-3"
                        ),
                        
                        # Time Window Controls
                        html.H6("Time Window", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Start Time (s)", size="sm"),
                                dbc.Input(
                                    id="resp-start-time",
                                    type="number",
                                    value=0,
                                    min=0,
                                    step=0.1,
                                    className="mb-2"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Label("End Time (s)", size="sm"),
                                dbc.Input(
                                    id="resp-end-time",
                                    type="number",
                                    value=10,
                                    min=0,
                                    step=0.1,
                                    className="mb-2"
                                )
                            ], md=6)
                        ]),
                        
                        # Time Range Slider
                        html.Div([
                            dbc.Label("Time Range Slider", size="sm"),
                            dcc.RangeSlider(
                                id="resp-time-range-slider",
                                min=0,
                                max=100,
                                step=0.1,
                                value=[0, 10],
                                marks={},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-3"),
                        
                        # Nudge Buttons
                        html.Div([
                            dbc.Button("-10s", id="resp-btn-nudge-m10", size="sm", color="outline-secondary", className="me-1"),
                            dbc.Button("-1s", id="resp-btn-nudge-m1", size="sm", color="outline-secondary", className="me-1"),
                            dbc.Button("+1s", id="resp-btn-nudge-p1", size="sm", color="outline-secondary", className="me-1"),
                            dbc.Button("+10s", id="resp-btn-nudge-p10", size="sm", color="outline-secondary")
                        ], className="mb-3"),
                        
                        # Signal Type Selection
                        html.H6("Signal Type", className="mb-3"),
                        dbc.Select(
                            id="resp-signal-type",
                            options=[
                                {"label": "Auto-detect", "value": "auto"},
                                {"label": "PPG", "value": "ppg"},
                                {"label": "ECG", "value": "ecg"},
                                {"label": "Respiratory Belt", "value": "respiratory"},
                                {"label": "Nasal Cannula", "value": "nasal"}
                            ],
                            value="auto",
                            className="mb-3"
                        ),
                        
                        # Respiratory Rate Estimation Methods
                        html.H6("Estimation Methods", className="mb-3"),
                        dbc.Checklist(
                            id="resp-estimation-methods",
                            options=[
                                {"label": "Peak Detection", "value": "peaks"},
                                {"label": "Zero Crossing", "value": "zero_crossing"},
                                {"label": "Time Domain", "value": "time_domain"},
                                {"label": "Frequency Domain", "value": "frequency_domain"},
                                {"label": "FFT-based", "value": "fft_based"},
                                {"label": "Counting Method", "value": "counting"}
                            ],
                            value=["peaks", "fft_based"],
                            className="mb-3"
                        ),
                        
                        # Advanced Analysis Options
                        html.H6("Advanced Analysis", className="mb-3"),
                        dbc.Checklist(
                            id="resp-advanced-options",
                            options=[
                                {"label": "Sleep Apnea Detection", "value": "sleep_apnea"},
                                {"label": "Breathing Pattern Analysis", "value": "breathing_pattern"},
                                {"label": "Respiratory Variability", "value": "respiratory_variability"},
                                {"label": "Multimodal Fusion", "value": "multimodal_fusion"},
                                {"label": "Respiratory-Cardiac Fusion", "value": "cardiac_fusion"},
                                {"label": "Quality Assessment", "value": "quality_assessment"}
                            ],
                            value=["breathing_pattern", "respiratory_variability"],
                            className="mb-3"
                        ),
                        
                        # Preprocessing Options
                        html.H6("Preprocessing", className="mb-3"),
                        dbc.Checklist(
                            id="resp-preprocessing-options",
                            options=[
                                {"label": "Bandpass Filter", "value": "bandpass"},
                                {"label": "Wavelet Denoising", "value": "wavelet"},
                                {"label": "Moving Average", "value": "moving_average"},
                                {"label": "Baseline Correction", "value": "baseline_correction"},
                                {"label": "Artifact Removal", "value": "artifact_removal"}
                            ],
                            value=["bandpass", "baseline_correction"],
                            className="mb-3"
                        ),
                        
                        # Filter Parameters
                        html.H6("Filter Parameters", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Low Cut (Hz)", size="sm"),
                                dbc.Input(
                                    id="resp-low-cut",
                                    type="number",
                                    value=0.1,
                                    min=0.01,
                                    max=1.0,
                                    step=0.01,
                                    className="mb-2"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Label("High Cut (Hz)", size="sm"),
                                dbc.Input(
                                    id="resp-high-cut",
                                    type="number",
                                    value=0.5,
                                    min=0.1,
                                    max=2.0,
                                    step=0.01,
                                    className="mb-2"
                                )
                            ], md=6)
                        ]),
                        
                        # Breath Duration Constraints
                        html.H6("Breath Duration Constraints", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Min Duration (s)", size="sm"),
                                dbc.Input(
                                    id="resp-min-breath-duration",
                                    type="number",
                                    value=0.5,
                                    min=0.1,
                                    max=2.0,
                                    step=0.1,
                                    className="mb-2"
                                )
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Max Duration (s)", size="sm"),
                                dbc.Input(
                                    id="resp-max-breath-duration",
                                    type="number",
                                    value=6.0,
                                    min=2.0,
                                    max=20.0,
                                    step=0.5,
                                    className="mb-2"
                                )
                            ], md=6)
                        ]),
                        
                        # Action Buttons
                        html.Div([
                            dbc.Button("🔄 Update Analysis", id="resp-btn-update-analysis", 
                                      color="primary", className="w-100 mb-2"),
                            dbc.Button("📊 Export Results", id="resp-btn-export-results", 
                                      color="success", className="w-100")
                        ], className="mt-4")
                    ])
                ], className="h-100")
            ], md=3),
            
            # Right Panel - Plots & Results
            dbc.Col([
                # Main Respiratory Signal Display
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("📈 Respiratory Signal Analysis", className="mb-0"),
                        html.Small("Raw signal with breathing pattern detection and annotations", className="text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="resp-main-signal-plot",
                                style={"height": "400px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ], className="mb-4"),
                
                # Respiratory Analysis Results
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("🔍 Respiratory Analysis Results", className="mb-0"),
                        html.Small("Comprehensive respiratory rate estimation and pattern analysis", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.Div(id="resp-analysis-results", className="mb-3"),
                        dcc.Loading(
                            dcc.Graph(
                                id="resp-analysis-plots",
                                style={"height": "500px"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                                    "displaylogo": False
                                }
                            ),
                            type="default"
                        )
                    ])
                ])
            ], md=9)
        ]),
        
        # Bottom Section - Additional Analysis
        html.Div(id="resp-additional-analysis-section", className="mt-4"),
        
        # Stores for data management
        dcc.Store(id="store-resp-data"),
        dcc.Store(id="store-resp-features"),
        dcc.Store(id="store-resp-analysis")
    ])


def features_layout():
    """Create the features page layout."""
    return html.Div([
        # Header Section
        html.Div([
            html.H1("🎯 Feature Engineering", className="text-center mb-4"),
            html.P("Comprehensive feature extraction and analysis using vitalDSP feature engineering modules", 
                   className="text-center text-muted mb-4")
        ]),
        
        # Configuration Section - Balanced Layout
        dbc.Row([
            # Left Panel - Analysis Configuration
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("⚙️ Analysis Configuration", className="mb-0"),
                        html.Small("Configure feature extraction parameters", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Signal Type Selection
                        html.Label("Signal Type", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id="features-signal-type",
                            options=[
                                {"label": "PPG Signal", "value": "ppg"},
                                {"label": "ECG Signal", "value": "ecg"},
                                {"label": "Both Signals", "value": "both"}
                            ],
                            value="ppg",
                            placeholder="Select signal type",
                            className="mb-3"
                        ),
                        
                        # Preprocessing Options
                        html.Label("Preprocessing Options", className="form-label fw-bold"),
                        dcc.Checklist(
                            id="features-preprocessing",
                            options=[
                                {"label": "Bandpass Filter", "value": "bandpass"},
                                {"label": "Denoising", "value": "denoising"},
                                {"label": "Baseline Correction", "value": "baseline"}
                            ],
                            value=["bandpass"],
                            className="mb-4"
                        ),
                        
                        # Analysis Button
                        dbc.Button(
                            "🚀 Analyze Features",
                            id="features-analyze-btn",
                            color="primary",
                            size="lg",
                            className="w-100 shadow-sm"
                        ),
                        
                        # Quick Info
                        html.Hr(className="my-3"),
                        html.Div([
                            html.Small("💡 Tip: Select signal type and preprocessing options, then click analyze to extract comprehensive features.", 
                                     className="text-muted fst-italic")
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            ], md=5),
            
            # Right Panel - Available Features (Compact)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("ℹ️ Available Features", className="mb-0"),
                        html.Small("Feature engineering modules in vitalDSP", className="text-muted")
                    ]),
                    dbc.CardBody([
                        # Compact feature display in columns
                        dbc.Row([
                            # Left column
                            dbc.Col([
                                html.H6("🧠 Autonomic", className="text-primary mb-2 fw-bold"),
                                html.Div([
                                    html.Small("• RRV, RSA", className="d-block text-muted mb-1"),
                                    html.Small("• Fractal Dimension", className="d-block text-muted mb-1"),
                                    html.Small("• DFA Analysis", className="d-block text-muted")
                                ], className="mb-3 p-2 bg-light rounded"),
                                
                                html.H6("📊 Morphology", className="text-success mb-2 fw-bold"),
                                html.Div([
                                    html.Small("• Peak Detection", className="d-block text-muted mb-1"),
                                    html.Small("• Trend Analysis", className="d-block text-muted mb-1"),
                                    html.Small("• Waveform Analysis", className="d-block text-muted")
                                ], className="mb-3 p-2 bg-light rounded")
                            ], md=6),
                            
                            # Right column
                            dbc.Col([
                                html.H6("💡 Light-Based", className="text-warning mb-2 fw-bold"),
                                html.Div([
                                    html.Small("• SpO2, PI", className="d-block text-muted mb-1"),
                                    html.Small("• Respiratory Rate", className="d-block text-muted mb-1"),
                                    html.Small("• PPR Ratio", className="d-block text-muted")
                                ], className="mb-3 p-2 bg-light rounded"),
                                
                                html.H6("🔗 Synchronization", className="text-info mb-2 fw-bold"),
                                html.Div([
                                    html.Small("• PTT, PAT", className="d-block text-muted mb-1"),
                                    html.Small("• EMD", className="d-block text-muted mb-1"),
                                    html.Small("• ECG-PPG Sync", className="d-block text-muted")
                                ], className="mb-3 p-2 bg-light rounded")
                            ], md=6)
                        ]),
                        
                        # ECG Features in a compact row
                        html.Hr(className="my-3"),
                        html.H6("❤️ ECG Features", className="text-danger mb-2 fw-bold"),
                        html.Div([
                            html.Small("P-wave, PR, QRS, ST, QT, Arrhythmia Detection", className="text-muted")
                        ], className="p-2 bg-light rounded"),
                        
                        # Feature count summary
                        html.Hr(className="my-3"),
                        html.Div([
                            html.Small("📊 Total: 25+ features across 5 categories", className="text-success fw-bold")
                        ], className="text-center")
                    ])
                ], className="mb-4 shadow-sm")
            ], md=7)
        ]),
        
        # Analysis Results Section
        html.Div(id="features-analysis-results", className="mb-4"),
        
        # Visualization Section
        dbc.Card([
            dbc.CardHeader([
                html.H4("📊 Feature Analysis Visualization", className="mb-0"),
                html.Small("Comprehensive feature analysis plots and results", className="text-muted")
            ]),
            dbc.CardBody([
                dcc.Loading(
                    dcc.Graph(
                        id="features-analysis-plots",
                        style={"height": "800px"},
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                            "displaylogo": False
                        }
                    ),
                    type="default"
                )
            ])
        ]),
        
        # Stores for data management
        dcc.Store(id="store-features-data"),
        dcc.Store(id="store-features-features")
    ])


def transforms_layout():
    """Create the transforms page layout."""
    return html.Div([
        html.H1("🔄 Signal Transforms", className="text-center mb-4"),
        html.P("Signal transformation tools coming soon...", className="text-center text-muted")
    ])


def quality_layout():
    """Create the signal quality page layout."""
    return html.Div([
        html.H1("🎯 Signal Quality Assessment", className="text-center mb-4"),
        html.P("Signal quality assessment tools coming soon...", className="text-center text-muted")
    ])


def advanced_layout():
    """Create the advanced analysis page layout."""
    return html.Div([
        html.H1("🧠 Advanced Analysis", className="text-center mb-4"),
        html.P("Advanced analysis tools coming soon...", className="text-center text-muted")
    ])


def health_report_layout():
    """Create the health report page layout."""
    return html.Div([
        html.H1("📋 Health Report Generator", className="text-center mb-4"),
        html.P("Health report generation tools coming soon...", className="text-center text-muted")
    ])


def settings_layout():
    """Create the settings page layout."""
    return html.Div([
        html.H1("⚙️ Settings", className="text-center mb-4"),
        html.P("Application settings and configuration coming soon...", className="text-center text-muted")
    ])
