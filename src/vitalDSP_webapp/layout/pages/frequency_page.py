"""
Frequency Domain Analysis page layout for vitalDSP webapp.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def frequency_layout():
    """Create the frequency domain analysis page layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("Frequency Domain Analysis", className="text-center mb-2"),
                    html.P(
                        "Analyze the spectral content of your filtered physiological signal — FFT, PSD, STFT, and HRV frequency bands.",
                        className="text-center text-muted mb-4",
                    ),
                ],
                className="mb-3",
            ),

            # Top Action Bar
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Row(
                                    [
                                        # Run button
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-wave-square me-2"), "Run Analysis"],
                                                id="freq-btn-update-analysis",
                                                color="primary",
                                                size="lg",
                                                className="w-100",
                                            ),
                                            md=3,
                                        ),
                                        # Start position
                                        dbc.Col(
                                            [
                                                html.Label("Start Position", className="form-label fw-semibold small mb-1"),
                                                dcc.Slider(
                                                    id="freq-start-position-slider",
                                                    min=0,
                                                    max=100,
                                                    step=1,
                                                    value=0,
                                                    marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className="mt-1",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        # Window duration
                                        dbc.Col(
                                            [
                                                html.Label("Window", className="form-label fw-semibold small mb-1"),
                                                dbc.Select(
                                                    id="freq-duration-select",
                                                    options=[
                                                        {"label": "30 s", "value": 30},
                                                        {"label": "1 min", "value": 60},
                                                        {"label": "2 min", "value": 120},
                                                        {"label": "5 min", "value": 300},
                                                    ],
                                                    value=60,
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        # Quick navigation
                                        dbc.Col(
                                            [
                                                html.Label("Navigate", className="form-label fw-semibold small mb-1"),
                                                html.Div(
                                                    [
                                                        dbc.Button("« -10%", id="freq-btn-nudge-m10", color="outline-secondary", size="sm", className="me-1"),
                                                        dbc.Button("‹ -5%",  id="freq-btn-nudge-m5",  color="outline-secondary", size="sm", className="me-1"),
                                                        dbc.Button("+5% ›",  id="freq-btn-nudge-p5",  color="outline-secondary", size="sm", className="me-1"),
                                                        dbc.Button("+10% »", id="freq-btn-nudge-p10", color="outline-secondary", size="sm"),
                                                    ],
                                                    className="d-flex",
                                                ),
                                            ],
                                            md=3,
                                        ),
                                    ],
                                    align="end",
                                )
                            ),
                            className="mb-4",
                        ),
                        md=12,
                    )
                ]
            ),

            # Analysis settings bar
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Spectrum Method", className="form-label fw-semibold small mb-1"),
                                                        dbc.RadioItems(
                                                            id="freq-analysis-type",
                                                            options=[
                                                                {"label": "FFT", "value": "fft"},
                                                                {"label": "PSD (Welch)", "value": "psd"},
                                                            ],
                                                            value="psd",
                                                            inline=True,
                                                            inputClassName="me-1",
                                                            labelClassName="me-3",
                                                        ),
                                                    ],
                                                    md=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Window Function", className="form-label fw-semibold small mb-1"),
                                                        dbc.Select(
                                                            id="freq-window-type",
                                                            options=[
                                                                {"label": "Hann (recommended)", "value": "hann"},
                                                                {"label": "Hamming", "value": "hamming"},
                                                                {"label": "Blackman", "value": "blackman"},
                                                                {"label": "Rectangular", "value": "none"},
                                                            ],
                                                            value="hann",
                                                        ),
                                                    ],
                                                    md=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Max Frequency (Hz)", className="form-label fw-semibold small mb-1"),
                                                        dbc.Input(
                                                            id="freq-max-hz",
                                                            type="number",
                                                            value=8,
                                                            min=1,
                                                            max=200,
                                                            step=1,
                                                        ),
                                                    ],
                                                    md=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Scale", className="form-label fw-semibold small mb-1"),
                                                        dbc.Select(
                                                            id="freq-scale",
                                                            options=[
                                                                {"label": "dB (log)", "value": "db"},
                                                                {"label": "Linear", "value": "linear"},
                                                            ],
                                                            value="db",
                                                        ),
                                                    ],
                                                    md=2,
                                                ),
                                                # signal-type hidden (auto-detected; kept for callback State compat)
                                                html.Div(
                                                    dbc.Select(
                                                        id="freq-signal-type",
                                                        options=[
                                                            {"label": "PPG", "value": "PPG"},
                                                            {"label": "ECG", "value": "ECG"},
                                                            {"label": "EEG", "value": "EEG"},
                                                        ],
                                                        value="PPG",
                                                    ),
                                                    style={"display": "none"},
                                                ),
                                            ],
                                            align="end",
                                        )
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        md=12,
                    )
                ]
            ),

            # ── Plots row: Spectrogram | Frequency Spectrum ─────────
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader([
                                    html.H5("Time-Frequency Map", className="mb-0"),
                                    html.Small("Spectrogram — energy vs. time and frequency", className="text-muted"),
                                ]),
                                dbc.CardBody(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="freq-spectrogram-plot",
                                            style={"height": "300px"},
                                            config={"displayModeBar": True, "displaylogo": False},
                                        ),
                                        type="default",
                                    ),
                                    style={"padding": "0"},
                                ),
                            ],
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader([
                                    html.H5("Frequency Spectrum", className="mb-0"),
                                    html.Small("FFT / PSD power vs. frequency", className="text-muted"),
                                ]),
                                dbc.CardBody(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="freq-main-plot",
                                            style={"height": "300px"},
                                            config={"displayModeBar": True, "displaylogo": False},
                                        ),
                                        type="default",
                                    ),
                                    style={"padding": "0"},
                                ),
                            ],
                        ),
                        md=6,
                    ),
                ],
                className="mb-4 g-3",
            ),

            # ── Key metric cards ────────────────────────────────────
            html.Div(id="freq-metric-cards", className="mb-4"),

            # ── Full Analysis Results — collapsible card ────────────
            dbc.Card(
                [
                    dbc.CardHeader(
                        dbc.Button(
                            [html.I(className="fas fa-table me-2"), "Analysis Results"],
                            id="freq-btn-collapse-analysis-results",
                            color="link",
                            className="p-0 text-decoration-none fw-semibold",
                        ),
                        className="py-2",
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            dcc.Loading(html.Div(id="freq-peak-analysis-table"), type="default")
                        ),
                        id="freq-collapse-analysis-results",
                        is_open=False,
                    ),
                ],
                className="mb-4",
            ),

            # ── Hidden outputs kept for callback compat ─────────────
            html.Div(id="freq-band-power-table",  style={"display": "none"}),
            html.Div(id="freq-harmonics-table",   style={"display": "none"}),
            html.Div(id="freq-stability-table",   style={"display": "none"}),

            # ── Signal Summary (collapsible) ────────────────────────
            dbc.Card(
                [
                    dbc.CardHeader(
                        dbc.Button(
                            [html.I(className="fas fa-info-circle me-2"), "Signal Summary"],
                            id="freq-btn-collapse-summary",
                            color="link",
                            className="p-0 text-decoration-none fw-semibold",
                        ),
                        className="py-2",
                    ),
                    dbc.Collapse(
                        dbc.CardBody(html.Div(id="freq-analysis-results")),
                        id="freq-collapse-summary",
                        is_open=False,
                    ),
                ],
                className="mb-4",
            ),

            # Data stores
            dcc.Store(id="store-frequency-data"),
            dcc.Store(id="store-time-freq-data"),
            dcc.Store(id="store-freq-analysis-results"),

            # Hidden compat components for cross-page callback compat
            dcc.Store(id="store-filtered-signal"),
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),

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
            dcc.Graph(id="filter-original-plot", style={"display": "none"}),
            dcc.Graph(id="filter-filtered-plot", style={"display": "none"}),
            dcc.Graph(id="filter-comparison-plot", style={"display": "none"}),
            dcc.Graph(id="filter-quality-plots", style={"display": "none"}),
            html.Div(id="filter-quality-metrics", style={"display": "none"}),
            dcc.Graph(id="frequency-filtered-signal-plot", style={"display": "none"}),
            dcc.Graph(id="filter-response-plot", style={"display": "none"}),
            html.Div(id="filter-stats", style={"display": "none"}),

            # Hidden nav button IDs referenced by other page callbacks
            html.Button(id="btn-nudge-m10", style={"display": "none"}),
            html.Button(id="btn-center", style={"display": "none"}),
            html.Button(id="btn-nudge-p10", style={"display": "none"}),
            html.Div(
                dcc.Slider(id="start-position-slider", min=0, max=100, value=0),
                style={"display": "none"},
            ),
            dcc.Dropdown(id="duration-select", style={"display": "none"}),

            # Old IDs kept for callback compat — no longer wired to UI
            html.Div(id="freq-signal-source-select", style={"display": "none"}),
            html.Div(id="freq-analysis-options", style={"display": "none"}),
            # Old parameter IDs (now replaced by unified controls above)
            dbc.Input(id="fft-n-points", style={"display": "none"}),
            dbc.Input(id="psd-window", style={"display": "none"}),
            dbc.Input(id="psd-overlap", style={"display": "none"}),
            dbc.Input(id="psd-freq-max", style={"display": "none"}),
            dcc.Checklist(id="psd-log-scale", style={"display": "none"}),
            dcc.Checklist(id="psd-normalize", style={"display": "none"}),
            dcc.Dropdown(id="psd-channel", style={"display": "none"}),
            dbc.Input(id="stft-window-size", style={"display": "none"}),
            dbc.Input(id="stft-hop-size", style={"display": "none"}),
            dcc.Dropdown(id="stft-window-type", style={"display": "none"}),
            dbc.Input(id="stft-overlap", style={"display": "none"}),
            dcc.Dropdown(id="stft-scaling", style={"display": "none"}),
            dbc.Input(id="stft-freq-max", style={"display": "none"}),
            dcc.Dropdown(id="stft-colormap", style={"display": "none"}),
            dcc.Dropdown(id="wavelet-type", style={"display": "none"}),
            dbc.Input(id="wavelet-levels", style={"display": "none"}),
            dbc.Input(id="freq-min", style={"display": "none"}),
            dbc.Input(id="freq-max", style={"display": "none"}),
            # Old nudge IDs
            html.Button(id="freq-btn-nudge-m1", style={"display": "none"}),
            html.Button(id="freq-btn-nudge-p1", style={"display": "none"}),
        ]
    )
