"""
Respiratory Rate Analysis page layout for vitalDSP webapp.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def _method_card(title, plot_id, result_id):
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H6(title, className="mb-0"), width=8),
                        dbc.Col(
                            html.Div(
                                id=result_id,
                                className="text-end fw-bold text-primary small",
                            ),
                            width=4,
                        ),
                    ],
                    align="center",
                )
            ),
            dbc.CardBody(
                dcc.Loading(
                    dcc.Graph(
                        id=plot_id,
                        style={"height": "180px"},
                        config={"displayModeBar": False, "displaylogo": False},
                    ),
                    type="default",
                ),
                style={"padding": "0"},
            ),
        ],
        className="shadow-sm h-100",
    )


def respiratory_layout():
    """Create the respiratory rate analysis page layout."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H1("Respiratory Rate Analysis", className="text-center mb-2"),
                    html.P(
                        "All respiratory rate extraction methods run simultaneously — compare results and confidence across methods.",
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
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(
                                                        className="fas fa-play me-2"
                                                    ),
                                                    "Run Analysis",
                                                ],
                                                id="resp-analyze-btn",
                                                color="primary",
                                                size="lg",
                                                className="w-100",
                                            ),
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Start Position",
                                                    className="form-label fw-semibold small mb-1",
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
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                    className="mt-1",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Window",
                                                    className="form-label fw-semibold small mb-1",
                                                ),
                                                dbc.Select(
                                                    id="resp-duration-select",
                                                    options=[
                                                        {"label": "30 s", "value": 30},
                                                        {"label": "1 min", "value": 60},
                                                        {
                                                            "label": "2 min",
                                                            "value": 120,
                                                        },
                                                        {
                                                            "label": "5 min",
                                                            "value": 300,
                                                        },
                                                        {
                                                            "label": "10 min",
                                                            "value": 600,
                                                        },
                                                    ],
                                                    value=60,
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Navigate",
                                                    className="form-label fw-semibold small mb-1",
                                                ),
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button(
                                                            "⏮",
                                                            id="resp-btn-nudge-m10",
                                                            color="secondary",
                                                            size="sm",
                                                            title="-10%",
                                                        ),
                                                        dbc.Button(
                                                            "◀",
                                                            id="resp-btn-nudge-m1",
                                                            color="secondary",
                                                            size="sm",
                                                            title="-5%",
                                                        ),
                                                        dbc.Button(
                                                            "▶",
                                                            id="resp-btn-nudge-p1",
                                                            color="secondary",
                                                            size="sm",
                                                            title="+5%",
                                                        ),
                                                        dbc.Button(
                                                            "⏭",
                                                            id="resp-btn-nudge-p10",
                                                            color="secondary",
                                                            size="sm",
                                                            title="+10%",
                                                        ),
                                                    ],
                                                    className="w-100",
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Min Breath (s)",
                                                    className="form-label fw-semibold small mb-1",
                                                ),
                                                dbc.Input(
                                                    id="resp-min-breath-duration",
                                                    type="number",
                                                    value=1.8,
                                                    min=0.5,
                                                    max=5.0,
                                                    step=0.1,
                                                    size="sm",
                                                ),
                                            ],
                                            md=1,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Max Breath (s)",
                                                    className="form-label fw-semibold small mb-1",
                                                ),
                                                dbc.Input(
                                                    id="resp-max-breath-duration",
                                                    type="number",
                                                    value=6.0,
                                                    min=2.0,
                                                    max=20.0,
                                                    step=0.5,
                                                    size="sm",
                                                ),
                                            ],
                                            md=1,
                                        ),
                                    ],
                                    align="end",
                                    className="g-2",
                                ),
                            ),
                            className="shadow-sm",
                        ),
                    )
                ],
                className="mb-3",
            ),
            # Methods Comparison & Ensemble — shown first
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H6(
                                        "Methods Comparison & Ensemble",
                                        className="mb-0",
                                    )
                                ),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Div(
                                            id="resp-analysis-results",
                                            children=html.P(
                                                "Click Run Analysis to compute all methods.",
                                                className="text-muted small mb-0",
                                            ),
                                        ),
                                        type="default",
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                    )
                ],
                className="mb-3",
            ),
            # Hidden graph kept for callback output compatibility
            dcc.Graph(id="resp-main-plot", style={"display": "none"}),
            # Row 1: counting + fft_based
            dbc.Row(
                [
                    dbc.Col(
                        _method_card(
                            "Counting (Peak Detection RR)",
                            "resp-plot-counting",
                            "resp-result-counting",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _method_card(
                            "FFT-Based RR",
                            "resp-plot-fft-based",
                            "resp-result-fft-based",
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # Row 2: frequency_domain + time_domain
            dbc.Row(
                [
                    dbc.Col(
                        _method_card(
                            "Frequency Domain RR (Welch PSD)",
                            "resp-plot-freq-domain",
                            "resp-result-freq-domain",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _method_card(
                            "Time Domain RR (Autocorrelation)",
                            "resp-plot-time-domain",
                            "resp-result-time-domain",
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # Row 3: peaks + zero_crossing
            dbc.Row(
                [
                    dbc.Col(
                        _method_card(
                            "Peak Interval Detection",
                            "resp-plot-peaks",
                            "resp-result-peaks",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _method_card(
                            "Zero-Crossing Detection",
                            "resp-plot-zero-crossing",
                            "resp-result-zero-crossing",
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # Stores
            dcc.Store(id="resp-data-store"),
            dcc.Store(id="resp-features-store"),
            # Hidden compat components for signal_filtering_callbacks
            html.Div(id="filter-btn-apply", style={"display": "none"}),
            dcc.Dropdown(id="filter-type-select", style={"display": "none"}),
            dcc.Dropdown(id="filter-family-advanced", style={"display": "none"}),
            dcc.Dropdown(id="filter-response-advanced", style={"display": "none"}),
            dcc.Input(
                id="filter-low-freq-advanced", type="number", style={"display": "none"}
            ),
            dcc.Input(
                id="filter-high-freq-advanced", type="number", style={"display": "none"}
            ),
            dcc.Input(
                id="filter-order-advanced", type="number", style={"display": "none"}
            ),
            dcc.Checklist(id="filter-quality-options", style={"display": "none"}),
            dcc.Dropdown(id="filter-signal-type-select", style={"display": "none"}),
            dcc.Dropdown(id="advanced-filter-method", style={"display": "none"}),
            dcc.Input(
                id="advanced-iterations", type="number", style={"display": "none"}
            ),
            dcc.Input(
                id="advanced-learning-rate", type="number", style={"display": "none"}
            ),
            dcc.Input(
                id="advanced-noise-level", type="number", style={"display": "none"}
            ),
            dcc.Input(
                id="artifact-removal-strength", type="number", style={"display": "none"}
            ),
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dcc.Dropdown(id="detrend-option", style={"display": "none"}),
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            dcc.Input(
                id="ensemble-n-filters", type="number", style={"display": "none"}
            ),
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
            dcc.Store(id="store-filtered-signal"),
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            dcc.Graph(id="filter-original-plot", style={"display": "none"}),
            dcc.Graph(id="filter-filtered-plot", style={"display": "none"}),
            dcc.Graph(id="filter-comparison-plot", style={"display": "none"}),
            html.Div(id="filter-quality-metrics", style={"display": "none"}),
            dcc.Graph(id="filter-quality-plots", style={"display": "none"}),
            html.Button(id="btn-nudge-m10", style={"display": "none"}),
            html.Button(id="btn-center", style={"display": "none"}),
            html.Button(id="btn-nudge-p10", style={"display": "none"}),
            html.Div(
                children=[
                    dcc.Slider(id="start-position-slider", min=0, max=100, value=0)
                ],
                style={"display": "none"},
            ),
            dcc.Dropdown(id="duration-select", style={"display": "none"}),
        ]
    )
