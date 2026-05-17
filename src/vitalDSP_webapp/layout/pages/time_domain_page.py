"""
Time Domain Analysis page layout for vitalDSP webapp.
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
                    html.H1("Time Domain Analysis", className="text-center mb-2"),
                    html.P(
                        "Extract and visualize time-domain features from your filtered physiological signal.",
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
                                                [html.I(className="fas fa-play me-2"), "Run Analysis"],
                                                id="btn-update-analysis",
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
                                                    id="start-position-slider",
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
                                        # Duration
                                        dbc.Col(
                                            [
                                                html.Label("Window", className="form-label fw-semibold small mb-1"),
                                                dbc.Select(
                                                    id="duration-select",
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
                                                        dbc.Button("« -10%", id="btn-nudge-m10", color="outline-secondary", size="sm", className="me-1"),
                                                        dbc.Button("‹ -5%",  id="btn-nudge-m5",  color="outline-secondary", size="sm", className="me-1"),
                                                        dbc.Button("+5% ›",  id="btn-nudge-p5",  color="outline-secondary", size="sm", className="me-1"),
                                                        dbc.Button("+10% »", id="btn-nudge-p10", color="outline-secondary", size="sm"),
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

            # Main Two-Column Layout
            dbc.Row(
                [
                    # ── RIGHT PANEL (signal plot + results) ──────────────────
                    dbc.Col(
                        [
                            # Signal Plot Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H5("Signal with Critical Points", className="mb-0"),
                                                        html.Small("Morphological features detected by vitalDSP", className="text-muted"),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="main-signal-plot",
                                                style={"height": "420px"},
                                                config={"displayModeBar": True, "displaylogo": False},
                                            ),
                                            type="default",
                                        )
                                    ),
                                ],
                                className="mb-4",
                            ),

                            # Signal summary (collapsible)
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        dbc.Button(
                                            [html.I(className="fas fa-info-circle me-2"), "Signal Summary"],
                                            id="btn-collapse-summary",
                                            color="link",
                                            className="p-0 text-decoration-none fw-semibold",
                                        ),
                                        className="py-2",
                                    ),
                                    dbc.Collapse(
                                        dbc.CardBody(html.Div(id="analysis-results")),
                                        id="collapse-summary",
                                        is_open=False,
                                    ),
                                ],
                                className="mb-4",
                            ),

                            # Analysis Results Card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H5("Analysis Results", className="mb-0"),
                                            html.Small(
                                                "Time-domain features and HRV metrics from vitalDSP",
                                                className="text-muted",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            html.Div(id="peak-analysis-table"),
                                            type="default",
                                        )
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),

            # Hidden outputs kept for callback compat
            dcc.Graph(id="signal-comparison-plot", style={"display": "none"}),
            html.Div(id="signal-quality-table", style={"display": "none"}),
            html.Div(id="signal-source-table", style={"display": "none"}),
            html.Div(id="additional-metrics-table", style={"display": "none"}),
            html.Div(id="additional-analysis-section"),

            # btn-center is used by signal_filtering_callbacks as an Input
            html.Button(id="btn-center", style={"display": "none"}),

            # Hidden filtering-page shadow components (for cross-page callback compat)
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
            dcc.Input(id="advanced-noise-level", style={"display": "none"}),
            dcc.Input(id="advanced-iterations", style={"display": "none"}),
            dcc.Input(id="advanced-learning-rate", style={"display": "none"}),
            dcc.Dropdown(id="artifact-type", style={"display": "none"}),
            dcc.Input(id="artifact-removal-strength", style={"display": "none"}),
            dcc.Dropdown(id="neural-network-type", style={"display": "none"}),
            dcc.Input(id="neural-model-complexity", style={"display": "none"}),
            dcc.Dropdown(id="ensemble-method", style={"display": "none"}),
            dcc.Checklist(id="detrend-option", style={"display": "none"}),
            dcc.Input(id="savgol-window", style={"display": "none"}),
            dcc.Input(id="savgol-polyorder", style={"display": "none"}),
            dcc.Input(id="moving-avg-window", style={"display": "none"}),
            dcc.Input(id="gaussian-sigma", style={"display": "none"}),
            dcc.Dropdown(id="wavelet-type", style={"display": "none"}),
            dcc.Input(id="wavelet-level", style={"display": "none"}),
            dcc.Dropdown(id="threshold-type", style={"display": "none"}),
            dcc.Input(id="threshold-value", style={"display": "none"}),
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

            # Data stores
            dcc.Store(id="store-time-domain-data"),
            dcc.Store(id="store-filtered-data"),
            dcc.Store(id="store-filtered-signal"),
            dcc.Store(id="store-analysis-results"),
            dcc.Store(id="store-time-domain-features"),

            # Removed: download components (export CSV/JSON removed)
            # Kept as empty placeholders so any stale callback IDs don't crash
            html.Div(id="download-time-domain-csv", style={"display": "none"}),
            html.Div(id="download-time-domain-json", style={"display": "none"}),

            # Hidden inputs removed from left panel - kept for callback State() compat
            dcc.Dropdown(id="signal-source-select", value="filtered", style={"display": "none"}),
            dcc.Checklist(id="analysis-options", value=["peaks", "critical_points", "hr"], style={"display": "none"}),
            dcc.Dropdown(id="signal-type-select", value="PPG", style={"display": "none"}),
            dcc.Dropdown(id="data-source-select", value="uploaded", style={"display": "none"}),
        ]
    )
