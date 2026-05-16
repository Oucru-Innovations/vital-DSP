"""
Filtering page layout for vitalDSP webapp.

This module provides the layout for the signal filtering page.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def filtering_layout():
    """Create the comprehensive signal filtering page layout with optimized space usage."""
    return html.Div(
        [
            # Visual polish lives in ``assets/filtering_page.css``,
            # auto-loaded by Dash.  Everything below scopes to the
            # ``signal-filtering-page`` class on the outer Div.
            # Page header — single tight line, no over-explained subtitle.
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fas fa-wave-square me-2 text-primary"),
                            html.Span("Signal Filtering", className="fw-semibold"),
                        ],
                        className="h4 mb-0 d-flex align-items-center",
                    ),
                    html.Small(
                        "Pick a segment in the timeline, build a filter chain, then Apply.",
                        className="text-muted",
                    ),
                ],
                className="mb-3",
            ),
            # ------------------------------------------------------------------
            # Top controller bar.
            #
            # Two rows, 12-column grid, never overflowing:
            #   Row 1 : timeline strip + scoring toggle (full width)
            #   Row 2 : filter-family radios | view-zoom | nudge buttons | Apply
            # ------------------------------------------------------------------
            dbc.Card(
                dbc.CardBody(
                    [
                        # Row 1 — segment timeline + scoring toggle.  The
                        # timeline IS the position picker; clicking a cell
                        # jumps the comparison plot below.
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    id="filter-segment-headline-top",
                                                    className="small text-muted",
                                                    children="Upload data to see segments.",
                                                ),
                                                dbc.Checkbox(
                                                    id="filter-segment-scoring-enabled",
                                                    label=" Show accept / reject scoring",
                                                    value=False,
                                                    className="small ms-auto mb-0",
                                                ),
                                            ],
                                            className="d-flex align-items-center justify-content-between mb-1",
                                        ),
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="filter-segment-timeline-top",
                                                config={
                                                    "displayModeBar": False,
                                                    "staticPlot": False,
                                                },
                                                className="segment-timeline-graph",
                                            ),
                                            type="default",
                                        ),
                                    ],
                                    md=12,
                                ),
                            ],
                            className="g-2 mb-3",
                        ),

                        # Row 2 — primary controls.  Filter family takes
                        # the wide column; view-zoom + nudge + Apply share
                        # the right half.
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Filter family",
                                            className="form-label section-label mb-1",
                                        ),
                                        # Pill-style toggle: Bootstrap's
                                        # btn-check input + btn label
                                        # idiom, plus our own pill CSS.
                                        dbc.RadioItems(
                                            id="filter-type-select",
                                            options=[
                                                {"label": "Traditional", "value": "traditional"},
                                                {"label": "Smoothing", "value": "smoothing"},
                                                {"label": "Advanced", "value": "advanced"},
                                                {"label": "Artifact", "value": "artifact"},
                                                {"label": "Neural", "value": "neural"},
                                                {"label": "Ensemble", "value": "ensemble"},
                                            ],
                                            value="traditional",
                                            inline=True,
                                            className="filter-family-pills",
                                            inputClassName="btn-check",
                                            labelClassName="btn",
                                            labelCheckedClassName="active",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "View zoom",
                                            className="form-label section-label mb-1",
                                        ),
                                        dbc.Select(
                                            id="duration-select",
                                            options=[
                                                {"label": "1 segment", "value": 1},
                                                {"label": "3 segments", "value": 3},
                                                {"label": "5 segments", "value": 5},
                                            ],
                                            value=1,
                                            size="sm",
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Nudge",
                                            className="form-label section-label mb-1",
                                        ),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    html.I(className="fas fa-chevron-left"),
                                                    id="btn-nudge-m10",
                                                    color="light",
                                                    size="sm",
                                                    title="Back 10%",
                                                ),
                                                dbc.Button(
                                                    html.I(className="fas fa-compress-arrows-alt"),
                                                    id="btn-center",
                                                    color="light",
                                                    size="sm",
                                                    title="Center",
                                                ),
                                                dbc.Button(
                                                    html.I(className="fas fa-chevron-right"),
                                                    id="btn-nudge-p10",
                                                    color="light",
                                                    size="sm",
                                                    title="Forward 10%",
                                                ),
                                            ],
                                            size="sm",
                                            className="w-100",
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Action",
                                            className="form-label section-label mb-1",
                                        ),
                                        dbc.Button(
                                            [
                                                html.I(className="fas fa-play me-1"),
                                                "Apply Filter",
                                            ],
                                            id="filter-btn-apply",
                                            color="primary",
                                            size="sm",
                                            className="w-100 apply-btn",
                                        ),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-2",
                        ),

                        # Hidden carriers — IDs preserved for callbacks
                        # that still read them as State, but no visible
                        # widgets.  ``start-position-slider`` is driven
                        # by the timeline click handler.
                        html.Div(
                            [
                                dcc.Slider(
                                    id="start-position-slider",
                                    min=0, max=100, step=1, value=0,
                                    marks={},
                                ),
                                dcc.Store(id="store-picked-segment", data=0),
                                dbc.Select(
                                    id="filter-signal-source",
                                    options=[{"label": "Original", "value": "original"}],
                                    value="original",
                                ),
                                dbc.Input(id="filter-application-count", type="number", value=1),
                                dbc.Select(
                                    id="filter-signal-type-select",
                                    options=[
                                        {"label": "PPG", "value": "PPG"},
                                        {"label": "ECG", "value": "ECG"},
                                        {"label": "Other", "value": "Other"},
                                    ],
                                    value="PPG",
                                ),
                                dbc.Checklist(
                                    id="filter-quality-options",
                                    options=[{"label": "SNR", "value": "snr"}],
                                    value=["snr"],
                                ),
                            ],
                            style={"display": "none"},
                        ),
                    ]
                ),
                className="mb-3 shadow-sm border-0",
            ),
            # Hidden carrier for the legacy saved-banner ID — kept alive
            # so the apply callback's Output list resolves; the banner
            # itself was removed because it was pure information noise.
            html.Div(id="filter-saved-banner", style={"display": "none"}),
            # Filter chain — optional ordered stages applied on Apply.
            # Wrapped in an accordion so the panel only takes one line
            # of vertical space when the user just wants to see what's
            # queued (the common case after they've built their chain).
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.Div(
                                id="filter-chain-list",
                                children=html.Small(
                                    "Chain is empty. Apply runs the single filter "
                                    "configured below; click + Add as stage to queue more.",
                                    className="text-muted",
                                ),
                                className="mb-2",
                            ),
                            html.Div(
                                id="filter-chain-panel-preview",
                                className="mb-2 small text-muted",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        [html.I(className="fas fa-plus me-1"), "Add as stage"],
                                        id="filter-chain-add",
                                        color="primary",
                                        outline=True,
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        "Clear",
                                        id="filter-chain-clear",
                                        color="secondary",
                                        outline=True,
                                        size="sm",
                                    ),
                                ]
                            ),
                            dcc.Store(id="filter-chain-store", data=[]),
                        ],
                        title="Filter chain  ·  optional, apply several filters in sequence",
                        item_id="filter-chain",
                    ),
                ],
                start_collapsed=True,
                className="mb-3 shadow-sm border-0",
            ),
            # Segment-quality controls: collapsed accordion (minimalist).
            # Auto-populates on page load from the uploaded recording; the
            # green/red timeline at the top is its read-out.
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Segment length:", className="form-label small"),
                                            dbc.Select(
                                                id="filter-segment-length",
                                                options=[
                                                    {"label": "5 s", "value": 5},
                                                    {"label": "10 s", "value": 10},
                                                    {"label": "15 s", "value": 15},
                                                    {"label": "30 s", "value": 30},
                                                    {"label": "60 s", "value": 60},
                                                ],
                                                value=30,
                                                size="sm",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Overlap:", className="form-label small"),
                                            dbc.Select(
                                                id="filter-segment-overlap",
                                                options=[
                                                    {"label": "0%", "value": 0},
                                                    {"label": "25%", "value": 25},
                                                    {"label": "50%", "value": 50},
                                                    {"label": "75%", "value": 75},
                                                ],
                                                value=0,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Mode:", className="form-label small"),
                                            dbc.RadioItems(
                                                id="filter-segment-mode",
                                                options=[
                                                    {"label": " Auto-tune", "value": "tune"},
                                                    {"label": " Quantile", "value": "quantile"},
                                                    {"label": " Manual", "value": "manual"},
                                                ],
                                                value="tune",
                                                inline=True,
                                            ),
                                        ],
                                        md=7,
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Joint accept target:", className="form-label small"),
                                                    dcc.Slider(
                                                        id="filter-segment-tune-slider",
                                                        min=0.5, max=0.99, step=0.01,
                                                        value=0.90,
                                                        marks={
                                                            0.5: "50%", 0.7: "70%",
                                                            0.85: "85%", 0.9: "90%",
                                                            0.95: "95%",
                                                        },
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                    ),
                                                ],
                                                id="filter-segment-tune-row",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Per-rule trim (each tail):", className="form-label small"),
                                                    dcc.Slider(
                                                        id="filter-segment-quantile-slider",
                                                        min=0.0, max=0.25, step=0.005,
                                                        value=0.05,
                                                        marks={
                                                            0.0: "p0", 0.01: "p1",
                                                            0.025: "p2.5", 0.05: "p5",
                                                            0.10: "p10", 0.25: "p25",
                                                        },
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                    ),
                                                ],
                                                id="filter-segment-quantile-row",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        md=12,
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("SQIs:", className="form-label small mb-0 me-2 d-inline"),
                                                    html.Span(id="filter-segment-rules-summary", className="small text-muted"),
                                                ]
                                            ),
                                            dbc.Checklist(
                                                id="filter-segment-rules-checklist",
                                                options=[], value=[], inline=True, switch=False,
                                            ),
                                            html.Div(id="filter-segment-rules-skipped", className="small text-muted"),
                                        ],
                                        md=12,
                                    ),
                                ],
                            ),
                        ],
                        title="Segment-quality tuning  ·  thresholds, SQIs, segmentation",
                        item_id="segment-quality-tuning",
                    )
                ],
                start_collapsed=True,
                className="mb-3 shadow-sm border-0",
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
                                            html.Span(
                                                "Filter configuration",
                                                className="fw-semibold small text-uppercase text-muted",
                                            ),
                                        ],
                                        className="bg-white border-0 pb-1",
                                    ),
                                    dbc.CardBody(
                                        className="filter-config-body",
                                        children=[
                                            # Preprocessing Options
                                            html.Div(
                                                "Preprocessing",
                                                className="config-section-title",
                                            ),
                                            dbc.Checklist(
                                                id="detrend-option",
                                                options=[
                                                    {
                                                        "label": "Detrending",
                                                        "value": "detrend",
                                                    }
                                                ],
                                                value=[],  # Default to unchecked
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
                                                                        "Low cutoff (Hz)",
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
                                                                        "High cutoff (Hz)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="filter-high-freq-advanced",
                                                                        type="number",
                                                                        value=40,
                                                                        min=0,
                                                                        step=0.1,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-1",
                                                    ),
                                                    html.Small(
                                                        [
                                                            html.Strong("Bandpass: "),
                                                            "keeps low–high.  ",
                                                            html.Strong("Low pass: "),
                                                            "uses High cutoff.  ",
                                                            html.Strong("High pass: "),
                                                            "uses Low cutoff.",
                                                        ],
                                                        className="text-muted d-block mb-2",
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
                                                    # The legacy "Additional Filters"
                                                    # subsection (Savgol/MovAvg/Gaussian)
                                                    # used to live here.  Promoted to a
                                                    # top-level "Smoothing" family - see
                                                    # smoothing-filter-params below.
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
                                                    # Method Selection
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
                                                                                "label": "Kalman Filter",
                                                                                "value": "kalman",
                                                                            },
                                                                            {
                                                                                "label": "Optimization-Based",
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
                                                                                "label": "Attention-Based",
                                                                                "value": "attention",
                                                                            },
                                                                            {
                                                                                "label": "Adaptive (LMS)",
                                                                                "value": "adaptive",
                                                                            },
                                                                        ],
                                                                        value="kalman",
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                width=12,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    # Kalman Filter Parameters
                                                    html.Div(
                                                        [
                                                            html.H6(
                                                                "Kalman Parameters",
                                                                className="mb-2 text-primary",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "R (Measurement Noise):",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="kalman-r",
                                                                                type="number",
                                                                                value=1.0,
                                                                                min=0.001,
                                                                                max=10.0,
                                                                                step=0.01,
                                                                                size="sm",
                                                                            ),
                                                                            html.Small(
                                                                                "Lower = trust measurements more",
                                                                                className="text-muted",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Q (Process Noise):",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="kalman-q",
                                                                                type="number",
                                                                                value=1.0,
                                                                                min=0.001,
                                                                                max=10.0,
                                                                                step=0.01,
                                                                                size="sm",
                                                                            ),
                                                                            html.Small(
                                                                                "Lower = trust model more",
                                                                                className="text-muted",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        id="kalman-params",
                                                    ),
                                                    # Optimization-Based Parameters
                                                    html.Div(
                                                        [
                                                            html.H6(
                                                                "Optimization Parameters",
                                                                className="mb-2 text-primary",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Loss Function:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="optimization-loss-type",
                                                                                options=[
                                                                                    {
                                                                                        "label": "MSE (Mean Squared Error)",
                                                                                        "value": "mse",
                                                                                    },
                                                                                    {
                                                                                        "label": "MAE (Mean Absolute Error)",
                                                                                        "value": "mae",
                                                                                    },
                                                                                    {
                                                                                        "label": "Huber Loss",
                                                                                        "value": "huber",
                                                                                    },
                                                                                    {
                                                                                        "label": "Smooth L1",
                                                                                        "value": "smooth_l1",
                                                                                    },
                                                                                    {
                                                                                        "label": "Log-Cosh",
                                                                                        "value": "log_cosh",
                                                                                    },
                                                                                    {
                                                                                        "label": "Quantile Loss",
                                                                                        "value": "quantile",
                                                                                    },
                                                                                ],
                                                                                value="mse",
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Initial Guess:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="optimization-initial-guess",
                                                                                type="number",
                                                                                value=0.0,
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
                                                                                "Learning Rate:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="optimization-learning-rate",
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
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Iterations:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="optimization-iterations",
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
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        id="optimization-params",
                                                        style={"display": "none"},
                                                    ),
                                                    # Gradient Descent Parameters
                                                    html.Div(
                                                        [
                                                            html.H6(
                                                                "Gradient Descent Parameters",
                                                                className="mb-2 text-primary",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Learning Rate:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="gradient-learning-rate",
                                                                                type="number",
                                                                                value=0.01,
                                                                                min=0.001,
                                                                                max=1.0,
                                                                                step=0.001,
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Iterations:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="gradient-iterations",
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
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        id="gradient-params",
                                                        style={"display": "none"},
                                                    ),
                                                    # Convolution Parameters
                                                    html.Div(
                                                        [
                                                            html.H6(
                                                                "Convolution Parameters",
                                                                className="mb-2 text-primary",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Kernel Type:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="convolution-kernel-type",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Smoothing",
                                                                                        "value": "smoothing",
                                                                                    },
                                                                                    {
                                                                                        "label": "Sharpening",
                                                                                        "value": "sharpening",
                                                                                    },
                                                                                    {
                                                                                        "label": "Edge Detection",
                                                                                        "value": "edge_detection",
                                                                                    },
                                                                                ],
                                                                                value="smoothing",
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Kernel Size:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="convolution-kernel-size",
                                                                                type="number",
                                                                                value=3,
                                                                                min=3,
                                                                                max=15,
                                                                                step=2,
                                                                                size="sm",
                                                                            ),
                                                                            html.Small(
                                                                                "Must be odd number",
                                                                                className="text-muted",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        id="convolution-params",
                                                        style={"display": "none"},
                                                    ),
                                                    # Attention-Based Parameters
                                                    html.Div(
                                                        [
                                                            html.H6(
                                                                "Attention Parameters",
                                                                className="mb-2 text-primary",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Attention Type:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="attention-type",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Uniform",
                                                                                        "value": "uniform",
                                                                                    },
                                                                                    {
                                                                                        "label": "Linear",
                                                                                        "value": "linear",
                                                                                    },
                                                                                    {
                                                                                        "label": "Gaussian",
                                                                                        "value": "gaussian",
                                                                                    },
                                                                                    {
                                                                                        "label": "Exponential",
                                                                                        "value": "exponential",
                                                                                    },
                                                                                ],
                                                                                value="uniform",
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Window Size:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="attention-size",
                                                                                type="number",
                                                                                value=5,
                                                                                min=3,
                                                                                max=21,
                                                                                step=2,
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                            # Gaussian-specific parameter
                                                            html.Div(
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            [
                                                                                html.Label(
                                                                                    "Sigma (σ):",
                                                                                    className="form-label",
                                                                                ),
                                                                                dbc.Input(
                                                                                    id="attention-sigma",
                                                                                    type="number",
                                                                                    value=1.0,
                                                                                    min=0.1,
                                                                                    max=5.0,
                                                                                    step=0.1,
                                                                                    size="sm",
                                                                                ),
                                                                            ],
                                                                            width=6,
                                                                        ),
                                                                    ],
                                                                    className="mb-2",
                                                                ),
                                                                id="attention-gaussian-params",
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                            # Linear/Exponential-specific parameters
                                                            html.Div(
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            [
                                                                                html.Label(
                                                                                    "Direction:",
                                                                                    className="form-label",
                                                                                ),
                                                                                dbc.Select(
                                                                                    id="attention-ascending",
                                                                                    options=[
                                                                                        {
                                                                                            "label": "Ascending",
                                                                                            "value": "true",
                                                                                        },
                                                                                        {
                                                                                            "label": "Descending",
                                                                                            "value": "false",
                                                                                        },
                                                                                    ],
                                                                                    value="true",
                                                                                    size="sm",
                                                                                ),
                                                                            ],
                                                                            width=6,
                                                                        ),
                                                                    ],
                                                                    className="mb-2",
                                                                ),
                                                                id="attention-linear-params",
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                            # Exponential-specific parameter
                                                            html.Div(
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            [
                                                                                html.Label(
                                                                                    "Base:",
                                                                                    className="form-label",
                                                                                ),
                                                                                dbc.Input(
                                                                                    id="attention-base",
                                                                                    type="number",
                                                                                    value=2.0,
                                                                                    min=1.1,
                                                                                    max=10.0,
                                                                                    step=0.1,
                                                                                    size="sm",
                                                                                ),
                                                                            ],
                                                                            width=6,
                                                                        ),
                                                                    ],
                                                                    className="mb-2",
                                                                ),
                                                                id="attention-exponential-params",
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                        ],
                                                        id="attention-params",
                                                        style={"display": "none"},
                                                    ),
                                                    # Adaptive Filter Parameters
                                                    html.Div(
                                                        [
                                                            html.H6(
                                                                "Adaptive (LMS) Parameters",
                                                                className="mb-2 text-primary",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Step Size (μ):",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="adaptive-mu",
                                                                                type="number",
                                                                                value=0.01,
                                                                                min=0.001,
                                                                                max=1.0,
                                                                                step=0.001,
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Filter Order:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="adaptive-order",
                                                                                type="number",
                                                                                value=4,
                                                                                min=2,
                                                                                max=20,
                                                                                step=1,
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        id="adaptive-params",
                                                        style={"display": "none"},
                                                    ),
                                                ],
                                                id="advanced-filter-params",
                                                style={"display": "none"},
                                            ),
                                            # Smoothing Filter Parameters
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Smoothing",
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
                                                                        id="smoothing-method-select",
                                                                        options=[
                                                                            {
                                                                                "label": "Savitzky-Golay",
                                                                                "value": "savgol",
                                                                            },
                                                                            {
                                                                                "label": "Moving Average",
                                                                                "value": "moving_avg",
                                                                            },
                                                                            {
                                                                                "label": "Gaussian",
                                                                                "value": "gaussian",
                                                                            },
                                                                        ],
                                                                        value="savgol",
                                                                    ),
                                                                ],
                                                                width=12,
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    # Savitzky-Golay params
                                                    html.Div(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Window length (odd):",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="savgol-window",
                                                                                type="number",
                                                                                value=11,
                                                                                min=3,
                                                                                step=2,
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Polyorder:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="savgol-polyorder",
                                                                                type="number",
                                                                                value=2,
                                                                                min=1,
                                                                                step=1,
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                        id="smoothing-savgol-params",
                                                    ),
                                                    # Moving average params
                                                    html.Div(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Window size:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="moving-avg-window",
                                                                                type="number",
                                                                                value=5,
                                                                                min=2,
                                                                                step=1,
                                                                            ),
                                                                        ],
                                                                        width=12,
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                        id="smoothing-movavg-params",
                                                        style={"display": "none"},
                                                    ),
                                                    # Gaussian params
                                                    html.Div(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Sigma:",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="gaussian-sigma",
                                                                                type="number",
                                                                                value=1.0,
                                                                                min=0.1,
                                                                                step=0.1,
                                                                            ),
                                                                        ],
                                                                        width=12,
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                        id="smoothing-gaussian-params",
                                                        style={"display": "none"},
                                                    ),
                                                ],
                                                id="smoothing-filter-params",
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
                                            # Multi-modal filtering UI is
                                            # hidden in the slim panel - the
                                            # underlying ``apply_multi_modal_filtering``
                                            # implementation remains in the
                                            # callback for future use.  IDs
                                            # are kept alive (default values
                                            # = the historic defaults) so the
                                            # callback's State reads still
                                            # find them.
                                            html.Div(
                                                [
                                                    dbc.Select(
                                                        id="reference-signal",
                                                        options=[
                                                            {"label": "None", "value": "none"},
                                                        ],
                                                        value="none",
                                                    ),
                                                    dbc.Select(
                                                        id="fusion-method",
                                                        options=[
                                                            {"label": "Weighted", "value": "weighted"},
                                                        ],
                                                        value="weighted",
                                                    ),
                                                ],
                                                style={"display": "none"},
                                            ),
                                        ]
                                    ),
                                ],
                                className="h-100 shadow-sm border-0",
                            )
                        ],
                        md=3,
                    ),
                    # Right Panel — Comparison plot (the primary
                    # surface).  Shows the picked segment, original vs
                    # filtered, with critical-point markers.  Replaces
                    # the old triplet of Original / Filtered / Comparison
                    # cards — all redundant with this single overlay.
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Span(
                                                "Filter comparison",
                                                className="fw-semibold small text-uppercase text-muted",
                                            ),
                                            html.Span(
                                                "original vs filtered, critical points overlaid",
                                                className="text-muted small ms-2",
                                            ),
                                        ],
                                        className="bg-white border-0 pb-1",
                                    ),
                                    dbc.CardBody(
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="filter-comparison-plot",
                                                className="comparison-plot-graph",
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
                                    ),
                                ],
                                className="shadow-sm border-0",
                            ),
                            # Hidden carriers for dead-but-still-wired Outputs.
                            html.Div(
                                [
                                    dcc.Graph(id="filter-original-plot"),
                                    dcc.Graph(id="filter-filtered-plot"),
                                    html.Div(id="filter-quality-metrics"),
                                    dcc.Graph(id="filter-quality-plots"),
                                ],
                                style={"display": "none"},
                            ),
                        ],
                        md=9,
                    ),
                ]
            ),
            # Hidden carriers for the IDs the segment-quality callback
            # still writes to.  The visible copies are the
            # ``-top`` widgets in the controller bar.  A small bridge
            # callback copies the hidden Output into the visible widget
            # each time the segment-quality engine re-renders.
            html.Div(
                [
                    html.Div(id="filter-segment-headline"),
                    dcc.Graph(
                        id="filter-segment-timeline",
                        config={"displayModeBar": False, "staticPlot": True},
                    ),
                    dbc.Checkbox(
                        id="filter-segment-auto-sync",
                        value=True,
                    ),
                ],
                style={"display": "none"},
            ),
            # Stores for data management
            dcc.Store(id="store-filtering-data"),
            dcc.Store(id="store-filter-comparison"),
            dcc.Store(id="store-filter-quality-metrics"),
            dcc.Store(id="store-filtered-signal"),  # For export
            # Segment-quality shared stores (read by Quality page accordion).
            dcc.Store(id="store-segment-decisions"),
            dcc.Store(id="store-segment-sqis"),
            dcc.Store(id="store-segment-milestones"),
            # The whole-signal filtered-through-the-chain payload the
            # segment-quality compute produced.  Per-segment waveform
            # inspection slices this rather than ``store-filtered-signal``
            # (which is the displayed-window slice, not the recording).
            dcc.Store(id="store-segment-filtered-signal"),
        ],
        className="signal-filtering-page",
    )
