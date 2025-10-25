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
                                    # Modern Time Range Controls
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Start Position (%)",
                                                className="form-label mb-1",
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
                                                className="mb-2",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    # Duration Selection
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Duration",
                                                className="form-label mb-1",
                                            ),
                                            dbc.Select(
                                                id="duration-select",
                                                options=[
                                                    {"label": "30s", "value": 30},
                                                    {"label": "1min", "value": 60},
                                                    {"label": "2min", "value": 120},
                                                    {"label": "5min", "value": 300},
                                                ],
                                                value=60,  # Default to 1 minute
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
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
                                                        "-10%",
                                                        id="btn-nudge-m10",
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
                                                        "+10%",
                                                        id="btn-nudge-p10",
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
