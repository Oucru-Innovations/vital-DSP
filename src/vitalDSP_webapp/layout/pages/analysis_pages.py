"""
Analysis pages layout for vitalDSP webapp.

This module provides layouts for various analysis pages.
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


# NOTE: physiological_layout has been moved to physiological_page.py
# This keeps the codebase organized with each major analysis page in its own file.
# Import from vitalDSP_webapp.layout.pages.physiological_page instead.


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
                                                                        style={
                                                                            "font-size": "12px",
                                                                            "color": "gray",
                                                                            "margin-top": "10px",
                                                                        },
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
