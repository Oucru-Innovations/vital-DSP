"""
Physiological Analysis page layout for vitalDSP webapp.

This module provides the layout for comprehensive physiological feature extraction and analysis,
including HRV analysis, morphological features, beat-to-beat analysis, and advanced signal processing.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


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
                                            # Start Position Slider - NEW PATTERN
                                            html.Label(
                                                "Start Position (%)",
                                                className="form-label fw-bold small",
                                            ),
                                            html.Small(
                                                "Position in data (0% = start, 100% = end)",
                                                className="text-muted d-block mb-1",
                                            ),
                                            dcc.Slider(
                                                id="physio-start-position-slider",
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
                                            # Duration Dropdown - NEW PATTERN
                                            html.Label(
                                                "Duration",
                                                className="form-label fw-bold small",
                                            ),
                                            html.Small(
                                                "Analysis window size",
                                                className="text-muted d-block mb-1",
                                            ),
                                            dbc.Select(
                                                id="physio-duration-select",
                                                options=[
                                                    {
                                                        "label": "10 seconds",
                                                        "value": 10,
                                                    },
                                                    {
                                                        "label": "30 seconds",
                                                        "value": 30,
                                                    },
                                                    {"label": "1 minute", "value": 60},
                                                    {
                                                        "label": "2 minutes",
                                                        "value": 120,
                                                    },
                                                    {
                                                        "label": "5 minutes",
                                                        "value": 300,
                                                    },
                                                    {
                                                        "label": "10 minutes",
                                                        "value": 600,
                                                    },
                                                ],
                                                value=60,  # Default to 1 minute
                                                className="mb-3",
                                            ),
                                            # Quick Navigation Buttons
                                            html.Label(
                                                "Quick Navigation",
                                                className="form-label fw-bold small",
                                            ),
                                            html.Small(
                                                "Adjust start position",
                                                className="text-muted d-block mb-1",
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "⏪ -10%",
                                                        id="physio-btn-nudge-m10",
                                                        size="sm",
                                                        color="secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "⏪ -5%",
                                                        id="physio-btn-nudge-m1",
                                                        size="sm",
                                                        color="secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+5% ⏩",
                                                        id="physio-btn-nudge-p1",
                                                        size="sm",
                                                        color="secondary",
                                                        className="me-1",
                                                    ),
                                                    dbc.Button(
                                                        "+10% ⏩",
                                                        id="physio-btn-nudge-p10",
                                                        size="sm",
                                                        color="secondary",
                                                    ),
                                                ],
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
