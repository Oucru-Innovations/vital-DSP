"""
Pipeline visualization page layout.

This page provides a comprehensive interface for the 8-stage vitalDSP processing pipeline,
allowing users to visualize, configure, and monitor multi-stage signal processing.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from vitalDSP_webapp.layout.common import (
    create_step_progress_indicator,
    create_progress_bar,
    create_interval_component,
)


def pipeline_layout():
    """
    Create the pipeline visualization page layout.

    Returns
    -------
    html.Div
        The complete pipeline page layout
    """
    # Define the 8 pipeline stages
    pipeline_stages = [
        "Data Ingestion",
        "Quality Screening",
        "Parallel Processing",
        "Quality Validation",
        "Segmentation",
        "Feature Extraction",
        "Intelligent Output",
        "Output Package",
    ]

    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H2(
                        [
                            html.I(className="fas fa-project-diagram mr-2"),
                            "8-Stage Processing Pipeline",
                        ],
                        className="mb-3",
                    ),
                    html.P(
                        "Visualize and monitor the complete signal processing pipeline with "
                        "quality screening, parallel processing paths, and intelligent output selection.",
                        className="text-muted",
                    ),
                ],
                className="mb-4",
            ),
            # Main content area with 3-panel layout
            dbc.Row(
                [
                    # Left Panel - Configuration
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Pipeline Configuration", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Signal Type Selection
                                            html.Label("Signal Type"),
                                            dcc.Dropdown(
                                                id="pipeline-signal-type",
                                                options=[
                                                    {"label": "ECG", "value": "ecg"},
                                                    {"label": "PPG", "value": "ppg"},
                                                    {"label": "EEG", "value": "eeg"},
                                                    {
                                                        "label": "Respiratory",
                                                        "value": "respiratory",
                                                    },
                                                    {
                                                        "label": "Generic",
                                                        "value": "generic",
                                                    },
                                                ],
                                                value="ecg",
                                                className="mb-3",
                                            ),
                                            # Processing Path Selection
                                            html.Label("Processing Paths"),
                                            dcc.Checklist(
                                                id="pipeline-paths",
                                                options=[
                                                    {
                                                        "label": " RAW (No filtering)",
                                                        "value": "raw",
                                                    },
                                                    {
                                                        "label": " FILTERED (Bandpass filtering)",
                                                        "value": "filtered",
                                                    },
                                                    {
                                                        "label": " PREPROCESSED (Filtered + Artifact Removal)",
                                                        "value": "preprocessed",
                                                    },
                                                ],
                                                value=["filtered", "preprocessed"],
                                                className="mb-3",
                                                inline=False,
                                            ),
                                            html.Hr(),
                                            # Stage-Specific Parameters (Collapsible Accordion)
                                            html.H6(
                                                "Stage Parameters", className="mb-3"
                                            ),
                                            dbc.Accordion(
                                                [
                                                    # Stage 2: Quality Screening
                                                    dbc.AccordionItem(
                                                        [
                                                            html.Label(
                                                                "Enable Quality Screening"
                                                            ),
                                                            dbc.Switch(
                                                                id="pipeline-enable-quality",
                                                                value=True,
                                                                className="mb-3",
                                                            ),
                                                            html.H6(
                                                                "SQI Methods Selection",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-sqi-methods",
                                                                options=[
                                                                    {
                                                                        "label": " Amplitude Variability SQI",
                                                                        "value": "amplitude_variability",
                                                                    },
                                                                    {
                                                                        "label": " Baseline Wander SQI",
                                                                        "value": "baseline_wander",
                                                                    },
                                                                    {
                                                                        "label": " SNR SQI",
                                                                        "value": "snr",
                                                                    },
                                                                    {
                                                                        "label": " Zero-Crossing SQI",
                                                                        "value": "zero_crossing",
                                                                    },
                                                                    {
                                                                        "label": " Entropy SQI",
                                                                        "value": "entropy",
                                                                    },
                                                                    {
                                                                        "label": " Kurtosis SQI",
                                                                        "value": "kurtosis",
                                                                    },
                                                                    {
                                                                        "label": " Skewness SQI",
                                                                        "value": "skewness",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "amplitude_variability",
                                                                    "baseline_wander",
                                                                    "snr",
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.H6(
                                                                "SQI Parameters",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            html.Label(
                                                                "SQI Window Size (seconds)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-sqi-window",
                                                                type="number",
                                                                value=5,
                                                                min=1,
                                                                max=30,
                                                                step=1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "SQI Step Size (seconds)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-sqi-step",
                                                                type="number",
                                                                value=2.5,
                                                                min=0.5,
                                                                max=15,
                                                                step=0.5,
                                                                className="mb-2",
                                                            ),
                                                            html.Label("SQI Threshold"),
                                                            dbc.Input(
                                                                id="pipeline-quality-threshold",
                                                                type="number",
                                                                value=0.7,
                                                                min=0,
                                                                max=1,
                                                                step=0.05,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "SQI Scaling Method"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="pipeline-sqi-scale",
                                                                options=[
                                                                    {
                                                                        "label": "Z-Score",
                                                                        "value": "zscore",
                                                                    },
                                                                    {
                                                                        "label": "IQR (Interquartile Range)",
                                                                        "value": "iqr",
                                                                    },
                                                                    {
                                                                        "label": "Min-Max",
                                                                        "value": "minmax",
                                                                    },
                                                                ],
                                                                value="zscore",
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 2: Quality Screening",
                                                    ),
                                                    # Stage 3: Filtering
                                                    dbc.AccordionItem(
                                                        [
                                                            # Filter Mode Toggle
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardBody(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Label(
                                                                                        "Filter Mode",
                                                                                        style={
                                                                                            "fontWeight": "bold"
                                                                                        },
                                                                                    ),
                                                                                    dbc.RadioItems(
                                                                                        id="pipeline-filter-mode",
                                                                                        options=[
                                                                                            {
                                                                                                "label": " Basic (Butterworth only)",
                                                                                                "value": "basic",
                                                                                            },
                                                                                            {
                                                                                                "label": " Advanced (All filter types)",
                                                                                                "value": "advanced",
                                                                                            },
                                                                                        ],
                                                                                        value="basic",
                                                                                        className="mb-2",
                                                                                        inline=False,
                                                                                    ),
                                                                                ],
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ],
                                                                color="light",
                                                                className="mb-3",
                                                            ),
                                                            # Basic Filter Settings (always visible)
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Basic Filter Settings",
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Low Cutoff Frequency (Hz)"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-filter-lowcut",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0.1,
                                                                        max=10,
                                                                        step=0.1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "High Cutoff Frequency (Hz)"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-filter-highcut",
                                                                        type="number",
                                                                        value=40,
                                                                        min=1,
                                                                        max=100,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Filter Order"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-filter-order",
                                                                        type="number",
                                                                        value=4,
                                                                        min=2,
                                                                        max=10,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                ],
                                                                id="basic-filter-params",
                                                            ),
                                                            # Advanced Filter Settings (shown only in advanced mode)
                                                            html.Div(
                                                                [
                                                                    html.Hr(),
                                                                    html.H6(
                                                                        "Advanced Filter Settings",
                                                                        className="mt-3 mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Filter Type"
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="pipeline-filter-type",
                                                                        options=[
                                                                            {
                                                                                "label": "Butterworth",
                                                                                "value": "butter",
                                                                            },
                                                                            {
                                                                                "label": "Chebyshev Type I",
                                                                                "value": "cheby1",
                                                                            },
                                                                            {
                                                                                "label": "Chebyshev Type II",
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
                                                                        className="mb-3",
                                                                    ),
                                                                    html.Label(
                                                                        "Passband Ripple (dB) - Chebyshev/Elliptic"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-filter-rp",
                                                                        type="number",
                                                                        value=1,
                                                                        min=0.1,
                                                                        max=5,
                                                                        step=0.1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Stopband Attenuation (dB) - Elliptic"
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-filter-rs",
                                                                        type="number",
                                                                        value=40,
                                                                        min=20,
                                                                        max=80,
                                                                        step=5,
                                                                        className="mb-2",
                                                                    ),
                                                                ],
                                                                id="advanced-filter-params",
                                                                style={
                                                                    "display": "none"
                                                                },  # Hidden by default
                                                            ),
                                                            html.H6(
                                                                "Artifact Removal",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            html.Label(
                                                                "Artifact Removal Method"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="pipeline-artifact-method",
                                                                options=[
                                                                    {
                                                                        "label": "Baseline Correction (High-pass)",
                                                                        "value": "baseline_correction",
                                                                    },
                                                                    {
                                                                        "label": "Mean Subtraction",
                                                                        "value": "mean_subtraction",
                                                                    },
                                                                    {
                                                                        "label": "Median Filter",
                                                                        "value": "median_filter",
                                                                    },
                                                                    {
                                                                        "label": "Wavelet Denoising",
                                                                        "value": "wavelet",
                                                                    },
                                                                ],
                                                                value="baseline_correction",
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Baseline Correction Cutoff (Hz)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-baseline-cutoff",
                                                                type="number",
                                                                value=0.5,
                                                                min=0.1,
                                                                max=5,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Median Filter Kernel Size"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-median-kernel",
                                                                type="number",
                                                                value=5,
                                                                min=3,
                                                                max=15,
                                                                step=2,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Wavelet Type (for wavelet denoising)"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="pipeline-wavelet-type",
                                                                options=[
                                                                    {
                                                                        "label": "Daubechies (db)",
                                                                        "value": "db",
                                                                    },
                                                                    {
                                                                        "label": "Haar",
                                                                        "value": "haar",
                                                                    },
                                                                    {
                                                                        "label": "Symlet (sym)",
                                                                        "value": "sym",
                                                                    },
                                                                    {
                                                                        "label": "Coiflet (coif)",
                                                                        "value": "coif",
                                                                    },
                                                                ],
                                                                value="db",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Wavelet Order"),
                                                            dbc.Input(
                                                                id="pipeline-wavelet-order",
                                                                type="number",
                                                                value=4,
                                                                min=1,
                                                                max=10,
                                                                step=1,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 3: Filtering & Artifact Removal",
                                                    ),
                                                    # Stage 4: Quality Validation
                                                    dbc.AccordionItem(
                                                        [
                                                            html.H6(
                                                                "SQI-Based Quality Metrics",
                                                                className="mb-2",
                                                            ),
                                                            html.P(
                                                                "Select which SQI methods to use for path quality comparison:",
                                                                className="text-muted small",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-stage4-sqi-methods",
                                                                options=[
                                                                    {
                                                                        "label": " Amplitude Variability SQI",
                                                                        "value": "amplitude_variability",
                                                                    },
                                                                    {
                                                                        "label": " Baseline Wander SQI",
                                                                        "value": "baseline_wander",
                                                                    },
                                                                    {
                                                                        "label": " SNR SQI",
                                                                        "value": "snr",
                                                                    },
                                                                    {
                                                                        "label": " Zero-Crossing SQI",
                                                                        "value": "zero_crossing",
                                                                    },
                                                                    {
                                                                        "label": " Entropy SQI",
                                                                        "value": "entropy",
                                                                    },
                                                                    {
                                                                        "label": " Kurtosis SQI",
                                                                        "value": "kurtosis",
                                                                    },
                                                                    {
                                                                        "label": " Skewness SQI",
                                                                        "value": "skewness",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "snr",
                                                                    "amplitude_variability",
                                                                    "baseline_wander",
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.Hr(),
                                                            html.H6(
                                                                "Traditional Quality Metrics",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            html.Label(
                                                                "SNR Calculation Method"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="pipeline-snr-method",
                                                                options=[
                                                                    {
                                                                        "label": "Standard SNR (Signal/Noise)",
                                                                        "value": "standard",
                                                                    },
                                                                    {
                                                                        "label": "RMS-based SNR",
                                                                        "value": "rms",
                                                                    },
                                                                    {
                                                                        "label": "Peak SNR",
                                                                        "value": "peak",
                                                                    },
                                                                ],
                                                                value="standard",
                                                                className="mb-3",
                                                            ),
                                                            html.Hr(),
                                                            html.H6(
                                                                "Quality Metric Weights",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Adjust importance of each quality metric category",
                                                                className="text-muted mb-2 d-block",
                                                            ),
                                                            html.Label(
                                                                "SQI Metrics Weight",
                                                                className="mt-2",
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-sqi-metrics-weight",
                                                                type="number",
                                                                value=0.5,
                                                                min=0,
                                                                max=1,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Traditional SNR Weight",
                                                                className="mt-2",
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-snr-weight",
                                                                type="number",
                                                                value=0.25,
                                                                min=0,
                                                                max=1,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Smoothness Weight"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-smoothness-weight",
                                                                type="number",
                                                                value=0.15,
                                                                min=0,
                                                                max=1,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Artifact Level Weight"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-artifact-weight",
                                                                type="number",
                                                                value=0.1,
                                                                min=0,
                                                                max=1,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 4: Quality Validation",
                                                    ),
                                                    # Stage 5: Segmentation
                                                    dbc.AccordionItem(
                                                        [
                                                            html.H6(
                                                                "Windowing Parameters",
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Window Size (seconds)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-window-size",
                                                                type="number",
                                                                value=30,
                                                                min=5,
                                                                max=300,
                                                                step=5,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Overlap Ratio (0 = no overlap, 0.5 = 50%)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-overlap-ratio",
                                                                type="number",
                                                                value=0.5,
                                                                min=0,
                                                                max=0.9,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Minimum Segment Length (seconds)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-min-segment-length",
                                                                type="number",
                                                                value=5,
                                                                min=1,
                                                                max=60,
                                                                step=1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Apply Window Function"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="pipeline-window-function",
                                                                options=[
                                                                    {
                                                                        "label": "None (Rectangular)",
                                                                        "value": "none",
                                                                    },
                                                                    {
                                                                        "label": "Hamming",
                                                                        "value": "hamming",
                                                                    },
                                                                    {
                                                                        "label": "Hanning",
                                                                        "value": "hanning",
                                                                    },
                                                                    {
                                                                        "label": "Blackman",
                                                                        "value": "blackman",
                                                                    },
                                                                    {
                                                                        "label": "Gaussian",
                                                                        "value": "gaussian",
                                                                    },
                                                                ],
                                                                value="none",
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 5: Segmentation",
                                                    ),
                                                    # Stage 6: Feature Extraction
                                                    dbc.AccordionItem(
                                                        [
                                                            html.H6(
                                                                "Feature Categories",
                                                                className="mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-feature-types",
                                                                options=[
                                                                    {
                                                                        "label": " Time Domain Features",
                                                                        "value": "time",
                                                                    },
                                                                    {
                                                                        "label": " Frequency Domain Features",
                                                                        "value": "frequency",
                                                                    },
                                                                    {
                                                                        "label": " Nonlinear Features",
                                                                        "value": "nonlinear",
                                                                    },
                                                                    {
                                                                        "label": " Morphological Features",
                                                                        "value": "morphology",
                                                                    },
                                                                    {
                                                                        "label": " Statistical Features",
                                                                        "value": "statistical",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "time",
                                                                    "frequency",
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.H6(
                                                                "Time Domain Features",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-time-features",
                                                                options=[
                                                                    {
                                                                        "label": " Mean",
                                                                        "value": "mean",
                                                                    },
                                                                    {
                                                                        "label": " Standard Deviation",
                                                                        "value": "std",
                                                                    },
                                                                    {
                                                                        "label": " RMS (Root Mean Square)",
                                                                        "value": "rms",
                                                                    },
                                                                    {
                                                                        "label": " Peak-to-Peak",
                                                                        "value": "ptp",
                                                                    },
                                                                    {
                                                                        "label": " Zero Crossings",
                                                                        "value": "zero_crossings",
                                                                    },
                                                                    {
                                                                        "label": " Mean Absolute Deviation",
                                                                        "value": "mad",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "mean",
                                                                    "std",
                                                                    "rms",
                                                                    "ptp",
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.H6(
                                                                "Frequency Domain Features",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-frequency-features",
                                                                options=[
                                                                    {
                                                                        "label": " Spectral Centroid",
                                                                        "value": "spectral_centroid",
                                                                    },
                                                                    {
                                                                        "label": " Dominant Frequency",
                                                                        "value": "dominant_freq",
                                                                    },
                                                                    {
                                                                        "label": " Spectral Entropy",
                                                                        "value": "spectral_entropy",
                                                                    },
                                                                    {
                                                                        "label": " Band Power",
                                                                        "value": "band_power",
                                                                    },
                                                                    {
                                                                        "label": " Peak Frequency",
                                                                        "value": "peak_freq",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "spectral_centroid",
                                                                    "dominant_freq",
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.H6(
                                                                "Nonlinear Features",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-nonlinear-features",
                                                                options=[
                                                                    {
                                                                        "label": " Sample Entropy",
                                                                        "value": "sample_entropy",
                                                                    },
                                                                    {
                                                                        "label": " Approximate Entropy",
                                                                        "value": "approx_entropy",
                                                                    },
                                                                    {
                                                                        "label": " Fractal Dimension",
                                                                        "value": "fractal_dim",
                                                                    },
                                                                    {
                                                                        "label": " Lyapunov Exponent",
                                                                        "value": "lyapunov",
                                                                    },
                                                                    {
                                                                        "label": " DFA (Detrended Fluctuation Analysis)",
                                                                        "value": "dfa",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "sample_entropy"
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.H6(
                                                                "Statistical Features",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-statistical-features",
                                                                options=[
                                                                    {
                                                                        "label": " Skewness",
                                                                        "value": "skewness",
                                                                    },
                                                                    {
                                                                        "label": " Kurtosis",
                                                                        "value": "kurtosis",
                                                                    },
                                                                    {
                                                                        "label": " Variance",
                                                                        "value": "variance",
                                                                    },
                                                                    {
                                                                        "label": " Median",
                                                                        "value": "median",
                                                                    },
                                                                    {
                                                                        "label": " IQR (Interquartile Range)",
                                                                        "value": "iqr",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "skewness",
                                                                    "kurtosis",
                                                                ],
                                                                className="mb-2",
                                                                inline=False,
                                                            ),
                                                        ],
                                                        title="Stage 6: Feature Extraction",
                                                    ),
                                                    # Stage 7: Intelligent Output
                                                    dbc.AccordionItem(
                                                        [
                                                            html.H6(
                                                                "Path Selection Strategy",
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Selection Criterion"
                                                            ),
                                                            dcc.Dropdown(
                                                                id="pipeline-selection-criterion",
                                                                options=[
                                                                    {
                                                                        "label": "Best Overall Quality",
                                                                        "value": "best_quality",
                                                                    },
                                                                    {
                                                                        "label": "Highest SNR",
                                                                        "value": "highest_snr",
                                                                    },
                                                                    {
                                                                        "label": "Lowest Artifact Level",
                                                                        "value": "lowest_artifact",
                                                                    },
                                                                    {
                                                                        "label": "Weighted Combination",
                                                                        "value": "weighted",
                                                                    },
                                                                ],
                                                                value="best_quality",
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Confidence Threshold"
                                                            ),
                                                            html.Small(
                                                                "Minimum confidence to recommend a path",
                                                                className="text-muted mb-2",
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-confidence-threshold",
                                                                type="number",
                                                                value=0.7,
                                                                min=0.5,
                                                                max=1.0,
                                                                step=0.05,
                                                                className="mb-2",
                                                            ),
                                                            html.Label(
                                                                "Generate Recommendations"
                                                            ),
                                                            dbc.Switch(
                                                                id="pipeline-generate-recommendations",
                                                                value=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 7: Intelligent Output",
                                                    ),
                                                    # Stage 8: Output Package
                                                    dbc.AccordionItem(
                                                        [
                                                            html.H6(
                                                                "Output Configuration",
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Output Format"),
                                                            dcc.Checklist(
                                                                id="pipeline-output-formats",
                                                                options=[
                                                                    {
                                                                        "label": " JSON",
                                                                        "value": "json",
                                                                    },
                                                                    {
                                                                        "label": " CSV",
                                                                        "value": "csv",
                                                                    },
                                                                    {
                                                                        "label": " HDF5",
                                                                        "value": "hdf5",
                                                                    },
                                                                    {
                                                                        "label": " MAT (MATLAB)",
                                                                        "value": "mat",
                                                                    },
                                                                ],
                                                                value=["json", "csv"],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.Label(
                                                                "Include in Output"
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-output-contents",
                                                                options=[
                                                                    {
                                                                        "label": " Raw Signal",
                                                                        "value": "raw_signal",
                                                                    },
                                                                    {
                                                                        "label": " Processed Signals",
                                                                        "value": "processed_signals",
                                                                    },
                                                                    {
                                                                        "label": " Quality Metrics",
                                                                        "value": "quality_metrics",
                                                                    },
                                                                    {
                                                                        "label": " Extracted Features",
                                                                        "value": "features",
                                                                    },
                                                                    {
                                                                        "label": " Segment Data",
                                                                        "value": "segments",
                                                                    },
                                                                    {
                                                                        "label": " Processing Metadata",
                                                                        "value": "metadata",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "processed_signals",
                                                                    "quality_metrics",
                                                                    "features",
                                                                    "metadata",
                                                                ],
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.Label("Compression"),
                                                            dbc.Switch(
                                                                id="pipeline-compress-output",
                                                                value=True,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 8: Output Package",
                                                    ),
                                                ],
                                                id="pipeline-stage-params-accordion",
                                                start_collapsed=True,
                                                always_open=True,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        md=3,
                        className="mb-4",
                    ),
                    # Right Panel - Visualization and Results
                    dbc.Col(
                        [
                            # Pipeline Progress Indicator
                            create_step_progress_indicator(
                                step_id="pipeline-progress",
                                steps=pipeline_stages,
                                current_step=0,
                            ),
                            # Processing Progress Bar
                            create_progress_bar(
                                progress_id="pipeline-processing-progress",
                                label="Overall Progress",
                            ),
                            # Interval for progress updates
                            create_interval_component(
                                interval_id="pipeline-progress-interval",
                                interval_ms=500,
                                disabled=True,
                            ),
                            # Visualization Controls
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Visualization Controls", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Time Window (seconds)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-viz-window",
                                                                type="number",
                                                                value=300,
                                                                min=10,
                                                                max=3600,
                                                                step=10,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Start Time (seconds)"
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-viz-start",
                                                                type="number",
                                                                value=0,
                                                                min=0,
                                                                step=10,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Mode"),
                                                            dcc.Dropdown(
                                                                id="pipeline-viz-mode",
                                                                options=[
                                                                    {
                                                                        "label": "From Start",
                                                                        "value": "start",
                                                                    },
                                                                    {
                                                                        "label": "Random Interval",
                                                                        "value": "random",
                                                                    },
                                                                    {
                                                                        "label": "Custom Range",
                                                                        "value": "custom",
                                                                    },
                                                                ],
                                                                value="start",
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        md=4,
                                                    ),
                                                ],
                                            ),
                                            dbc.Button(
                                                "Refresh Visualizations",
                                                id="pipeline-viz-refresh-btn",
                                                color="info",
                                                size="sm",
                                                className="mt-2",
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Stage Details
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Stage Details", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="pipeline-stage-details"),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Processing Paths Comparison
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Processing Paths Comparison",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="pipeline-paths-comparison",
                                                config={"displayModeBar": True},
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Quality Screening Results
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Quality Screening Results",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="pipeline-quality-results"),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Feature Extraction Results
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Feature Extraction Summary",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="pipeline-features-summary"),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Stage-Specific Visualizations
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.H5(
                                                            "Stage Visualizations",
                                                            className="mb-0",
                                                        ),
                                                        width=8,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            [
                                                                html.I(
                                                                    className="fas fa-download mr-1"
                                                                ),
                                                                "Export Stage Data",
                                                            ],
                                                            id="pipeline-export-stage-btn",
                                                            color="success",
                                                            size="sm",
                                                            disabled=True,
                                                        ),
                                                        width=4,
                                                        className="text-right",
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Stage 1: Data Ingestion Plot
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 1: Raw Signal",
                                                        className="mb-3",
                                                    ),
                                                    dcc.Graph(
                                                        id="pipeline-stage1-plot",
                                                        config={"displayModeBar": True},
                                                    ),
                                                ],
                                                id="pipeline-stage1-container",
                                                style={"display": "none"},
                                            ),
                                            # Stage 2: SQI Metrics Plot
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 2: SQI Metrics Over Time",
                                                        className="mb-3",
                                                    ),
                                                    dcc.Graph(
                                                        id="pipeline-stage2-plot",
                                                        config={"displayModeBar": True},
                                                    ),
                                                ],
                                                id="pipeline-stage2-container",
                                                style={"display": "none"},
                                            ),
                                            # Stage 3: Processing Paths Plot (already exists, just reference)
                                            # Stage 4: Quality Validation Plot
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 4: Path Quality Comparison",
                                                        className="mb-3",
                                                    ),
                                                    dcc.Graph(
                                                        id="pipeline-stage4-plot",
                                                        config={"displayModeBar": True},
                                                    ),
                                                ],
                                                id="pipeline-stage4-container",
                                                style={"display": "none"},
                                            ),
                                            # Stage 5: Segmentation Plot
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 5: Signal Segmentation",
                                                        className="mb-3",
                                                    ),
                                                    dcc.Graph(
                                                        id="pipeline-stage5-plot",
                                                        config={"displayModeBar": True},
                                                    ),
                                                ],
                                                id="pipeline-stage5-container",
                                                style={"display": "none"},
                                            ),
                                            # Stage 6: Feature Heatmap
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 6: Feature Extraction Heatmap",
                                                        className="mb-3",
                                                    ),
                                                    dcc.Graph(
                                                        id="pipeline-stage6-plot",
                                                        config={"displayModeBar": True},
                                                    ),
                                                ],
                                                id="pipeline-stage6-container",
                                                style={"display": "none"},
                                            ),
                                            # Stage 7: Path Selection Plot
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 7: Path Selection Analysis",
                                                        className="mb-3",
                                                    ),
                                                    dcc.Graph(
                                                        id="pipeline-stage7-plot",
                                                        config={"displayModeBar": True},
                                                    ),
                                                ],
                                                id="pipeline-stage7-container",
                                                style={"display": "none"},
                                            ),
                                            # Stage 8: Pipeline Summary
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "Stage 8: Pipeline Summary",
                                                        className="mb-3",
                                                    ),
                                                    html.Div(
                                                        id="pipeline-stage8-summary"
                                                    ),
                                                ],
                                                id="pipeline-stage8-container",
                                                style={"display": "none"},
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Intelligent Output Recommendations
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Intelligent Output Recommendations",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="pipeline-output-recommendations"
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                        ],
                        md=9,
                    ),
                ],
                className="mb-4",
            ),
            # Action Buttons
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-play mr-2"),
                                            "Run Pipeline",
                                        ],
                                        id="pipeline-run-btn",
                                        color="primary",
                                        size="lg",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-stop mr-2"),
                                            "Stop",
                                        ],
                                        id="pipeline-stop-btn",
                                        color="danger",
                                        size="lg",
                                        disabled=True,
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-redo mr-2"),
                                            "Reset",
                                        ],
                                        id="pipeline-reset-btn",
                                        color="secondary",
                                        size="lg",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-download mr-2"),
                                            "Export Results",
                                        ],
                                        id="pipeline-export-btn",
                                        color="success",
                                        size="lg",
                                        disabled=True,
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-file-pdf mr-2"),
                                            "Generate Report",
                                        ],
                                        id="pipeline-report-btn",
                                        color="info",
                                        size="lg",
                                        disabled=True,
                                    ),
                                ],
                                className="ml-2",
                            ),
                        ],
                        className="text-center",
                    ),
                ],
                className="mb-4",
            ),
            # Stores for pipeline state
            dcc.Store(id="pipeline-state", data={}),
            dcc.Store(id="pipeline-results", data={}),
            dcc.Store(id="pipeline-current-stage", data=0),
            # Download component for exports
            dcc.Download(id="download-dataframe"),
        ],
        style={"padding": "20px"},
    )


# Helper function to create stage details view
def create_stage_details_view(stage_name: str, stage_info: dict) -> html.Div:
    """
    Create a detailed view for a pipeline stage.

    Parameters
    ----------
    stage_name : str
        Name of the pipeline stage
    stage_info : dict
        Information about the stage execution

    Returns
    -------
    html.Div
        Formatted stage details
    """
    return html.Div(
        [
            html.H5(stage_name, className="mb-3"),
            html.P(stage_info.get("description", ""), className="text-muted"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Strong("Status: "),
                            dbc.Badge(
                                stage_info.get("status", "Pending"),
                                color=(
                                    "success"
                                    if stage_info.get("status") == "Completed"
                                    else (
                                        "primary"
                                        if stage_info.get("status") == "Running"
                                        else "secondary"
                                    )
                                ),
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Strong("Duration: "),
                            html.Span(f"{stage_info.get('duration_ms', 0):.2f} ms"),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            html.Hr(),
            html.H6("Stage Metrics:"),
            html.Pre(
                str(stage_info.get("metrics", {})),
                style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "10px",
                    "borderRadius": "5px",
                },
            ),
        ]
    )
