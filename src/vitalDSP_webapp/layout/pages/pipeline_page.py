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
                                        html.H5("Pipeline Configuration", className="mb-0")
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
                                                    {"label": "Generic", "value": "generic"},
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
                                            html.H6("Stage Parameters", className="mb-3"),
                                            dbc.Accordion(
                                                [
                                                    # Stage 2: Quality Screening
                                                    dbc.AccordionItem(
                                                        [
                                                            html.Label("Enable Quality Screening"),
                                                            dbc.Switch(
                                                                id="pipeline-enable-quality",
                                                                value=True,
                                                                className="mb-2",
                                                            ),
                                                            html.Label("SQI Window Size (seconds)"),
                                                            dbc.Input(
                                                                id="pipeline-sqi-window",
                                                                type="number",
                                                                value=5,
                                                                min=1,
                                                                max=30,
                                                                step=1,
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
                                                        ],
                                                        title="Stage 2: Quality Screening",
                                                    ),
                                                    # Stage 3: Filtering
                                                    dbc.AccordionItem(
                                                        [
                                                            html.Label("Bandpass Low Cutoff (Hz)"),
                                                            dbc.Input(
                                                                id="pipeline-filter-lowcut",
                                                                type="number",
                                                                value=0.5,
                                                                min=0.1,
                                                                max=10,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Bandpass High Cutoff (Hz)"),
                                                            dbc.Input(
                                                                id="pipeline-filter-highcut",
                                                                type="number",
                                                                value=40,
                                                                min=1,
                                                                max=100,
                                                                step=1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Filter Order"),
                                                            dbc.Input(
                                                                id="pipeline-filter-order",
                                                                type="number",
                                                                value=4,
                                                                min=2,
                                                                max=8,
                                                                step=1,
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Baseline Correction Cutoff (Hz)"),
                                                            dbc.Input(
                                                                id="pipeline-baseline-cutoff",
                                                                type="number",
                                                                value=0.5,
                                                                min=0.1,
                                                                max=5,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 3: Filtering & Artifact Removal",
                                                    ),
                                                    # Stage 5: Segmentation
                                                    dbc.AccordionItem(
                                                        [
                                                            html.Label("Window Size (seconds)"),
                                                            dbc.Input(
                                                                id="pipeline-window-size",
                                                                type="number",
                                                                value=30,
                                                                min=5,
                                                                max=300,
                                                                step=5,
                                                                className="mb-2",
                                                            ),
                                                            html.Label("Overlap Ratio"),
                                                            dbc.Input(
                                                                id="pipeline-overlap-ratio",
                                                                type="number",
                                                                value=0.5,
                                                                min=0,
                                                                max=0.9,
                                                                step=0.1,
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 5: Segmentation",
                                                    ),
                                                    # Stage 6: Feature Extraction
                                                    dbc.AccordionItem(
                                                        [
                                                            html.Label("Feature Types"),
                                                            dcc.Checklist(
                                                                id="pipeline-feature-types",
                                                                options=[
                                                                    {
                                                                        "label": " Time Domain",
                                                                        "value": "time",
                                                                    },
                                                                    {
                                                                        "label": " Frequency Domain",
                                                                        "value": "frequency",
                                                                    },
                                                                    {
                                                                        "label": " Nonlinear",
                                                                        "value": "nonlinear",
                                                                    },
                                                                ],
                                                                value=["time", "frequency"],
                                                                className="mb-2",
                                                                inline=False,
                                                            ),
                                                        ],
                                                        title="Stage 6: Feature Extraction",
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
                                            "Processing Paths Comparison", className="mb-0"
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
                                            "Quality Screening Results", className="mb-0"
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
                                            "Feature Extraction Summary", className="mb-0"
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
                                            html.Div(id="pipeline-output-recommendations"),
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
                                color="success"
                                if stage_info.get("status") == "Completed"
                                else "primary"
                                if stage_info.get("status") == "Running"
                                else "secondary",
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
