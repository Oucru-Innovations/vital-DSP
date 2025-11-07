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


def create_interactive_flow_diagram():
    """
    Create an interactive pipeline flow diagram showing all 8 stages.

    Returns
    -------
    dbc.Card
        Interactive flow diagram card with stage indicators
    """
    pipeline_stages = [
        {"num": 1, "name": "Data Ingestion", "short": "Data", "icon": "fa-database"},
        {
            "num": 2,
            "name": "Quality Screening",
            "short": "QScreen",
            "icon": "fa-filter",
        },
        {
            "num": 3,
            "name": "Parallel Processing",
            "short": "3 Paths",
            "icon": "fa-code-branch",
        },
        {
            "num": 4,
            "name": "Quality Validation",
            "short": "QValid",
            "icon": "fa-check-circle",
        },
        {
            "num": 5,
            "name": "Segmentation",
            "short": "Segment",
            "icon": "fa-grip-vertical",
        },
        {
            "num": 6,
            "name": "Feature Extraction",
            "short": "Features",
            "icon": "fa-chart-bar",
        },
        {"num": 7, "name": "Intelligent Output", "short": "IntOut", "icon": "fa-brain"},
        {"num": 8, "name": "Output Package", "short": "Package", "icon": "fa-box"},
    ]

    # Create stage boxes
    stage_boxes = []
    for idx, stage in enumerate(pipeline_stages):
        stage_box = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.I(
                                            className=f"fas {stage['icon']} fa-2x mb-2",
                                            id=f"pipeline-stage-icon-{stage['num']}",
                                        ),
                                    ],
                                    className="text-center",
                                ),
                                html.Div(
                                    [
                                        html.Strong(
                                            f"{stage['num']}", className="d-block"
                                        ),
                                        html.Small(
                                            stage["short"], className="text-muted"
                                        ),
                                    ],
                                    className="text-center",
                                ),
                            ],
                            className="p-2",
                            id=f"pipeline-stage-box-{stage['num']}",
                        ),
                    ],
                    className="shadow-sm",
                    id=f"pipeline-stage-card-{stage['num']}",
                    style={
                        "cursor": "pointer",
                        "transition": "all 0.3s ease",
                        "border": "2px solid #dee2e6",
                    },
                ),
                # Tooltip for stage
                dbc.Tooltip(
                    stage["name"],
                    target=f"pipeline-stage-card-{stage['num']}",
                    placement="top",
                ),
            ],
            width="auto",
            className="px-1",
        )
        stage_boxes.append(stage_box)

        # Add arrow between stages
        if idx < len(pipeline_stages) - 1:
            arrow = dbc.Col(
                html.I(className="fas fa-arrow-right text-muted fa-lg"),
                width="auto",
                className="d-flex align-items-center px-0",
            )
            stage_boxes.append(arrow)

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(
                        [
                            html.I(className="fas fa-project-diagram mr-2"),
                            "Pipeline Flow Diagram",
                        ],
                        className="mb-0",
                    ),
                ],
            ),
            dbc.CardBody(
                [
                    # Stage flow
                    dbc.Row(
                        stage_boxes,
                        className="mb-3 justify-content-center",
                    ),
                    # Current status
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Strong("Current Stage: "),
                                    html.Span(
                                        "Not Started", id="pipeline-current-stage-text"
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Strong("Status: "),
                                    html.Span("Idle", id="pipeline-status-text"),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Strong("Overall Progress: "),
                                    html.Span(
                                        "0%", id="pipeline-overall-progress-text"
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Progress bar
                    dbc.Progress(
                        id="pipeline-flow-progress-bar",
                        value=0,
                        striped=True,
                        animated=False,
                        className="mb-3",
                    ),
                    # Path processing status
                    html.Div(
                        id="pipeline-paths-status-container",
                        style={"display": "none"},
                        children=[
                            html.Hr(),
                            html.H6("Processing Paths (Stage 3-4):", className="mb-2"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Span("PATH 1: RAW"),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Progress(
                                                            id="pipeline-path-raw-progress",
                                                            value=0,
                                                            color="secondary",
                                                            style={"height": "20px"},
                                                        ),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "--",
                                                            id="pipeline-path-raw-quality",
                                                        ),
                                                        width=2,
                                                        className="text-right",
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                        ],
                                        md=12,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Span("PATH 2: FILTERED"),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Progress(
                                                            id="pipeline-path-filtered-progress",
                                                            value=0,
                                                            color="info",
                                                            style={"height": "20px"},
                                                        ),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "--",
                                                            id="pipeline-path-filtered-quality",
                                                        ),
                                                        width=2,
                                                        className="text-right",
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                        ],
                                        md=12,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Span(
                                                            "PATH 3: PREPROCESSED"
                                                        ),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Progress(
                                                            id="pipeline-path-preprocessed-progress",
                                                            value=0,
                                                            color="success",
                                                            style={"height": "20px"},
                                                        ),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "--",
                                                            id="pipeline-path-preprocessed-quality",
                                                        ),
                                                        width=2,
                                                        className="text-right",
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                        ],
                                        md=12,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        className="mb-4",
    )


def create_live_status_panel():
    """
    Create a live status panel showing current pipeline execution status.

    Returns
    -------
    dbc.Card
        Live status panel card
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span("🔴", id="pipeline-status-indicator", className="mr-2"),
                    html.Strong("LIVE PIPELINE STATUS"),
                ],
            ),
            dbc.CardBody(
                [
                    # Run information
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(html.Small("Run ID:"), width=4),
                                    dbc.Col(
                                        html.Small("--", id="pipeline-run-id"),
                                        width=8,
                                    ),
                                ],
                                className="mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(html.Small("Started:"), width=4),
                                    dbc.Col(
                                        html.Small("--", id="pipeline-start-time"),
                                        width=8,
                                    ),
                                ],
                                className="mb-1",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(html.Small("Elapsed:"), width=4),
                                    dbc.Col(
                                        html.Small("--", id="pipeline-elapsed-time"),
                                        width=8,
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ],
                    ),
                    # Current stage
                    html.Div(
                        [
                            html.Strong("Current Stage:", className="d-block mb-1"),
                            html.Div(
                                "0/8",
                                id="pipeline-current-stage-indicator",
                                className="mb-2",
                            ),
                            dbc.Progress(
                                id="pipeline-stage-progress-mini",
                                value=0,
                                striped=True,
                                className="mb-3",
                            ),
                        ],
                    ),
                    # Stage status list
                    html.Hr(),
                    html.Strong("STAGE STATUS:", className="d-block mb-2"),
                    html.Div(
                        id="pipeline-stage-status-list",
                        children=[
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-circle text-muted mr-2",
                                        id=f"pipeline-stage-status-icon-{i}",
                                    ),
                                    html.Small(f"{i}. {stage}"),
                                ],
                                className="mb-1",
                            )
                            for i, stage in enumerate(
                                [
                                    "Data Ingestion",
                                    "Quality Screening",
                                    "Parallel Processing",
                                    "Quality Validation",
                                    "Segmentation",
                                    "Feature Extraction",
                                    "Intelligent Output",
                                    "Output Package",
                                ],
                                1,
                            )
                        ],
                    ),
                    # Paths processed
                    html.Hr(),
                    html.Strong("PATHS PROCESSED:", className="d-block mb-2"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-circle text-muted mr-2",
                                        id="pipeline-path-status-raw",
                                    ),
                                    html.Small("RAW"),
                                ],
                                className="mb-1",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-circle text-muted mr-2",
                                        id="pipeline-path-status-filtered",
                                    ),
                                    html.Small("FILTERED"),
                                ],
                                className="mb-1",
                            ),
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-circle text-muted mr-2",
                                        id="pipeline-path-status-preprocessed",
                                    ),
                                    html.Small("PREPROCESSED"),
                                ],
                                className="mb-3",
                            ),
                        ],
                    ),
                    # Action buttons
                    dbc.ButtonGroup(
                        [
                            dbc.Button(
                                "View Logs",
                                id="pipeline-view-logs-btn",
                                color="secondary",
                                size="sm",
                                outline=True,
                            ),
                        ],
                        className="d-flex",
                        style={"width": "100%"},
                    ),
                ],
                className="p-3",
            ),
        ],
        className="mb-4",
        style={"position": "sticky", "top": "20px"},
    )


# ============================================================================
# PHASE B: PATH COMPARISON DASHBOARD COMPONENTS
# ============================================================================


def create_path_comparison_dashboard():
    """
    Create an enhanced path comparison dashboard showing quality metrics
    and signal visualizations for all 3 processing paths.

    Returns
    -------
    dbc.AccordionItem
        Collapsible accordion item with path comparison
    """
    return dbc.AccordionItem(
        [
            # Path Overview Table
            html.Div(
                [
                    html.H6("Path Overview", className="mb-3"),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Path"),
                                        html.Th("Processing"),
                                        html.Th("Quality Score"),
                                        html.Th("Status"),
                                    ]
                                )
                            ),
                            html.Tbody(
                                [
                                    # RAW Path
                                    html.Tr(
                                        [
                                            html.Td(
                                                [
                                                    html.I(
                                                        className="fas fa-circle text-secondary me-2"
                                                    ),
                                                    "RAW",
                                                ]
                                            ),
                                            html.Td("No filtering (baseline)"),
                                            html.Td(
                                                id="path-table-raw-quality",
                                                children="N/A",
                                            ),
                                            html.Td(
                                                id="path-table-raw-status",
                                                children="⚪ Pending",
                                            ),
                                        ]
                                    ),
                                    # FILTERED Path
                                    html.Tr(
                                        [
                                            html.Td(
                                                [
                                                    html.I(
                                                        className="fas fa-circle text-primary me-2"
                                                    ),
                                                    "FILTERED",
                                                ]
                                            ),
                                            html.Td("Bandpass filtering"),
                                            html.Td(
                                                id="path-table-filtered-quality",
                                                children="N/A",
                                            ),
                                            html.Td(
                                                id="path-table-filtered-status",
                                                children="⚪ Pending",
                                            ),
                                        ]
                                    ),
                                    # PREPROCESSED Path
                                    html.Tr(
                                        [
                                            html.Td(
                                                [
                                                    html.I(
                                                        className="fas fa-circle text-success me-2"
                                                    ),
                                                    "PREPROCESSED",
                                                ]
                                            ),
                                            html.Td("Filtering + Artifact Removal"),
                                            html.Td(
                                                id="path-table-preprocessed-quality",
                                                children="N/A",
                                            ),
                                            html.Td(
                                                id="path-table-preprocessed-status",
                                                children="⚪ Pending",
                                            ),
                                        ]
                                    ),
                                ],
                                id="path-comparison-table-body",
                            ),
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mb-4",
                    ),
                ]
            ),
            # Quality Metrics Comparison
            html.Div(
                [
                    html.H6("Quality Metrics by Path", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "SNR (Signal-to-Noise)",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div(
                                                    id="metric-snr-raw",
                                                    children="RAW: N/A",
                                                    className="mb-1",
                                                ),
                                                html.Div(
                                                    id="metric-snr-filtered",
                                                    children="FILTERED: N/A",
                                                    className="mb-1",
                                                ),
                                                html.Div(
                                                    id="metric-snr-preprocessed",
                                                    children="PREPROCESSED: N/A",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="h-100",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Baseline Wander",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div(
                                                    id="metric-baseline-raw",
                                                    children="RAW: N/A",
                                                    className="mb-1",
                                                ),
                                                html.Div(
                                                    id="metric-baseline-filtered",
                                                    children="FILTERED: N/A",
                                                    className="mb-1",
                                                ),
                                                html.Div(
                                                    id="metric-baseline-preprocessed",
                                                    children="PREPROCESSED: N/A",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="h-100",
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Artifact Level",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div(
                                                    id="metric-artifact-raw",
                                                    children="RAW: N/A",
                                                    className="mb-1",
                                                ),
                                                html.Div(
                                                    id="metric-artifact-filtered",
                                                    children="FILTERED: N/A",
                                                    className="mb-1",
                                                ),
                                                html.Div(
                                                    id="metric-artifact-preprocessed",
                                                    children="PREPROCESSED: N/A",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="h-100",
                                ),
                                md=4,
                            ),
                        ],
                        className="mb-4",
                    ),
                ],
                id="path-metrics-container",
                style={"display": "none"},  # Hidden until pipeline runs
            ),
            # Recommendation Panel
            dbc.Alert(
                [
                    html.H6(
                        [html.I(className="fas fa-lightbulb me-2"), "Recommendation"],
                        className="mb-2",
                    ),
                    html.Div(
                        id="path-recommendation-text",
                        children="Run pipeline to see path recommendation",
                    ),
                ],
                id="path-recommendation-alert",
                color="info",
                className="mb-3",
            ),
            # Signal Visualization Placeholder
            html.Div(
                [
                    html.H6("Signal Comparison", className="mb-3"),
                    dcc.Graph(
                        id="path-comparison-signal-plot",
                        config={"displayModeBar": True},
                        style={"height": "400px"},
                    ),
                ],
                id="path-signal-viz-container",
                style={"display": "none"},  # Hidden until pipeline runs
            ),
        ],
        title="Processing Path Comparison",
        item_id="accordion-path-comparison",
    )


def create_path_flow_diagrams():
    """
    Create detailed flow diagrams for each of the 3 processing paths.

    Returns
    -------
    dbc.AccordionItem
        Collapsible accordion item with path workflow tabs
    """
    # Define path workflows
    paths_info = [
        {
            "name": "RAW",
            "color": "secondary",
            "icon": "fa-stream",
            "description": "No filtering - baseline for comparison",
            "steps": [
                {
                    "name": "Data Ingestion",
                    "icon": "fa-database",
                    "desc": "Load signal from Enhanced Data Service",
                },
                {
                    "name": "Quality Screening",
                    "icon": "fa-filter",
                    "desc": "Calculate baseline SQI metrics",
                },
                {
                    "name": "RAW Path",
                    "icon": "fa-stream",
                    "desc": "No filtering applied",
                },
                {
                    "name": "Quality Validation",
                    "icon": "fa-check-circle",
                    "desc": "Measure raw signal quality",
                },
                {
                    "name": "Segmentation",
                    "icon": "fa-grip-vertical",
                    "desc": "Window signal for analysis",
                },
                {
                    "name": "Feature Extraction",
                    "icon": "fa-chart-bar",
                    "desc": "Extract features from raw signal",
                },
                {"name": "Output", "icon": "fa-box", "desc": "Export results"},
            ],
        },
        {
            "name": "FILTERED",
            "color": "primary",
            "icon": "fa-wave-square",
            "description": "Bandpass filtering only",
            "steps": [
                {
                    "name": "Data Ingestion",
                    "icon": "fa-database",
                    "desc": "Load signal from Enhanced Data Service",
                },
                {
                    "name": "Quality Screening",
                    "icon": "fa-filter",
                    "desc": "Calculate baseline SQI metrics",
                },
                {
                    "name": "Bandpass Filter",
                    "icon": "fa-wave-square",
                    "desc": "Apply Butterworth filter (0.5-40 Hz)",
                },
                {
                    "name": "Quality Validation",
                    "icon": "fa-check-circle",
                    "desc": "Measure filtered signal quality",
                },
                {
                    "name": "Segmentation",
                    "icon": "fa-grip-vertical",
                    "desc": "Window filtered signal",
                },
                {
                    "name": "Feature Extraction",
                    "icon": "fa-chart-bar",
                    "desc": "Extract features from filtered signal",
                },
                {"name": "Output", "icon": "fa-box", "desc": "Export results"},
            ],
        },
        {
            "name": "PREPROCESSED",
            "color": "success",
            "icon": "fa-magic",
            "description": "Filtering + Artifact Removal",
            "steps": [
                {
                    "name": "Data Ingestion",
                    "icon": "fa-database",
                    "desc": "Load signal from Enhanced Data Service",
                },
                {
                    "name": "Quality Screening",
                    "icon": "fa-filter",
                    "desc": "Calculate baseline SQI metrics",
                },
                {
                    "name": "Bandpass Filter",
                    "icon": "fa-wave-square",
                    "desc": "Apply Butterworth filter",
                },
                {
                    "name": "Artifact Removal",
                    "icon": "fa-magic",
                    "desc": "Remove baseline + wavelet denoising",
                },
                {
                    "name": "Quality Validation",
                    "icon": "fa-check-circle",
                    "desc": "Measure preprocessed quality",
                },
                {
                    "name": "Segmentation",
                    "icon": "fa-grip-vertical",
                    "desc": "Window preprocessed signal",
                },
                {
                    "name": "Feature Extraction",
                    "icon": "fa-chart-bar",
                    "desc": "Extract features from clean signal",
                },
                {"name": "Output", "icon": "fa-box", "desc": "Export results"},
            ],
        },
    ]

    # Create tabs for each path
    tabs = []
    for path in paths_info:
        # Create flow diagram for this path
        flow_steps = []
        for i, step in enumerate(path["steps"]):
            # Add step box
            flow_steps.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.I(
                                    className=f"fas {step['icon']} fa-2x mb-2 text-{path['color']}"
                                ),
                                html.Div(step["name"], className="fw-bold"),
                                html.Small(step["desc"], className="text-muted"),
                            ],
                            className="text-center p-3 border rounded bg-light",
                            style={"minWidth": "150px"},
                        )
                    ],
                    className="d-inline-block",
                )
            )

            # Add arrow between steps
            if i < len(path["steps"]) - 1:
                flow_steps.append(
                    html.Div(
                        html.I(
                            className=f"fas fa-arrow-right fa-2x text-{path['color']}"
                        ),
                        className="d-inline-block mx-2 align-middle",
                    )
                )

        tab = dbc.Tab(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.I(
                                    className=f"fas {path['icon']} fa-3x text-{path['color']} mb-3"
                                ),
                                html.H5(f"PATH: {path['name']}", className="mb-2"),
                                html.P(
                                    path["description"], className="text-muted mb-4"
                                ),
                            ],
                            className="text-center",
                        ),
                        # Flow diagram
                        html.Div(
                            flow_steps,
                            className="d-flex flex-wrap justify-content-center align-items-center",
                            style={"overflowX": "auto"},
                        ),
                    ]
                ),
                className="border-0",
            ),
            label=path["name"],
            tab_id=f"path-tab-{path['name'].lower()}",
            activeTabClassName="fw-bold",
        )
        tabs.append(tab)

    return dbc.AccordionItem(
        [
            dbc.Tabs(
                tabs,
                id="path-workflow-tabs",
                active_tab="path-tab-raw",
            )
        ],
        title="Processing Path Workflows",
        item_id="accordion-path-workflows",
    )


def create_path_selector():
    """
    Create an interactive path selector for choosing which paths to compare.

    Returns
    -------
    dbc.Card
        Card with path selection checkboxes
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H6(
                    [
                        html.I(className="fas fa-tasks me-2"),
                        "Select Processing Paths",
                    ],
                    className="mb-0",
                )
            ),
            dbc.CardBody(
                [
                    dbc.Checklist(
                        options=[
                            {
                                "label": html.Div(
                                    [
                                        html.Strong("PATH 1: RAW"),
                                        html.Br(),
                                        html.Small(
                                            "No filtering - baseline for comparison",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                "value": "raw",
                            },
                            {
                                "label": html.Div(
                                    [
                                        html.Strong("PATH 2: FILTERED"),
                                        html.Br(),
                                        html.Small(
                                            "Bandpass filtering (0.5-40 Hz)",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                "value": "filtered",
                            },
                            {
                                "label": html.Div(
                                    [
                                        html.Strong("PATH 3: PREPROCESSED"),
                                        html.Br(),
                                        html.Small(
                                            "Filtered + Artifact Removal",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                                "value": "preprocessed",
                            },
                        ],
                        value=[
                            "raw",
                            "filtered",
                            "preprocessed",
                        ],  # All selected by default
                        id="pipeline-path-selector",
                        className="mb-3",
                    ),
                    html.Hr(),
                    html.Div(
                        [
                            html.Label("Comparison Mode:", className="mb-2"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "All Paths", "value": "all"},
                                    {"label": "Best 2 Paths", "value": "best2"},
                                    {"label": "Custom Selection", "value": "custom"},
                                ],
                                value="all",
                                id="pipeline-comparison-mode",
                                inline=False,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="mb-3",
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
            # NEW: Interactive Flow Diagram (Phase A)
            create_interactive_flow_diagram(),
            # Action Buttons (Run, Stop, Reset, Export, Report)
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
                                            html.Hr(),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Phase B: Interactive Path Selector
                            create_path_selector(),
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H6("Stage Parameters", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Stage-Specific Parameters (Collapsible Accordion)
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
                                                            # SQI Type Selection (matching Quality Page)
                                                            html.H6(
                                                                "Signal Quality Index (SQI) Type",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            html.Small(
                                                                "Select which SQI method to use for quality screening",
                                                                className="text-muted d-block mb-2",
                                                            ),
                                                            dbc.Select(
                                                                id="pipeline-stage2-sqi-type",
                                                                options=[
                                                                    {
                                                                        "label": "SNR SQI - Signal-to-Noise Ratio",
                                                                        "value": "snr_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Baseline Wander SQI",
                                                                        "value": "baseline_wander_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Amplitude Variability SQI",
                                                                        "value": "amplitude_variability_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Zero Crossing SQI",
                                                                        "value": "zero_crossing_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Waveform Similarity SQI",
                                                                        "value": "waveform_similarity_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Signal Entropy SQI",
                                                                        "value": "signal_entropy_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Energy SQI",
                                                                        "value": "energy_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Kurtosis SQI",
                                                                        "value": "kurtosis_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Skewness SQI",
                                                                        "value": "skewness_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Peak-to-Peak Amplitude SQI",
                                                                        "value": "peak_to_peak_amplitude_sqi",
                                                                    },
                                                                    {
                                                                        "label": "PPG Signal Quality SQI",
                                                                        "value": "ppg_signal_quality_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Respiratory Signal Quality SQI",
                                                                        "value": "respiratory_signal_quality_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Heart Rate Variability SQI",
                                                                        "value": "heart_rate_variability_sqi",
                                                                    },
                                                                    {
                                                                        "label": "EEG Band Power SQI",
                                                                        "value": "eeg_band_power_sqi",
                                                                    },
                                                                ],
                                                                value="snr_sqi",
                                                                className="mb-3",
                                                            ),
                                                            # SQI Parameters (matching Quality Page)
                                                            html.H6(
                                                                "SQI Parameters",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Window Size (samples)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage2-window-size",
                                                                                type="number",
                                                                                value=1000,
                                                                                min=100,
                                                                                step=100,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Step Size (samples)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage2-step-size",
                                                                                type="number",
                                                                                value=500,
                                                                                min=50,
                                                                                step=50,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Threshold Type",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="pipeline-stage2-threshold-type",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Below Threshold = Abnormal",
                                                                                        "value": "below",
                                                                                    },
                                                                                    {
                                                                                        "label": "Above Threshold = Abnormal",
                                                                                        "value": "above",
                                                                                    },
                                                                                    {
                                                                                        "label": "Range (min-max)",
                                                                                        "value": "range",
                                                                                    },
                                                                                ],
                                                                                value="below",
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Threshold Value",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage2-threshold",
                                                                                type="number",
                                                                                value=0.7,
                                                                                min=0,
                                                                                max=1,
                                                                                step=0.05,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Scaling Method",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="pipeline-stage2-scale",
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
                                                                        md=12,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                        ],
                                                        title="Stage 2: Quality Screening",
                                                    ),
                                                    # Stage 3: Filtering
                                                    dbc.AccordionItem(
                                                        [
                                                            # Filter Type Selection (Main)
                                                            html.H6(
                                                                "Filter Type",
                                                                className="mb-2",
                                                            ),
                                                            dbc.Select(
                                                                id="pipeline-stage3-filter-type",
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
                                                                className="mb-3",
                                                            ),
                                                            # Common Parameters Section
                                                            html.H6(
                                                                "Common Parameters",
                                                                className="mb-2",
                                                            ),
                                                            # Detrending
                                                            dbc.Checklist(
                                                                id="pipeline-stage3-detrend",
                                                                options=[
                                                                    {
                                                                        "label": "Apply Detrending",
                                                                        "value": "detrend",
                                                                    }
                                                                ],
                                                                value=[],
                                                                className="mb-2",
                                                            ),
                                                            # Signal Source
                                                            html.Label(
                                                                "Signal Source",
                                                                className="form-label",
                                                            ),
                                                            dbc.Select(
                                                                id="pipeline-stage3-signal-source",
                                                                options=[
                                                                    {
                                                                        "label": "Original Signal",
                                                                        "value": "original",
                                                                    },
                                                                    {
                                                                        "label": "Filtered Signal (Iterative)",
                                                                        "value": "filtered",
                                                                    },
                                                                ],
                                                                value="original",
                                                                className="mb-2",
                                                            ),
                                                            # Filter Application Count
                                                            html.Label(
                                                                "Apply Filter (n times)",
                                                                className="form-label",
                                                            ),
                                                            dbc.Input(
                                                                id="pipeline-stage3-application-count",
                                                                type="number",
                                                                value=1,
                                                                min=1,
                                                                max=10,
                                                                step=1,
                                                                className="mb-3",
                                                            ),
                                                            # Traditional Filter Parameters
                                                            html.Div(
                                                                id="pipeline-stage3-traditional-params",
                                                                children=[
                                                                    html.Hr(),
                                                                    html.H6(
                                                                        "Traditional Filters",
                                                                        className="mb-2",
                                                                    ),
                                                                    # Filter Family
                                                                    html.Label(
                                                                        "Filter Family",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-filter-family",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    # Filter Response
                                                                    html.Label(
                                                                        "Filter Response",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-filter-response",
                                                                        options=[
                                                                            {
                                                                                "label": "Lowpass",
                                                                                "value": "lowpass",
                                                                            },
                                                                            {
                                                                                "label": "Highpass",
                                                                                "value": "highpass",
                                                                            },
                                                                            {
                                                                                "label": "Bandpass",
                                                                                "value": "bandpass",
                                                                            },
                                                                            {
                                                                                "label": "Bandstop",
                                                                                "value": "bandstop",
                                                                            },
                                                                        ],
                                                                        value="bandpass",
                                                                        className="mb-2",
                                                                    ),
                                                                    # Low Cutoff Frequency
                                                                    html.Label(
                                                                        "Low Cutoff Frequency (Hz)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-filter-lowcut",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0.1,
                                                                        max=50,
                                                                        step=0.1,
                                                                        className="mb-2",
                                                                    ),
                                                                    # High Cutoff Frequency
                                                                    html.Label(
                                                                        "High Cutoff Frequency (Hz)",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-filter-highcut",
                                                                        type="number",
                                                                        value=40,
                                                                        min=1,
                                                                        max=100,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                    # Filter Order
                                                                    html.Label(
                                                                        "Filter Order",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-filter-order",
                                                                        type="number",
                                                                        value=4,
                                                                        min=2,
                                                                        max=10,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                    # Passband Ripple (Chebyshev/Elliptic)
                                                                    html.Label(
                                                                        "Passband Ripple (dB) - Chebyshev/Elliptic",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-filter-rp",
                                                                        type="number",
                                                                        value=1,
                                                                        min=0.1,
                                                                        max=5,
                                                                        step=0.1,
                                                                        className="mb-2",
                                                                    ),
                                                                    # Stopband Attenuation (Elliptic)
                                                                    html.Label(
                                                                        "Stopband Attenuation (dB) - Elliptic",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-filter-rs",
                                                                        type="number",
                                                                        value=40,
                                                                        min=20,
                                                                        max=80,
                                                                        step=5,
                                                                        className="mb-3",
                                                                    ),
                                                                    # Additional Filters
                                                                    html.H6(
                                                                        "Additional Filters",
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Savitzky-Golay Window",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-savgol-window",
                                                                        type="number",
                                                                        placeholder="Window (leave empty to skip)",
                                                                        min=3,
                                                                        step=2,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Savitzky-Golay Polyorder",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-savgol-polyorder",
                                                                        type="number",
                                                                        value=2,
                                                                        min=1,
                                                                        max=5,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Moving Average Window",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-moving-avg-window",
                                                                        type="number",
                                                                        placeholder="Window (leave empty to skip)",
                                                                        min=3,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Gaussian Sigma",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-gaussian-sigma",
                                                                        type="number",
                                                                        placeholder="Sigma (leave empty to skip)",
                                                                        min=0.1,
                                                                        max=10.0,
                                                                        step=0.1,
                                                                        className="mb-2",
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "block"
                                                                },
                                                            ),
                                                            # Advanced Filter Parameters
                                                            html.Div(
                                                                id="pipeline-stage3-advanced-params",
                                                                children=[
                                                                    html.Hr(),
                                                                    html.H6(
                                                                        "Advanced Filters",
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Method",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-advanced-method",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    # Kalman Parameters
                                                                    html.Div(
                                                                        id="pipeline-stage3-kalman-params",
                                                                        children=[
                                                                            html.Label(
                                                                                "R (Measurement Noise)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage3-kalman-r",
                                                                                type="number",
                                                                                value=1.0,
                                                                                min=0.001,
                                                                                max=10.0,
                                                                                step=0.01,
                                                                                className="mb-2",
                                                                            ),
                                                                            html.Label(
                                                                                "Q (Process Noise)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage3-kalman-q",
                                                                                type="number",
                                                                                value=1.0,
                                                                                min=0.001,
                                                                                max=10.0,
                                                                                step=0.01,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                    ),
                                                                    # Adaptive Parameters
                                                                    html.Div(
                                                                        id="pipeline-stage3-adaptive-params",
                                                                        children=[
                                                                            html.Label(
                                                                                "Step Size (μ)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage3-adaptive-mu",
                                                                                type="number",
                                                                                value=0.01,
                                                                                min=0.001,
                                                                                max=1.0,
                                                                                step=0.001,
                                                                                className="mb-2",
                                                                            ),
                                                                            html.Label(
                                                                                "Filter Order",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage3-adaptive-order",
                                                                                type="number",
                                                                                value=4,
                                                                                min=2,
                                                                                max=20,
                                                                                step=1,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        style={
                                                                            "display": "none"
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                            # Artifact Removal Parameters
                                                            html.Div(
                                                                id="pipeline-stage3-artifact-params",
                                                                children=[
                                                                    html.Hr(),
                                                                    html.H6(
                                                                        "Artifact Removal",
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Artifact Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-artifact-type",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Removal Strength",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-artifact-strength",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0.1,
                                                                        max=1.0,
                                                                        step=0.1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Wavelet Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-wavelet-type",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Wavelet Level",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-wavelet-level",
                                                                        type="number",
                                                                        value=3,
                                                                        min=1,
                                                                        max=8,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Threshold Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-threshold-type",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Threshold Value",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-threshold-value",
                                                                        type="number",
                                                                        value=0.1,
                                                                        min=0.01,
                                                                        max=1.0,
                                                                        step=0.01,
                                                                        className="mb-2",
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                            # Neural Network Parameters
                                                            html.Div(
                                                                id="pipeline-stage3-neural-params",
                                                                children=[
                                                                    html.Hr(),
                                                                    html.H6(
                                                                        "Neural Network",
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Network Type",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-neural-type",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Model Complexity",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-neural-complexity",
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
                                                                        className="mb-2",
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                            # Ensemble Parameters
                                                            html.Div(
                                                                id="pipeline-stage3-ensemble-params",
                                                                children=[
                                                                    html.Hr(),
                                                                    html.H6(
                                                                        "Ensemble Methods",
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Ensemble Method",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Select(
                                                                        id="pipeline-stage3-ensemble-method",
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
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Label(
                                                                        "Number of Filters",
                                                                        className="form-label",
                                                                    ),
                                                                    dbc.Input(
                                                                        id="pipeline-stage3-ensemble-n-filters",
                                                                        type="number",
                                                                        value=3,
                                                                        min=2,
                                                                        max=10,
                                                                        step=1,
                                                                        className="mb-2",
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "none"
                                                                },
                                                            ),
                                                        ],
                                                        title="Stage 3: Filtering & Artifact Removal",
                                                    ),
                                                    # Stage 4: Quality Validation
                                                    dbc.AccordionItem(
                                                        [
                                                            html.Small(
                                                                "Compare processing paths using comprehensive quality assessment",
                                                                className="text-muted d-block mb-3",
                                                            ),
                                                            # SQI Type Selection (matching Quality Page)
                                                            html.H6(
                                                                "Signal Quality Index (SQI) Type",
                                                                className="mb-2",
                                                            ),
                                                            html.Small(
                                                                "Select which SQI method to use for path quality comparison",
                                                                className="text-muted d-block mb-2",
                                                            ),
                                                            dbc.Select(
                                                                id="pipeline-stage4-sqi-type",
                                                                options=[
                                                                    {
                                                                        "label": "SNR SQI - Signal-to-Noise Ratio",
                                                                        "value": "snr_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Baseline Wander SQI",
                                                                        "value": "baseline_wander_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Amplitude Variability SQI",
                                                                        "value": "amplitude_variability_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Zero Crossing SQI",
                                                                        "value": "zero_crossing_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Waveform Similarity SQI",
                                                                        "value": "waveform_similarity_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Signal Entropy SQI",
                                                                        "value": "signal_entropy_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Energy SQI",
                                                                        "value": "energy_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Kurtosis SQI",
                                                                        "value": "kurtosis_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Skewness SQI",
                                                                        "value": "skewness_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Peak-to-Peak Amplitude SQI",
                                                                        "value": "peak_to_peak_amplitude_sqi",
                                                                    },
                                                                    {
                                                                        "label": "PPG Signal Quality SQI",
                                                                        "value": "ppg_signal_quality_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Respiratory Signal Quality SQI",
                                                                        "value": "respiratory_signal_quality_sqi",
                                                                    },
                                                                    {
                                                                        "label": "Heart Rate Variability SQI",
                                                                        "value": "heart_rate_variability_sqi",
                                                                    },
                                                                    {
                                                                        "label": "EEG Band Power SQI",
                                                                        "value": "eeg_band_power_sqi",
                                                                    },
                                                                ],
                                                                value="snr_sqi",
                                                                className="mb-3",
                                                            ),
                                                            # SQI Parameters (matching Quality Page)
                                                            html.H6(
                                                                "SQI Parameters",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Window Size (samples)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage4-window-size",
                                                                                type="number",
                                                                                value=1000,
                                                                                min=100,
                                                                                step=100,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Step Size (samples)",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage4-step-size",
                                                                                type="number",
                                                                                value=500,
                                                                                min=50,
                                                                                step=50,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Threshold Type",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="pipeline-stage4-threshold-type",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Below Threshold = Abnormal",
                                                                                        "value": "below",
                                                                                    },
                                                                                    {
                                                                                        "label": "Above Threshold = Abnormal",
                                                                                        "value": "above",
                                                                                    },
                                                                                    {
                                                                                        "label": "Range (min-max)",
                                                                                        "value": "range",
                                                                                    },
                                                                                ],
                                                                                value="below",
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Threshold Value",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="pipeline-stage4-threshold",
                                                                                type="number",
                                                                                value=0.7,
                                                                                min=0,
                                                                                max=1,
                                                                                step=0.05,
                                                                                className="mb-2",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Scaling Method",
                                                                                className="form-label",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="pipeline-stage4-scale",
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
                                                                        md=12,
                                                                    ),
                                                                ],
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
                                                                        "label": " Statistical Features",
                                                                        "value": "statistical",
                                                                    },
                                                                    {
                                                                        "label": " Nonlinear Features",
                                                                        "value": "nonlinear",
                                                                    },
                                                                    {
                                                                        "label": " Morphological Features",
                                                                        "value": "morphological",
                                                                    },
                                                                ],
                                                                value=[
                                                                    "time",
                                                                    "frequency",
                                                                    "statistical",
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
                                                                    {
                                                                        "label": " Energy",
                                                                        "value": "energy",
                                                                    },
                                                                    {
                                                                        "label": " Power",
                                                                        "value": "power",
                                                                    },
                                                                    {
                                                                        "label": " Crest Factor",
                                                                        "value": "crest_factor",
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
                                                                    {
                                                                        "label": " Spectral Bandwidth",
                                                                        "value": "spectral_bandwidth",
                                                                    },
                                                                    {
                                                                        "label": " Spectral Rolloff",
                                                                        "value": "spectral_rolloff",
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
                                                                className="mb-3",
                                                                inline=False,
                                                            ),
                                                            html.H6(
                                                                "Morphological Features",
                                                                className="mt-3 mb-2",
                                                            ),
                                                            dcc.Checklist(
                                                                id="pipeline-morphological-features",
                                                                options=[
                                                                    {
                                                                        "label": " Number of Peaks",
                                                                        "value": "num_peaks",
                                                                    },
                                                                    {
                                                                        "label": " Number of Valleys",
                                                                        "value": "num_valleys",
                                                                    },
                                                                    {
                                                                        "label": " Mean Peak Height",
                                                                        "value": "mean_peak_height",
                                                                    },
                                                                    {
                                                                        "label": " Mean Valley Depth",
                                                                        "value": "mean_valley_depth",
                                                                    },
                                                                ],
                                                                value=[],
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
                    # Center Panel - Visualization and Results
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
                            # Phase B: Enhanced Path Comparison Dashboard and Flow Diagrams
                            dbc.Accordion(
                                [
                                    create_path_comparison_dashboard(),
                                    create_path_flow_diagrams(),
                                ],
                                id="pipeline-path-accordion",
                                start_collapsed=True,
                                always_open=False,
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
                        md=6,
                    ),
                    # Right Panel - Live Status (Phase A)
                    dbc.Col(
                        [
                            create_live_status_panel(),
                        ],
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            # Stores for pipeline state
            dcc.Store(id="pipeline-state", data={}),
            dcc.Store(id="pipeline-results", data={}),
            dcc.Store(id="pipeline-current-stage", data=0),
            # Interval for real-time status updates (1 second)
            dcc.Interval(
                id="pipeline-progress-interval",
                interval=1000,  # 1 second in milliseconds
                n_intervals=0,
                disabled=True,  # Start disabled, enable when pipeline runs
            ),
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
