"""
Progress indicator components for long-running operations.

This module provides reusable progress indicators for the vitalDSP webapp.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_progress_bar(
    progress_id: str,
    label: str = "Processing...",
    show_percentage: bool = True,
    animated: bool = True,
    striped: bool = True,
    color: str = "primary",
) -> html.Div:
    """
    Create a progress bar component for long-running operations.

    Parameters
    ----------
    progress_id : str
        Unique ID for the progress bar
    label : str, optional
        Label text to display above the progress bar
    show_percentage : bool, optional
        Whether to show percentage in the bar
    animated : bool, optional
        Whether to animate the progress bar
    striped : bool, optional
        Whether to use striped styling
    color : str, optional
        Bootstrap color variant (primary, success, info, warning, danger)

    Returns
    -------
    html.Div
        Div containing the progress indicator
    """
    return html.Div(
        id=f"{progress_id}-container",
        style={"display": "none"},  # Hidden by default
        children=[
            html.Label(label, className="mb-2"),
            dbc.Progress(
                id=progress_id,
                value=0,
                striped=striped,
                animated=animated,
                color=color,
                className="mb-3",
                style={"height": "25px"},
            ),
            html.Div(
                id=f"{progress_id}-status",
                className="text-muted",
                style={"fontSize": "0.9em"},
            ),
        ],
    )


def create_spinner_overlay(
    spinner_id: str,
    message: str = "Processing your request...",
    spinner_type: str = "border",
    color: str = "primary",
) -> html.Div:
    """
    Create a full-screen spinner overlay for operations without progress tracking.

    Parameters
    ----------
    spinner_id : str
        Unique ID for the spinner overlay
    message : str, optional
        Message to display with the spinner
    spinner_type : str, optional
        Type of spinner (border, grow)
    color : str, optional
        Bootstrap color variant

    Returns
    -------
    html.Div
        Div containing the spinner overlay
    """
    return html.Div(
        id=f"{spinner_id}-overlay",
        style={
            "display": "none",  # Hidden by default
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0, 0, 0, 0.5)",
            "zIndex": 9999,
            "display": "none",
        },
        children=[
            html.Div(
                style={
                    "position": "absolute",
                    "top": "50%",
                    "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "textAlign": "center",
                    "backgroundColor": "white",
                    "padding": "30px",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                },
                children=[
                    dbc.Spinner(
                        spinner_type=spinner_type,
                        color=color,
                        size="lg",
                    ),
                    html.H5(message, className="mt-3"),
                    html.Div(
                        id=f"{spinner_id}-message",
                        className="text-muted mt-2",
                    ),
                ],
            )
        ],
    )


def create_step_progress_indicator(
    step_id: str,
    steps: list,
    current_step: int = 0,
) -> html.Div:
    """
    Create a multi-step progress indicator (e.g., for pipeline stages).

    Parameters
    ----------
    step_id : str
        Unique ID for the step indicator
    steps : list
        List of step names/labels
    current_step : int, optional
        Current step index (0-based)

    Returns
    -------
    html.Div
        Div containing the step progress indicator
    """
    step_items = []
    for i, step_name in enumerate(steps):
        # Determine step status
        if i < current_step:
            icon = "fa-check-circle"
            icon_color = "text-success"
            status = "completed"
        elif i == current_step:
            icon = "fa-circle-notch fa-spin"
            icon_color = "text-primary"
            status = "in-progress"
        else:
            icon = "fa-circle"
            icon_color = "text-muted"
            status = "pending"

        step_items.append(
            html.Div(
                className=f"step-item {status}",
                style={
                    "display": "inline-block",
                    "textAlign": "center",
                    "flex": "1",
                    "position": "relative",
                },
                children=[
                    html.I(
                        className=f"fas {icon} {icon_color}",
                        style={"fontSize": "24px"},
                    ),
                    html.Div(
                        step_name,
                        className="mt-2",
                        style={
                            "fontSize": "12px",
                            "fontWeight": "bold" if i == current_step else "normal",
                        },
                    ),
                ],
            )
        )

        # Add connector line between steps (except after last step)
        if i < len(steps) - 1:
            connector_color = "#28a745" if i < current_step else "#dee2e6"
            step_items.append(
                html.Div(
                    style={
                        "flex": "0.5",
                        "height": "2px",
                        "backgroundColor": connector_color,
                        "alignSelf": "center",
                        "marginTop": "-20px",
                    }
                )
            )

    return html.Div(
        id=f"{step_id}-container",
        style={"display": "none"},
        children=[
            html.Div(
                className="step-progress",
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "justifyContent": "space-between",
                    "padding": "20px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "5px",
                    "marginBottom": "20px",
                },
                children=step_items,
            ),
            html.Div(
                id=f"{step_id}-status",
                className="text-center text-muted",
                style={"fontSize": "0.9em"},
            ),
        ],
    )


def create_interval_component(
    interval_id: str,
    interval_ms: int = 1000,
    max_intervals: int = -1,
    disabled: bool = True,
) -> dcc.Interval:
    """
    Create an interval component for periodic updates.

    Parameters
    ----------
    interval_id : str
        Unique ID for the interval component
    interval_ms : int, optional
        Update interval in milliseconds (default: 1000ms = 1 second)
    max_intervals : int, optional
        Maximum number of intervals before stopping (-1 = infinite)
    disabled : bool, optional
        Whether the interval is initially disabled

    Returns
    -------
    dcc.Interval
        Interval component for periodic callbacks
    """
    return dcc.Interval(
        id=interval_id,
        interval=interval_ms,
        n_intervals=0,
        max_intervals=max_intervals,
        disabled=disabled,
    )


# Example usage in a page layout:
"""
def example_layout():
    return html.Div([
        # For simple operations with progress
        create_progress_bar(
            progress_id="filter-progress",
            label="Filtering signal...",
        ),

        # For operations without progress tracking
        create_spinner_overlay(
            spinner_id="processing",
            message="Applying advanced filters...",
        ),

        # For multi-step operations (e.g., pipeline)
        create_step_progress_indicator(
            step_id="pipeline-steps",
            steps=[
                "Data Ingestion",
                "Quality Screening",
                "Parallel Processing",
                "Quality Validation",
                "Segmentation",
                "Feature Extraction",
                "Intelligent Output",
                "Output Package",
            ],
        ),

        # Interval for progress updates
        create_interval_component(
            interval_id="progress-update",
            interval_ms=500,  # Update every 500ms
        ),
    ])
"""
