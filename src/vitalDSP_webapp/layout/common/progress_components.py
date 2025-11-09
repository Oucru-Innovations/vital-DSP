"""
Progress Indicator Components for vitalDSP Webapp

This module provides reusable progress indicator components for long-running operations.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Optional, Dict, Any, List


def create_progress_indicator(
    progress_id: str,
    label: str = "Processing...",
    show_percentage: bool = True,
    animated: bool = True,
    striped: bool = True,
    color: str = "primary",
    height: str = "20px",
    style: Optional[Dict[str, Any]] = None,
) -> html.Div:
    """
    Create a progress indicator component.

    Args:
        progress_id: Unique ID for the progress bar
        label: Label text to display
        show_percentage: Whether to show percentage text
        animated: Whether to animate the progress bar
        striped: Whether to show stripes
        color: Bootstrap color variant
        height: Height of the progress bar
        style: Additional CSS styles

    Returns:
        html.Div containing the progress indicator
    """
    progress_style = {"height": height, "border-radius": "10px", "overflow": "hidden"}

    if style:
        progress_style.update(style)

    return html.Div(
        [
            # Progress bar
            dbc.Progress(
                id=progress_id,
                value=0,
                max=100,
                animated=animated,
                striped=striped,
                color=color,
                style=progress_style,
                className="mb-2",
            ),
            # Status text
            html.Div(
                id=f"{progress_id}-status",
                children=label,
                className="text-muted small text-center",
                style={"font-size": "0.9rem"},
            ),
            # Percentage text (if enabled)
            (
                html.Div(
                    id=f"{progress_id}-percentage",
                    children="0%",
                    className="text-muted small text-center",
                    style={"font-size": "0.8rem", "margin-top": "2px"},
                )
                if show_percentage
                else None
            ),
        ],
        style={"display": "none"},
        id=f"{progress_id}-container",
    )


def create_progress_overlay(
    overlay_id: str,
    message: str = "Processing your request...",
    spinner_type: str = "border",
    color: str = "primary",
    size: str = "lg",
) -> html.Div:
    """
    Create a full-screen progress overlay.

    Args:
        overlay_id: Unique ID for the overlay
        message: Message to display
        spinner_type: Type of spinner (border, grow)
        color: Bootstrap color variant
        size: Size of the spinner (sm, md, lg)

    Returns:
        html.Div containing the progress overlay
    """
    return html.Div(
        [
            # Overlay background
            html.Div(
                style={
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100%",
                    "height": "100%",
                    "background-color": "rgba(0, 0, 0, 0.5)",
                    "z-index": 9999,
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                },
                children=[
                    # Progress card
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    # Spinner
                                    dbc.Spinner(
                                        html.Div(),
                                        type=spinner_type,
                                        color=color,
                                        size=size,
                                        className="mb-3",
                                    ),
                                    # Message
                                    html.P(
                                        message,
                                        className="text-center mb-0",
                                        style={"font-size": "1.1rem"},
                                    ),
                                    # Cancel button
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Cancel",
                                                id=f"{overlay_id}-cancel",
                                                color="secondary",
                                                size="sm",
                                                className="mt-3",
                                            )
                                        ],
                                        className="text-center",
                                    ),
                                ]
                            )
                        ],
                        style={"min-width": "300px"},
                    )
                ],
            )
        ],
        id=overlay_id,
        style={"display": "none"},
    )


def create_progress_card(
    card_id: str,
    title: str = "Processing Progress",
    show_details: bool = True,
    show_cancel_button: bool = True,
) -> dbc.Card:
    """
    Create a progress card with detailed information.

    Args:
        card_id: Unique ID for the card
        title: Title of the card
        show_details: Whether to show detailed progress info
        show_cancel_button: Whether to show cancel button

    Returns:
        dbc.Card containing the progress information
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(title, className="mb-0"),
                    (
                        dbc.Button(
                            "×",
                            id=f"{card_id}-close",
                            color="link",
                            className="float-right p-0",
                            style={"font-size": "1.5rem", "line-height": "1"},
                        )
                        if show_cancel_button
                        else None
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    # Progress bar
                    create_progress_indicator(
                        progress_id=f"{card_id}-progress",
                        label="Initializing...",
                        show_percentage=True,
                        animated=True,
                        striped=True,
                    ),
                    # Details section
                    (
                        html.Div(
                            [
                                html.H6("Details:", className="mt-3 mb-2"),
                                html.Div(
                                    id=f"{card_id}-details",
                                    children=[
                                        html.P(
                                            "Operation: Initializing", className="mb-1"
                                        ),
                                        html.P("Status: Starting", className="mb-1"),
                                        html.P("Time elapsed: 0s", className="mb-0"),
                                    ],
                                ),
                            ]
                        )
                        if show_details
                        else None
                    ),
                    # Cancel button
                    (
                        html.Div(
                            [
                                dbc.Button(
                                    "Cancel Operation",
                                    id=f"{card_id}-cancel",
                                    color="danger",
                                    size="sm",
                                    className="mt-3",
                                )
                            ],
                            className="text-center",
                        )
                        if show_cancel_button
                        else None
                    ),
                ]
            ),
        ],
        id=card_id,
        style={"display": "none"},
    )


def create_progress_list(
    list_id: str, title: str = "Active Operations", max_items: int = 5
) -> dbc.Card:
    """
    Create a list of active progress operations.

    Args:
        list_id: Unique ID for the list
        title: Title of the list
        max_items: Maximum number of items to show

    Returns:
        dbc.Card containing the progress list
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H5(title, className="mb-0"),
                    dbc.Badge(
                        "0", id=f"{list_id}-count", color="primary", className="ml-2"
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    html.Div(
                        id=f"{list_id}-items",
                        children=[
                            html.P(
                                "No active operations",
                                className="text-muted text-center",
                            )
                        ],
                    )
                ]
            ),
        ],
        id=list_id,
    )


def create_progress_interval(
    interval_id: str,
    interval_ms: int = 500,
    max_intervals: int = -1,
    disabled: bool = True,
) -> dcc.Interval:
    """
    Create an interval component for progress updates.

    Args:
        interval_id: Unique ID for the interval
        interval_ms: Update interval in milliseconds
        max_intervals: Maximum number of intervals (-1 for infinite)
        disabled: Whether to start disabled

    Returns:
        dcc.Interval component
    """
    return dcc.Interval(
        id=interval_id,
        interval=interval_ms,
        max_intervals=max_intervals,
        disabled=disabled,
    )


def create_progress_store(
    store_id: str, initial_data: Optional[Dict[str, Any]] = None
) -> dcc.Store:
    """
    Create a store component for progress data.

    Args:
        store_id: Unique ID for the store
        initial_data: Initial data for the store

    Returns:
        dcc.Store component
    """
    return dcc.Store(id=store_id, data=initial_data or {})


def create_progress_components(
    base_id: str,
    include_overlay: bool = True,
    include_card: bool = True,
    include_interval: bool = True,
    include_store: bool = True,
) -> List:
    """
    Create a complete set of progress components.

    Args:
        base_id: Base ID for all components
        include_overlay: Whether to include progress overlay
        include_card: Whether to include progress card
        include_interval: Whether to include interval component
        include_store: Whether to include store component

    Returns:
        List of progress components
    """
    components = []

    if include_overlay:
        components.append(
            create_progress_overlay(
                overlay_id=f"{base_id}-overlay", message="Processing your request..."
            )
        )

    if include_card:
        components.append(
            create_progress_card(card_id=f"{base_id}-card", title="Processing Progress")
        )

    if include_interval:
        components.append(
            create_progress_interval(interval_id=f"{base_id}-interval", interval_ms=500)
        )

    if include_store:
        components.append(create_progress_store(store_id=f"{base_id}-store"))

    return components
