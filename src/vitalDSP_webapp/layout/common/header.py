"""
Header component for vitalDSP webapp.

This module provides the main header component for the application.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def Header():
    """Create the main header component."""
    return html.Header(
        className="header",
        style={
            "position": "fixed",
            "top": "0",
            "left": "0",
            "right": "0",
            "height": "60px",
            "backgroundColor": "#2c3e50",
            "color": "white",
            "zIndex": 1000,
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "0 20px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
        children=[
            # Left side - Logo and title
            html.Div(
                [
                    html.H1(
                        "vitalDSP",
                        className="mb-0",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "color": "white",
                        },
                    ),
                    html.Span(
                        "Digital Signal Processing for Vital Signs",
                        className="ms-3",
                        style={"fontSize": "14px", "opacity": "0.8"},
                    ),
                ]
            ),
            # Right side - System status
            html.Div(
                [
                    # Memory usage indicator
                    dbc.Badge(
                        id="memory-usage-badge",
                        children="Memory: Loading...",
                        color="success",
                        className="me-2",
                        style={"font-size": "0.8rem"},
                    ),
                    # Active tasks indicator
                    dbc.Badge(
                        id="active-tasks-badge",
                        children="Tasks: 0",
                        color="info",
                        className="me-2",
                        style={"font-size": "0.8rem"},
                    ),
                    # System status
                    dbc.Badge(
                        id="system-status-badge",
                        children="System: Ready",
                        color="success",
                        style={"font-size": "0.8rem"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            # Hidden stores for system monitoring
            dcc.Store(id="memory-usage-store", data={}),
            dcc.Store(id="system-status-store", data={}),
            dcc.Interval(
                id="header-monitor-interval",
                interval=5000,  # Update every 5 seconds
                n_intervals=0,
                disabled=False,
            ),
        ],
    )
