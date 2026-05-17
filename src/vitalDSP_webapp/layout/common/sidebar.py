"""
Sidebar component for vitalDSP webapp.

This module provides the main navigation sidebar component.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def Sidebar():
    """Create the main sidebar component."""
    return html.Div(
        id="sidebar",
        className="sidebar sidebar-expanded",
        children=[
            # Sidebar toggle button
            html.Button(
                id="sidebar-toggle",
                className="sidebar-toggle btn btn-link text-white",
                children=[html.I(id="sidebar-toggle-icon", className="fas fa-bars")],
            ),
            # Navigation menu (expanded view)
            html.Div(
                className="sidebar-content",
                children=[
                    html.H5("Navigation", className="mb-3"),
                    # Introduction
                    html.Div(
                        [
                            dcc.Link(
                                [html.I(className="fas fa-home"), " Introduction"],
                                href="/preview",
                                className="nav-link",
                            )
                        ],
                        className="mb-2",
                    ),
                    # Preprocessing section
                    html.H6("Preprocessing", className="mt-3 mb-2"),
                    html.Div(
                        [
                            dcc.Link(
                                [html.I(className="fas fa-upload"), " Upload Data"],
                                href="/upload",
                                className="nav-link",
                            ),
                            dcc.Link(
                                [html.I(className="fas fa-filter"), " Filtering"],
                                href="/filtering",
                                className="nav-link",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Analysis section
                    html.H6("Analysis", className="mt-3 mb-2"),
                    html.Div(
                        [
                            dcc.Link(
                                [html.I(className="fas fa-chart-line"), " Time Domain"],
                                href="/time-domain",
                                className="nav-link",
                            ),
                            dcc.Link(
                                [
                                    html.I(className="fas fa-wave-square"),
                                    " Frequency Domain",
                                ],
                                href="/frequency",
                                className="nav-link",
                            ),
                            dcc.Link(
                                [html.I(className="fas fa-lungs"), " Respiratory Rate"],
                                href="/respiratory",
                                className="nav-link",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Theme toggle
                    html.Div(
                        [
                            html.Button(
                                id="theme-toggle",
                                className="btn btn-outline-light btn-sm w-100",
                                children=[
                                    html.I(
                                        id="theme-icon", className="fas fa-sun me-2"
                                    ),
                                    html.Span(id="theme-text", children="Light"),
                                ],
                                style={"font-size": "0.9rem"},
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
            ),
            # Icon-only navigation menu (collapsed view)
            html.Div(
                className="sidebar-icons",
                children=[
                    # Introduction
                    html.Div(
                        [
                            dcc.Link(
                                html.I(className="fas fa-home"),
                                href="/preview",
                                className="nav-icon",
                                title="Introduction",
                            )
                        ],
                        className="mb-3",
                    ),
                    # Preprocessing + Analysis icons
                    html.Div(
                        [
                            dcc.Link(
                                html.I(className="fas fa-upload"),
                                href="/upload",
                                className="nav-icon",
                                title="Upload Data",
                            ),
                            dcc.Link(
                                html.I(className="fas fa-filter"),
                                href="/filtering",
                                className="nav-icon",
                                title="Filtering",
                            ),
                            dcc.Link(
                                html.I(className="fas fa-chart-line"),
                                href="/time-domain",
                                className="nav-icon",
                                title="Time Domain",
                            ),
                            dcc.Link(
                                html.I(className="fas fa-wave-square"),
                                href="/frequency",
                                className="nav-icon",
                                title="Frequency Domain",
                            ),
                            dcc.Link(
                                html.I(className="fas fa-lungs"),
                                href="/respiratory",
                                className="nav-icon",
                                title="Respiratory Rate",
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Theme toggle (collapsed view)
                    html.Div(
                        [
                            html.Button(
                                id="theme-toggle-collapsed",
                                className="btn btn-outline-light btn-sm",
                                children=[
                                    html.I(
                                        id="theme-icon-collapsed",
                                        className="fas fa-sun",
                                    )
                                ],
                                title="Toggle Theme",
                                style={
                                    "width": "40px",
                                    "height": "40px",
                                    "padding": "0",
                                },
                            ),
                        ],
                        className="mb-3",
                    ),
                ],
            ),
        ],
    )
