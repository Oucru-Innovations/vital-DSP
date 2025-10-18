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
                    # Upload section
                    html.Div(
                        [
                            html.A(
                                [html.I(className="fas fa-upload"), " Upload Data"],
                                href="/upload",
                                className="nav-link text-white",
                            )
                        ],
                        className="mb-2",
                    ),
                    # Analysis section
                    html.H6("Analysis", className="mt-4 mb-2 text-muted"),
                    html.Div(
                        [
                            html.A(
                                [html.I(className="fas fa-filter"), " Filtering"],
                                href="/filtering",
                                className="nav-link text-white",
                            ),
                            html.A(
                                [html.I(className="fas fa-chart-line"), " Time Domain"],
                                href="/time-domain",
                                className="nav-link text-white",
                            ),
                            html.A(
                                [
                                    html.I(className="fas fa-wave-square"),
                                    " Frequency Domain",
                                ],
                                href="/frequency",
                                className="nav-link text-white",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Features section
                    html.H6("Features", className="mt-4 mb-2 text-muted"),
                    html.Div(
                        [
                            html.A(
                                [
                                    html.I(className="fas fa-heartbeat"),
                                    " Physiological",
                                ],
                                href="/physiological",
                                className="nav-link text-white",
                            ),
                            html.A(
                                [html.I(className="fas fa-lungs"), " Respiratory"],
                                href="/respiratory",
                                className="nav-link text-white",
                            ),
                            html.A(
                                [
                                    html.I(className="fas fa-chart-bar"),
                                    " Advanced Features",
                                ],
                                href="/features",
                                className="nav-link text-white",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Pipeline section
                    html.H6("Pipeline", className="mt-4 mb-2 text-muted"),
                    html.Div(
                        [
                            html.A(
                                [html.I(className="fas fa-project-diagram"), " Processing Pipeline"],
                                href="/pipeline",
                                className="nav-link text-white",
                            ),
                            html.A(
                                [html.I(className="fas fa-tasks"), " Background Tasks"],
                                href="/tasks",
                                className="nav-link text-white",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Other sections
                    html.H6("Other", className="mt-4 mb-2 text-muted"),
                    html.Div(
                        [
                            html.A(
                                [html.I(className="fas fa-eye"), " Preview"],
                                href="/preview",
                                className="nav-link text-white",
                            ),
                            html.A(
                                [html.I(className="fas fa-cog"), " Settings"],
                                href="/settings",
                                className="nav-link text-white",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Theme toggle section
                    html.H6("Theme", className="mt-4 mb-2 text-muted"),
                    html.Div(
                        [
                            html.Button(
                                id="theme-toggle",
                                className="btn btn-outline-light btn-sm w-100",
                                children=[
                                    html.I(id="theme-icon", className="fas fa-sun me-2"),
                                    html.Span(id="theme-text", children="Light")
                                ],
                                style={"font-size": "0.9rem"}
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
                    # Upload section
                    html.Div(
                        [
                            html.A(
                                html.I(className="fas fa-upload"),
                                href="/upload",
                                className="nav-icon text-white",
                                title="Upload Data",
                            )
                        ],
                        className="mb-3",
                    ),
                    # Analysis section
                    html.Div(
                        [
                            html.A(
                                html.I(className="fas fa-filter"),
                                href="/filtering",
                                className="nav-icon text-white",
                                title="Filtering",
                            ),
                            html.A(
                                html.I(className="fas fa-chart-line"),
                                href="/time-domain",
                                className="nav-icon text-white",
                                title="Time Domain",
                            ),
                            html.A(
                                html.I(className="fas fa-wave-square"),
                                href="/frequency",
                                className="nav-icon text-white",
                                title="Frequency Domain",
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Features section
                    html.Div(
                        [
                            html.A(
                                html.I(className="fas fa-heartbeat"),
                                href="/physiological",
                                className="nav-icon text-white",
                                title="Physiological",
                            ),
                            html.A(
                                html.I(className="fas fa-lungs"),
                                href="/respiratory",
                                className="nav-icon text-white",
                                title="Respiratory",
                            ),
                            html.A(
                                html.I(className="fas fa-chart-bar"),
                                href="/features",
                                className="nav-icon text-white",
                                title="Advanced Features",
                            ),
                        ],
                        className="mb-3",
                    ),
                    # Pipeline section
                    html.Div(
                        [
                            html.A(
                                html.I(className="fas fa-project-diagram"),
                                href="/pipeline",
                                className="nav-icon text-white",
                                title="Processing Pipeline",
                            ),
                            html.A(
                                html.I(className="fas fa-tasks"),
                                href="/tasks",
                                className="nav-icon text-white",
                                title="Background Tasks",
                            )
                        ],
                        className="mb-3",
                    ),
                    # Other sections
                    html.Div(
                        [
                            html.A(
                                html.I(className="fas fa-eye"),
                                href="/preview",
                                className="nav-icon text-white",
                                title="Preview",
                            ),
                            html.A(
                                html.I(className="fas fa-cog"),
                                href="/settings",
                                className="nav-icon text-white",
                                title="Settings",
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
                                    html.I(id="theme-icon-collapsed", className="fas fa-sun")
                                ],
                                title="Toggle Theme",
                                style={"width": "40px", "height": "40px", "padding": "0"}
                            ),
                        ],
                        className="mb-3",
                    ),
                ],
            ),
        ],
    )
