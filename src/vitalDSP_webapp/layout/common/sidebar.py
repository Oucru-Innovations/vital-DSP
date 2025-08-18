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
                children=[
                    html.I(
                        id="sidebar-toggle-icon",
                        className="fas fa-bars"
                    )
                ]
            ),
            
            # Navigation menu
            html.Div(
                className="sidebar-content",
                children=[
                    html.H5("Navigation", className="mb-3"),
                    
                    # Upload section
                    html.Div([
                        html.A(
                            "📁 Upload Data",
                            href="/upload",
                            className="nav-link text-white"
                        )
                    ], className="mb-2"),
                    
                    # Analysis section
                    html.H6("Analysis", className="mt-4 mb-2 text-muted"),
                    html.Div([
                        html.A(
                            "⏱️ Time Domain",
                            href="/time-domain",
                            className="nav-link text-white"
                        ),
                        html.A(
                            "📊 Frequency Domain",
                            href="/frequency",
                            className="nav-link text-white"
                        ),
                        html.A(
                            "🔧 Filtering",
                            href="/filtering",
                            className="nav-link text-white"
                        )
                    ], className="mb-2"),
                    
                    # Features section
                    html.H6("Features", className="mt-4 mb-2 text-muted"),
                    html.Div([
                        html.A(
                            "❤️ Physiological",
                            href="/physiological",
                            className="nav-link text-white"
                        ),
                        html.A(
                            "🫁 Respiratory",
                            href="/respiratory",
                            className="nav-link text-white"
                        ),
                        html.A(
                            "⚡ Advanced Features",
                            href="/features",
                            className="nav-link text-white"
                        )
                    ], className="mb-2"),
                    
                    # Other sections
                    html.H6("Other", className="mt-4 mb-2 text-muted"),
                    html.Div([
                        html.A(
                            "👁️ Preview",
                            href="/preview",
                            className="nav-link text-white"
                        ),
                        html.A(
                            "⚙️ Settings",
                            href="/settings",
                            className="nav-link text-white"
                        )
                    ], className="mb-2")
                ]
            )
        ]
    )
