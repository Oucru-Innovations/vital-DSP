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
            
            # Navigation menu (expanded view)
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
            ),
            
            # Icon-only navigation menu (collapsed view)
            html.Div(
                className="sidebar-icons",
                children=[
                    # Upload section
                    html.Div([
                        html.A(
                            "📁",
                            href="/upload",
                            className="nav-icon text-white",
                            title="Upload Data"
                        )
                    ], className="mb-3"),
                    
                    # Analysis section
                    html.Div([
                        html.A(
                            "⏱️",
                            href="/time-domain",
                            className="nav-icon text-white",
                            title="Time Domain"
                        ),
                        html.A(
                            "📊",
                            href="/frequency",
                            className="nav-icon text-white",
                            title="Frequency Domain"
                        ),
                        html.A(
                            "🔧",
                            href="/filtering",
                            className="nav-icon text-white",
                            title="Filtering"
                        )
                    ], className="mb-3"),
                    
                    # Features section
                    html.Div([
                        html.A(
                            "❤️",
                            href="/physiological",
                            className="nav-icon text-white",
                            title="Physiological"
                        ),
                        html.A(
                            "🫁",
                            href="/respiratory",
                            className="nav-icon text-white",
                            title="Respiratory"
                        ),
                        html.A(
                            "⚡",
                            href="/features",
                            className="nav-icon text-white",
                            title="Advanced Features"
                        )
                    ], className="mb-3"),
                    
                    # Other sections
                    html.Div([
                        html.A(
                            "👁️",
                            href="/preview",
                            className="nav-icon text-white",
                            title="Preview"
                        ),
                        html.A(
                            "⚙️",
                            href="/settings",
                            className="nav-icon text-white",
                            title="Settings"
                        )
                    ], className="mb-3")
                ]
            )
        ]
    )
