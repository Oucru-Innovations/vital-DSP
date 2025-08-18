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
        style={
            "position": "fixed",
            "top": "60px",
            "left": "0",
            "width": "250px",
            "height": "calc(100vh - 60px)",
            "backgroundColor": "#34495e",
            "color": "white",
            "zIndex": 999,
            "transition": "width 0.3s ease",
            "overflowY": "auto"
        },
        children=[
            # Sidebar toggle button
            html.Button(
                id="sidebar-toggle",
                className="sidebar-toggle btn btn-link text-white",
                style={
                    "position": "absolute",
                    "top": "10px",
                    "right": "10px",
                    "border": "none",
                    "background": "none"
                },
                children=[
                    html.I(
                        id="sidebar-toggle-icon",
                        className="fas fa-bars",
                        style={"fontSize": "18px"}
                    )
                ]
            ),
            
            # Navigation menu
            html.Div(
                className="sidebar-content",
                style={"padding": "20px"},
                children=[
                    html.H5("Navigation", className="mb-3"),
                    
                    # Upload section
                    html.Div([
                        html.A(
                            "📁 Upload Data",
                            href="/upload",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        )
                    ], className="mb-2"),
                    
                    # Analysis section
                    html.H6("Analysis", className="mt-4 mb-2 text-muted"),
                    html.Div([
                        html.A(
                            "⏱️ Time Domain",
                            href="/time-domain",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        ),
                        html.A(
                            "📊 Frequency Domain",
                            href="/frequency",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        ),
                        html.A(
                            "🔧 Filtering",
                            href="/filtering",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        )
                    ], className="mb-2"),
                    
                    # Features section
                    html.H6("Features", className="mt-4 mb-2 text-muted"),
                    html.Div([
                        html.A(
                            "❤️ Physiological",
                            href="/physiological",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        ),
                        html.A(
                            "🫁 Respiratory",
                            href="/respiratory",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        ),
                        html.A(
                            "⚡ Advanced Features",
                            href="/features",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        )
                    ], className="mb-2"),
                    
                    # Other sections
                    html.H6("Other", className="mt-4 mb-2 text-muted"),
                    html.Div([
                        html.A(
                            "👁️ Preview",
                            href="/preview",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        ),
                        html.A(
                            "⚙️ Settings",
                            href="/settings",
                            className="nav-link text-white",
                            style={"textDecoration": "none", "padding": "8px 0"}
                        )
                    ], className="mb-2")
                ]
            )
        ]
    )
