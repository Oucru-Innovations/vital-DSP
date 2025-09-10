"""
Header component for vitalDSP webapp.

This module provides the main header component for the application.
"""

from dash import html
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
            "padding": "0 20px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
        children=[
            html.H1(
                "vitalDSP",
                className="mb-0",
                style={"fontSize": "24px", "fontWeight": "bold", "color": "white"},
            ),
            html.Span(
                "Digital Signal Processing for Vital Signs",
                className="ms-3",
                style={"fontSize": "14px", "opacity": "0.8"},
            ),
        ],
    )
