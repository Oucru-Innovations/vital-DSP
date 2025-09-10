"""
Footer component for vitalDSP webapp.

This module provides the main footer component for the application.
"""

from dash import html


def Footer():
    """Create the main footer component."""
    return html.Footer(
        className="footer",
        style={
            "position": "fixed",
            "bottom": "0",
            "left": "0",
            "right": "0",
            "height": "40px",
            "backgroundColor": "#2c3e50",
            "color": "white",
            "zIndex": 1000,
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontSize": "12px",
            "opacity": "0.8",
        },
        children=[
            html.Span("© 2024 vitalDSP - Digital Signal Processing for Vital Signs")
        ],
    )
