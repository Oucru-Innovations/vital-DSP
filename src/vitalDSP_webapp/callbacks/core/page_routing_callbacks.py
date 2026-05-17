"""
Core page routing callbacks for vitalDSP webapp.

This module handles page navigation and routing based on URL paths.
"""

from dash.dependencies import Input, Output
from dash import html, no_update
import logging

# Import layouts from the new modular structure
from vitalDSP_webapp.layout.pages.upload_page import upload_layout
from vitalDSP_webapp.layout.pages.filtering_page import filtering_layout
from vitalDSP_webapp.layout.pages.time_domain_page import time_domain_layout
from vitalDSP_webapp.layout.pages.frequency_page import frequency_layout
from vitalDSP_webapp.layout.pages.respiratory_page import respiratory_layout

logger = logging.getLogger(__name__)


def display_page(pathname: str) -> html.Div:
    """
    Function to dynamically update the content of the main page based on the current URL path.

    Parameters
    ----------
    pathname : str
        The current URL path to determine which page to display.

    Returns
    -------
    html.Div
        The HTML content for the selected page.
    """
    logger.info("=== PAGE ROUTING CALLBACK TRIGGERED ===")
    logger.info(f"Page routing callback triggered with pathname: {pathname}")

    try:
        if pathname == "/" or pathname is None:
            # Default welcome page
            logger.info("Returning default welcome layout")
            return _get_welcome_layout()
        elif pathname == "/upload":
            logger.info("Returning upload layout")
            return upload_layout()
        elif pathname == "/time-domain":
            logger.info("Returning time domain layout")
            return time_domain_layout()
        elif pathname == "/frequency":
            logger.info("Returning frequency layout")
            return frequency_layout()
        elif pathname == "/filtering":
            logger.info("Returning filtering layout")
            return filtering_layout()
        elif pathname == "/preview":
            logger.info("Returning welcome layout for preview")
            return _get_welcome_layout()
        elif pathname == "/respiratory":
            logger.info("Returning respiratory layout")
            return respiratory_layout()
        else:
            logger.info("Returning default welcome page")
            return _get_welcome_layout()
    except Exception as e:
        logger.error(f"Error in page routing callback: {e}")
        import traceback

        traceback.print_exc()
        return _get_error_layout(str(e))


def _get_welcome_layout():
    """Returns the welcome page layout."""
    return html.Div(
        [
            html.H1(
                "Welcome to vitalDSP Dashboard",
                className="text-center mb-4",
            ),
            html.Div(
                [
                    html.H3(
                        "Digital Signal Processing for Vital Signs",
                        className="text-center mb-4",
                    ),
                    html.P(
                        [
                            "This dashboard provides access to vitalDSP features including:",
                            html.Br(),
                            "• Time and frequency domain analysis",
                            html.Br(),
                            "• Advanced signal filtering and processing",
                            html.Br(),
                            "• Signal preview and visualization",
                        ],
                        className="text-center",
                    ),
                    html.Hr(),
                    html.Div(
                        [
                            html.H4("Getting Started:"),
                            html.Ol(
                                [
                                    html.Li(
                                        "Upload your PPG/ECG data using the Upload page"
                                    ),
                                    html.Li(
                                        "Configure your data parameters (sampling frequency, etc.)"
                                    ),
                                    html.Li(
                                        "Navigate to Filtering, Time Domain, or Frequency Domain"
                                    ),
                                    html.Li(
                                        "Adjust parameters and view results in real-time"
                                    ),
                                ]
                            ),
                        ],
                        className="text-left",
                    ),
                ],
                className="container",
            ),
        ]
    )


def _get_error_layout(error_msg: str):
    """Returns the error page layout."""
    return html.Div(
        [
            html.H1("Error Loading Page", className="text-center text-danger"),
            html.P(f"An error occurred while loading the page: {error_msg}"),
            html.P("Please check the console for more details."),
        ]
    )


def register_page_routing_callbacks(app):
    """
    Registers the callback for page routing based on the URL path.

    Parameters
    ----------
    app : Dash
        The Dash app object where the callback is being registered.
    """

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page_callback(pathname: str) -> html.Div:
        """Callback wrapper for display_page function."""
        return display_page(pathname)
