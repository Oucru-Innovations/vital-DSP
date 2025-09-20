"""
Main application module for vitalDSP webapp.

This module provides the main Dash and FastAPI application setup.
"""

from dash import Dash, html, dcc
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
import dash_bootstrap_components as dbc

# Import configuration
from vitalDSP_webapp.config.settings import app_config, ui_styles

# Import layout components from new modular structure
from vitalDSP_webapp.layout import Header, Sidebar, Footer

# Import FastAPI routes
from vitalDSP_webapp.api.endpoints import router as api_router

# Import callbacks from new modular structure
from vitalDSP_webapp.callbacks import (
    register_sidebar_callbacks,
    register_page_routing_callbacks,
    register_upload_callbacks,
    register_vitaldsp_callbacks,
    register_frequency_filtering_callbacks,
    register_signal_filtering_callbacks,
    register_respiratory_callbacks,
    register_physiological_callbacks,
    register_features_callbacks,
    register_preview_callbacks,
)


def create_dash_app() -> Dash:
    """
    Creates and configures a Dash application instance with the required layout and callback setup.

    Returns
    -------
    app : Dash
        A Dash application object configured with layout and callbacks.
    """
    # Initialize Dash app with Bootstrap CSS theme
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
        ],
        suppress_callback_exceptions=True,
    )
    app.title = app_config.APP_NAME

    # Set the layout of the app, including header, sidebar, and footer
    app.layout = html.Div(
        id="body",
        children=[
            # Stores for data management
            dcc.Store(id="store_file_path"),
            dcc.Store(id="store_total_rows"),
            dcc.Store(id="store_window"),
            dcc.Store(id="store_theme", data=app_config.THEME),
            dcc.Store(id="store_processed_data"),
            dcc.Store(id="store_analysis_results"),
            # Global stores that persist across page changes
            dcc.Store(id="store-uploaded-data", storage_type="memory"),
            dcc.Store(id="store-data-config", storage_type="memory"),
            dcc.Location(id="url", refresh=False),  # For page routing
            Header(),
            Sidebar(),
            # Placeholder for main content area, dynamically updated by the callback
            html.Div(
                id="page-content",
                style={
                    "position": "absolute",
                    "top": f"{app_config.HEADER_HEIGHT}px",
                    "left": f"{app_config.SIDEBAR_WIDTH}px",
                    "right": "0",
                    "padding": ui_styles.SECTION_MARGIN,
                    "backgroundColor": "#ffffff",
                    "minHeight": f"calc(100vh - {app_config.HEADER_HEIGHT}px)",
                    "zIndex": 100,
                    "transition": "left 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                },
                children=[
                    # Default welcome content - will be replaced by routing callback
                    html.H1(app_config.APP_NAME, className="text-center mb-4"),
                    html.Div(
                        [
                            html.H3(
                                app_config.APP_DESCRIPTION, className="text-center mb-4"
                            ),
                            html.P(
                                [
                                    "This dashboard provides comprehensive access to all vitalDSP features including:",
                                    html.Br(),
                                    "• Time and frequency domain analysis",
                                    html.Br(),
                                    "• Advanced signal filtering and processing",
                                    html.Br(),
                                    "• Physiological feature extraction",
                                    html.Br(),
                                    "• Respiratory analysis",
                                    html.Br(),
                                    "• Signal quality assessment",
                                    html.Br(),
                                    "• Advanced computational methods",
                                    html.Br(),
                                    "• Health report generation",
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
                                                "Navigate to the analysis page of your choice"
                                            ),
                                            html.Li(
                                                "Adjust parameters and view results in real-time"
                                            ),
                                            html.Li(
                                                "Export your analysis results and reports"
                                            ),
                                        ]
                                    ),
                                ],
                                className="text-left",
                            ),
                        ],
                        className="container",
                    ),
                ],
            ),
            Footer(),
        ],
    )

    # Register callbacks AFTER app is created
    register_sidebar_callbacks(app)
    register_page_routing_callbacks(app)
    register_upload_callbacks(app)
    register_vitaldsp_callbacks(app)  # Register vitalDSP analysis callbacks
    register_frequency_filtering_callbacks(
        app
    )  # Register frequency and filtering callbacks
    register_signal_filtering_callbacks(app)  # Register signal filtering callbacks
    register_respiratory_callbacks(app)  # Register respiratory analysis callbacks
    register_physiological_callbacks(app)  # Register physiological features callbacks
    register_features_callbacks(app)  # Register feature engineering callbacks
    register_preview_callbacks(app)  # Register preview callbacks

    return app


def create_fastapi_app() -> FastAPI:
    """
    Creates and configures a FastAPI application instance with the Dash app mounted at the root ("/")
    and the FastAPI routes mounted under "/api/".

    Returns
    -------
    fastapi_app : FastAPI
        A FastAPI application object with Dash integrated at the root ("/") and FastAPI at "/api/".
    """
    # Initialize the FastAPI app
    fastapi_app = FastAPI(
        title="Vital-DSP API",
        description=app_config.APP_DESCRIPTION,
        version=app_config.APP_VERSION,
    )

    # Include FastAPI routes from the router module under "/api/"
    fastapi_app.include_router(api_router, prefix="/api")

    # Create the Dash app
    dash_app = create_dash_app()

    # Mount Dash app at the root ("/")
    fastapi_app.mount("/", WSGIMiddleware(dash_app.server))

    return fastapi_app


def run_app(debug: bool = None, host: str = None, port: int = None):
    """
    Run the application with specified parameters.

    Parameters
    ----------
    debug : bool, optional
        Enable debug mode. Defaults to app_config.DEBUG
    host : str, optional
        Host to bind to. Defaults to app_config.HOST
    port : int, optional
        Port to bind to. Defaults to app_config.PORT
    """
    import uvicorn
    import logging

    logger = logging.getLogger(__name__)

    debug = debug if debug is not None else app_config.DEBUG
    host = host or app_config.HOST
    port = port or app_config.PORT

    # Update config if needed
    if debug != app_config.DEBUG:
        app_config.DEBUG = debug

    logger.info(f"Starting {app_config.APP_NAME} v{app_config.APP_VERSION}")
    logger.info(f"Server: {host}:{port}")
    logger.info(f"Debug mode: {debug}")

    # Run the FastAPI app
    uvicorn.run(
        "vitalDSP_webapp.app:create_fastapi_app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
    )


if __name__ == "__main__":
    run_app()
