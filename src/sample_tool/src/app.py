"""
Main Dash application for the PPG analysis tool.

This module creates and configures the Dash application with proper error handling,
logging, and configuration management.
"""

import logging

# Configure logging
import os
import sys
from pathlib import Path
from typing import Optional

from dash import Dash, html

from .callbacks import register_data_callbacks, register_plot_callbacks, register_window_callbacks
from .components import APP_INDEX_STRING, create_layout
from .config.settings import settings
from .utils.exceptions import ConfigurationError, PPGError

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging with error handling for file handler
try:
    logging.basicConfig(
        level=logging.INFO if not settings.debug else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/app.log", mode="a")],
    )
except Exception as e:
    # Fallback to console-only logging if file logging fails
    logging.basicConfig(
        level=logging.INFO if not settings.debug else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    print(f"Warning: File logging failed, using console-only logging: {e}")

logger = logging.getLogger(__name__)


def create_app(config: Optional[dict] = None) -> Dash:
    """
    Create and configure the Dash application.

    Args:
        config: Optional configuration dictionary to override settings

    Returns:
        Configured Dash application instance

    Raises:
        ConfigurationError: If app configuration fails
        PPGError: For other application creation errors
    """
    try:
        # Create app with configuration
        app_config = {
            "name": __name__,
            "title": settings.app_name,
            "suppress_callback_exceptions": True,
            "external_stylesheets": [],
            "external_scripts": [],
        }

        if config:
            app_config.update(config)

        app = Dash(**app_config)

        # Set app properties
        app.title = f"{settings.app_name} — Window Mode (Wide)"
        app.index_string = APP_INDEX_STRING

        # Configure app server
        app.server.config["MAX_CONTENT_LENGTH"] = settings.max_file_size_mb * 1024 * 1024

        # Set up the layout
        app.layout = create_layout()

        # Register all callbacks
        logger.info("Registering application callbacks...")
        register_data_callbacks(app)
        register_window_callbacks(app)
        register_plot_callbacks(app)

        # Add health check endpoint
        @app.server.route("/health")
        def health_check():
            """Health check endpoint for Docker and load balancers."""
            try:
                return {
                    "status": "healthy",
                    "service": "ppg-analysis-tool",
                    "version": settings.app_version,
                    "debug": settings.debug,
                }, 200
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {"status": "unhealthy", "error": str(e)}, 500

        # Add error handlers
        @app.server.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            logger.warning(f"404 error: {error}")
            return {"error": "Not found", "status": 404}, 404

        @app.server.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            logger.error(f"Internal server error: {error}")
            return {"error": "Internal server error", "status": 500}, 500

        logger.info(f"Application '{settings.app_name}' created successfully")
        return app

    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        if isinstance(e, (ConfigurationError, PPGError)):
            raise
        raise PPGError(f"Failed to create application: {e}") from e


def create_production_app() -> Dash:
    """
    Create a production-ready version of the application.

    Returns:
        Production-configured Dash application
    """
    production_config = {
        "suppress_callback_exceptions": False,
        "external_stylesheets": [],
        "external_scripts": [],
    }

    return create_app(production_config)


# Create the app instance
try:
    app = create_app()
    logger.info("Application instance created successfully")
except Exception as e:
    logger.error(f"Failed to create application instance: {e}")
    # Create a minimal fallback app for error display
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("Application Error"),
            html.P(f"Failed to initialize the PPG analysis tool: {e}"),
            html.P("Please check the logs for more details."),
            html.P("If this problem persists, please contact support."),
        ]
    )
    logger.error("Fallback application created due to initialization failure")


if __name__ == "__main__":
    # This block runs when the script is executed directly
    try:
        logger.info("Starting PPG analysis tool...")
        app.run_server(
            debug=settings.debug, host="0.0.0.0", port=8050, dev_tools_hot_reload=settings.debug
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
