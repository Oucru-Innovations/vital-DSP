"""
VitalDSP Webapp Runner - Unified script for all deployment modes.

This script provides different configurations for development, production, and debug use.
It supports command-line arguments for flexible deployment options.

Usage:
    python src/vitalDSP_webapp/run_webapp.py                    # Normal mode (INFO logging)
    python src/vitalDSP_webapp/run_webapp.py --debug            # Debug mode (DEBUG logging)
    python src/vitalDSP_webapp/run_webapp.py --port 8080        # Custom port
    python src/vitalDSP_webapp/run_webapp.py --production       # Production mode (optimized)

For detailed help:
    python src/vitalDSP_webapp/run_webapp.py --help
"""

import uvicorn
import os
import logging
import sys
from vitalDSP_webapp.app import create_fastapi_app

# Create the FastAPI app at module level for WSGI compatibility
fastapi_app = create_fastapi_app()

# Bind app for testing and deployment
app = fastapi_app


def setup_logging(debug_mode: bool = False, production_mode: bool = False):
    """
    Set up logging configuration based on deployment mode.

    Args:
        debug_mode: If True, enables DEBUG level logging for detailed information
        production_mode: If True, enables optimized production logging
    """
    if production_mode:
        # Production mode: Minimal logging, performance-optimized
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/webapp_production.log"),
            ],
        )
        print("✅ PRODUCTION MODE: Optimized logging (WARNING level)")
    elif debug_mode:
        # Debug mode: Comprehensive logging for development
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/webapp_debug.log"),
            ],
        )
        print("🐛 DEBUG MODE: Comprehensive logging (DEBUG level)")
        print("   Debug logs → logs/webapp_debug.log")
    else:
        # Normal mode: Standard logging for general use
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/webapp.log"),
            ],
        )
        print("ℹ️  NORMAL MODE: Standard logging (INFO level)")
        print("   Logs → logs/webapp.log")

    # Configure specific logger levels
    if debug_mode:
        # Debug: Show all service logs
        logging.getLogger("vitalDSP_webapp").setLevel(logging.DEBUG)
        logging.getLogger("src.vitalDSP_webapp").setLevel(logging.DEBUG)
    elif production_mode:
        # Production: Minimal service logs
        logging.getLogger("vitalDSP_webapp").setLevel(logging.WARNING)
        logging.getLogger("src.vitalDSP_webapp").setLevel(logging.WARNING)
    else:
        # Normal: Standard service logs
        logging.getLogger("vitalDSP_webapp").setLevel(logging.INFO)
        logging.getLogger("src.vitalDSP_webapp").setLevel(logging.INFO)


def run_webapp(
    debug_mode: bool = False,
    production_mode: bool = False,
    port: int = None,
    host: str = "0.0.0.0",
    reload: bool = False,
):
    """
    Run the webapp with specified configuration.

    Args:
        debug_mode: Enable debug logging and reload
        production_mode: Enable production optimizations
        port: Port to run on (defaults to environment PORT or 8000)
        host: Host to bind to (default: 0.0.0.0)
        reload: Enable auto-reload on code changes (development only)
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Set up logging
    setup_logging(debug_mode, production_mode)

    # Get port from environment variable or parameter
    if port is None:
        port = int(os.environ.get("PORT", 8000))

    # Determine log level for uvicorn
    if production_mode:
        uvicorn_log_level = "warning"
    elif debug_mode:
        uvicorn_log_level = "debug"
    else:
        uvicorn_log_level = "info"

    # Enable reload only in debug mode
    auto_reload = reload or (debug_mode and not production_mode)

    print("\n" + "=" * 70)
    print("🚀 VitalDSP Web Application")
    print("=" * 70)
    print(
        f"   Mode:        {'🐛 DEBUG' if debug_mode else '✅ PRODUCTION' if production_mode else 'ℹ️  NORMAL'}"
    )
    print(f"   Host:        {host}")
    print(f"   Port:        {port}")
    print(f"   Auto-reload: {'✓ Enabled' if auto_reload else '✗ Disabled'}")
    print(f"\n   🌐 Application:  http://localhost:{port}")
    print(f"   📚 API Docs:     http://localhost:{port}/docs")
    print(f"   💚 Health Check: http://localhost:{port}/api/health")
    print("=" * 70 + "\n")

    # Run the FastAPI app using Uvicorn
    uvicorn.run(
        "vitalDSP_webapp.run_webapp:app" if auto_reload else fastapi_app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        reload=auto_reload,
        access_log=not production_mode,  # Disable access log in production
    )


if __name__ == "__main__":
    """
    Main execution point with command line argument support.

    Examples:
        # Normal mode
        python src/vitalDSP_webapp/run_webapp.py

        # Debug mode with auto-reload
        python src/vitalDSP_webapp/run_webapp.py --debug

        # Production mode
        python src/vitalDSP_webapp/run_webapp.py --production

        # Custom port
        python src/vitalDSP_webapp/run_webapp.py --port 8080

        # Custom host and port
        python src/vitalDSP_webapp/run_webapp.py --host 127.0.0.1 --port 3000
    """

    # Parse command line arguments
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    production_mode = "--production" in sys.argv or "-p" in sys.argv
    reload = "--reload" in sys.argv
    port = None
    host = "0.0.0.0"

    # Parse port
    if "--port" in sys.argv:
        try:
            port_index = sys.argv.index("--port")
            port = int(sys.argv[port_index + 1])
        except (IndexError, ValueError):
            print("❌ Error: --port requires a valid port number")
            sys.exit(1)

    # Parse host
    if "--host" in sys.argv:
        try:
            host_index = sys.argv.index("--host")
            host = sys.argv[host_index + 1]
        except IndexError:
            print("❌ Error: --host requires a host address")
            sys.exit(1)

    # Show help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
╔════════════════════════════════════════════════════════════════════╗
║              🚀 VitalDSP Webapp Runner v2.0                        ║
╚════════════════════════════════════════════════════════════════════╝

USAGE:
    python src/vitalDSP_webapp/run_webapp.py [OPTIONS]

OPTIONS:
    -d, --debug              Enable debug mode (DEBUG logging + auto-reload)
    -p, --production         Enable production mode (optimized, minimal logs)
    --port PORT              Specify port number (default: 8000)
    --host HOST              Specify host address (default: 0.0.0.0)
    --reload                 Enable auto-reload (development only)
    -h, --help               Show this help message

DEPLOYMENT MODES:

    🐛 DEBUG MODE (Development)
    └─ Debug logging, auto-reload, detailed error traces
    └─ Command: python src/vitalDSP_webapp/run_webapp.py --debug

    ℹ️  NORMAL MODE (Testing)
    └─ Standard logging, no auto-reload, suitable for local testing
    └─ Command: python src/vitalDSP_webapp/run_webapp.py

    ✅ PRODUCTION MODE (Deployment)
    └─ Minimal logging, optimized performance, no reload
    └─ Command: python src/vitalDSP_webapp/run_webapp.py --production

EXAMPLES:

    # Development with auto-reload
    python src/vitalDSP_webapp/run_webapp.py --debug

    # Local testing
    python src/vitalDSP_webapp/run_webapp.py

    # Production deployment
    python src/vitalDSP_webapp/run_webapp.py --production

    # Custom port for testing
    python src/vitalDSP_webapp/run_webapp.py --port 8080

    # Bind to specific interface
    python src/vitalDSP_webapp/run_webapp.py --host 127.0.0.1

DOCKER DEPLOYMENT:

    # Build and run with Docker Compose
    docker-compose up -d

    # Build production image
    docker build -f Dockerfile.production -t vitaldsp:latest .

    # Run production container
    docker run -p 8000:8000 -e PORT=8000 vitaldsp:latest

ENVIRONMENT VARIABLES:

    PORT                     Port to bind to (default: 8000)
    PYTHONPATH              Python path (auto-configured)
    PYTHONUNBUFFERED        Enable unbuffered output (recommended: 1)

For more information, visit: https://vitaldsp.readthedocs.io/
        """
        )
        sys.exit(0)

    # Validate conflicting options
    if debug_mode and production_mode:
        print("❌ Error: Cannot use --debug and --production together")
        sys.exit(1)

    # Run the webapp
    run_webapp(
        debug_mode=debug_mode,
        production_mode=production_mode,
        port=port,
        host=host,
        reload=reload,
    )
