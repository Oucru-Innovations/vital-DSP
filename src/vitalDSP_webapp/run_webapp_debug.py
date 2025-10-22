"""
Enhanced webapp runner with debug and normal mode support.

This script provides different logging configurations for development and production use.
"""

import uvicorn
import os
import logging
import sys
from vitalDSP_webapp.app import create_fastapi_app

def setup_logging(debug_mode: bool = False):
    """
    Set up logging configuration based on mode.
    
    Args:
        debug_mode: If True, enables DEBUG level logging for detailed information
    """
    if debug_mode:
        # Debug mode: Show all logs including DEBUG level
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('webapp_debug.log')
            ]
        )
        print("🔍 DEBUG MODE: All logs enabled (DEBUG level)")
        print("📝 Debug logs will be saved to: webapp_debug.log")
    else:
        # Normal mode: Show only INFO and above
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('webapp.log')
            ]
        )
        print("🚀 NORMAL MODE: Essential logs only (INFO level)")
        print("📝 Logs will be saved to: webapp.log")
    
    # Set specific logger levels for our services
    if debug_mode:
        # Debug mode: Show all enhanced data service logs
        logging.getLogger('src.vitalDSP_webapp.services.data.enhanced_data_service').setLevel(logging.DEBUG)
        logging.getLogger('vitalDSP_webapp.services.data.enhanced_data_service').setLevel(logging.DEBUG)
    else:
        # Normal mode: Only show important enhanced data service logs
        logging.getLogger('src.vitalDSP_webapp.services.data.enhanced_data_service').setLevel(logging.INFO)
        logging.getLogger('vitalDSP_webapp.services.data.enhanced_data_service').setLevel(logging.INFO)

def run_webapp(debug_mode: bool = False, port: int = None, host: str = "0.0.0.0"):
    """
    Run the webapp with specified configuration.
    
    Args:
        debug_mode: Enable debug logging
        port: Port to run on (defaults to environment PORT or 8000)
        host: Host to bind to
    """
    # Set up logging
    setup_logging(debug_mode)
    
    # Get port from environment variable or parameter
    if port is None:
        port = int(os.environ.get("PORT", 8000))
    
    # Create the FastAPI app
    fastapi_app = create_fastapi_app()
    
    print(f"\n🌐 Starting webapp on {host}:{port}")
    print(f"📊 Mode: {'DEBUG' if debug_mode else 'NORMAL'}")
    print(f"🔗 Access at: http://localhost:{port}")
    print(f"🔗 API docs at: http://localhost:{port}/docs")
    print("\n" + "="*60)
    
    # Run the FastAPI app (which also serves the Dash app) using Uvicorn
    uvicorn.run(
        fastapi_app, 
        host=host, 
        port=port,
        log_level="debug" if debug_mode else "info",
        reload=debug_mode  # Enable auto-reload in debug mode
    )

if __name__ == "__main__":
    """
    Main execution point with command line argument support.
    
    Usage:
        python src/vitalDSP_webapp/run_webapp_debug.py          # Normal mode
        python src/vitalDSP_webapp/run_webapp_debug.py --debug  # Debug mode
        python src/vitalDSP_webapp/run_webapp_debug.py --debug --port 8080  # Debug mode on port 8080
    """
    
    # Parse command line arguments
    debug_mode = "--debug" in sys.argv
    port = None
    host = "0.0.0.0"
    
    # Parse port if provided
    if "--port" in sys.argv:
        try:
            port_index = sys.argv.index("--port")
            port = int(sys.argv[port_index + 1])
        except (IndexError, ValueError):
            print("❌ Error: --port requires a valid port number")
            sys.exit(1)
    
    # Parse host if provided
    if "--host" in sys.argv:
        try:
            host_index = sys.argv.index("--host")
            host = sys.argv[host_index + 1]
        except IndexError:
            print("❌ Error: --host requires a host address")
            sys.exit(1)
    
    # Show help if requested
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
🚀 VitalDSP Webapp Runner

Usage:
    python src/vitalDSP_webapp/run_webapp_debug.py [options]

Options:
    --debug              Enable debug mode (DEBUG logging level)
    --port PORT          Specify port number (default: 8000)
    --host HOST          Specify host address (default: 0.0.0.0)
    --help, -h           Show this help message

Examples:
    python src/vitalDSP_webapp/run_webapp_debug.py                    # Normal mode
    python src/vitalDSP_webapp/run_webapp_debug.py --debug            # Debug mode
    python src/vitalDSP_webapp/run_webapp_debug.py --debug --port 8080 # Debug mode on port 8080
        """)
        sys.exit(0)
    
    # Run the webapp
    run_webapp(debug_mode=debug_mode, port=port, host=host)
