import uvicorn
from webapp.app import create_fastapi_app

fastapi_app = create_fastapi_app()

# Bind app for testing
app = fastapi_app

if __name__ == "__main__":
    """
    Main execution point to run both FastAPI and Dash apps. This script uses Uvicorn to serve the FastAPI app,
    which also mounts the Dash app at the root ("/") and the FastAPI routes at "/api/".

    Example
    -------
    >>> python -m webapp.run_webapp
    """

    # Run the FastAPI app (which also serves the Dash app) using Uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

    # Expose app for testing
