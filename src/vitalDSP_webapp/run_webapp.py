import uvicorn
import os
from vitalDSP_webapp.app import create_fastapi_app

fastapi_app = create_fastapi_app()

# Bind app for testing
app = fastapi_app

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/healthz")  # <- match your Render setting
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    """
    Main execution point to run both FastAPI and Dash apps. This script uses Uvicorn to serve the FastAPI app,
    which also mounts the Dash app at the root ("/") and the FastAPI routes at "/api/".

    Example
    -------
    >>> python -m webapp.run_webapp
    """

    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))

    # Run the FastAPI app (which also serves the Dash app) using Uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

    # Expose app for testing
