from fastapi import FastAPI

# from fastapi.staticfiles import StaticFiles
from starlette.middleware.wsgi import WSGIMiddleware
from uvicorn import run as uvicorn_run

# import os
# Import the Dash app and its layout
from webapp.app import app as dash_app
# from webapp.layout import layout

# import webapp.callbacks  # Register Dash callbacks

# Import FastAPI routes
from webapp.api.endpoints import router as api_router

# Initialize FastAPI app
fastapi_app = FastAPI()

# Include FastAPI routes
fastapi_app.include_router(api_router)

# Set up the Dash app layout
# dash_app.layout = layout

# Mount the Dash app onto FastAPI at the root ("/") instead of "/dash"
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))

# Run FastAPI server
if __name__ == "__main__":
    uvicorn_run(fastapi_app, host="0.0.0.0", port=8008)
