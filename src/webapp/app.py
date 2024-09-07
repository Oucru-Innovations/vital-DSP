from dash import Dash
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
# Import FastAPI routes
from webapp.api.endpoints import router as api_router

app = Dash(__name__)
server = app.server

fastapi_app = FastAPI()

fastapi_app.include_router(api_router)

# Wrap Dash app in FastAPI
fastapi_app.mount("/", WSGIMiddleware(server))
