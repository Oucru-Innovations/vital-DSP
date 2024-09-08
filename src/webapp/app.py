from dash import Dash, html, dcc
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
import dash_bootstrap_components as dbc

# Import layout components
from webapp.layout.header import Header
from webapp.layout.footer import Footer
from webapp.layout.sidebar import Sidebar

# Import FastAPI routes
from webapp.api.endpoints import router as api_router


def create_dash_app() -> Dash:
    """
    Creates and configures a Dash application instance with the required layout and callback setup.

    Returns
    -------
    app : Dash
        A Dash application object configured with layout and callbacks.

    Example
    -------
    >>> dash_app = create_dash_app()
    >>> dash_app.run_server(debug=True)
    """
    # Initialize Dash app with Bootstrap CSS theme
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Vital-DSP Dashboard"

    # Set the layout of the app, including header, sidebar, and footer
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),  # For page routing
            Header(),
            Sidebar(),
            # Placeholder for main content area, dynamically updated by the callback
            html.Div(
                id="page-content",
                style={
                    "margin-left": "18rem",  # Adjust for sidebar
                    "margin-right": "2rem",
                    "padding": "2rem 1rem",
                },
            ),
            Footer(),
        ]
    )

    # Import the callback functions from the callbacks folder
    from webapp.callbacks.page_routing_callbacks import register_page_routing_callbacks

    register_page_routing_callbacks(app)

    return app


app = create_dash_app()


def create_fastapi_app() -> FastAPI:
    """
    Creates and configures a FastAPI application instance with the Dash app mounted at the root ("/")
    and the FastAPI routes mounted under "/api/".

    Returns
    -------
    fastapi_app : FastAPI
        A FastAPI application object with Dash integrated at the root ("/") and FastAPI at "/api/".

    Example
    -------
    >>> fastapi_app = create_fastapi_app()
    >>> uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
    """
    # Initialize the FastAPI app
    fastapi_app = FastAPI()

    # Include FastAPI routes from the router module under "/api/"
    fastapi_app.include_router(api_router, prefix="/api")

    # Create the Dash app
    dash_app = create_dash_app()

    # Mount Dash app at the root ("/")
    fastapi_app.mount("/", WSGIMiddleware(dash_app.server))

    return fastapi_app
