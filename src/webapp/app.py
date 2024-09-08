from dash import Dash, html, dcc, Input, Output
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
# Import layout components
from webapp.layout.header import Header
from webapp.layout.footer import Footer
from webapp.layout.sidebar import Sidebar
from webapp.layout.main_content import MainContent
from webapp.layout.upload_section import upload_layout
import dash_bootstrap_components as dbc
# Import FastAPI routes
from webapp.api.endpoints import router as api_router

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Vital-DSP Dashboard"

# Set the layout
# Define the app layout with a Location component for routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    Header(),
    Sidebar(),
    # MainContent(),
    html.Div(id='page-content', style={
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }),
    Footer(),
])
# Define callback for page routing
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/upload':
        return upload_layout
    elif pathname == '/visualize':
        return html.Div([html.H3("Data Visualization Page")])
    elif pathname == '/settings':
        return html.Div([html.H3("Settings Page")])
    else:
        return html.Div([html.H3("Welcome to vitalDSP Dashboard")])

server = app.server

fastapi_app = FastAPI()

fastapi_app.include_router(api_router)

# Wrap Dash app in FastAPI
fastapi_app.mount("/", WSGIMiddleware(server))
