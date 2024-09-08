from dash import html
import dash_bootstrap_components as dbc


def Sidebar():
    """
    Generates the sidebar component for navigation and filtering in the Dash web application.

    Returns
    -------
    html.Div
        A Dash HTML Div component containing navigation links and filters in the sidebar.

    Example
    -------
    >>> sidebar = Sidebar()
    """
    sidebar = html.Div(
        [
            html.H2("Sidebar", className="display-4"),
            html.Hr(),
            html.P("Navigation and Filters", className="lead"),
            dbc.Nav(
                [
                    dbc.NavLink("Upload Data", href="/upload", active="exact"),
                    dbc.NavLink(
                        "Data Visualization", href="/visualize", active="exact"
                    ),
                    dbc.NavLink("Settings", href="/settings", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
            html.Hr(),
            # Additional sidebar content (e.g., filters) can go here
        ],
        style={
            "position": "fixed",
            "top": "56px",  # Height of the header
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "background-color": "#f8f9fa",
        },
    )
    return sidebar
