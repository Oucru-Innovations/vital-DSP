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
            html.H2("Menu", className="display-4"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.I(className="fas fa-upload"), " Upload Data"],
                        href="/upload",
                        active="exact",
                    ),
                    dbc.NavLink(
                        [html.I(className="fas fa-chart-line"), " Data Visualization"],
                        href="/visualize",
                        active="exact",
                    ),
                    dbc.NavLink(
                        [html.I(className="fas fa-cog"), " Settings"],
                        href="/settings",
                        active="exact",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
            html.Hr(),
            dbc.Button(
                "Toggle Sidebar", id="sidebar-toggle", className="mb-3", color="primary"
            ),
            dbc.Collapse(
                dbc.Nav(
                    [
                        html.P("Additional content here..."),
                        dbc.NavLink("More Links", href="#", className="nav-item-icon"),
                    ],
                    vertical=True,
                ),
                id="sidebar-collapse",
                is_open=True,
            ),
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
        className="sidebar",
    )
    return sidebar
