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
            # Stylish hamburger toggle button
            html.Div(
                dbc.Button(
                    html.I(
                        className="fas fa-bars", id="toggle-icon"
                    ),  # 3-dash "hamburger" icon
                    id="sidebar-toggle",
                    n_clicks=0,
                    className="toggle-button",
                ),
                style={"text-align": "left"},  # Center the button horizontally
            ),
            # Navigation items with icons and text
            dbc.Nav(
                [
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-home"),  # Icon
                            html.Span(" Home", className="nav-text"),  # Text
                        ],
                        href="/",
                        active="exact",
                        className="nav-item",
                    ),
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-upload"),  # Icon
                            html.Span(" Upload", className="nav-text"),  # Text
                        ],
                        href="/upload",
                        active="exact",
                        className="nav-item",
                    ),
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-chart-line"),  # Icon
                            html.Span(" Visualize", className="nav-text"),  # Text
                        ],
                        href="/visualize",
                        active="exact",
                        className="nav-item",
                    ),
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-cog"),  # Icon
                            html.Span(" Settings", className="nav-text"),  # Text
                        ],
                        href="/settings",
                        active="exact",
                        className="nav-item",
                    ),
                ],
                vertical=True,
                pills=True,
                className="sidebar-nav",
            ),
        ],
        id="sidebar",
        className="sidebar-expanded",  # Start in expanded state
    )
    return sidebar
