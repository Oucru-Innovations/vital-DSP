import dash_bootstrap_components as dbc
from dash import html


def Header():
    """
    Generates the header component for the Dash web application.

    Returns
    -------
    dbc.Navbar
        A Dash Bootstrap component (dbc) Navbar containing the brand and navigation links.

    Example
    -------
    >>> header = Header()
    """
    header = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [
                        html.Img(
                            src="/assets/logo.png", height="40px"
                        ),  # Add your logo here
                        " VitalDSP Dashboard",
                    ],
                    href="#",
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.NavLink("Home", href="#", className="nav-item-icon")
                        ),
                        dbc.NavItem(
                            dbc.NavLink("About", href="#", className="nav-item-icon")
                        ),
                        dbc.NavItem(
                            dbc.NavLink("Contact", href="#", className="nav-item-icon")
                        ),
                    ],
                    navbar=True,
                ),
            ]
        ),
        color="dark",
        dark=True,
        sticky="top",
        className="header-nav",
    )
    return header
