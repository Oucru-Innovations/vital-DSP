import dash_bootstrap_components as dbc


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
                dbc.NavbarBrand("vitalDSP Dashboard", href="#"),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Home", href="#")),
                        dbc.NavItem(dbc.NavLink("About", href="#")),
                        dbc.NavItem(dbc.NavLink("Contact", href="#")),
                    ],
                    navbar=True,
                ),
            ]
        ),
        color="dark",
        dark=True,
        sticky="top",
    )
    return header
