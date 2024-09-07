import dash_bootstrap_components as dbc


def Header():
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
