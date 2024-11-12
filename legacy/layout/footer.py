from dash import html
import dash_bootstrap_components as dbc


def Footer():
    """
    Generates the footer component for the Dash web application.

    Returns
    -------
    dbc.Container
        A Dash Bootstrap component (dbc) container with footer content, including links
        to the privacy policy and terms of service.

    Example
    -------
    >>> footer = Footer()
    """
    footer = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.P(
                    [
                        "© 2024 Oucru - VitalDSP. All rights reserved.",
                        html.Br(),
                        html.A("Privacy Policy", href="#", className="footer-link"),
                        " | ",
                        html.A("Terms of Service", href="#", className="footer-link"),
                        html.Br(),
                        html.A(
                            html.I(className="fas fa-arrow-up"),
                            href="#",
                            className="scroll-top",  # Scroll-to-top button
                        ),
                    ],
                    className="text-center",
                )
            )
        ),
        fluid=True,
        className="footer",
    )
    return footer
