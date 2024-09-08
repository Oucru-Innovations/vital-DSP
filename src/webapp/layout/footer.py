from dash import html
import dash_bootstrap_components as dbc

def Footer():
    footer = dbc.Container(
        dbc.Row(
            dbc.Col(
                html.P(
                    [
                        "© 2024 vitalDSP. All rights reserved.",
                        html.Br(),
                        html.A("Privacy Policy", href="#"),
                        " | ",
                        html.A("Terms of Service", href="#"),
                    ],
                    className="text-center",
                )
            )
        ),
        fluid=True,
        className="footer",
    )
    return footer