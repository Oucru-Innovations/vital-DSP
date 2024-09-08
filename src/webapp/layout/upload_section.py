from dash import html
import dash_bootstrap_components as dbc

# Layout for file upload interface
upload_layout = html.Div(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Data Upload", className="card-title"),
                    html.P("Upload your data files here."),
                    dbc.Button("Upload", color="primary"),
                ]
            ),
            className="mb-3",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Visualization", className="card-title"),
                    html.P("Visualize your data after uploading."),
                    dbc.Button("Visualize", color="primary"),
                ]
            ),
            className="mb-3",
        ),
    ],
    style={
        "margin-left": "18rem",  # Adjust for sidebar width
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    },
    className="main-content",
)
