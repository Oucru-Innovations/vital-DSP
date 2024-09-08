from dash import dcc, html
import dash_bootstrap_components as dbc

# Layout for file upload interface
upload_layout = html.Div(
    [
        # Hidden store to store the full dataset
        dcc.Store(id="uploaded-data-store"),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Data Upload", className="card-title"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drag and Drop or ", html.A("Select Files")]
                        ),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "margin": "10px",
                        },
                        multiple=False,  # Single file upload
                    ),
                    html.Div(id="upload-status"),
                    # Input to control the chunk size
                    dcc.Input(
                        id="chunk-size-input",
                        type="number",
                        value=100,  # Default chunk size
                        min=10,
                        step=10,
                        style={"marginTop": "10px"},
                    ),
                ]
            ),
            className="mb-3",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Visualization", className="card-title"),
                    dcc.Graph(id="uploaded-data-plot", figure={}),
                ]
            ),
            className="mb-3",
        ),
    ],
    # style={
    #     "margin-left": "18rem",  # Adjust for sidebar width
    #     "margin-right": "2rem",
    #     "padding": "2rem 1rem",
    # },
    className="main-content",
)
