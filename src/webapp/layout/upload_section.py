from dash import dcc, html

# Layout for file upload interface
upload_layout = html.Div(
    [
        html.H3("Upload Data Files"),
        # File Upload Component
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
            multiple=False,  # Allow only one file to be uploaded at a time
        ),
        # Display file upload status
        html.Div(id="upload-status"),
        # Display the graph for data chunks
        dcc.Graph(id="uploaded-data-plot", figure={}),
    ]
)
