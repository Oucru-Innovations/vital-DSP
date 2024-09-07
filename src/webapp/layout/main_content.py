from dash import html, dcc

main_content = html.Div(
    [
        dcc.Graph(id="signal-graph"),
        dcc.Upload(id="upload-data", children=html.Button("Upload File")),
        html.Div(id="output-data"),
    ],
    className="main-content",
)
