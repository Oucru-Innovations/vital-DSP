import base64
import io
import pandas as pd
# from dash import dcc, html
from dash.dependencies import Input, Output, State
from webapp.app import app


# Callback to handle file upload
@app.callback(
    Output("upload-status", "children"),
    Output("uploaded-data-plot", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def handle_file_upload(contents, filename):
    if contents is None:
        return "No file uploaded.", {}

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # Handle different file types: CSV, TXT, EDF (using Pandas)
        if filename.endswith(".csv"):
            # Load CSV file into a DataFrame
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith(".txt"):
            # Load TXT file into a DataFrame
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter="\t")
        else:
            return "Unsupported file type: {}".format(filename), {}

        # Process and visualize the first chunk of the data
        figure = create_plot(df)

        return "File {} uploaded successfully.".format(filename), figure

    except Exception as e:
        return "There was an error processing the file: {}".format(e), {}


# Helper function to create a basic plot for the first chunk of data
def create_plot(df):
    # Here you can load and process the first chunk of data for visualization
    fig = {
        "data": [
            {
                "x": df.index[:100],
                "y": df.iloc[:100, 1],
                "type": "line",
                "name": "Data Chunk",
            }
        ],
        "layout": {"title": "Uploaded Data (First Chunk)"},
    }
    return fig
