import base64
import io
import pandas as pd
from dash.dependencies import Input, Output, State
from webapp.app import app


def handle_file_upload(contents: str, filename: str) -> tuple:
    """
    Callback to handle file upload from the Dash front end. Supports CSV and TXT file types for now.

    Parameters
    ----------
    contents : str
        The content of the uploaded file, encoded as a base64 string.
    filename : str
        The name of the uploaded file, used to determine file type.

    Returns
    -------
    tuple
        A message indicating the status of the upload and a Plotly figure object visualizing the first
        chunk of the data.

    Example
    -------
    >>> handle_file_upload("data:text/csv;base64,SGVsbG8sV29ybGQ=", "sample.csv")
    ("File sample.csv uploaded successfully.", figure)

    Notes
    -----
    - The callback will automatically trigger upon file upload in the Dash interface.
    - For now, only CSV and TXT files are supported. Other file types can be added.
    """
    if contents is None:
        return "No file uploaded.", {}

    # Split the base64 contents and decode the data
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # Handle CSV and TXT file formats, load them into a DataFrame using Pandas
        if filename.endswith(".csv"):
            # Load CSV file into a Pandas DataFrame
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith(".txt"):
            # Load TXT file (tab-separated) into a Pandas DataFrame
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter="\t")
        else:
            return f"Unsupported file type: {filename}", {}

        # Create a visualization for the first chunk of data
        figure = create_plot(df)

        return f"File {filename} uploaded successfully.", figure

    except Exception as e:
        return f"There was an error processing the file: {e}", {}


def create_plot(df: pd.DataFrame) -> dict:
    """
    Helper function to generate a simple Plotly figure for the first chunk of the uploaded data.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame containing the uploaded data. The first chunk of data will be visualized.

    Returns
    -------
    dict
        A dictionary containing the Plotly figure object for visualizing the first 100 rows of data.

    Example
    -------
    >>> df = pd.DataFrame({"A": range(100), "B": range(100, 200)})
    >>> create_plot(df)
    {'data': [{'x': [0, 1, 2, ..., 99], 'y': [100, 101, 102, ..., 199], 'type': 'line', 'name': 'Data Chunk'}],
     'layout': {'title': 'Uploaded Data (First Chunk)'}}
    """
    # Generate a Plotly figure for the first 100 rows of the data
    fig = {
        "data": [
            {
                "x": df.index[:100],  # X-axis: Index of the first 100 rows
                "y": df.iloc[:100, 1],  # Y-axis: Second column of data
                "type": "line",
                "name": "Data Chunk",
            }
        ],
        "layout": {"title": "Uploaded Data (First Chunk)"},  # Layout with title
    }
    return fig


# Register the callback in the Dash app
@app.callback(
    Output("upload-status", "children"),
    Output("uploaded-data-plot", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def upload_file_callback(contents: str, filename: str) -> tuple:
    """
    Dash callback function to handle file uploads. This function is registered in the Dash app and
    triggered when a file is uploaded via the `dcc.Upload` component.

    Parameters
    ----------
    contents : str
        The content of the uploaded file, encoded as a base64 string.
    filename : str
        The name of the uploaded file, used to determine file type.

    Returns
    -------
    tuple
        A status message indicating the result of the upload, and a figure object for visualizing the
        uploaded data.
    """
    return handle_file_upload(contents, filename)
