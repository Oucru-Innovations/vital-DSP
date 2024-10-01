import pytest
import base64
from dash.dependencies import Input, Output, State
from vitalDSP_webapp.callbacks.upload_callbacks import handle_file_upload


# Test file upload callback
def test_handle_file_upload():
    # Simulate file content (CSV)
    file_content = "data:text/csv;base64," + base64.b64encode(b"x,y\n1,2\n3,4").decode(
        "utf-8"
    )
    filename = "test.csv"

    # Call the callback function with mock inputs
    status, figure, data = handle_file_upload(
        file_content, filename
    )  # Expecting 3 return values now

    # Check if the status is correct
    assert (
        status == "File test.csv uploaded successfully."
    ), "File upload status is incorrect."

    # Check if the figure is a dictionary (as expected by Plotly)
    assert isinstance(figure, dict), "Figure should be a dictionary."

    # Check if data is a list of dictionaries (as expected for dcc.Store)
    assert isinstance(data, list), "Data should be a list of dictionaries."
    assert len(data) == 2, "Data should contain two rows."
