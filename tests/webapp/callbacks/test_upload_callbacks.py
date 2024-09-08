import pytest
import base64
from dash.dependencies import Input, Output, State
from webapp.app import app
from webapp.callbacks.upload_callbacks import handle_file_upload

# Test file upload callback
def test_handle_file_upload():
    # Simulate file content (CSV)
    file_content = "data:text/csv;base64," + base64.b64encode(b"x,y\n1,2\n3,4").decode('utf-8')
    filename = 'test.csv'
    
    # Call the callback function with mock inputs
    status, figure = handle_file_upload(file_content, filename)
    
    # Check the status message
    assert status == "File test.csv uploaded successfully."
    
    # Check that a valid figure is returned
    assert figure is not None
    assert 'data' in figure, "The figure should contain data for plotting"
    assert len(figure['data']) > 0, "The figure should have data for plotting"
