import pytest
from webapp.layout.upload_section import upload_layout
from dash import html, dcc

def test_upload_section():
    """
    Test that the UploadSection layout is correctly created and contains the file upload component and a graph.
    """
    assert isinstance(upload_layout, html.Div), "UploadSection layout is not rendering correctly."
    
    # Check if upload section contains the Upload component and Graph
    upload_component = upload_layout.children[1]
    graph_component = upload_layout.children[3]

    assert isinstance(upload_component, dcc.Upload), "Upload section should contain dcc.Upload component."
    assert isinstance(graph_component, dcc.Graph), "Upload section should contain dcc.Graph for plotting data."
