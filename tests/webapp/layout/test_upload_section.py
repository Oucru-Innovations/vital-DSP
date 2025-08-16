import pytest
from vitalDSP_webapp.layout.upload_section import upload_layout
from dash import html, dcc
import dash_bootstrap_components as dbc


def test_upload_section():
    """
    Test that the UploadSection layout is correctly created and contains the expected components.
    """
    # upload_layout is a function, so we need to call it first
    layout = upload_layout()
    assert (
        type(layout) == html.Div
    ), "UploadSection layout is not rendering correctly."

    # Ensure that the children are present and are of expected type
    children = layout.children
    assert len(children) == 7, "Upload layout should have 7 children: header, main row, preview div, status div, and 3 stores."

    # Check the first child (header section)
    assert type(children[0]) == html.Div, "First child should be a header div."
    
    # Check the second child (main row with upload controls and configuration)
    assert type(children[1]) == dbc.Row, "Second child should be a dbc.Row containing upload controls and configuration."
    
    # Check the third child (data preview section div)
    assert type(children[2]) == html.Div, "Third child should be a data preview section div."
    assert children[2].id == "data-preview-section", "Third child should have id 'data-preview-section'."
    
    # Check the fourth child (processing status div)
    assert type(children[3]) == html.Div, "Fourth child should be a processing status div."
    assert children[3].id == "processing-status", "Fourth child should have id 'processing-status'."
    
    # Check the stores
    assert type(children[4]) == dcc.Store, "Fifth child should be a dcc.Store for uploaded data."
    assert children[4].id == "store-uploaded-data", "Fifth child should have id 'store-uploaded-data'."
    
    assert type(children[5]) == dcc.Store, "Sixth child should be a dcc.Store for data config."
    assert children[5].id == "store-data-config", "Sixth child should have id 'store-data-config'."
    
    assert type(children[6]) == dcc.Store, "Seventh child should be a dcc.Store for column mapping."
    assert children[6].id == "store-column-mapping", "Seventh child should have id 'store-column-mapping'."

    # Test that the main row contains the expected structure
    main_row = children[1]
    assert len(main_row.children) == 2, "Main row should have 2 columns: upload controls and configuration."
    
    # Check the first column (upload controls)
    upload_col = main_row.children[0]
    assert type(upload_col) == dbc.Col, "First column should be a dbc.Col."
    assert upload_col.md == 4, "Upload column should have md=4."
    
    # Check the second column (configuration)
    config_col = main_row.children[1]
    assert type(config_col) == dbc.Col, "Second column should be a dbc.Col."
    assert config_col.md == 8, "Configuration column should have md=8."
