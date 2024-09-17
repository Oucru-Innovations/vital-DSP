import pytest
from webapp.layout.upload_section import upload_layout
from dash import html, dcc
import dash_bootstrap_components as dbc


def test_upload_section():
    """
    Test that the UploadSection layout is correctly created and contains the file upload component and a graph.
    """
    assert (
        type(upload_layout) == html.Div
    ), "UploadSection layout is not rendering correctly."

    # Ensure that the children are present and are of expected type
    children = upload_layout.children
    assert len(children) == 3, "Upload layout should have two cards."

    assert type(children[0]) == (
        dcc.Store
    ), "First child should be a dcc.Store Component."

    # Access the first Card (containing the file upload)
    card_1 = children[1]
    assert type(card_1) == (dbc.Card), "First child should be a dbc.Card."

    card_body_1 = card_1.children
    assert type(card_body_1) == (
        dbc.CardBody
    ), "First card should contain dbc.CardBody."
    # assert isinstance(card_body_1, type(dbc.CardBody)), "First card should contain dbc.CardBody."

    upload_component = card_body_1.children[1]  # Access the dcc.Upload component
    assert type(upload_component) == (
        dcc.Upload
    ), "First card should contain dcc.Upload component."
    # assert isinstance(upload_component, type(dcc.Upload)), "First card should contain dcc.Upload component."

    # Access the second Card (containing the graph)
    card_2 = children[2]
    assert type(card_2) == (dbc.Card), "Second child should be a dbc.Card."
    # assert isinstance(card_2, type(dbc.Card)), "Second child should be a dbc.Card."

    card_body_2 = card_2.children
    assert type(card_body_2) == (
        dbc.CardBody
    ), "Second card should contain dbc.CardBody."
    # assert isinstance(card_body_2, type(dbc.CardBody)), "Second card should contain dbc.CardBody."

    graph_component = card_body_2.children[1]  # Access the dcc.Graph component
    assert type(graph_component) == (
        dcc.Graph
    ), "Second card should contain dcc.Graph for data visualization."
    # assert isinstance(graph_component, type(dcc.Graph)), "Second card should contain dcc.Graph for data visualization."
