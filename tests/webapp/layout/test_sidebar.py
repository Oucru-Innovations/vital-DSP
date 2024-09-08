import pytest
from webapp.layout.sidebar import Sidebar
from dash import html

def test_sidebar():
    """
    Test that the Sidebar component is correctly created and returns an html.Div with navigation links.
    """
    sidebar = Sidebar()
    assert isinstance(sidebar, html.Div), "Sidebar is not rendering correctly."
    assert hasattr(sidebar, 'children'), "Sidebar should have children."
    assert len(sidebar.children) > 0, "Sidebar should contain navigation links and filters."
