import pytest
from webapp.layout.header import Header
import dash_bootstrap_components as dbc

def test_header():
    """
    Test that the Header component is correctly created and returns a dbc.Navbar.
    """
    header = Header()
    assert hasattr(header, 'children'), "Header should have children."
    assert isinstance(header, dbc.Navbar), "Header is not rendering correctly."
