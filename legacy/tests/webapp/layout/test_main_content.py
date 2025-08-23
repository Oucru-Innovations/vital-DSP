import pytest
from vitalDSP_webapp.layout.main_content import MainContent
from dash import html


def test_main_content():
    """
    Test that the MainContent component is correctly created and returns an html.Div with the upload layout.
    """
    main_content = MainContent()
    assert isinstance(main_content, html.Div), "MainContent is not rendering correctly."
    assert hasattr(main_content, "children"), "MainContent should have children."
    assert (
        len(main_content.children) > 0
    ), "MainContent should contain at least one child (upload section)."
