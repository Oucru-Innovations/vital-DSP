import pytest
from vitalDSP_webapp.layout.footer import Footer
import dash_bootstrap_components as dbc


def test_footer():
    """
    Test that the Footer component is correctly created and returns a dbc.Container.
    """
    footer = Footer()
    assert hasattr(footer, "children"), "Footer should have children."
    assert isinstance(footer, dbc.Container), "Footer should return a dbc.Container."
    assert "© 2024 Oucru - VitalDSP" in str(
        footer
    ), "Footer should contain copyright information."
