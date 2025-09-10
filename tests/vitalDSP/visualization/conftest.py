"""
Configuration for visualization tests.
Provides mocking for plotting functions to prevent display issues in CI.
"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_plotly_show():
    """Mock plotly figure.show() to prevent display issues in headless CI environment."""
    with patch('plotly.graph_objs.Figure.show') as mock_show:
        mock_show.return_value = None
        yield mock_show


@pytest.fixture(autouse=True)
def mock_matplotlib_show():
    """Mock matplotlib pyplot.show() to prevent display issues in headless CI environment."""
    with patch('matplotlib.pyplot.show') as mock_show:
        mock_show.return_value = None
        yield mock_show


@pytest.fixture(autouse=True)
def mock_display():
    """Mock IPython display functions to prevent display issues in headless CI environment."""
    try:
        with patch('IPython.display.display') as mock_display:
            mock_display.return_value = None
            yield mock_display
    except ImportError:
        # IPython not available, skip mocking
        yield None
