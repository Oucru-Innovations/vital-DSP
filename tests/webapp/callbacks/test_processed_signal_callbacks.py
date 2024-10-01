import pytest
from dash import dcc, html
from dash.testing.application_runners import import_app
from selenium import webdriver
from vitalDSP_webapp.app import create_dash_app


# Fixture for setting up the Dash app
@pytest.fixture
def dash_app():
    # Import the app directly from your app file
    app = create_dash_app()

    # Define the layout for testing
    app.layout = html.Div(
        [dcc.Input(id="input-box", value=""), html.Div(id="output-div")]
    )
    return app


# Hook to set up WebDriver options for headless mode
def pytest_setup_options():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Enable headless mode
    options.add_argument("--disable-gpu")  # Disable GPU acceleration
    options.add_argument("--no-sandbox")  # Bypass OS security model
    options.add_argument(
        "--disable-dev-shm-usage"
    )  # Overcome limited resource problems
    options.add_argument(
        "--window-size=1920,1080"
    )  # Set screen size for headless Chrome
    return options
