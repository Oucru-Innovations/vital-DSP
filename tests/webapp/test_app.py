import pytest
import httpx
from fastapi.testclient import TestClient
from webapp.app import create_fastapi_app, create_dash_app
from httpx import WSGITransport  # Import WSGITransport for Dash testing


@pytest.fixture
def fastapi_client():
    """
    Fixture that creates a TestClient for the FastAPI app.
    
    Returns
    -------
    client : TestClient
        A TestClient for the FastAPI app.
    """
    app = create_fastapi_app()
    return TestClient(app)


@pytest.fixture
def dash_client():
    """
    Fixture that creates a Dash test client.
    
    Returns
    -------
    client : TestClient
        A TestClient for the Dash app.
    """
    app = create_dash_app()
    transport = WSGITransport(app.server)  # Use WSGITransport for Dash
    return httpx.Client(transport=transport)


@pytest.mark.asyncio
async def test_dash_homepage(dash_client):
    """
    Test that the Dash app homepage loads successfully.
    
    Parameters
    ----------
    dash_client : TestClient
        The Dash test client to send requests to the app.
    """
    response = dash_client.get("http://localhost/")
    assert response.status_code == 200, "Dash homepage should load successfully."
    assert "Vital-DSP Dashboard" in response.text, "Homepage should contain 'Vital-DSP Dashboard'."


# @pytest.mark.asyncio
# async def test_fastapi_routes(fastapi_client):
#     """
#     Test that the FastAPI routes load successfully.
    
#     Parameters
#     ----------
#     fastapi_client : TestClient
#         The FastAPI test client to send requests to the app.
#     """
#     response = fastapi_client.get("/api/some-endpoint")  # Replace with an actual FastAPI route
#     assert response.status_code == 200, "FastAPI route should load successfully."
