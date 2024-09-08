import pytest
import httpx
import responses
from fastapi.testclient import TestClient
from webapp.app import create_fastapi_app, create_dash_app
from httpx import WSGITransport  # Import WSGITransport for Dash testing

@pytest.fixture
def fastapi_client():
    app = create_fastapi_app()
    return TestClient(app)


@pytest.fixture
def dash_client():
    app = create_dash_app()
    transport = WSGITransport(app.server)
    return httpx.Client(transport=transport)


@responses.activate
@pytest.mark.asyncio
async def test_dash_homepage(dash_client):
    """
    Test that the Dash app homepage loads successfully.
    
    In CI environments, the request is mocked to avoid real server dependency.
    """
    # Mock response for http://localhost/
    responses.add(
        responses.GET, "http://localhost/", 
        body="Vital-DSP Dashboard",
        status=200
    )

    response = dash_client.get("http://localhost/")
    assert response.status_code == 200, "Dash homepage should load successfully."
    assert "Vital-DSP Dashboard" in response.text, "Homepage should contain 'Vital-DSP Dashboard'."
