import pytest
import subprocess
import asyncio
import httpx


@pytest.mark.asyncio
async def test_uvicorn_server():
    """
    Test that the Uvicorn server starts and responds correctly.
    """
    # Run the Uvicorn server as a subprocess
    process = subprocess.Popen(
        ["uvicorn", "vitalDSP_webapp.run_webapp:fastapi_app", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Allow the server some time to start up
    await asyncio.sleep(25)

    try:
        async with httpx.AsyncClient() as client:
            # Test the Dash homepage
            response = await client.get("http://localhost:8000/")
            assert (
                response.status_code == 200
            ), "Uvicorn should serve Dash homepage successfully."
            assert (
                "Vital-DSP Dashboard" in response.text
            ), "Dash homepage should contain 'Vital-DSP Dashboard'."

            # Test the FastAPI route
            response = await client.get(
                "http://localhost:8000/api/some-endpoint"
            )  # Replace with an actual FastAPI route
            assert (
                response.status_code == 200
            ), "Uvicorn should serve FastAPI route successfully."

    finally:
        # Terminate the Uvicorn process after the test
        process.terminate()
        process.wait()
