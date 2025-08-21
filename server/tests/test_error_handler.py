import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_http_exception_handler():
    response = await client.get("/nonexistent")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": 404, "message": "Not Found"}}


@pytest.mark.asyncio
async def test_general_exception_handler():
    # Simulate an endpoint that raises an unhandled exception
    async def broken_endpoint():
        raise Exception("Test error")
    app.get("/broken")(broken_endpoint)
    response = await client.get("/broken")
    assert response.status_code == 500
    assert response.json() == {"error": {"code": 500, "message": "Internal server error"}}
