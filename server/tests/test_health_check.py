import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_full_health_check():
    response = await client.get("/health/full")
    assert response.status_code == 200
    assert "redis" in response.json()
    assert "mongo" in response.json()
    assert "api" in response.json()
    assert response.json()["api"]["status"] == "ok"
