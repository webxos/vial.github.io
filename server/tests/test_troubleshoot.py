import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_troubleshoot():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.get(
        "/troubleshoot",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "health" in response.json()
    assert "recent_logs" in response.json()
    assert response.json()["health"]["status"] == "ok"


@pytest.mark.asyncio
async def test_troubleshoot_unauthorized():
    response = await client.get("/troubleshoot")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == 401
