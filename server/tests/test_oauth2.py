import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_token_endpoint():
    response = await client.post("/auth/token")
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_generate_credentials():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/auth/generate-credentials",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "key" in response.json()
    assert "secret" in response.json()
