import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.auth_agent import AuthAgent


client = TestClient(app)


@pytest.mark.asyncio
async def test_void_network():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    auth_agent = AuthAgent()
    await auth_agent.assign_role(token_response.json()["user_id"], "admin")
    response = await client.post(
        "/void",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "voided"


@pytest.mark.asyncio
async def test_void_unauthorized():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/void",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["error"]["message"] == "Insufficient permissions"
