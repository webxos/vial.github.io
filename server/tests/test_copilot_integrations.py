import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_suggest_code():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    code_snippet = {"code": "def hello(): print('Hello')"}
    response = await client.post(
        "/copilot/suggest",
        json={"code_snippet": code_snippet},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "suggestion" in response.json()
