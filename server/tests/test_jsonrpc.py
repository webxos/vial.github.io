import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_jsonrpc_create_repo():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    payload = {
        "jsonrpc": "2.0",
        "method": "create_repo",
        "params": {"repo_name": "test-repo", "private": False},
        "id": "1"
    }
    response = await client.post(
        "/jsonrpc",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["jsonrpc"] == "2.0"
    assert response.json()["id"] == "1"
    assert "result" in response.json()
    assert response.json()["result"]["status"] == "created"


@pytest.mark.asyncio
async def test_jsonrpc_invalid_method():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    payload = {
        "jsonrpc": "2.0",
        "method": "invalid_method",
        "params": {},
        "id": "2"
    }
    response = await client.post(
        "/jsonrpc",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["jsonrpc"] == "2.0"
    assert response.json()["id"] == "2"
    assert "error" in response.json()
    assert response.json()["error"]["code"] == -32601
