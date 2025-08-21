import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WebXOSWallet


client = TestClient(app)


@pytest.mark.asyncio
async def test_stream_vials():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.get(
        "/stream/test_network",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    content = response.text
    assert "data:" in content
    wallet_manager = WebXOSWallet()
    balance = await wallet_manager.get_balance("test_network")
    assert balance > 0.0


@pytest.mark.asyncio
async def test_stream_unauthorized():
    response = await client.get("/stream/test_network")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == 401
