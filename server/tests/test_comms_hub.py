import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WebXOSWallet


client = TestClient(app)


@pytest.mark.asyncio
async def test_comms_hub():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/comms_hub",
        json={"message": "Test prompt", "network_id": "test_network"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["response"].startswith("Simulated NanoGPT response")
    wallet_manager = WebXOSWallet()
    balance = await wallet_manager.get_balance("test_network")
    assert balance > 0.0


@pytest.mark.asyncio
async def test_comms_hub_empty_message():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/comms_hub",
        json={"message": "", "network_id": "test_network"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "No prompt entered"
