import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WebXOSWallet


client = TestClient(app)


@pytest.mark.asyncio
async def test_train_vials():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/vials/train",
        json={"network_id": "test_network", "content": "test_content", "filename": "test.py"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "trained"
    assert "balance_earned" in response.json()
    assert "vials" in response.json()
    assert response.json()["vials"]["vial1"]["trained"] is True


@pytest.mark.asyncio
async def test_reset_vials():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/vials/reset",
        json={"network_id": "test_network"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "reset"


@pytest.mark.asyncio
async def test_wallet_compatibility():
    wallet_manager = WebXOSWallet()
    old_format = {
        "Network ID": "test_network",
        "Balance": "100.0 $WEBXOS",
        "Transactions": [{"id": "tx1", "amount": 0.1, "timestamp": "2025-08-21T12:00:00Z"}]
    }
    response = await client.post(
        "/import",
        json=old_format,
        headers={"Authorization": f"Bearer {token_response.json()['access_token']}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "imported"
    balance = await wallet_manager.get_balance("test_network")
    assert balance >= 100.0
