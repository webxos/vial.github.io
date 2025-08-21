import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_export_wallet():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    response = await client.post(
        "/export",
        json={"user_id": "test_user"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "exported"
    assert "wallet" in response.json()
    assert "Wallet" in response.json()["wallet"]
    assert "API Credentials" in response.json()["wallet"]
    assert "Instructions" in response.json()["wallet"]


@pytest.mark.asyncio
async def test_import_wallet():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    wallet_data = {
        "Wallet": {
            "Wallet Key": "test_user",
            "Session Balance": "100.0000 $WEBXOS",
            "Address": "test_address",
            "Hash": "test_hash"
        },
        "API Credentials": {
            "Key": "test_key",
            "Secret": "test_secret"
        }
    }
    response = await client.post(
        "/import",
        json=wallet_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "imported"
    assert response.json()["wallet"]["user_id"] == "test_user"
