import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from server.api.endpoints import app
from server.security.crypto_engine import EncryptionParams

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_oauth():
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock:
        mock.return_value.status_code = 200
        mock.return_value.json.return_value = {"sub": "test_user"}
        yield mock

@pytest.mark.asyncio
async def test_auth_token(client, mock_oauth):
    response = client.post("/v1/auth/token", data={"access_token": "mock_token"})
    assert response.status_code == 200
    assert "jwt" in response.json()

@pytest.mark.asyncio
async def test_wallet_export(client, mock_oauth):
    with patch("server.security.crypto_engine.CryptoEngine.encrypt") as mock_encrypt:
        mock_encrypt.return_value = EncryptionParams(data=b"encrypted", key=b"key", iv=b"iv")
        response = client.post("/v1/wallet/export", json={"address": "0x123", "amount": 100.0}, headers={"Authorization": "Bearer mock_token"})
        assert response.status_code == 200
        assert "wallet_md" in response.json()

@pytest.mark.asyncio
async def test_prompt_injection(client):
    response = client.post("/v1/wallet/export", json={"address": "DROP TABLE users; --", "amount": 100.0}, headers={"Authorization": "Bearer mock_token"})
    assert response.status_code == 400
    assert "Malicious input detected" in response.json()["detail"]
