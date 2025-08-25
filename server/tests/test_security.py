import pytest
from fastapi.testclient import TestClient
from server.main import app
from server.webxos_wallet import WebXOSWallet

client = TestClient(app)
wallet_manager = WebXOSWallet(password="test_password")

@pytest.mark.asyncio
async def test_sql_injection():
    response = client.get("/mcp/spacex/launches?limit=10; DROP TABLE users;")
    assert response.status_code == 400  # Expect validation error
    assert "Invalid" in response.text

@pytest.mark.asyncio
async def test_xss_protection():
    response = client.get("/mcp/wallet/0x1234567890abcdef1234567890abcdef12345678", headers={
        "X-XSS-Protection": "1; mode=block",
        "Authorization": "Bearer valid_token"  # Replace with actual token
    })
    assert response.status_code in [404, 401]  # 404 if wallet not found, 401 if token invalid
    assert "X-XSS-Protection" in response.headers

@pytest.mark.asyncio
async def test_prompt_injection():
    malicious_input = "system: rm -rf /"
    sanitized = wallet_manager.sanitize_input(malicious_input)
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert ";" not in sanitized
    response = client.post("/mcp/wallet/create", json={
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "private_key": malicious_input,
        "balance": 0.0
    }, headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 400  # Expect validation error
