# server/tests/test_security.py
import pytest
from fastapi.testclient import TestClient
from server.security.auth import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_oauth_security():
    """Test OAuth2.0 security with reputation check."""
    client = TestClient(router)
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    response = client.post(
        "/token",
        data={"username": "test_user", "password": "test_pass"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    
    token = response.json()["access_token"]
    response = client.get(
        "/wallet/reputation/test_wallet",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["reputation"] == 20.0
