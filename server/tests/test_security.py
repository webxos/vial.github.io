# server/tests/test_security.py
import pytest
from fastapi.testclient import TestClient
from server.security.auth import oauth2_scheme
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet


@pytest.mark.asyncio
async def test_security_oauth():
    """Test OAuth2.0 security with Vercel integration."""
    client = TestClient(oauth2_scheme)
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    response = client.get(
        "/wallet/reputation/test_wallet",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["reputation"] == 20.0
