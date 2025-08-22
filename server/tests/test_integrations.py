# server/tests/test_integrations.py
import pytest
from fastapi.testclient import TestClient
from server.api.webxos_wallet import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_wallet_integration():
    """Test wallet API integration with reputation."""
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
        "/wallet/transaction",
        json={
            "address": "test_wallet",
            "amount": 10.0,
            "action": "deposit"
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["balance"] == 110.0
    
    response = client.get(
        "/wallet/reputation/test_wallet",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["reputation"] == 20.0
