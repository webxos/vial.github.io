# server/tests/test_copilot_integrations.py
import pytest
from server.api.copilot_integration import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_copilot_integration():
    """Test copilot integration with reputation check."""
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
        "/copilot",
        json={"wallet_address": "test_wallet", "action": "generate"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
