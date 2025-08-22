# server/tests/test_visual_router.py
import pytest
from fastapi.testclient import TestClient
from server.api.visual_router import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_visual_router():
    """Test visual router with reputation visualization."""
    client = TestClient(router)
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    response = client.get(
        "/visual/data",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "visualization" in response.json()
