# server/tests/test_stream.py
import pytest
from fastapi.testclient import TestClient
from server.api.stream import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_stream_endpoint():
    """Test streaming endpoint with reputation check."""
    client = TestClient(router)
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    token = "test_token"
    response = client.get(
        "/stream/data",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "data" in response.json()
