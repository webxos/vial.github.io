# server/tests/test_stream.py
import pytest
from fastapi.testclient import TestClient
from server.api.stream import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet


@pytest.mark.asyncio
async def test_stream_diagram():
    """Test streaming diagram updates with wallet data."""
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
        "/stream/diagram",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "reputation" in response.json()
