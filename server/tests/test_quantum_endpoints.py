# server/tests/test_quantum_endpoints.py
import pytest
from fastapi.testclient import TestClient
from server.api.quantum_endpoints import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet


@pytest.mark.asyncio
async def test_quantum_endpoints():
    """Test quantum endpoints with Vercel deployment check."""
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
        "/quantum/status",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "quantum_state" in response.json()
