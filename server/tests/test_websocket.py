# server/tests/test_websocket.py
import pytest
from fastapi.testclient import TestClient
from server.api.websocket import router
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_websocket():
    """Test WebSocket with reputation updates."""
    client = TestClient(router)
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({
            "type": "wallet_update",
            "wallet_address": "test_wallet"
        })
        response = websocket.receive_json()
        assert response["status"] == "success"
        assert response["reputation"] == 20.0
