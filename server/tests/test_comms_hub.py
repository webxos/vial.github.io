# server/tests/test_comms_hub.py
import pytest
from fastapi.testclient import TestClient
from server.api.comms_hub import router
from server.models.webxos_wallet import Wallet
from server.services.database import SessionLocal

@pytest.mark.asyncio
async def test_websocket_comms():
    """Test WebSocket communication for wallet updates."""
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
            "type": "wallet_transaction",
            "wallet_address": "test_wallet",
            "amount": 10.0
        })
        response = websocket.receive_json()
        assert response["status"] == "success"
        assert response["reputation"] == 20.0
