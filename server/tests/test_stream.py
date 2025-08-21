import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WebXOSWallet, WalletModel
from server.api.vial_manager import VialManager
from unittest.mock import AsyncMock, patch
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_wallet():
    wallet = WebXOSWallet()
    wallet.update_wallet = AsyncMock(return_value=WalletModel(
        user_id="test_user",
        balance=0.0001,
        transactions=[{"type": "stream_update", "vial_id": "vial1", "amount": 0.0001}]
    ))
    return wallet

@pytest.fixture
async def mock_vial_manager():
    manager = VialManager()
    manager.get_vial_states = AsyncMock(return_value={
        "vial1": {"status": "running", "qubits": []}
    })
    return manager

@pytest.mark.asyncio
async def test_stream_endpoint(client, mock_wallet, mock_vial_manager):
    with patch("server.api.stream.WebXOSWallet", return_value=mock_wallet), \
         patch("server.api.stream.VialManager", return_value=mock_vial_manager):
        with client.websocket_connect("/stream?token=test_token") as ws:
            data = ws.receive_json()
            assert data["vial_id"] == "vial1"
            assert data["state"] == {"status": "running", "qubits": []}
            assert data["wallet"]["user_id"] == "test_user"
            assert data["wallet"]["balance"] == 0.0001
            mock_wallet.update_wallet.assert_called_once()
            mock_vial_manager.get_vial_states.assert_called_once()
