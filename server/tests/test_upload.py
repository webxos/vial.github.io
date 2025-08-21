import pytest
import aiofiles
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WebXOSWallet, WalletModel
from unittest.mock import AsyncMock, patch
import json
from datetime import datetime

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_wallet():
    wallet = WebXOSWallet()
    wallet.update_wallet = AsyncMock(return_value=WalletModel(
        user_id="test_user",
        balance=0.0001,
        transactions=[{"type": "upload", "amount": 0.0001, "timestamp": datetime.utcnow().isoformat()}]
    ))
    return wallet

@pytest.mark.asyncio
async def test_upload_new_wallet_format(client, mock_wallet):
    with patch("server.api.upload.WebXOSWallet", return_value=mock_wallet):
        async with aiofiles.tempfile.NamedTemporaryFile(suffix=".md", mode="w+") as f:
            await f.write(
                "# WebXOS Vial and Wallet Export\n\n"
                "## Wallet\n- Wallet Key: test_user\n- Session Balance: 0.0000 $WEBXOS\n"
            )
            await f.seek(0)
            response = client.post("/upload", files={"file": ("wallet.md", f, "text/markdown")})
            assert response.status_code == 200
            assert response.json()["user_id"] == "test_user"
            assert response.json()["balance"] == 0.0001
            mock_wallet.update_wallet.assert_called_once()

@pytest.mark.asyncio
async def test_upload_old_wallet_format(client, mock_wallet):
    with patch("server.api.upload.WebXOSWallet", return_value=mock_wallet):
        async with aiofiles.tempfile.NamedTemporaryFile(suffix=".md", mode="w+") as f:
            await f.write(
                "# Wallet Export\n\n"
                "- Network ID: test_network\n- Balance: 0.0000 $WEBXOS\n"
            )
            await f.seek(0)
            response = client.post("/upload", files={"file": ("wallet.md", f, "text/markdown")})
            assert response.status_code == 200
            assert response.json()["user_id"] == "test_user"
            assert response.json()["balance"] == 0.0001
            mock_wallet.update_wallet.assert_called_once()

@pytest.mark.asyncio
async def test_upload_invalid_file(client, mock_wallet):
    with patch("server.api.upload.WebXOSWallet", return_value=mock_wallet):
        async with aiofiles.tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as f:
            await f.write("Invalid content")
            await f.seek(0)
            response = client.post("/upload", files={"file": ("invalid.txt", f, "text/plain")})
            assert response.status_code == 400
            assert "Invalid wallet format" in response.json()["detail"]
