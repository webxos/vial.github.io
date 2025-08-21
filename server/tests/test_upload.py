import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WebXOSWallet
from fastapi import UploadFile
import io


client = TestClient(app)


@pytest.mark.asyncio
async def test_upload_file():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    file_content = b"test content"
    response = await client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        data={"network_id": "test_network"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["filePath"] == "/uploads/test.txt"
    wallet_manager = WebXOSWallet()
    balance = await wallet_manager.get_balance("test_network")
    assert balance > 0.0


@pytest.mark.asyncio
async def test_upload_unauthorized():
    response = await client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(b"test content"), "text/plain")},
        data={"network_id": "test_network"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == 401
