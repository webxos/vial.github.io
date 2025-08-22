import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.api.upload import UploadManager


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_file():
    client = TestClient(app)
    with open("test_wallet.md", "w") as f:
        f.write("user_id: test_user\nbalance: 100.0")
    with open("test_wallet.md", "rb") as f:
        response = client.post("/upload", files={"file": f})
    assert response.status_code == 200
    assert response.json()["user_id"] == "test_user"
    assert response.json()["balance"] == 100.0


def test_invalid_upload():
    client = TestClient(app)
    response = client.post("/upload", files={"file": ("invalid.txt", b"invalid")})
    assert response.status_code == 400
