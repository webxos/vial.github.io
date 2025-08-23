import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from server.api.eight_bim import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_pil():
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value._getexif.return_value = {1: "test_data"}
        yield mock_open

@pytest.mark.asyncio
async def test_8bim_process(client, mock_pil):
    with patch("pymongo.collection.Collection.insert_one") as mock_insert:
        with open("test_image.png", "wb") as f:
            f.write(b"mock_image_data")
        with open("test_image.png", "rb") as f:
            response = client.post("/v1/8bim/process", files={"image": f})
        assert response.status_code == 200
        assert "metadata" in response.json()
        assert mock_insert.called

@pytest.mark.asyncio
async def test_8bim_prompt_shield(client):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
        mock_shield.return_value.json.return_value = {"malicious": True}
        with open("test_image.png", "rb") as f:
            response = client.post("/v1/8bim/process", files={"image": f})
        assert response.status_code == 400
        assert "Malicious input detected" in response.json()["detail"]
