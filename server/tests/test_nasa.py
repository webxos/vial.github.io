import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from server.api.nasa import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_nasa_fetch(client):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"title": "Test Image"}
        with patch("pymongo.collection.Collection.insert_one") as mock_insert:
            response = client.get("/v1/nasa/fetch?dataset=apod")
            assert response.status_code == 200
            assert "data" in response.json()
            assert mock_insert.called

@pytest.mark.asyncio
async def test_nasa_prompt_shield(client):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
        mock_shield.return_value.json.return_value = {"malicious": True}
        response = client.get("/v1/nasa/fetch?dataset=DROP TABLE users")
        assert response.status_code == 400
        assert "Malicious input detected" in response.json()["detail"]
