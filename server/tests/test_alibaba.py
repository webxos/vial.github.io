import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from server.api.alibaba import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_alibaba_query(client):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"result": [{"id": "123"}]}
        with patch("pymongo.collection.Collection.insert_one") as mock_insert:
            response = client.post("/v1/alibaba/query", json={"service": "dataworks", "query": "project=test"}, headers={"Authorization": "Bearer mock_token"})
            assert response.status_code == 200
            assert "data" in response.json()
            assert mock_insert.called

@pytest.mark.asyncio
async def test_alibaba_prompt_shield(client):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
        mock_shield.return_value.json.return_value = {"malicious": True}
        response = client.post("/v1/alibaba/query", json={"service": "dataworks", "query": "DROP TABLE users"}, headers={"Authorization": "Bearer mock_token"})
        assert response.status_code == 400
        assert "Malicious input detected" in response.json()["detail"]
