from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_training_scheduler_high_load(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    async def train_vial(vial_id):
        response = client.post("/agent/train", json={"vial_id": vial_id}, headers={"Authorization": f"Bearer {token}"})
        return response.json()
    
    tasks = [train_vial(f"vial{i % 4 + 1}") for i in range(20)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = await asyncio.gather(*tasks)
    
    for result in results:
        assert result["status"] == "trained"
        assert "vial_id" in result
