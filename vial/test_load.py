import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_high_load_auth(client):
    request_id = str(uuid.uuid4())
    
    async def simulate_requests():
        for _ in range(50):
            response = client.post("/v1/auth/token", json={"network_id": "test_net", "session_id": "test_sess"})
            assert response.status_code == 200
        logger.info(f"High load auth test passed", request_id=request_id)
    
    await simulate_requests()


@pytest.mark.asyncio
async def test_high_load_wallet(client):
    request_id = str(uuid.uuid4())
    
    async def simulate_requests():
        for _ in range(50):
            response = client.post("/v1/crud/create_wallet", json={"network_id": "test_net", "balance": 100})
            assert response.status_code == 200
        logger.info(f"High load wallet test passed", request_id=request_id)
    
    await simulate_requests()
