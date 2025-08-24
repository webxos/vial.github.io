```python
import pytest
import httpx
from fastapi.testclient import TestClient
from ..api.fastapi_router import app
from ..agents.astronomy import AstronomyAgent

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_gibs_fetch(client):
    agent = AstronomyAgent()
    response = await agent.fetch_gibs_data({
        "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "time": "2023-01-01",
        "wallet_id": "test-wallet"
    })
    assert response["gibs"]["layer"] == "MODIS_Terra_CorrectedReflectance_TrueColor"
    assert response["gibs"]["time"] == "2023-01-01"
    assert "url" in response["gibs"]

@pytest.mark.asyncio
async def test_gibs_endpoint(client):
    headers = {"X-API-Key": "test-key"}  # Replace with actual key in tests
    response = client.post("/tools/gibs_data", json={"layer": "MODIS_Terra_CorrectedReflectance_TrueColor", "time": "2023-01-01"}, headers=headers)
    assert response.status_code == 401  # Expect failure without correct key
```
