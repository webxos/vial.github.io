```python
import pytest
from fastapi.testclient import TestClient
from ..api.fastapi_router import app
from ..agents.astronomy import AstronomyAgent

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_astronomy_fetch(client):
    agent = AstronomyAgent()
    response = await agent.fetch_data({"query": "2023-01-01", "wallet_id": "test-wallet"})
    assert "apod" in response
    assert "eonet" in response
    assert "spacex" in response
    assert response["wallet_id"] == "test-wallet"

@pytest.mark.asyncio
async def test_gibs_fetch(client):
    agent = AstronomyAgent()
    response = await agent.fetch_gibs_data({"layer": "MODIS_Terra_CorrectedReflectance_TrueColor", "time": "2023-01-01"})
    assert response["gibs"]["layer"] == "MODIS_Terra_CorrectedReflectance_TrueColor"
```
