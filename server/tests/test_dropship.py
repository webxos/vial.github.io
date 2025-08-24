```python
import pytest
from fastapi.testclient import TestClient
from ..api.fastapi_router import app
from ..services.dropship_service import DropshipService

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_dropship_simulation(client):
    service = DropshipService()
    response = await service.simulate_supply_chain({"route": "moon-mars", "time": "2023-01-01"}, "test-wallet")
    assert "gibs" in response
    assert "spacex" in response
    assert "higress" in response
    assert "solar" in response
    assert "obs" in response
    assert response["wallet_id"] == "test-wallet"
```
