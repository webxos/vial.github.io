```python
import pytest
from fastapi.testclient import TestClient
from ..api.fastapi_router import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_galaxycraft_data(client):
    response = client.post(
        "/api/mcp/tools/dropship_data",
        json={"route": "mars-exploration", "time": "2023-01-01", "wallet_id": "test-wallet"},
        headers={"X-API-Key": "your_nasa_key"}
    )
    assert response.status_code == 200
    assert "data" in response.json()
```
