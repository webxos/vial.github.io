```python
import pytest
from fastapi.testclient import TestClient
from ..api.fastapi_router import app
from ..api.mcp_alchemist import MCPAlchemist

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_alchemist_coordination(client):
    alchemist = MCPAlchemist()
    response = await alchemist.coordinate_agents({"route": "moon-mars", "time": "2023-01-01"}, "test-wallet")
    assert "agents" in response
    assert response["agents"] == "coordinated"
    assert "data" in response
    assert response["data"]["wallet_id"] == "test-wallet"
```
