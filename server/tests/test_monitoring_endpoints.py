```python
import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_check_success():
    """Test successful health check."""
    with patch("psutil.cpu_percent", return_value=10.0), \
         patch("psutil.virtual_memory", return_value=type('obj', (), {'percent': 20.0})), \
         patch("psutil.disk_usage", return_value=type('obj', (), {'percent': 30.0})), \
         patch("server.config.settings.settings", return_value={
             "ANTHROPIC_API_KEY": "key",
             "OBS_HOST": "localhost",
             "OBS_PORT": 4455,
             "SERVICENOW_INSTANCE": "instance"
         }):
        response = client.get("/mcp/monitoring/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["cpu_usage_percent"] == 10.0
        assert response.json()["services"]["llm_router"] == "healthy"

@pytest.mark.asyncio
async def test_health_check_failure():
    """Test health check failure."""
    with patch("psutil.cpu_percent", side_effect=Exception("System error")):
        response = client.get("/mcp/monitoring/health")
        assert response.status_code == 500
        assert "System error" in response.json()["detail"]
```
