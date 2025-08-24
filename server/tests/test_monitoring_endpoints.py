import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch

client = TestClient(app)

@pytest.mark.asyncio
async def test_metrics_success():
    """Test successful metrics retrieval."""
    with patch("prometheus_client.generate_latest", return_value=b"mocked_metrics"):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.text == "mocked_metrics"

@pytest.mark.asyncio
async def test_metrics_failure():
    """Test metrics endpoint failure."""
    with patch("prometheus_client.generate_latest", side_effect=Exception("Metrics error")):
        response = client.get("/metrics")
        assert response.status_code == 500
        assert "Metrics error" in response.json()["detail"]
