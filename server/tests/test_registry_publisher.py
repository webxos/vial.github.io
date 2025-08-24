import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_publish_registry_success():
    """Test successful registry publishing."""
    with patch("server.services.registry_publisher.publish_to_registry", new=AsyncMock()) as mock_publish:
        mock_publish.return_value = {"status": "published"}
        response = client.post(
            "/mcp/registry/publish",
            json={"package_name": "vial-mcp", "version": "1.0.0", "metadata": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"status": "published"}}

@pytest.mark.asyncio
async def test_publish_registry_failure():
    """Test registry publishing failure."""
    with patch("server.services.registry_publisher.publish_to_registry", new=AsyncMock()) as mock_publish:
        mock_publish.side_effect = Exception("Registry error")
        response = client.post(
            "/mcp/registry/publish",
            json={"package_name": "vial-mcp", "version": "1.0.0", "metadata": {}}
        )
        assert response.status_code == 500
        assert "Registry error" in response.json()["detail"]
