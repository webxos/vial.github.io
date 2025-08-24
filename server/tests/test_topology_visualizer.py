import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_topology_visualization_success():
    """Test successful topology visualization."""
    with patch("server.services.topology_visualizer.generate_topology_visualization", new=AsyncMock()) as mock_visualize:
        mock_visualize.return_value = "<svg>mocked_svg</svg>"
        response = client.post(
            "/mcp/topology/visualize",
            json={"qubits": 2, "circuit_data": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "svg": "<svg>mocked_svg</svg>"}

@pytest.mark.asyncio
async def test_topology_visualization_failure():
    """Test topology visualization failure."""
    with patch("server.services.topology_visualizer.generate_topology_visualization", new=AsyncMock()) as mock_visualize:
        mock_visualize.side_effect = Exception("Visualization error")
        response = client.post(
            "/mcp/topology/visualize",
            json={"qubits": 2, "circuit_data": {}}
        )
        assert response.status_code == 500
        assert "Visualization error" in response.json()["detail"]
