import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import AsyncMock, patch

client = TestClient(app)

@pytest.mark.asyncio
async def test_stream_svg_success():
    """Test successful SVG streaming."""
    with patch("server.video.obs_handler.connect_obs", new=AsyncMock()) as mock_connect:
        mock_connect.return_value.post.return_value.json.return_value = {"status": "rendered"}
        response = client.post(
            "/obs/stream",
            json={"svg_data": "<svg></svg>", "output_format": "mp4"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "output": {"status": "rendered"}}

@pytest.mark.asyncio
async def test_stream_svg_failure():
    """Test SVG streaming failure."""
    with patch("server.video.obs_handler.connect_obs", new=AsyncMock()) as mock_connect:
        mock_connect.return_value.post.side_effect = Exception("OBS error")
        response = client.post(
            "/obs/stream",
            json={"svg_data": "<svg></svg>", "output_format": "mp4"}
        )
        assert response.status_code == 500
        assert "OBS error" in response.json()["detail"]
