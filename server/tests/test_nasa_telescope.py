import pytest
from fastapi.testclient import TestClient
from server.main import app
from server.nasa_tools.nasa_telescope import NASATelescopeProcessor

client = TestClient(app)
telescope = NASATelescopeProcessor()

@pytest.mark.asyncio
async def test_fetch_telescope_data():
    data = await telescope.fetch_telescope_data()
    assert "date" in data
    assert "url" in data

@pytest.mark.asyncio
async def test_process_image():
    # Mock image data
    mock_image_url = "https://example.com/image.jpg"
    result = await telescope.process_image(mock_image_url)
    assert "processed_image" in result
    assert len(result["processed_image"]) > 0
    assert len(result["processed_image"][0]) == 1  # Single-channel mean
