import pytest
from fastapi.testclient import TestClient
from server.main import app
from server.nasa_tools.nasa_orchestration import NASAOrchestrator

client = TestClient(app)
orchestrator = NASAOrchestrator()

@pytest.mark.asyncio
async def test_orchestrate_workflow():
    query = "earth"
    image_data = b"fake_image_data"
    result = await orchestrator.orchestrate_workflow(query, image_data)
    assert "datasets" in result
    assert len(result["datasets"]) > 0
    assert "correlations" in result
    assert "analysis" in result
    assert result["analysis"]["prediction"] is not None

@pytest.mark.asyncio
async def test_orchestrate_without_image():
    query = "mars"
    result = await orchestrator.orchestrate_workflow(query)
    assert "datasets" in result
    assert len(result["datasets"]) > 0
    assert "correlations" in result
    assert result["analysis"]["prediction"] is None
