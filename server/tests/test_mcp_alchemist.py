from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.mcp_alchemist import Alchemist


@pytest.fixture
def client():
    return TestClient(app)


def test_alchemist_process():
    alchemist = Alchemist()
    result = alchemist.process_prompt("Test prompt", {"data": "test"})
    assert result is not None
