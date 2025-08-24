import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch

client = TestClient(app)

@pytest.mark.asyncio
async def test_rbac_access_granted():
    """Test RBAC access for authorized user."""
    with patch("server.utils.rbac.get_current_user_roles", return_value=["admin"]):
        response = client.post("/mcp/tools/servicenow", json={"endpoint": "table/incident", "params": {}})
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_rbac_access_denied():
    """Test RBAC access denial for unauthorized user."""
    with patch("server.utils.rbac.get_current_user_roles", return_value=["user"]):
        response = client.post("/mcp/quantum", json={"qubits": 2, "circuit": {}})
        assert response.status_code == 403
        assert response.json() == {"detail": "Insufficient permissions"}
