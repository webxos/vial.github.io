# server/tests/test_deployment_vps.py
import pytest
from fastapi.testclient import TestClient
from server.automation.deployment import deploy_application
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_deployment_vps():
    """Test VPS deployment with reputation validation."""
    app = TestClient(deploy_application.__self__)
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    result = deploy_application(app)
    assert result is True
    
    response = app.get("/health")
    assert response.status_code == 200
