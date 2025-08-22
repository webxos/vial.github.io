# server/tests/test_performance_tuner.py
import pytest
from server.optimization.performace_tuner import PerformanceTuner
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_performance_tuner():
    """Test performance tuner with reputation check."""
    tuner = PerformanceTuner()
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    result = await tuner.optimize_training("test_vial")
    assert result["status"] == "success"
    assert "performance" in result
