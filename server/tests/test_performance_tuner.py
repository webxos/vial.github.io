# server/tests/test_performance_tuner.py
import pytest
from server.optimization.performance_tuner import PerformanceTuner
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet


@pytest.mark.asyncio
async def test_performance_tuner():
    """Test performance tuning with wallet reputation."""
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    tuner = PerformanceTuner()
    result = await tuner.optimize("test_model")
    assert result["status"] == "success"
    assert "optimized_metrics" in result
