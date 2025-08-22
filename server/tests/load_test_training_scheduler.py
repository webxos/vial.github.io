# server/tests/load_test_training_scheduler.py
import pytest
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import asyncio

@pytest.mark.asyncio
async def test_load_training_scheduler():
    """Test load on training scheduler with reputation check."""
    vial_manager = VialManager()
    
    async def simulate_training(vial_id: str):
        with SessionLocal() as session:
            wallet = session.query(Wallet).filter_by(
                address="test_wallet"
            ).first()
            if not wallet or wallet.reputation < 5.0:
                return {"status": "failed", "reason": "low reputation"}
            return await vial_manager.train_vial(vial_id)
    
    tasks = [
        simulate_training(f"vial_{i}")
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    assert all(r["status"] == "success" for r in results)
