# server/tests/test_prompt_training.py
import pytest
from server.services.prompt_training import train_prompt
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet


@pytest.mark.asyncio
async def test_prompt_training():
    """Test prompt training with reputation data."""
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    result = await train_prompt("test prompt")
    assert result["status"] == "success"
    assert "trained_model" in result
