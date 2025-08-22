# server/tests/test_agent_tasks.py
import pytest
from server.services.agent_tasks import AgentTasks
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_agent_task_execution():
    """Test agent task execution with reputation check."""
    agent_tasks = AgentTasks()
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    result = await agent_tasks.execute_task(
        "test_task",
        "test_wallet"
    )
    assert result["status"] == "success"
