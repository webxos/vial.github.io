# server/tests/test_terminal_commands.py
import pytest
from server.utils.terminal_commands import execute_terminal_command
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet


@pytest.mark.asyncio
async def test_terminal_commands():
    """Test terminal command execution with reputation check."""
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    result = execute_terminal_command("echo test")
    assert result["status"] == "success"
    assert "test" in result["output"]
