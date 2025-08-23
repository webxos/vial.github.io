import pytest
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

@pytest.fixture
def memory_manager():
    return MemoryManager()


@pytest.mark.asyncio
async def test_save_session(memory_manager):
    request_id = str(uuid.uuid4())
    token = "test_token"
    session_data = {"network_id": "test_net", "session_id": "test_sess"}
    await memory_manager.save_session(token, session_data, request_id)
    session = await memory_manager.get_session(token, request_id)
    assert session["network_id"] == "test_net"
    logger.info(f"Save session test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_save_wallet(memory_manager):
    request_id = str(uuid.uuid4())
    wallet_id = str(uuid.uuid4())
    wallet_data = {"network_id": "test_net", "balance": 100}
    await memory_manager.save_wallet(wallet_id, wallet_data, request_id)
    wallet = await memory_manager.get_wallet(wallet_id, request_id)
    assert wallet["balance"] == 100
    logger.info(f"Save wallet test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_save_tool(memory_manager):
    request_id = str(uuid.uuid4())
    tool_id = str(uuid.uuid4())
    tool_data = {"name": "agentic_search", "type": "crewai"}
    await memory_manager.save_tool(tool_id, tool_data, request_id)
    tool = await memory_manager.get_tool(tool_id, request_id)
    assert tool["name"] == "agentic_search"
    logger.info(f"Save tool test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_delete_wallet(memory_manager):
    request_id = str(uuid.uuid4())
    wallet_id = str(uuid.uuid4())
    wallet_data = {"network_id": "test_net", "balance": 100}
    await memory_manager.save_wallet(wallet_id, wallet_data, request_id)
    await memory_manager.delete_wallet(wallet_id, request_id)
    wallet = await memory_manager.get_wallet(wallet_id, request_id)
    assert wallet is None
    logger.info(f"Delete wallet test passed", request_id=request_id)
