import pytest
from server.services.logging import Logger


@pytest.mark.asyncio
async def test_logger_info():
    logger = Logger("test_logger")
    await logger.info("Test info log", user_id="test_user")
    response = await logger.audit.log_action(
        action="test_log",
        user_id="test_user",
        details={"message": "Test info log"}
    )
    assert response["status"] == "saved"


@pytest.mark.asyncio
async def test_logger_error():
    logger = Logger("test_logger")
    await logger.error("Test error log", user_id="test_user")
    response = await logger.audit.log_action(
        action="test_log",
        user_id="test_user",
        details={"message": "Test error log"}
    )
    assert response["status"] == "saved"
