import pytest
from server.services.audit_log import AuditLog


@pytest.mark.asyncio
async def test_log_action():
    audit = AuditLog()
    response = await audit.log_action(
        action="test_action",
        user_id="test_user",
        details={"key": "value"}
    )
    assert response["status"] == "saved"
    assert "id" in response
