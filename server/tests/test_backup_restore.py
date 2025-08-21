import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.backup_restore import BackupRestore


client = TestClient(app)


@pytest.mark.asyncio
async def test_backup_data():
    backup_restore = BackupRestore()
    response = await backup_restore.backup_data()
    assert response["status"] == "backup_completed"
    assert os.path.exists(response["file"])


@pytest.mark.asyncio
async def test_restore_data():
    backup_restore = BackupRestore()
    backup_response = await backup_restore.backup_data()
    backup_file = backup_response["file"]
    response = await backup_restore.restore_data(backup_file)
    assert response["status"] == "restore_completed"
