import pytest
import os
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.backup_restore import BackupRestore


@pytest.fixture
def client():
    return TestClient(app)


def test_backup_restore():
    backup = BackupRestore()
    temp_file = "temp_backup.db"
    backup.backup_database(temp_file)
    assert os.path.exists(temp_file)
    os.remove(temp_file)
