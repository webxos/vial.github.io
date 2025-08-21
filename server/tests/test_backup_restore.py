from server.services.backup_restore import backup_restore
import pytest
from unittest.mock import patch
import subprocess

@patch("subprocess.run")
def test_backup(mock_subprocess):
    mock_subprocess.return_value = None
    result = backup_restore.backup()
    assert result["status"] == "backup created"
    assert "file" in result
    mock_subprocess.assert_called_once()

@patch("subprocess.run")
def test_restore(mock_subprocess):
    mock_subprocess.return_value = None
    result = backup_restore.restore("backups/vial_backup_20230820_123456.gz")
    assert result["status"] == "restore complete"
    mock_subprocess.assert_called_once()

@patch("subprocess.run")
def test_backup_failure(mock_subprocess):
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "mongodump")
    with pytest.raises(ValueError, match="Backup failed"):
        backup_restore.backup()
