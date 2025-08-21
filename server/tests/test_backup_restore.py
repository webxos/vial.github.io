from server.services.backup_restore import backup_restore
from unittest.mock import patch


def test_backup():
    with patch("gzip.open") as mock_gzip:
        mock_gzip.return_value.__enter__.return_value.write = None
        result = backup_restore.backup()
        assert result["status"] == "success"
        assert "file" in result


def test_restore():
    with patch("gzip.open") as mock_gzip:
        mock_gzip.return_value.__enter__.return_value.read.return_value = "{'collection': 'test'}"
        result = backup_restore.restore("test.gz")
        assert result["status"] == "success"


def test_restore_failure():
    with patch("gzip.open") as mock_gzip:
        mock_gzip.return_value.__enter__.side_effect = Exception("Restore error")
        result = backup_restore.restore("test.gz")
        assert result["status"] == "failed"
