from server.sdk.vial_sdk import vial_sdk
import pytest
from unittest.mock import patch

@patch("docker.from_env")
def test_sdk_initialize_system(mock_docker):
    mock_docker.return_value.containers.run.return_value = None
    result = vial_sdk.initialize_system({"username": "test_user"})
    assert result["status"] == "system initialized"
    assert "wallet_address" in result
    mock_docker.return_value.containers.run.assert_called_once()
