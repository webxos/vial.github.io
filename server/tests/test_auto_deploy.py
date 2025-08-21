from server.automation.auto_deploy import auto_deploy
import pytest
from unittest.mock import patch

@patch("subprocess.run")
@patch("docker.from_env")
def test_auto_deploy(mock_docker, mock_subprocess):
    mock_subprocess.return_value = None
    mock_docker.return_value.containers.run.return_value = None
    result = auto_deploy.deploy("app")
    assert result["status"] == "app deployed"
    mock_subprocess.assert_called_once()

@patch("subprocess.run")
def test_auto_deploy_failure(mock_subprocess):
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "docker-compose")
    with pytest.raises(ValueError, match="Deployment failed"):
        auto_deploy.deploy("app")
