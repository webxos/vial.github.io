from server.automation.auto_deploy import auto_deploy
from unittest.mock import patch


def test_auto_deploy():
    with patch("docker.APIClient") as mock_docker:
        mock_docker.return_value.containers.run.return_value = None
        result = auto_deploy.deploy("test_repo")
        assert result["status"] == "deployed"


def test_auto_deploy_failure():
    with patch("docker.APIClient") as mock_docker:
        mock_docker.return_value.containers.run.side_effect = Exception("Docker error")
        try:
            auto_deploy.deploy("test_repo")
        except ValueError as e:
            assert "Deployment failed" in str(e)
