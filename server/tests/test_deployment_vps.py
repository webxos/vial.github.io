import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
import subprocess
import os
import requests


@pytest.fixture
def client():
    return TestClient(app)


def test_vps_deployment():
    try:
        subprocess.run(["git", "push", "origin", "main"], check=True)
        response = requests.get("http://vps-ip/public/index.html")
        assert response.status_code == 200
        assert "Vial MCP" in response.text
    except (subprocess.CalledProcessError, requests.RequestException):
        pytest.fail("VPS deployment or GitHub Pages sync failed")


def test_docker_compose_vps():
    result = subprocess.run(["docker-compose", "-f", "docker-compose.yml", "up", "-d"],
                           capture_output=True, text=True)
    assert result.returncode == 0, f"Docker Compose failed: {result.stderr}"
    subprocess.run(["docker-compose", "down"], capture_output=True)
