import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
import subprocess
import os


@pytest.fixture
def client():
    return TestClient(app)


def test_vps_deployment():
    # Simulate VPS deployment with GitHub Pages
    try:
        subprocess.run(["git", "push", "origin", "main"], check=True)
        assert os.path.exists("public/index.html")
    except subprocess.CalledProcessError:
        pytest.fail("Git push to GitHub Pages failed")
    except FileNotFoundError:
        pytest.fail("public/index.html not found")


def test_docker_compose():
    result = subprocess.run(["docker-compose", "up", "-d"], capture_output=True, text=True)
    assert result.returncode == 0, f"Docker Compose failed: {result.stderr}"
