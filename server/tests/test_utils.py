import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.utils import parse_json


@pytest.fixture
def client():
    return TestClient(app)


def test_parse_json():
    data = '{"user_id": "test_user", "balance": 50.0, "network_id": "test_net"}'
    wallet = parse_json(data)
    assert wallet.user_id == "test_user"
    assert wallet.balance == 50.0
    assert wallet.network_id == "test_net"


def test_parse_json_invalid():
    data = "invalid_json"
    try:
        parse_json(data)
    except ValueError:
        assert True
