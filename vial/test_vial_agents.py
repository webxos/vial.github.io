import pytest
from unittest.mock import patch
from server.services.mcp_alchemist import Alchemist
from server.webxos_wallet import WebXOSWallet

@pytest.fixture
def alchemist():
    return Alchemist()

@pytest.fixture
def wallet():
    return WebXOSWallet()

def test_train_vial(alchemist):
    network_id = "54965687-3871-4f3d-a803-ac9840af87c4"
    vial_id = "vial1"
    result = alchemist.train_vial({"vial_id": vial_id, "network_id": network_id}, str(uuid.uuid4()))
    assert result["status"] == "trained"
    assert "request_id" in result

def test_coordinate_agents(alchemist, wallet):
    network_id = "54965687-3871-4f3d-a803-ac9840af87c4"
    wallet.import_wallet("""
# Vial Wallet Export
**Timestamp**: 2025-08-23T01:21:00Z
**User ID**: 54965687-3871-4f3d-a803-ac9840af87c4
**Wallet Address**: 54965687-3871-4f3d-a803-ac9840af87c4
**Balance**: 764.0000 $WEBXOS
**Vials**: vial1, vial2, vial3, vial4
""")
    result = alchemist.coordinate_agents({"network_id": network_id}, str(uuid.uuid4()))
    assert result["status"] == "coordinated"
    assert len(result["results"]) == 4
    assert "request_id" in result
