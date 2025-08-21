from server.models.webxos_wallet import webxos_wallet
import pytest
from pymongo import MongoClient
from server.config import get_settings

settings = get_settings()

@pytest.fixture
def mongo_client():
    client = MongoClient(settings.MONGO_URL)
    yield client
    client.close()

def test_create_wallet():
    wallet = webxos_wallet.create_wallet("test_user")
    assert "address" in wallet
    assert wallet["user_id"] == "test_user"
    assert wallet["balance"] == 0.0

def test_update_balance(mongo_client):
    wallet = webxos_wallet.create_wallet("test_user")
    new_balance = webxos_wallet.update_balance(wallet["address"], 100.0)
    assert new_balance == 100.0
    db_wallet = mongo_client.vial.wallets.find_one({"address": wallet["address"]})
    assert db_wallet["balance"] == 100.0

def test_export_import_wallet():
    wallet = webxos_wallet.create_wallet("test_user")
    exported = webxos_wallet.export_wallet(wallet["address"])
    imported_address = webxos_wallet.import_wallet(exported)
    assert imported_address == wallet["address"]
