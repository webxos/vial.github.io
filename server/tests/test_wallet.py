import pytest
from unittest.mock import patch, AsyncMock
from server.webxos_wallet import WebXOSWallet, DAOProposal, WalletData

@pytest.fixture
def wallet():
    return WebXOSWallet()

@pytest.fixture
def proposal():
    return DAOProposal(proposal_id="prop1", description="Test proposal", votes={"vial1": True})

@pytest.mark.asyncio
async def test_multi_sig_verification(wallet):
    with patch("server.security.wallet_crypto.WalletCrypto.verify_multi_sig") as mock_sig:
        mock_sig.return_value = True
        result = await wallet.create_proposal(DAOProposal(proposal_id="prop1", description="Test", votes={}), ["sig1", "sig2"])
        assert result is True
        assert mock_sig.called

@pytest.mark.asyncio
async def test_wallet_export(wallet):
    with patch("pymongo.collection.Collection.find_one") as mock_find:
        mock_find.return_value = {"_id": "wallet1", "address": "0x123", "balance": 100.0}
        result = await wallet.export_wallet("wallet1", "password")
        assert "# Encrypted Wallet" in result

@pytest.mark.asyncio
async def test_dao_proposal(wallet, proposal):
    with patch("pymongo.collection.Collection.insert_one") as mock_insert:
        result = await wallet.create_proposal(proposal, ["sig1", "sig2"])
        assert result is True
        assert mock_insert.called
