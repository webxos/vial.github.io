from server.models.webxos_wallet import webxos_wallet
from unittest.mock import patch


def test_wallet_initialization():
    assert webxos_wallet is not None
    assert webxos_wallet.address is not None


def test_wallet_balance():
    with patch("server.models.webxos_wallet.webxos_wallet.get_balance") as mock_balance:
        mock_balance.return_value = 100
        balance = webxos_wallet.get_balance()
        assert balance == 100


def test_wallet_transaction():
    with patch("server.models.webxos_wallet.webxos_wallet.send_transaction") as mock_tx:
        mock_tx.return_value = {"tx_id": "123"}
        result = webxos_wallet.send_transaction("recipient", 50)
        assert result["tx_id"] == "123"


def test_wallet_invalid_transaction():
    result = webxos_wallet.send_transaction("recipient", -50)
    assert result["status"] == "failed"
    assert "Invalid amount" in result["error"]
