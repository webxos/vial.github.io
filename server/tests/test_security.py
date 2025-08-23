import pytest
from unittest.mock import patch
from server.security.crypto_engine import CryptoEngine, EncryptionParams
from server.security.key_manager import KeyManager
from server.security.wallet_crypto import WalletCrypto, WalletData

@pytest.fixture
def crypto_engine():
    return CryptoEngine()

@pytest.fixture
def wallet_data():
    return WalletData(address="0x123", balance=100.0, signatures=["sig1", "sig2"])

@pytest.mark.asyncio
async def test_encryption(crypto_engine):
    params = EncryptionParams(data=b"test data")
    encrypted = crypto_engine.encrypt(params)
    decrypted = crypto_engine.decrypt(encrypted)
    assert decrypted == b"test data"

@pytest.mark.asyncio
async def test_key_rotation():
    km = KeyManager()
    key1 = km.generate_key_pair(version=1)
    with patch("datetime.datetime") as mock_dt:
        mock_dt.utcnow.return_value = key1.created_at + timedelta(days=31)
        key2 = km.rotate_keys("test_key")
        assert key2.version == 2

@pytest.mark.asyncio
async def test_prompt_injection_defense():
    with pytest.raises(ValueError, match="Invalid input"):
        wc = WalletCrypto()
        wc.encrypt_wallet(WalletData(address="0x123", balance=100.0, signatures=["sig1"]), "malicious; DROP TABLE users")
