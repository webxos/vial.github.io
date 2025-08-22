# server/tests/test_backup_restore.py
import pytest
from server.services.backup_restore import backup_data, restore_data
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

@pytest.mark.asyncio
async def test_backup_restore():
    """Test backup and restore of wallet data."""
    with SessionLocal() as session:
        wallet = Wallet(
            address="test_wallet",
            balance=100.0,
            staked_amount=50.0,
            reputation=20.0
        )
        session.add(wallet)
        session.commit()
    
    backup = await backup_data()
    assert len(backup["data"]) > 0
    assert backup["data"][0]["reputation"] == 20.0
    
    await restore_data(backup)
    with SessionLocal() as session:
        wallet = session.query(Wallet).filter_by(
            address="test_wallet"
        ).first()
        assert wallet.reputation == 20.0
