import asyncio
import schedule
from ..models.mcp_alchemist import mcp_alchemist
from ..quantum_sync import quantum_sync
from ..models.webxos_wallet import webxos_wallet
from ..security.oauth2 import get_current_user
from ..logging import logger

class AutoScheduler:
    def __init__(self):
        self.user = {"username": "system"}  # Default system user for automation
        schedule.every(10).minutes.do(self.sync_wallets)
        schedule.every(15).minutes.do(self.train_models)

    async def sync_wallets(self):
        logger.info("Auto-syncing wallets")
        await quantum_sync.broadcast_sync()
        wallet_status = webxos_wallet.get_wallet_status()
        logger.info(f"Wallet sync complete: {wallet_status}")

    async def train_models(self):
        logger.info("Auto-training models")
        data = "1,2,3,4,5,6,7,8,9,0"  # Sample data for automation
        result = await mcp_alchemist.train_wallet(data, self.user)
        if "error" not in result:
            logger.info(f"Training successful: {result}")
        else:
            logger.error(f"Training failed: {result}")

    async def run(self):
        while True:
            schedule.run_pending()
            await asyncio.sleep(1)

auto_scheduler = AutoScheduler()
